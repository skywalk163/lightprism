# -*- coding: utf-8 -*-
"""T-0494：真实决策接进 train_openi 的纯标准库回归。

覆盖任务卡全部接线点：

1. payload 的 ``decisions`` 键（内嵌 + --data-dir 两条路径）；
2. 旧 payload 缺 decisions 时行为完全不变（向后兼容）；
3. embed_payload.py 打包 decisions 且 --check 比对；
4. load_samples 把 241 条决策记录转成样本，字段缺失 / 未登记验证码报错；
5. build_ids 的 real 策略原样注入自带 decisions、逐点校验词表、
   无该字段退回旧逻辑；
6. sha256 确定性限量：每样本 1 点、可复现、密度 ≤ DENSITY_MAX；
7. FIX 稀有点在 train/test 两侧都有正样本（不靠 seed 试）；
8. --group-split 构造性 overlap=0、样本总数守恒；
9. 门禁解除：覆盖 7 点 + test 行 ≥30 + overlap=0 → gated=false，
   且 verdict 如实（该 fail 仍 fail，未改判据阈值）；
10. data/runtime 与 data/unified 只读（快照字节对齐）。

不依赖 torch（importlib 直接加载脚本，torch 只在 main 里延迟导入）。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "train_openi.py"
spec = importlib.util.spec_from_file_location("train_openi_wire", SCRIPT)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

POINTS = ["CTX", "EDIT", "FIX", "NEXT", "PLAN", "RETRY", "SAFE"]


def _args(data_dir: str = "") -> argparse.Namespace:
    return argparse.Namespace(data_dir=data_dir, dump_data="")


def _decisions_text() -> str:
    return (ROOT / "data" / "unified" / "decisions.jsonl").read_text(encoding="utf-8")


def _rows_count(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip())


# ---- 1. payload decisions 键 ----------------------------------------------

def test_payload_has_decisions_on_both_paths():
    embedded = t._load_payload(_args(""))
    external = t._load_payload(_args(str(ROOT / "data" / "unified")))
    for payload in (embedded, external):
        assert "decisions" in payload
        assert _rows_count(payload["decisions"]) == 241
        assert payload["meta"]["decisions_sha256"] == hashlib.sha256(
            payload["decisions"].encode("utf-8")).hexdigest()
    assert embedded["decisions"] == external["decisions"]


def test_old_payload_without_decisions_is_backward_compatible():
    """旧 payload 缺 decisions 键：补空串、不报错、不产生决策样本。"""
    vocab = json.loads((ROOT / "data" / "unified" / "vocab_4096.json").read_text(
        encoding="utf-8"))
    old = {
        "vocab": vocab,
        "json_tasks": "",
        "code_tasks": "",
        "err_tasks": "",
        # 故意没有 decisions
    }
    source = SCRIPT.read_text(encoding="utf-8")
    # 直接验证 _load_payload 的 setdefault 分支：用 monkeypatch 的 B64 太重，
    # 这里验证 load_samples 对缺键的处理（=不产生决策样本、不抛异常）
    rows = t.load_samples(dict(old))
    assert all(not r.get("_decision_row") for r in rows)
    assert "setdefault(\"decisions\", \"\")" in source


# ---- 2. embed_payload 打包与 --check --------------------------------------

def test_embed_payload_packs_decisions_and_checks():
    spec_e = importlib.util.spec_from_file_location(
        "embed_payload", ROOT / "scripts" / "embed_payload.py")
    ep = importlib.util.module_from_spec(spec_e)
    spec_e.loader.exec_module(ep)
    payload = ep.load_payload()
    assert _rows_count(payload["decisions"]) == 241
    assert payload["decisions"] == _decisions_text()
    assert payload["meta"]["decisions_sha256"] == hashlib.sha256(
        _decisions_text().encode("utf-8")).hexdigest()
    # 内嵌 payload 与磁盘一致（--check 的非写入等价断言）
    embedded = t._load_payload(_args(""))
    assert embedded["decisions"] == payload["decisions"]


# ---- 3. load_samples 接线与报错 -------------------------------------------

def test_load_samples_converts_241_decision_rows():
    payload = t._load_payload(_args(str(ROOT / "data" / "unified")))
    rows = t.load_samples(payload)
    dec = [r for r in rows if r.get("_decision_row")]
    assert len(dec) == 241
    r = dec[0]
    assert r["_category"] == "runtime_decision"
    assert r["meta"]["template"]
    s = r["steps"][0]
    assert s["code"] and s["validation"] and s["decisions"]
    # 限量字段已就位（每样本 1 点）
    assert len(s["_kept_decisions"]) == 1
    assert s["_kept_decisions"] == {p: s["decisions"][p]
                                    for p in s["_kept_decisions"]}


def test_load_samples_rejects_missing_fields_and_unknown_validation():
    vocab = t._load_payload(_args(""))["vocab"]
    base = json.loads(_decisions_text().splitlines()[0])
    good = dict(base)

    def _payload(rec):
        return {"vocab": vocab, "json_tasks": "", "code_tasks": "", "err_tasks": "",
                "decisions": json.dumps(rec, ensure_ascii=False)}

    with pytest.raises(ValueError, match="缺字段"):
        bad = dict(base); bad.pop("source")
        t.load_samples(_payload(bad))
    with pytest.raises(ValueError, match="未登记的验证码"):
        bad = dict(base); bad["validation_code"] = "ERR=NOT_IN_VOCAB"
        t.load_samples(_payload(bad))
    # 空 decisions 也会被缺字段检查拦下（空值 falsy），同样拒绝静默丢行
    with pytest.raises(ValueError, match="缺字段|不是非空 dict"):
        bad = dict(base); bad["decisions"] = {}
        t.load_samples(_payload(bad))
    # 正常记录不报错
    assert any(r.get("_decision_row") for r in t.load_samples(_payload(good)))


# ---- 4. build_ids real 策略 ------------------------------------------------

def test_build_ids_real_injects_own_decisions_and_validates_vocab():
    payload = t._load_payload(_args(str(ROOT / "data" / "unified")))
    v = t.Vocab(payload["vocab"])
    rows = t.load_samples(payload)
    dec = next(r for r in rows if r.get("_decision_row"))
    ids, labels = t.build_ids(dec, v, "real")
    assert labels == dec["steps"][0]["_kept_decisions"]
    assert sum(1 for i in ids if i in v.decision_ids) == len(labels)
    # 非法 point 直接报错
    bad_point = json.loads(json.dumps(dec))
    bad_point["steps"][0]["_kept_decisions"] = {"BOGUS": "X"}
    with pytest.raises(ValueError, match="不在词表 point_token_ids"):
        t.build_ids(bad_point, v, "real")
    # 非法 option 直接报错
    bad_opt = json.loads(json.dumps(dec))
    p = next(iter(bad_opt["steps"][0]["_kept_decisions"]))
    bad_opt["steps"][0]["_kept_decisions"] = {p: "NOT_AN_OPTION"}
    with pytest.raises(ValueError, match="不在封闭集合里"):
        t.build_ids(bad_opt, v, "real")


def test_build_ids_real_without_own_decisions_falls_back():
    """无自带 decisions 字段时退回旧逻辑（decision 类答案 JSON 取 EDIT）。"""
    v = t.Vocab(t._load_payload(_args(""))["vocab"])
    sample = {"task": "输出 JSON。", "steps": [{"code": '{"decision": "RW"}'}],
              "meta": {"category": "decision"}}
    _, labels = t.build_ids(sample, v, "real")
    assert labels == {"EDIT": "RW"}
    sample2 = {"task": "输出 JSON。", "steps": [{"code": "x = 1"}],
               "meta": {"category": "code"}}
    assert t.build_ids(sample2, v, "real")[1] == {}


# ---- 5. 确定性限量与密度 ---------------------------------------------------

def test_decision_budget_is_deterministic_and_under_density_cap():
    payload = t._load_payload(_args(str(ROOT / "data" / "unified")))
    v = t.Vocab(payload["vocab"])
    a = t.load_samples(payload)
    b = t.load_samples(payload)
    ba = a and payload["meta"]["decision_budget"]
    assert ba["decision_tokens_kept"] == 241  # 241 条 × 1 点
    for ra, rb in zip(a, b):
        if ra.get("_decision_row"):
            assert ra["steps"][0]["_kept_decisions"] == rb["steps"][0]["_kept_decisions"]
    seqs = {k: [t.build_ids(r, v, "real")[0] for r in
                t.stratified_split(a, 0, group_split=True)[k]]
            for k in ("train", "val", "test")}
    st = t.dataset_stats(seqs["train"], v)
    assert st["decision_density"] <= t.DENSITY_MAX
    assert st["decision_density_ok"] is True
    assert t.DENSITY_MAX == 0.02  # 阈值本身不许被改


# ---- 6. 稀有点双侧覆盖（构造性，不靠 seed） --------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_rare_fix_point_has_train_and_test_samples_across_seeds(seed):
    payload = t._load_payload(_args(str(ROOT / "data" / "unified")))
    v = t.Vocab(payload["vocab"])
    rows = t.load_samples(payload)
    splits = t.stratified_split(rows, seed, group_split=True)
    for side in ("train", "test"):
        n = sum(1 for r in splits[side]
                if r.get("_decision_row")
                and "FIX" in r["steps"][0].get("_kept_decisions", {}))
        assert n >= 1, f"seed={seed} {side} 侧无 FIX 正样本，探针无法 fit head"


# ---- 7. group-split 构造性零重合 + 样本守恒 --------------------------------

def test_group_split_zero_overlap_preserves_all_samples():
    payload = t._load_payload(_args(str(ROOT / "data" / "unified")))
    rows = t.load_samples(payload)
    cov: dict = {}
    splits = t.stratified_split(rows, 0, group_split=True, coverage=cov)
    assert cov["train_test_template_overlap"] == 0
    assert cov["val_test_template_overlap"] == 0
    # 样本守恒：切分前后行数一致（external data-dir 的 err_tasks 更全，
    # 实测 629 = 300 json + 63 code + 25 err + 241 decisions）
    assert sum(len(v) for v in splits.values()) == len(rows)
    assert len(rows) == 629


# ---- 8. 门禁解除 + verdict 如实 --------------------------------------------

def test_gate_clears_but_verdict_stays_truthful():
    """覆盖 7 点 / test 行 ≥30 / overlap=0 -> gated=false；③ fail 仍是 fail。"""
    metrics = {
        "C": {
            "probe": {
                "macro_accuracy": 0.6667,
                "points_with_test": POINTS,
                "per_point": {p: {"random_baseline": 0.3333,
                                  "majority_baseline": 0.4,
                                  "n_test": 30, "n_train": 100}
                              for p in POINTS},
                "feature_is_constant": False,
                "feature_hidden_std": 0.6,
                "macro_majority_baseline": 0.4,
                "n_train_rows": 100, "n_test_rows": 88,
            },
            "gen": {
                "json": {"n": 42, "exact": 8, "exact_rate": 0.1905},
                "free": {"structure_rate_all": 0.5385, "n_all": 52,
                         "n_truncated": 2},
                "decision_density": 0.0043, "speed_drop_vs_B": 0.01,
            },
        },
        "A": {"macro_accuracy": 0.8889},
        "B": {"gen": {"json": {"n": 42, "exact": 7, "exact_rate": 0.1667}}},
        "shuffle": {"probe": {"macro_accuracy": 0.35}},
        "coverage": {
            "points_with_test_labels": POINTS,
            "n_points_with_test_labels": 7,
            "n_test_decision_rows": 88,
            "validation_codes_seen": ["PASS", "ERR=RUNTIME"],
            "has_err_validation": True,
            "split_diagnostics": {"train_test_template_overlap": 0,
                                  "group_split": True},
        },
    }
    r = t.judge(metrics, {"decision_policy": "real", "min_decisive_gap": 3})
    assert r["gated"] is False
    assert r["gate_blockers"] == []
    assert r["evidence_level"] == "real_labels"
    # ③ 0.5385 < 0.95 是真实测量：fail 不因门禁解除而被吞掉
    c3 = next(c for c in r["checks"] if c["criterion"].startswith("③"))
    assert c3["result"] == "fail"
    assert r["verdict"] == "fail"  # 如实，不是 pass
    # 阈值常量未被改动
    assert t.MIN_PROBE_TEST_ROWS == 30
    assert t.MIN_DECISIVE_GAP == 3


# ---- 9. 数据快照只读（字节对齐） -------------------------------------------

def test_runtime_and_unified_decisions_are_byte_identical():
    rt = (ROOT / "data" / "runtime" / "decisions.jsonl").read_bytes()
    un = (ROOT / "data" / "unified" / "decisions.jsonl").read_bytes()
    assert rt.replace(b"\r\n", b"\n") == un.replace(b"\r\n", b"\n")
    assert _rows_count(un.decode("utf-8")) == 241
