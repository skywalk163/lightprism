# -*- coding: utf-8 -*-
"""train_openi.py v3 的纯标准库回归：门禁、分组切分、判据口径。

不依赖 torch：用 importlib 直接加载脚本（顶层 import 都是标准库，
torch 只在 main() 里延迟导入）。CI 的纯标准库门禁也能跑。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "train_openi.py"
spec = importlib.util.spec_from_file_location("train_openi_v3", SCRIPT)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

POINTS = ["CTX", "EDIT", "FIX", "NEXT", "PLAN", "RETRY", "SAFE"]


def _per_point(points, acc=0.85, n_test=6):
    out = {}
    for p in POINTS:
        if p in points:
            out[p] = {"status": "ok", "n_classes": 3, "random_baseline": 0.3333,
                      "n_train": 100, "n_test": n_test, "accuracy": acc,
                      "eval_split": "heldout_test"}
        else:
            out[p] = {"status": "no_samples", "n_classes": 3,
                      "random_baseline": 0.3333}
    return out


def _metrics(points, c_macro, a_macro, n_test_rows, sr_all,
             exact_c, exact_b, overlap, const_feature, has_err=True):
    return {
        "C": {
            "probe": {
                "macro_accuracy": c_macro, "points_with_test": points,
                "per_point": _per_point(points),
                "feature_is_constant": const_feature,
                "feature_prev_token_distribution": {"</TASK>": 7}
                if const_feature else {"<text>": 40},
                "self_position_control": {"macro_accuracy": 1.0, "per_point": {}},
                "params": 579, "n_train_rows": 100, "n_test_rows": n_test_rows,
            },
            "gen": {
                "json": {"n": 42, "exact": exact_c,
                         "exact_rate": round(exact_c / 42, 4)},
                "free": {"structure_rate": 0.99, "structure_rate_all": sr_all,
                         "n_all": 52, "n_counted": 50, "n_truncated": 2,
                         "n_struct_ok_all": round(sr_all * 52),
                         "tokens_per_s": 100.0},
                "decision_density": 0.0020, "speed_drop_vs_B": 0.010,
            },
        },
        "A": {"macro_accuracy": a_macro, "per_point": {},
              "points_with_test": points, "params": 390419},
        "B": {"gen": {"json": {"n": 42, "exact": exact_b,
                               "exact_rate": round(exact_b / 42, 4)},
                      "free": {"structure_rate": None, "tokens_per_s": 101.0}}},
        "shuffle": {"probe": {"macro_accuracy": 0.35, "per_point": {}}},
        "coverage": {
            "points_with_test_labels": points,
            "n_points_with_test_labels": len(points),
            "n_test_decision_rows": n_test_rows,
            "validation_codes_seen": ["PASS", "ERR=RUNTIME"] if has_err else ["PASS"],
            "has_err_validation": has_err,
            "split_diagnostics": {"train_test_template_overlap": overlap,
                                  "group_split": True},
        },
    }


CFG = {"decision_policy": "real", "min_decisive_gap": 3}


def _result(checks, prefix):
    for c in checks:
        if c["criterion"].startswith(prefix):
            return c["result"]
    raise AssertionError(f"缺判据 {prefix}")


# ---- ① 门禁：v2 的 fail 必须降级为 incomplete ----

def test_v2_numbers_are_incomplete_not_fail():
    """v2 OpenI 原始数字回填 → 门禁命中，整体 incomplete，① 记 unknown。"""
    m = _metrics(["EDIT"], 0.7143, 1.0, 7, 0.5385, 8, 7, 6, True, has_err=False)
    r = t.judge(m, CFG)
    assert r["verdict"] == "incomplete"
    assert r["gated"] is True
    assert len(r["gate_blockers"]) >= 4
    assert _result(r["checks"], "①") == "unknown"
    # ③ 用全分母后确实低于阈值，这是真实测量（不是门禁 artifact），仍是 fail
    assert _result(r["checks"], "③") == "fail"
    assert "不要" in r["next_step"] and "调超参" in r["next_step"]


def test_gate_blockers_are_specific():
    m = _metrics(["EDIT"], 0.7143, 1.0, 7, 0.5385, 8, 7, 6, True, has_err=False)
    r = t.judge(m, CFG)
    joined = "；".join(r["gate_blockers"])
    assert "1/7" in joined
    assert "< 30" in joined
    assert "模板簇重合" in joined
    assert "ERR" in joined
    assert "同一个 id" in joined


def test_gate_does_not_silence_3_and_4():
    """v3.1 回归：门禁只封 ①②；③④ 不依赖决策标签，仍是有效测量。

    所以被门禁时 ③ 可以是 fail、④ 可以是 pass，只有综合结论降为 incomplete。
    """
    m = _metrics(["EDIT"], 0.7143, 1.0, 7, 0.5385, 8, 7, 6, True, has_err=False)
    r = t.judge(m, CFG)
    results = {c["criterion"][:2]: c["result"] for c in r["checks"]}
    assert results["③ "] == "fail"      # 0.5385 < 0.95，真实测量
    assert results["④ "] == "pass"      # 密度/速度正常
    assert r["verdict"] == "incomplete"  # 综合结论仍无判别力
    # caveat 必须说明 ③④ 是有效测量，而不是「四条判据无一具备判别力」
    joined = " ".join(r["caveats"])
    assert "③④" in joined and "有效测量" in joined


# ---- ② 条件补齐后：全绿才 pass，真差才 fail ----

def test_all_conditions_met_passes():
    m = _metrics(POINTS, 0.85, 0.80, 40, 0.97, 20, 10, 0, False)
    r = t.judge(m, CFG)
    assert r["verdict"] == "pass"
    assert r["gated"] is False
    assert all(c["result"] == "pass" for c in r["checks"])


def test_real_failure_still_fails():
    """门禁不该把「真失败」也洗成 incomplete。"""
    m = _metrics(POINTS, 0.50, 0.80, 40, 0.90, 20, 10, 0, False)
    r = t.judge(m, CFG)
    assert r["verdict"] == "fail"
    assert r["gated"] is False
    assert _result(r["checks"], "①") == "fail"
    assert _result(r["checks"], "③") == "fail"


# ---- ③ 判据 ② 的擦边保护 ----

def test_one_sample_gap_is_unknown():
    """v2 的 8/42 vs 7/42 差 1 条，必须记 unknown 而不是 pass。"""
    m = _metrics(POINTS, 0.85, 0.80, 40, 0.97, 8, 7, 0, False)
    r = t.judge(m, CFG)
    assert _result(r["checks"], "②") == "unknown"
    assert r["verdict"] == "incomplete"


def test_rule_policy_locks_evidence_level():
    m = _metrics(POINTS, 0.85, 0.80, 40, 0.97, 20, 10, 0, False)
    r = t.judge(m, {**CFG, "decision_policy": "rule"})
    assert r["evidence_level"] == "mechanism_only"
    assert r["verdict"] == "incomplete"  # 证据不足不升级 pass


# ---- ④ 分组切分：模板簇不跨 split ----

def _payload_vocab_rows():
    import argparse
    p = t._load_payload(argparse.Namespace(data_dir="", dump_data=""))
    return t.Vocab(p["vocab"]), t.load_samples(p)


def test_group_split_zero_template_overlap():
    vocab, rows = _payload_vocab_rows()
    cov = {}
    splits = t.stratified_split(rows, 0, group_split=True, coverage=cov)
    assert cov["train_test_template_overlap"] == 0
    assert cov["val_test_template_overlap"] == 0
    assert cov["group_split"] is True
    assert sum(len(v) for v in splits.values()) == len(rows)


def test_plain_split_has_overlap():
    """保留 v2 行为作为反面证据：逐样本切分确实会漏模板。"""
    vocab, rows = _payload_vocab_rows()
    cov = {}
    t.stratified_split(rows, 0, group_split=False, coverage=cov)
    assert cov["train_test_template_overlap"] > 0


# ---- ⑤ build_ids：scattered 让 pos-1 不再是常数 ----

def _prev_token_ids(mode):
    import random
    vocab, rows = _payload_vocab_rows()
    splits = t.stratified_split(rows, 0, group_split=True)
    ids = set()
    for name in splits:
        for i, r in enumerate(splits[name]):
            s = t.build_ids(r, vocab, "real", decision_position=mode,
                            position_rng=random.Random(5000 + i))[0]
            for k, tid in enumerate(s):
                if k > 0 and tid in vocab.decision_ids:
                    ids.add(s[k - 1])
    return ids


def test_fixed_decision_position_has_single_prev_id():
    assert len(_prev_token_ids("fixed")) == 1


def test_scattered_decision_position_breaks_constant_feature():
    """v3.1 回归：scattered 必须让 pos-1 出现多个不同 token id。

    初版取 task_tail[-k:]，本库所有 task 均以「输出 JSON。」结尾，
    实测 48/48 条前一 token 都是同一个 ``。``，等于空转。
    """
    ids = _prev_token_ids("scattered")
    assert len(ids) > 1, f"scattered 仍是常数：{ids}"
    # 不应再退化成末尾固定标点
    assert len(ids) >= 5


def test_describe_prev_reflects_token_identity():
    """v3.1 回归：不同文本 token 必须产生不同描述，不能全折叠成 '<text>'。"""
    vocab, rows = _payload_vocab_rows()
    splits = t.stratified_split(rows, 0, group_split=True)
    import random
    desc = set()
    for name in splits:
        for i, r in enumerate(splits[name]):
            s = t.build_ids(r, vocab, "real", decision_position="scattered",
                            position_rng=random.Random(5000 + i))[0]
            for k, tid in enumerate(s):
                if k > 0 and tid in vocab.decision_ids:
                    desc.add(t.describe_prev(s, k, vocab))
    assert len(desc) > 1, f"describe_prev 仍折叠成单一描述：{desc}"
    assert "<text>" not in desc


def test_feature_is_constant_uses_real_id_count():
    """v3.1 回归：常量哨兵按真实 id 种类数判定，而不是描述字符串数。"""
    import random
    vocab, rows = _payload_vocab_rows()
    splits = t.stratified_split(rows, 0, group_split=True)
    seqs = {k: [t.build_ids(r, vocab, "real", decision_position="scattered",
                            position_rng=random.Random(5000 + i))[0]
                for i, r in enumerate(splits[k])] for k in splits}
    ids = set()
    for s in seqs["test"]:
        for k, tid in enumerate(s):
            if k > 0 and tid in vocab.decision_ids:
                ids.add(s[k - 1])
    assert len(ids) > 1


# ---- ⑧ 几何检查点：字段语义（v3.1 改名 + tied embedding 说明） ----

def test_embedding_geometry_field_semantics():
    """纯标准库检查字段命名与自洽性（不跑 torch）。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert '"n_decision_tokens_seen_as_input":' in src
    assert '"n_decision_tokens_never_seen_as_input":' in src
    # 旧字段名可以出现在「解释为何改名」的注释里，但不能作为返回字典的 key
    assert '"n_decision_tokens_never_trained":' not in src
    assert '"cosine_dec_vs_text_seen_as_input":' in src
    assert '"tied_embedding":' in src
    # 必须说明 tied embedding 下 never_seen_as_input 不等于权重没动
    assert "不等于权重没动" in src


# ---- ⑥ 判据 ③ 用全分母 ----

def test_structure_rate_all_counts_truncation_as_failure():
    """截断生成必然结构非法，必须计入分母。"""
    vocab, rows = _payload_vocab_rows()
    splits = t.stratified_split(rows, 0, group_split=True)
    seqs = {k: [t.build_ids(r, vocab, "real")[0] for r in splits[k]] for k in splits}
    # 构造一个未闭合 CODE 段，check_structure 必判非法
    gen = [vocab.structure["<CODE>"]] + [vocab.unk_id] * 5
    ok, problems = t.check_structure(gen, vocab, True)
    assert not ok
    assert any("CODE 段未闭合" in x for x in problems)


# ---- ⑦ 覆盖率诊断 ----

def test_label_coverage_surfaces_partial_coverage():
    vocab, rows = _payload_vocab_rows()
    splits = t.stratified_split(rows, 0, group_split=True)
    seqs = {k: [t.build_ids(r, vocab, "real")[0] for r in splits[k]] for k in splits}
    cov = t.label_coverage(splits, seqs, vocab)
    assert cov["n_points_with_test_labels"] < 7
    assert cov["has_err_validation"] is False
    assert cov["validation_codes_seen"] == ["PASS"]
