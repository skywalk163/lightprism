# -*- coding: utf-8 -*-
"""T-0330 demo 4096 实词表验收测试。

覆盖卡面「完成标准」四项：总数上限、L0 原子性、控制 token 对齐、UNK 率；
另加重复 token、分层元数据、``--check`` 端到端、以及 UNK 函数的判别力
（避免「恒返回 0」式假绿）。

注：本文件不在卡面 ``paths`` 列表里，但「完成标准」与 ``verify`` 命令都直接
要求它存在。交付说明见 ``.hive/summary.md`` / ``reports/词表报告.md``。
"""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from lightprism.demo import UNK_LIMIT, VOCAB_LIMIT, check_vocab, unk_rate
from lightprism.demo.vocab_build import (
    VOCAB64_SLOT_ORDER,
    VOCAB64_TOTAL,
    check_vocab_64k,
    coverage_compare,
    slot_quotas,
)
from lightprism.tokens import LIGHT_L0_CORE, VALIDATION_CODES, TokenSpace
from lightprism.tokenizer import STRUCTURE_MARKERS, ControlTokenLayout
from lightprism.vocab import DEFAULT_BUDGET, VocabBudget, check_l0_atomicity

ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "data" / "vocab_4096.json"
INDEX_PATH = ROOT / "data" / "corpus" / "index.jsonl"


@pytest.fixture(scope="module")
def vocab() -> dict:
    assert VOCAB_PATH.is_file(), f"缺少词表产物 {VOCAB_PATH}"
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def _expected_control() -> list[str]:
    space = TokenSpace()
    return (list(STRUCTURE_MARKERS)
            + [t.text for t in space.decision_tokens()]
            + [f"<V:{code}>" for code in VALIDATION_CODES])


# ---- 1. 总数上限与留白 -------------------------------------------------
def test_total_within_4096_and_headroom(vocab):
    tokens = vocab["tokens"]
    assert len(tokens) <= VOCAB_LIMIT
    assert len(tokens) == vocab["size"]
    # 卡面明确「不要把 4096 塞满，留 15% 余量」
    assert (VOCAB_LIMIT - len(tokens)) / VOCAB_LIMIT >= 0.15
    assert vocab["headroom"] == VOCAB_LIMIT - len(tokens)


# ---- 2. L0 原子性 -------------------------------------------------------
def test_l0_atomicity(vocab):
    tokens = vocab["tokens"]
    assert len(LIGHT_L0_CORE) == 44
    for ch in LIGHT_L0_CORE:
        assert tokens.count(ch) == 1, f"L0 字 {ch!r} 必须恰好占 1 个 token"
    problems = check_l0_atomicity(tokens)
    assert problems == [], f"L0 字被并入多字 token: {problems[:5]}"


def test_no_l0_char_inside_multichar_token(vocab):
    """把硬约束单独钉死：任何长度 >1 的 token 都不许含 L0 字。"""
    core = set(LIGHT_L0_CORE)
    bad = [t for t in vocab["tokens"] if len(t) > 1 and any(c in core for c in t)]
    assert bad == [], f"含 L0 字的多字 token: {bad[:5]}"


# ---- 3. 控制 token 与 TokenSpace 对齐 -----------------------------------
def test_control_tokens_align_with_tokenspace(vocab):
    tokens = vocab["tokens"]
    layout = vocab["control_layout"]
    expected = _expected_control()
    assert len(STRUCTURE_MARKERS) == 4
    assert len(TokenSpace().decision_tokens()) == 19
    assert len(VALIDATION_CODES) == len(TokenSpace().validation_tokens()) == 9

    start = layout["start"]
    assert tokens[start:] == expected, "控制 token 文本/数量/顺序必须与 TokenSpace 一致"
    assert layout["end"] == len(tokens)
    assert layout["size"] == len(expected)

    # id 区间必须连续，且与 ControlTokenLayout 的布局推导完全一致
    ref = ControlTokenLayout(start=start)
    ref.structure = {t: start + i for i, t in enumerate(STRUCTURE_MARKERS)}
    cursor = start + len(STRUCTURE_MARKERS)
    ref.decisions = {t.text: cursor + i
                     for i, t in enumerate(TokenSpace().decision_tokens())}
    cursor += len(ref.decisions)
    ref.validations = {f"<V:{c}>": cursor + i for i, c in enumerate(VALIDATION_CODES)}
    ref.finalize()
    assert layout["structure"] == ref.structure
    assert layout["decisions"] == ref.decisions
    assert layout["validations"] == ref.validations
    ids = (list(layout["structure"].values()) + list(layout["decisions"].values())
           + list(layout["validations"].values()))
    assert sorted(ids) == list(range(start, layout["end"])), "控制 id 区间有空洞/越界"


# ---- 4. UNK 率 ----------------------------------------------------------
def test_unk_rate_below_1_percent(vocab):
    freq = vocab["corpus_stats"]["char_freq"]
    rate, unk, total = unk_rate(freq, vocab["tokens"])
    assert total > 0
    assert rate < UNK_LIMIT, f"UNK {rate:.4%} 未达 < 1%"
    assert vocab["corpus_stats"]["total_chars"] == total


def test_unk_rate_is_discriminating():
    """判据必须有牙：删单字 token 会让 UNK 率上升，而不是恒为 0。"""
    freq = {"a": 90, "b": 10}
    assert unk_rate(freq, ["a", "b"])[0] == 0.0
    assert unk_rate(freq, ["a"])[0] == pytest.approx(0.1)
    assert unk_rate(freq, [])[0] == 1.0
    assert unk_rate({}, ["a"])[0] == 0.0


# ---- 5. 元数据自洽与端到端 ---------------------------------------------
def test_no_duplicate_tokens_and_layer_meta(vocab):
    tokens, kinds = vocab["tokens"], vocab["kinds"]
    assert len(tokens) == len(kinds)
    assert len(set(tokens)) == len(tokens), f"重复 token: {[t for t, n in Counter(tokens).items() if n > 1][:5]}"
    counts = vocab["counts"]
    assert set(counts) == set(vocab["layer_names"])
    assert sum(counts.values()) == len(tokens)
    for layer, n in counts.items():
        assert kinds.count(layer) == n
        assert len(vocab["layers"][layer]) == n


def test_corpus_index_fields(vocab):
    assert INDEX_PATH.is_file()
    rows = [json.loads(line) for line in
            INDEX_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert set(("src_path", "bytes", "md5", "kept", "reason_if_dropped")) <= set(rows[0])
    assert sum(1 for r in rows if r["kept"]) == vocab["corpus_stats"]["files_kept"]
    assert sum(1 for r in rows if not r["kept"]) == vocab["corpus_stats"]["files_dropped"]
    assert all(r["reason_if_dropped"] for r in rows if not r["kept"])
    assert len(rows) == vocab["corpus_stats"]["files_scanned"]


def test_check_mode_is_full_green():
    good, lines = check_vocab(VOCAB_PATH, ROOT / "data" / "corpus")
    assert good, "\n".join(lines)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(
        [sys.executable, "-m", "lightprism.demo.vocab_build",
         "--check", str(VOCAB_PATH)],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300)
    assert p.returncode == 0, f"--check 退出码 {p.returncode}\n{p.stdout}\n{p.stderr}"
    assert "全绿" in p.stdout


# =========================================================================
# T-0373 · 64K 实词表（DEFAULT_BUDGET 落地）
#
# 这一组是 64K 卡的验收判据：总数精确 65536、八槽逐项等于预算 recommended、
# 区间全覆盖无重叠、末尾控制块与 TokenSpace 对齐、manifest 逐槽 sha256 可复算、
# 字符级 UNK 为 0（可见字符口径）、以及 config 的 V 与词表实际尺寸一致。
# =========================================================================
VOCAB64_PATH = ROOT / "data" / "vocab_64k.json"
MANIFEST64_PATH = ROOT / "data" / "vocab_64k.manifest.json"
CONFIG64_PATH = ROOT / "configs" / "demo-3m-64k.json"
CONFIG4096_PATH = ROOT / "configs" / "demo-3m.json"


@pytest.fixture(scope="module")
def vocab64() -> dict:
    assert VOCAB64_PATH.is_file(), f"缺少 64K 词表产物 {VOCAB64_PATH}"
    return json.loads(VOCAB64_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest64() -> dict:
    assert MANIFEST64_PATH.is_file(), f"缺少 64K manifest {MANIFEST64_PATH}"
    return json.loads(MANIFEST64_PATH.read_text(encoding="utf-8"))


def test_vocab64_total_is_exactly_65536(vocab64):
    tokens = vocab64["tokens"]
    assert vocab64["size"] == VOCAB64_TOTAL == 65536
    assert len(tokens) == 65536
    assert len(set(tokens)) == len(tokens), "64K 词表存在重复 token"
    assert vocab64["schema_version"] == "1.0"


def test_vocab64_slots_match_default_budget(vocab64):
    """八个槽位的 count 必须逐项等于 DEFAULT_BUDGET 的 recommended。"""
    counts = vocab64["slot_counts"]
    recommended = {slot.name: slot.recommended for slot in DEFAULT_BUDGET}
    assert set(counts) == set(VOCAB64_SLOT_ORDER) == set(recommended)
    for name in VOCAB64_SLOT_ORDER:
        assert counts[name] == slot_quotas()[name] == recommended[name], \
            f"{name}: count={counts[name]} 与预算 recommended={recommended[name]} 不一致"
    assert sum(counts.values()) == 65536


def test_vocab64_budget_problems_empty(vocab64, manifest64):
    counts = vocab64["slot_counts"]
    budget = VocabBudget(alloc=dict(counts))
    problems = budget.problems(TokenSpace())
    assert problems == [], f"预算判据未通过: {problems}"
    assert vocab64["budget"]["vocab_problems"] == []
    assert vocab64["budget"]["slot_problems"] == []
    assert manifest64["budget"]["problems"] == []
    assert manifest64["budget"]["sum"] == manifest64["budget"]["total"] == 65536


def test_vocab64_slot_ranges_cover_every_id_once(vocab64):
    tokens = vocab64["tokens"]
    ranges = vocab64["slot_ranges"]
    seen: list[int] = []
    for name in VOCAB64_SLOT_ORDER:
        spans = ranges[name]
        assert spans, f"{name} 没有 slot range"
        n = sum(b - a for a, b in spans)
        assert n == vocab64["slot_counts"][name], f"{name} 区间长度 ≠ count"
        for a, b in spans:
            assert 0 <= a < b <= len(tokens)
            # 槽内 token 文本必须与 id 一一对应（防止 build 时错位）
            assert all(tokens[i] != "" for i in range(a, b))
            seen.extend(range(a, b))
    assert sorted(seen) == list(range(len(tokens))), "八槽区间未恰好覆盖全部 id"


def test_vocab64_l0_atomicity(vocab64):
    tokens = vocab64["tokens"]
    core = set(LIGHT_L0_CORE)
    for ch in LIGHT_L0_CORE:
        assert tokens.count(ch) == 1, f"L0 字 {ch!r} 必须恰好占 1 个 token"
    assert check_l0_atomicity(tokens) == []
    bad = [t for t in tokens if len(t) > 1 and any(c in core for c in t)]
    assert bad == [], f"含 L0 字的多字 token: {bad[:5]}"


def test_vocab64_control_block_is_tail_and_contiguous(vocab64):
    tokens = vocab64["tokens"]
    layout = vocab64["control_layout"]
    expected_tail = _expected_control()
    start, end = layout["start"], layout["end"]
    assert end == len(tokens) - 2, "控制块之后只允许 <UNK>/<PAD> 两个 id"
    assert tokens[start:end] == expected_tail
    assert sorted(list(layout["structure"].values())
                  + list(layout["decisions"].values())
                  + list(layout["validations"].values())) == list(range(start, end))
    assert layout["size"] == len(expected_tail) == 32
    # ControlTokenLayout.finalize() 的连续性要求：ids 必须恰好是 start..end-1
    ref = ControlTokenLayout(start=start)
    ref.structure = dict(layout["structure"])
    ref.decisions = dict(layout["decisions"])
    ref.validations = dict(layout["validations"])
    ref.finalize()  # 有空洞会抛断言


def test_vocab64_unk_pad_ids(vocab64):
    tokens = vocab64["tokens"]
    special = vocab64["special_ids"]
    assert tokens[-2] == "<UNK>" and tokens[-1] == "<PAD>"
    assert special == {"unk": 65534, "pad": 65535, "embedding_size": 65536}


def test_vocab64_has_blank_tokens(vocab64):
    """空格/换行必须各自有单字 token：缺了它们 encode 会把空白整段映射成 UNK。"""
    tokens = vocab64["tokens"]
    assert " " in tokens and "\n" in tokens


def test_vocab64_manifest_reconciles(vocab64, manifest64):
    tokens = vocab64["tokens"]
    manifest = manifest64
    assert manifest["size"] == vocab64["size"] == 65536
    assert manifest["schema_version"] == vocab64["schema_version"]
    assert manifest["budget"]["problems"] == []
    assert manifest["special_ids"] == vocab64["special_ids"]

    for name in VOCAB64_SLOT_ORDER:
        entry = manifest["slots"][name]
        spans = vocab64["slot_ranges"][name]
        slot_tokens = [tokens[i] for a, b in spans for i in range(a, b)]
        assert entry["count"] == len(slot_tokens) == vocab64["slot_counts"][name]
        assert entry["quota"] == slot_quotas()[name]
        expect = hashlib.sha256(("\n".join(slot_tokens) + "\n").encode("utf-8")).hexdigest()
        assert entry["sha256"] == expect, f"{name} 槽 sha256 复算不一致"
        assert entry["ranges"] == spans
        charged = entry.get("charged") or {}
        if charged:
            assert sum(charged.values()) == entry["count"], f"{name} charged 合计 ≠ count"
            own = charged.get("own_placeholders_and_specials")
            if own is not None:
                assert sum(entry["sources"].values()) == own
        else:
            assert sum(entry["sources"].values()) == len(slot_tokens), \
                f"{name} 来源分档合计 ≠ count"


def test_vocab64_check_mode_is_full_green():
    good, lines = check_vocab_64k(VOCAB64_PATH, MANIFEST64_PATH, ROOT / "data")
    assert good, "\n".join(lines)
    assert "全绿" in "\n".join(lines)


def test_vocab64_visible_char_unk_is_zero(vocab64):
    """可见字符（去掉空白）口径下 UNK 必须为 0；空白另有单字 token。"""
    freq_all = vocab64["corpus_stats"]["char_freq"]
    blanks = sum(n for ch, n in freq_all.items() if ch.isspace())
    freq = {ch: n for ch, n in freq_all.items() if not ch.isspace()}
    rate, unk, total = unk_rate(freq, vocab64["tokens"])
    assert unk == 0 and rate == 0.0, f"可见字符 UNK {rate:.4%} ({unk}/{total})"
    assert total + blanks == vocab64["corpus_stats"]["total_chars"]
    assert blanks > 0, "语料快照应当含空白字符（空格/换行）"


def test_vocab64_config_declares_matching_v():
    cfg = json.loads(CONFIG64_PATH.read_text(encoding="utf-8"))
    assert cfg["vocab_size"] == 65536
    assert cfg["name"] == "light-prism-demo-3m-64k"
    base = json.loads(CONFIG4096_PATH.read_text(encoding="utf-8"))
    assert base["vocab_size"] == 4096, "demo-3m.json 必须保持 V=4096，不能被 64K 卡顺手改掉"


def test_vocab64_demo_vocab_exposes_65536():
    """DemoVocab 读 64K 表后 embedding 尺寸必须是 65536（torch 可选）。"""
    pytest.importorskip("torch")
    from lightprism.demo.train_demo import DemoVocab

    vocab = DemoVocab(json.loads(VOCAB64_PATH.read_text(encoding="utf-8")))
    assert vocab.vocab_size == 65536
    assert vocab.unk_id == 65534 and vocab.pad_id == 65535
    ids = vocab.encode("设 等于 《甲》。\n")
    assert ids and vocab.unk_id not in ids, f"encode 出现 UNK: {ids}"


def test_vocab64_beats_4096_on_token_count():
    """同语料贪心最长匹配口径：64K 的 token 数必须少于 4096 的。"""
    pytest.importorskip("torch")
    stats = coverage_compare([VOCAB_PATH, VOCAB64_PATH], ROOT / "data")
    v4 = stats["vocab_4096.json"]
    v64 = stats["vocab_64k.json"]
    assert v64["tokens"] < v4["tokens"], f"64K {v64['tokens']} 未少于 4096 {v4['tokens']}"
    assert v64["chars_per_token"] > v4["chars_per_token"]
    assert v64["unk_token_rate"] <= v4["unk_token_rate"]

# ---- T-0371：unified 词表追加段标记 + 旧 checkpoint 兼容 ---------------
UNIFIED_VOCAB_PATH = ROOT / "data" / "unified" / "vocab_4096.json"


def test_unified_vocab_appends_segment_markers_without_moving_old_ids():
    old = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    new = json.loads(UNIFIED_VOCAB_PATH.read_text(encoding="utf-8"))
    # 旧 token 前缀完全不变（旧 id 语义不动，方案 b 的前提）
    assert new["tokens"][:len(old["tokens"])] == old["tokens"]
    assert new["kinds"][:len(old["kinds"])] == old["kinds"]
    assert new["tokens"][-2:] == ["<PY>", "<LIGHT>"]
    assert new["size"] == len(old["tokens"]) + 2 == 2902
    assert new["control_layout"]["start"] == old["control_layout"]["start"] == 2868
    assert new["control_layout"]["end"] == 2902
    assert new["control_layout"]["structure"]["<PY>"] == 2900
    assert new["control_layout"]["structure"]["<LIGHT>"] == 2901
    assert new["counts"]["control"] == old["counts"]["control"] + 2
    assert new["headroom"] == new["limit"] - new["size"]
    assert new["segment_markers"]["ids"] == {"<PY>": 2900, "<LIGHT>": 2901}


def test_unified_vocab_keeps_embedding_at_4096_and_loads_marked_tokens():
    from lightprism.demo.train_demo import DemoVocab

    old = DemoVocab(json.loads(VOCAB_PATH.read_text(encoding="utf-8")))
    new = DemoVocab(json.loads(UNIFIED_VOCAB_PATH.read_text(encoding="utf-8")))
    # 只加控制 token，不加嵌入维度：旧 checkpoint 的 4096 行嵌入仍可直接加载
    assert new.vocab_size == old.vocab_size == 4096
    assert (new.unk_id, new.pad_id) == (2902, 2903)
    assert new.kind_of(2900) == new.kind_of(2901) == "structure"
    ids = new.encode("<PY>x = 1\n<LIGHT>\n设 x 为 1。\n<V:PASS>")
    assert ids[0] == new.structure["<PY>"]
    assert new.structure["<LIGHT>"] in ids
    assert new.validations["<V:PASS>"] in ids


def test_old_checkpoint_state_dict_loads_with_new_vocab():
    """方案 b 的硬证据：旧 checkpoint（4096 嵌入）能 strict 加载进新词表模型。"""
    torch = pytest.importorskip("torch")
    from lightprism.demo.train_demo import DemoVocab
    from lightprism.model import LightPrismModel, PrismConfig

    old = DemoVocab(json.loads(VOCAB_PATH.read_text(encoding="utf-8")))
    new = DemoVocab(json.loads(UNIFIED_VOCAB_PATH.read_text(encoding="utf-8")))
    cfg = PrismConfig(vocab_size=old.vocab_size, dim=32, n_layers=1, n_heads=4,
                      max_seq=64, ffn_mult=2.0, qk_norm=True, tie_embedding=True)
    old_model = LightPrismModel(cfg)
    old_state = {k: v.detach().clone() for k, v in old_model.state_dict().items()}
    new_model = LightPrismModel(PrismConfig(**{**cfg.__dict__, "vocab_size": new.vocab_size}))
    missing, unexpected = new_model.load_state_dict(old_state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    assert new_model.embed.weight.shape[0] == 4096
