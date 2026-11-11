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
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from lightprism.demo import UNK_LIMIT, VOCAB_LIMIT, check_vocab, unk_rate
from lightprism.tokens import LIGHT_L0_CORE, VALIDATION_CODES, TokenSpace
from lightprism.tokenizer import STRUCTURE_MARKERS, ControlTokenLayout
from lightprism.vocab import check_l0_atomicity

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
