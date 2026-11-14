# -*- coding: utf-8 -*-
"""T-0345 · demo 约束解码接线的验收测试。

覆盖卡面「完成标准」四项：

1. 状态机掩码（``allowed_next`` 白名单真的被 -inf 掩掉）；
2. 空槽 mask（``vocab.size`` 之后、PAD/UNK 之外的槽位永久 -inf）；
3. 两列指标分离（约束版产品结构率 ≠ 无约束能力结构率）；
4. 空输出不得计入代码段合法率（防 ``light ai check`` 对空程序判 ok 的假绿）。

另加一条小模型端到端：随机权重 + 结构推进也必须能收出成对 ``<CODE>`` / ``<V:…>``，
证明 ``make demo`` 走的这条路径不是「只有测试夹具才合法」。
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="约束解码测试需要 torch")

from lightprism.demo.infer_demo import (  # noqa: E402
    DemoTokenizerAdapter,
    _DemoCore,
    evaluate_code_segments,
    evaluate_two_columns,
    extract_code,
    structure_legal,
    summarize_structure,
)
from lightprism.demo.infer_demo import ConstrainedDemoGenerator  # noqa: E402
from lightprism.demo.train_demo import load_demo_vocab  # noqa: E402
from lightprism.infer import GenConfig  # noqa: E402


@pytest.fixture(scope="module")
def vocab():
    return load_demo_vocab()


@pytest.fixture(scope="module")
def adapter(vocab):
    return DemoTokenizerAdapter(vocab)


def _core(vocab, adapter, *, constrain=True, mask_empty=False, force=False):
    return _DemoCore(
        model=object(),  # PrismGenerator 构造只保存引用，不调用模型
        tokenizer=adapter,
        cfg=GenConfig(max_new_tokens=8),
        vocab=vocab,
        constrain_structure=constrain,
        mask_empty_slots=mask_empty,
        force_structure=force,
    )


# ---------------------------------------------------------------------------
# 1. 状态机掩码
# ---------------------------------------------------------------------------
def test_allowed_next_reuses_global_state_machine(vocab, adapter):
    st = vocab.structure
    assert adapter.allowed_next([]) == [st["<TASK>"]]
    assert adapter.allowed_next([st["<TASK>"]]) == [st["</TASK>"]]

    after_task = adapter.allowed_next([st["<TASK>"], st["</TASK>"]])
    assert st["<CODE>"] in after_task
    assert set(after_task) <= set(vocab.decisions.values()) | {st["<CODE>"]}

    assert adapter.allowed_next([st["<TASK>"], st["</TASK>"], st["<CODE>"]]) == [
        st["</CODE>"]]
    after_code = adapter.allowed_next(
        [st["<TASK>"], st["</TASK>"], st["<CODE>"], st["</CODE>"]])
    assert set(after_code) == set(vocab.validations.values())


def test_apply_constraint_masks_only_control_tokens(vocab, adapter):
    """不强制推进时：非法控制 token 置 -inf，自由文本放行（与 infer.py 同口径）。"""
    core = _core(vocab, adapter, constrain=True, mask_empty=False, force=False)
    st = vocab.structure
    history = [st["<TASK>"], st["</TASK>"]]
    allowed = set(adapter.allowed_next(history))

    logits = torch.zeros(vocab.vocab_size)
    masked = core._apply_constraint(logits, history)

    for token_id in adapter.layout.id_range:
        if token_id in allowed:
            assert math.isfinite(float(masked[token_id]))
        else:
            assert float(masked[token_id]) == float("-inf")

    # 文本 token 不被状态机掩掉（否则模型永远进不了代码体）
    space_id = vocab.index[" "]
    assert float(masked[space_id]) == 0.0


def test_force_structure_masks_free_text_when_control_is_required(vocab, adapter):
    """结构推进：``</TASK>`` 之后必须出控制 token，自由文本也被掩掉。"""
    core = _core(vocab, adapter, constrain=True, mask_empty=False, force=True)
    st = vocab.structure
    history = [st["<TASK>"], st["</TASK>"]]
    logits = torch.zeros(vocab.vocab_size)
    masked = core._apply_constraint(logits, history)

    assert float(masked[vocab.index[" "]]) == float("-inf")
    allowed = set(adapter.allowed_next(history))
    for token_id in adapter.layout.id_range:
        if token_id in allowed:
            assert math.isfinite(float(masked[token_id]))
        else:
            assert float(masked[token_id]) == float("-inf")


# ---------------------------------------------------------------------------
# 2. 空槽 mask
# ---------------------------------------------------------------------------
def test_empty_slots_are_masked_but_pad_and_unk_are_kept(vocab, adapter):
    core = _core(vocab, adapter, constrain=False, mask_empty=True, force=False)
    assert core.empty_slot_ids, "2900..4096 之间应当有空槽"
    assert vocab.unk_id not in core.empty_slot_ids
    assert vocab.pad_id not in core.empty_slot_ids

    empty_id = core.empty_slot_ids[0]
    logits = torch.zeros(vocab.vocab_size)
    logits[empty_id] = 100.0
    logits[vocab.unk_id] = 50.0
    logits[vocab.pad_id] = 40.0
    masked = core._apply_constraint(logits, [])

    assert float(masked[empty_id]) == float("-inf")
    assert float(masked[vocab.unk_id]) == 50.0
    assert float(masked[vocab.pad_id]) == 40.0


def test_ablation_without_mask_keeps_empty_slot_logits(vocab, adapter):
    core = _core(vocab, adapter, constrain=False, mask_empty=False, force=False)
    empty_id = vocab.size  # 2900 是 UNK 之外第一个空槽
    logits = torch.zeros(vocab.vocab_size)
    logits[empty_id] = 100.0
    out = core._apply_constraint(logits, [])
    assert float(out[empty_id]) == 100.0


# ---------------------------------------------------------------------------
# 3. 两列指标分离
# ---------------------------------------------------------------------------
def test_two_columns_are_not_merged():
    constrained = [
        "<CODE>print(1)</CODE><V:PASS>",
        "<CODE></CODE><V:PASS>",
    ]
    unconstrained = [" " * 64, " " * 64]
    result = evaluate_two_columns(constrained, unconstrained, checker=lambda _c: True)

    assert result["product_structure"]["rate"] == 1.0
    assert result["product_structure"]["legal"] == 2
    assert result["capability_structure"]["rate"] == 0.0
    assert result["capability_structure"]["legal"] == 0
    # 约束版结构合法不能提高能力列
    assert result["capability_structure"]["rate"] != result["product_structure"]["rate"]


def test_structure_legal_requires_paired_tags_and_validation():
    assert structure_legal("<CODE>x</CODE><V:PASS>")
    assert not structure_legal("<CODE>x</CODE>")            # 缺验证 token
    assert not structure_legal("<CODE>x<V:PASS>")           # CODE 未闭合
    assert not structure_legal("<CODE>a</CODE><CODE>b</CODE><V:PASS>")
    assert not structure_legal("<TASK>t</TASK><CODE>x</CODE><V:PASS>")
    assert not structure_legal(" " * 64)


# ---------------------------------------------------------------------------
# 4. 空输出不得计入合法
# ---------------------------------------------------------------------------
def test_empty_code_is_never_valid_even_if_checker_says_ok():
    texts = ["<CODE></CODE><V:PASS>", " " * 64, "<CODE>   </CODE><V:PASS>",
             "<CODE>print(1)</CODE><V:PASS>"]
    called = []

    def checker(code: str) -> bool:
        called.append(code)
        return True  # 模拟 light ai check 对空程序判 ok 的盲区

    result = evaluate_code_segments(texts, checker)

    assert result["samples"] == 4, "分母不许剔除空输出"
    assert result["empty"] == 3
    assert result["checked"] == 1
    assert result["valid"] == 1
    assert result["rate"] == pytest.approx(0.25)
    assert called == ["print(1)"], "空输出不能送进 checker 充当合法"


def test_extract_code_only_reads_code_body():
    assert extract_code("<CODE>print(1)</CODE><V:PASS>") == "print(1)"
    assert extract_code(" " * 32) == ""
    assert extract_code("<CODE></CODE>") == ""


# ---------------------------------------------------------------------------
# 5. 小模型端到端：结构推进必须能收口
# ---------------------------------------------------------------------------
def test_tiny_model_constrained_generation_is_structurally_legal(vocab):
    from lightprism.model import LightPrismModel, PrismConfig

    torch.manual_seed(0)
    model = LightPrismModel(PrismConfig(
        vocab_size=vocab.vocab_size, dim=32, n_layers=1, n_heads=2, max_seq=128))
    model.eval()

    gen = ConstrainedDemoGenerator(model, vocab, max_new_tokens=64)
    out = gen.generate("写一个加法程序", max_new_tokens=64)

    assert out["generated_tokens"] == 64
    assert all(0 <= i < vocab.vocab_size for i in out["ids"])
    assert "<CODE>" in out["text"] and "</CODE>" in out["text"]
    assert "<V:" in out["text"]
    assert structure_legal(out["text"])

    summary = summarize_structure([out["text"]])
    assert summary["legal"] == 1
    assert summary["with_code_pair"] == 1


def test_tiny_model_unconstrained_path_does_not_force_structure(vocab):
    """能力对照列：关掉状态机后，约束不会替模型补齐结构。"""
    from lightprism.model import LightPrismModel, PrismConfig

    torch.manual_seed(0)
    model = LightPrismModel(PrismConfig(
        vocab_size=vocab.vocab_size, dim=32, n_layers=1, n_heads=2, max_seq=128))
    model.eval()

    gen = ConstrainedDemoGenerator(
        model, vocab, max_new_tokens=32,
        constrain_structure=False, mask_empty_slots=False)
    out = gen.generate("写一个加法程序", max_new_tokens=32)

    assert out["generated_tokens"] == 32
    assert out["constrain_structure"] is False
    # 这一列只做能力对照，不保证合法；只要求它不是由状态机硬凑出来的
    assert out["force_structure"] is False
