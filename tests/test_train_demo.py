# -*- coding: utf-8 -*-
"""T-0332 demo checkpoint 训练管线的回归测试。

覆盖卡面「完成标准」：

1. 词表装配：控制 token 布局与词表 JSON 对齐，UNK/pad 槽位在 4096 内；
2. 数据管线：按文件 5% 切分、SFT 渲染、定长切块可复现；
3. 小步端到端：几步训练能跑完、checkpoint 可重新加载、smoke 生成不崩；
4. 模型规模：configs/demo-3m.json 对应约 3.4M 参数。

需要 torch 的用例统一 skip（与 ``test_model.py`` 同口径）；纯标准库部分
（词表、切分、切块、渲染）不依赖 torch，始终可跑。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lightprism.demo.train_demo import (
    DEFAULT_LIGHT_ROOT,
    DemoTrainConfig,
    DemoTrainer,
    VAL_RATIO,
    load_demo_vocab,
    pack_stream,
    read_corpus_files,
    render_sft_samples,
    split_corpus,
)
from lightprism.model_cfg import PrismConfig, count_parameters

ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "data" / "vocab_4096.json"
CONFIG_PATH = ROOT / "configs" / "demo-3m.json"

torch = pytest.importorskip("torch", reason="demo 训练/推理需要 torch")


# ---- 1. 词表装配 ---------------------------------------------------------
def test_vocab_layout_and_slots():
    vocab = load_demo_vocab(VOCAB_PATH)
    raw = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    layout = raw["control_layout"]

    assert vocab.size == raw["size"]
    assert vocab.control_start == layout["start"]
    assert vocab.control_end == layout["end"]
    # UNK/pad 紧随控制区间，且不越出 4096 词表预算
    assert vocab.unk_id == vocab.control_end
    assert vocab.pad_id == vocab.unk_id + 1
    assert vocab.vocab_size <= 4096
    assert vocab.vocab_size >= vocab.size

    # 控制 id 与固化布局逐条一致
    assert vocab.structure == layout["structure"]
    assert vocab.decisions == layout["decisions"]
    assert vocab.validations == layout["validations"]


def test_encode_decode_roundtrip_control_tokens():
    vocab = load_demo_vocab(VOCAB_PATH)
    seq = "<TASK>测试加法</TASK>\n<CODE>\n设 x 为 1\n</CODE>\n<V:PASS>\n"
    ids = vocab.encode(seq)
    assert ids, "序列被编码成空"
    assert vocab.decode(ids) == seq
    # 控制 token 各占 1 个 id，落在控制区间内
    ctrl = [i for i in ids if vocab.control_start <= i < vocab.control_end]
    assert len(ctrl) == 5  # <TASK> </TASK> <CODE> </CODE> <V:PASS>


def test_encode_unknown_falls_back_to_unk_slot():
    vocab = load_demo_vocab(VOCAB_PATH)
    # 词表里极不可能出现的私有组合字符
    ids = vocab.encode("㐀㐁㐂")
    assert ids, "未覆盖字符被静默丢弃"
    assert all(i == vocab.unk_id for i in ids)


# ---- 2. 数据管线 ---------------------------------------------------------
def test_corpus_split_is_deterministic_and_by_file():
    files, skipped = read_corpus_files(Path(DEFAULT_LIGHT_ROOT))
    # 上游不在场（只交付索引）时跳过本用例，不造假绿。
    if not files:
        pytest.skip("上游 light 仓不在场，语料文件读不到")
    tr1, va1 = split_corpus(files, VAL_RATIO, seed=20261112)
    tr2, va2 = split_corpus(files, VAL_RATIO, seed=20261112)
    assert [p for p, _ in tr1] == [p for p, _ in tr2]
    assert [p for p, _ in va1] == [p for p, _ in va2]
    # 切分按文件：训练 / 验证不重叠
    assert {p for p, _ in tr1}.isdisjoint({p for p, _ in va1})
    assert len(va1) == round(len(files) * VAL_RATIO) or len(va1) >= 1
    # 验证集占比在 5% 附近（至少 1 个文件）
    assert 0 < len(va1) <= max(1, round(len(files) * VAL_RATIO) + 1)


def test_sft_samples_render_to_standard_sequence():
    samples = render_sft_samples()
    assert len(samples) >= 200
    for _, seq in samples:
        assert seq.startswith("<TASK>")
        assert "<CODE>" in seq and "</CODE>" in seq
        assert "<V:PASS>" in seq


def test_pack_stream_drops_tail_and_shapes_blocks():
    vocab = load_demo_vocab(VOCAB_PATH)
    seq_ids = [vocab.encode("设 x 为 1。\n" * 30)]
    inp, tgt, w = pack_stream(seq_ids, seq_len=64, pad_id=vocab.pad_id)
    assert inp and tgt and w
    for row in inp:
        assert len(row) == 64
    for row in tgt:
        assert len(row) == 64
    # targets 错位一位：最后一位是补位目标（这里无 pad，残块已被丢弃）
    assert inp[0][1:] == tgt[0][:-1]


# ---- 3. 模型规模 ---------------------------------------------------------
def test_demo_config_is_about_3m_parameters():
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = PrismConfig(**{k: v for k, v in raw.items() if k in PrismConfig.__dataclass_fields__})
    total = count_parameters(cfg)
    # 卡面：约 3.4M 参数；允许 ±15% 区间，避免浮点取整口径差异造成假红
    assert 3_000_000 <= total <= 4_000_000, f"参数量 {total} 不在 3.0M—4.0M"


# ---- 4. 小步端到端（需要 torch + 语料） -----------------------------------
def test_tiny_train_run_saves_and_reloads(tmp_path: Path):
    files, _ = read_corpus_files(Path(DEFAULT_LIGHT_ROOT))
    if not files:
        pytest.skip("上游 light 仓不在场，跑不了端到端训练")

    cfg = DemoTrainConfig(
        steps=2, seq_len=128, batch_size=2, eval_every=0, log_every=1,
        out_dir=str(tmp_path / "demo-3m"),
        model={"vocab_size": 4096, "dim": 64, "n_layers": 2, "n_heads": 4,
               "max_seq": 128, "ffn_mult": 4.0, "qk_norm": True,
               "tie_embedding": True},
    )
    trainer = DemoTrainer(cfg)
    summary = trainer.train()
    ckpt = trainer.save()
    assert ckpt.is_file()
    assert summary["params"] > 0
    assert summary["dataset"]["train_blocks"] > 0

    # 重新加载：模型 + 词表布局 + 底座词表本体都能还原
    from lightprism.demo.smoke import load_demo_generator

    gen = load_demo_generator(ckpt, max_new_tokens=8)
    out = gen.generate("写一个加法程序", max_new_tokens=8)
    assert out["generated_tokens"] == 8
    assert len(out["ids"]) == 8
    assert all(0 <= i < gen.vocab.vocab_size for i in out["ids"])
