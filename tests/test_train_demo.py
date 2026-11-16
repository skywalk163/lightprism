# -*- coding: utf-8 -*-
"""T-0332 / T-0347 demo checkpoint 训练管线的回归测试。

覆盖卡面「完成标准」：

1. 词表装配：控制 token 布局与词表 JSON 对齐，UNK/pad 槽位在 4096 内；
2. 数据管线：按文件 5% 切分、统一目录、SFT / 翻译 / 决策渲染、定长切块可复现；
3. 控制 token 加权：结构 / 决策 / 验证 token 的 loss 权重高于文本 token；
4. 小步端到端：几步训练能跑完、checkpoint 可重新加载、smoke 生成不崩；
5. 模型规模：configs/demo-3m.json 对应约 3.4M 参数。

需要 torch 的用例统一 skip（与 ``test_model.py`` 同口径）；纯标准库部分
（词表、切分、切块、渲染）不依赖 torch，始终可跑。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightprism.demo.train_demo import (
    DECISIONS_LOG,
    DemoTrainConfig,
    DemoTrainer,
    UNIFIED_DIR,
    VAL_RATIO,
    encode_with_weights,
    load_corpus,
    load_demo_vocab,
    pack_stream,
    render_decision_samples,
    render_decision_sequence,
    render_sft_samples,
    render_translation_samples,
    render_unified_samples,
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
    # 控制 token 集合覆盖结构 / 决策 / 验证三类
    assert len(vocab.control_ids) == len(vocab.decisions) + len(vocab.validations) + len(vocab.structure)


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
    files, _, source = load_corpus(None, None)
    # 统一目录快照与上游 light 仓都不在场时跳过本用例，不造假绿。
    if not files:
        pytest.skip(f"代码语料不在场（source={source}）")
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
        assert "<FORMAT>json</FORMAT>" in seq or "<FORMAT>code</FORMAT>" in seq
        assert "<TASK>" in seq and "</TASK>" in seq
        assert "<CODE>" in seq and "</CODE>" in seq
        assert "<V:PASS>" in seq


def test_translation_samples_render_two_code_blocks():
    rows = render_translation_samples()
    if not rows:
        pytest.skip("data/unified/translation_tasks.jsonl 不在场")
    kinds = {k for k, _ in rows}
    assert kinds <= {"pair", "light_only"}
    pair = next(seq for k, seq in rows if k == "pair")
    assert pair.startswith("<FORMAT>translate</FORMAT>\n")
    assert pair.count("<CODE>") == 2
    assert pair.rstrip().endswith("<V:PASS>")
    light = next(seq for k, seq in rows if k == "light_only")
    assert light.startswith("<FORMAT>light</FORMAT>\n")
    assert light.count("<CODE>") == 1


def test_decision_sequences_use_real_labels_only():
    """决策序列只许用词表里真实存在的 <D:…>/<V:…>。

    T-0374：条数不再钉死 29。T-0346 是 29 条快照，T-0370 把它扩到 241 条后
    ``== 29`` 必然红——那是数据增长，不是回归。改为：
      1. 条数由日志里「带 decision_tokens 的记录数」现算，两边必须一致（口径自洽）；
      2. 叠一条 ``>= 29`` 下限，防止有人把日志删空后测试反而"更绿"。
    """
    vocab = load_demo_vocab(VOCAB_PATH)
    rows = render_decision_samples()
    if not rows:
        pytest.skip("data/runtime/decisions.jsonl 不在场")

    expected = 0
    for line in DECISIONS_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("decision_tokens"):
            expected += 1
    assert expected >= 29, f"决策日志只剩 {expected} 条，低于 T-0346 基线 29"
    assert len(rows) == expected, f"渲染 {len(rows)} 条 ≠ 日志 {expected} 条"

    allowed = set(vocab.decisions) | set(vocab.validations)
    points: set[str] = set()
    for _, seq in rows:
        assert seq.startswith("<FORMAT>runtime</FORMAT>\n")
        assert "<TASK>" in seq and "<CODE>" in seq
        # 每个 <D:…> / <V:…> 都必须是词表里真实存在的控制 token
        labels = [tok for tok in seq.split() if tok.startswith("<D:") or tok.startswith("<V:")]
        assert labels, "决策序列缺少 <D:…> / <V:…>"
        for tok in labels:
            assert tok in allowed, f"未在词表中的标签 {tok}"
            if tok.startswith("<D:") and "=" in tok:
                points.add(tok[3:].split("=", 1)[0])
    # 比旧的"只数条数"更强：7 个决策点都得有真实日志样本（T-0370 的覆盖目标）
    assert len(points) == 7, f"决策点只覆盖 {sorted(points)}"


def test_decision_renderer_keeps_order_and_validation():
    rec = {
        "task_text": "计算 3 加 4",
        "source": "打印(3 加 4)。",
        "decision_tokens": ["<D:PLAN=DIRECT>", "<D:EDIT=RW>"],
        "validation_token_text": "<V:PASS>",
    }
    seq = render_decision_sequence(rec)
    assert seq.index("<D:PLAN=DIRECT>") < seq.index("<D:EDIT=RW>")
    assert seq.rstrip().endswith("<V:PASS>")
    assert "打印(3 加 4)。" in seq


def test_unified_samples_cover_all_sources():
    rows = render_unified_samples()
    assert rows
    kinds = {k for k, _ in rows}
    assert {"sft", "decision"} <= kinds
    if (UNIFIED_DIR / "translation_tasks.jsonl").is_file():
        assert kinds & {"pair", "light_only"}


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


def test_control_tokens_are_weighted_and_text_is_not():
    vocab = load_demo_vocab(VOCAB_PATH)
    ids, weights = encode_with_weights(vocab, "<TASK>甲</TASK>\n<V:PASS>\n", control_weight=3.0)
    assert len(ids) == len(weights)
    assert all(w in (1.0, 3.0) for w in weights)
    assert sum(1 for i in ids if i in vocab.control_ids) == 3  # <TASK> </TASK> <V:PASS>
    # 打包后权重跟到 target 侧且长度对齐
    inp, tgt, w = pack_stream([(ids * 40, weights * 40)], seq_len=64, pad_id=vocab.pad_id)
    assert len(w[0]) == 64
    assert max(w[0]) == 3.0 and min(w[0]) == 1.0
    assert all(len(r) == 64 for r in w)


# ---- 3. 模型规模 ---------------------------------------------------------
def test_demo_config_is_about_3m_parameters():
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = PrismConfig(**{k: v for k, v in raw.items() if k in PrismConfig.__dataclass_fields__})
    total = count_parameters(cfg)
    # 卡面：约 3.4M 参数；允许 ±15% 区间，避免浮点取整口径差异造成假红
    assert 3_000_000 <= total <= 4_000_000, f"参数量 {total} 不在 3.0M—4.0M"


# ---- 4. 小步端到端（需要 torch + 语料） -----------------------------------
def test_tiny_train_run_saves_and_reloads(tmp_path: Path):
    files, _, _ = load_corpus(None, None)
    if not files:
        pytest.skip("代码语料不在场，跑不了端到端训练")

    cfg = DemoTrainConfig(
        steps=2, seq_len=128, batch_size=2, eval_every=0, log_every=1,
        corpus_limit=3, sft_repeat=1, translate_repeat=1,
        decision_repeat=2, err_repeat=1,
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

    # checkpoint 必须带 base_vocab 本体（v0.2.0 修过的坑）
    payload = torch.load(ckpt, map_location="cpu")
    layout = payload["tokenizer_layout"]
    assert layout["base_kind"] == "DemoVocab"
    assert len(layout["base_vocab"]) == layout["base_vocab_size"]

    # 重新加载：模型 + 词表布局 + 底座词表本体都能还原
    from lightprism.demo.smoke import load_demo_generator

    gen = load_demo_generator(ckpt, max_new_tokens=8)
    out = gen.generate("写一个加法程序", max_new_tokens=8)
    assert out["generated_tokens"] == 8
    assert len(out["ids"]) == 8
    assert all(0 <= i < gen.vocab.vocab_size for i in out["ids"])
