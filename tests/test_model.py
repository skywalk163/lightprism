# -*- coding: utf-8 -*-
"""数据 / 词表装配 / 模型配置的测试。纯标准库可跑的放前面，需要 torch 的用 skip。"""

from __future__ import annotations

import json

import pytest

from lightprism.data import (
    BatchBuilder,
    Sample,
    Step,
    build_sequence,
    compute_stats,
    load_jsonl,
    write_jsonl,
)
from lightprism.model_cfg import CONFIG_1B, CONFIG_300M, PrismConfig, count_parameters, from_json_file
from lightprism.sequence import parse_sequence
from lightprism.tokenizer import CharTokenizer, PrismTokenizer
from lightprism.tokens import ParseError

torch = pytest.importorskip("torch", reason="训练/推理骨架需要 torch")


# ---- 模型配置（不需要 torch 的部分其实也能跑，但本文件统一 skip）--------
def test_config_formula_matches_actual_model():
    from lightprism.model import LightPrismModel

    cfg = PrismConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, max_seq=64)
    model = LightPrismModel(cfg)
    assert model.count_parameters() == count_parameters(cfg)
    assert model.count_parameters(non_embedding=True) == count_parameters(cfg, non_embedding=True)


def test_published_configs_have_expected_scale():
    # 320M / 950M 是 README 与 configs/*.json 里写死对外承诺的数字
    assert 300_000_000 < count_parameters(CONFIG_300M) < 340_000_000
    assert 900_000_000 < count_parameters(CONFIG_1B) < 1_000_000_000


def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        PrismConfig(dim=100, n_heads=3)


def test_from_json_file_ignores_annotation_fields():
    cfg, unknown = from_json_file("configs/1b.json")
    assert cfg.dim == 1536 and cfg.n_layers == 30
    assert unknown == {"name", "note"}


@pytest.mark.parametrize("cfg_name", ["configs/300m.json", "configs/1b.json"])
def test_shipped_configs_are_valid(cfg_name):
    cfg, _ = from_json_file(cfg_name)
    assert cfg.vocab_size == 65536
    assert cfg.qk_norm is True          # 方案 §8.1 的必要条件
    assert cfg.tie_embedding is True


# ---- 前向 -------------------------------------------------------------
def test_forward_shapes_and_loss():
    from lightprism.model import LightPrismModel

    cfg = PrismConfig(vocab_size=128, dim=64, n_layers=2, n_heads=4, max_seq=32)
    model = LightPrismModel(cfg)
    ids = torch.randint(0, 128, (2, 16))
    logits, loss = model(ids, ids)
    assert logits.shape == (2, 16, 128)
    assert loss.ndim == 0 and loss.item() > 0


def test_loss_weights_zero_out_padding():
    """补位 token 权重为 0 时，损失不应受补位内容影响。"""
    from lightprism.model import LightPrismModel

    cfg = PrismConfig(vocab_size=128, dim=64, n_layers=2, n_heads=4, max_seq=32)
    model = LightPrismModel(cfg)
    ids = torch.randint(0, 128, (1, 8))
    w = torch.ones(1, 8)
    w[0, 4:] = 0.0
    _, loss1 = model(ids, ids, loss_weights=w)
    mutated = ids.clone()
    mutated[0, 5] = (mutated[0, 5] + 7) % 128
    _, loss2 = model(mutated, mutated, loss_weights=w)
    # 第 5 位权重为 0，改它不影响损失
    assert loss1.item() == pytest.approx(loss2.item(), abs=1e-6)


def test_param_groups_reject_excessive_lr_multiplier():
    from lightprism.model import LightPrismModel

    model = LightPrismModel(PrismConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4))
    with pytest.raises(ValueError):
        model.param_groups(control_lr_mult=3.0)   # 方案 §8.4 硬上限 2 倍
    groups = model.param_groups(control_lr_mult=2.0)
    assert [g["name"] for g in groups] == ["embed", "weights"]


# ---- 分词器装配 -------------------------------------------------------
def _sample() -> Sample:
    return Sample(
        task="把测试文件按大小排序",
        source="light_ai_check",
        steps=[Step(decisions={"EDIT": "FR"}, code='设 甲 为 1。', validation="PASS")],
    )


def test_control_ids_are_one_contiguous_tail_block():
    tk = PrismTokenizer(CharTokenizer())
    layout = tk.layout
    assert layout.start == CharTokenizer().vocab_size
    # 区间必须连续且无空洞
    all_ids = sorted(list(layout.structure.values()) + list(layout.decisions.values())
                     + list(layout.validations.values()))
    assert all_ids == list(range(layout.start, layout.end))
    assert tk.vocab_size == layout.end


def test_control_token_always_one_id():
    tk = PrismTokenizer(CharTokenizer(extra="把测试文件按大小排序设甲为。"))
    ids = tk.encode_sequence(_sample().to_sequence())
    kinds = [tk.kind_of(i) for i in ids]
    # 决策 token 与验证 token 各占**一个** id：计数与出现次数一致
    assert kinds.count("decision") == 1
    assert kinds.count("validation") == 1
    assert kinds.count("structure") == 4          # <TASK> </TASK> <CODE> </CODE>
    assert kinds[0] == "structure" and kinds[-1] == "validation"


def test_encode_decode_roundtrip_of_control_tokens():
    tk = PrismTokenizer(CharTokenizer(extra="测试"))
    seq = _sample().to_sequence()
    ids = tk.encode_sequence(seq)
    pieces = tk.decode_tokens(ids)
    assert "<TASK>" in pieces and "</CODE>" in pieces
    assert "<D:EDIT=FR>" in pieces and "<V:PASS>" in pieces


def test_budget_violation_blocks_assembly():
    from lightprism.vocab import VocabBudget

    broken = VocabBudget(alloc={"decision": 1})     # 装不下 19 个决策 token
    with pytest.raises(ParseError):
        PrismTokenizer(CharTokenizer(), budget=broken)


# ---- 结构状态机（约束解码的核心）--------------------------------------
def test_state_machine_forbids_repeating_task_end():
    tk = PrismTokenizer(CharTokenizer())
    st = tk.layout.structure
    # 已经写过一次 </TASK>，下一个控制 token 不允许再来一个 </TASK>
    allowed = tk.allowed_next([st["<TASK>"], st["<TASK>"], st["</TASK>"]])
    assert allowed is not None
    assert st["</TASK>"] not in allowed


def test_state_machine_forces_validation_after_code_end():
    tk = PrismTokenizer(CharTokenizer())
    st = tk.layout.structure
    allowed = tk.allowed_next([st["<TASK>"], st["</TASK>"], st["<CODE>"], st["</CODE>"]])
    assert allowed == sorted(tk.layout.validations.values())


def test_state_machine_forbids_second_decision_of_same_point():
    tk = PrismTokenizer(CharTokenizer())
    st = tk.layout.structure
    edit_fr = tk.layout.decisions["<D:EDIT=FR>"]
    allowed = tk.allowed_next([st["<TASK>"], st["</TASK>"], edit_fr])
    assert edit_fr not in allowed
    # 但别的决策点仍可选
    assert tk.layout.decisions["<D:SAFE=L0>"] in allowed


# ---- 数据 -------------------------------------------------------------
def test_sample_validation_reports_every_problem():
    bad = Sample(task="", steps=[Step(decisions={"EDIT": "MAGIC"}, code="", validation="NOPE")])
    problems = bad.validate()
    assert any("task 为空" in p for p in problems)
    assert any("MAGIC" in p for p in problems)
    assert any("code 为空" in p for p in problems)
    assert any("NOPE" in p for p in problems)


def test_decision_order_is_registry_order_not_dict_order():
    """同一步的决策点顺序必须稳定，否则同一样本两次导出得到不同 target。"""
    a = Step(decisions={"NEXT": "TEST", "EDIT": "FR"}, code="设 甲 为 1。")
    b = Step(decisions={"EDIT": "FR", "NEXT": "TEST"}, code="设 甲 为 1。")
    from lightprism.tokens import TokenSpace
    assert [i.text for i in a.to_item_list(TokenSpace())] == \
           [i.text for i in b.to_item_list(TokenSpace())]


def test_jsonl_roundtrip_and_bad_line_is_reported_not_raised(tmp_path):
    good = _sample()
    path = tmp_path / "s.jsonl"
    write_jsonl([good], path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"task": "", "steps": []}\n')
        fh.write("not json\n")
    samples, problems = load_jsonl(path)
    assert len(samples) == 1
    assert len(problems) == 2


def test_batch_is_left_padded_with_zero_weight():
    tk = PrismTokenizer(CharTokenizer(extra="把测试文件按大小排序设甲为。"))
    builder = BatchBuilder(tokenizer=tk, seq_len=64)
    batch = builder.make_batch([_sample(), _sample()])
    assert len(batch["input_ids"]) == 2
    assert all(len(row) == 64 for row in batch["input_ids"])
    pad = 64 - (batch["lengths"][0] - 1)
    assert pad > 0
    assert all(w == 0.0 for w in batch["loss_weights"][0][-pad:]) if pad else True


def test_stats_count_density_on_real_ids():
    tk = PrismTokenizer(CharTokenizer(extra="把测试文件按大小排序设甲为。"))
    stats = compute_stats([_sample()], tk)
    assert stats.tokens > 0
    assert stats.decision_tokens == 1
    assert stats.validation_tokens == 1
    # 段计数与 id 计数不同：这里必须是 id 口径
    assert 0 < stats.decision_ratio < 1


def test_pack_stream_drops_partial_tail():
    tk = PrismTokenizer(CharTokenizer(extra="把测试文件按大小排序设甲为。"))
    builder = BatchBuilder(tokenizer=tk, seq_len=8)
    chunks = list(builder.pack_stream([_sample(), _sample()]))
    assert all(len(c) == 8 for c in chunks)
    assert chunks, "样本长度超过 8，至少要产出一个块"


# ---- 训练/推理端到端（微小模型，秒级）-------------------------------
def test_train_then_infer_end_to_end(tmp_path):
    from lightprism.infer import GenConfig, load_generator
    from lightprism.train import TrainConfig, Trainer

    data = tmp_path / "d.jsonl"
    write_jsonl([_sample(), _sample(), _sample()], data)
    cfg = TrainConfig(data=str(data), out_dir=str(tmp_path / "run"),
                      steps=3, seq_len=48, batch_size=2, log_every=1, warmup=1)
    trainer = Trainer(cfg)
    summary = trainer.train()
    assert summary["steps"] == 3
    ckpt = trainer.save()
    assert ckpt.exists()

    gen = load_generator(ckpt, GenConfig(max_new_tokens=6))
    out = gen.generate("测试", decisions={"EDIT": "FR"})
    assert out["decisions"] == ["<D:EDIT:FR>".replace(":", "")] or out["decisions"]
    # 钉死的决策必须出现且只出现一次（互斥性）
    assert out["decisions"].count("<D:EDIT=FR>") == 1


def test_rlvr_stage_is_rejected_explicitly():
    from lightprism.train import TrainConfig

    with pytest.raises(NotImplementedError):
        TrainConfig(stage="rlvr", data="x.jsonl")


def test_cosine_lr_schedule_shape():
    from lightprism.train import cosine_lr

    assert cosine_lr(0, 100, 1.0, warmup=10) < cosine_lr(9, 100, 1.0, warmup=10)
    assert cosine_lr(50, 100, 1.0, warmup=10) > cosine_lr(99, 100, 1.0, warmup=10)
    assert cosine_lr(100, 100, 1.0, warmup=10) == pytest.approx(0.1)


def test_checkpoint_without_base_vocab_is_rejected(tmp_path):
    """旧版 checkpoint 只存 base_vocab_size，必须明确报错而不是拿占位 token 凑数。"""
    from lightprism.infer import load_generator
    from lightprism.model import LightPrismModel

    model = LightPrismModel(PrismConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4))
    path = tmp_path / "bad.pt"
    torch.save({
        "model": model.state_dict(),
        "model_config": {"vocab_size": 64, "dim": 32, "n_layers": 1, "n_heads": 4},
        "train_config": {},
        "tokenizer_layout": {"base_vocab_size": 96, "layout": {"start": 96, "end": 130}},
        "history": [],
    }, path)
    with pytest.raises(ValueError, match="base_vocab"):
        load_generator(path)


def test_probe_marks_train_only_accuracy_as_not_generalization():
    from lightprism.probe import ProbeResult

    result = ProbeResult(macro_accuracy=1.0, points_with_test_split=0)
    data = result.as_dict()
    assert data["macro_accuracy_is_meaningful"] is False


def test_judge_treats_missing_metrics_as_incomplete_not_pass():
    from lightprism.probe import PassCriteria, judge

    verdict = judge(PassCriteria(probe_macro_accuracy=0.9))
    assert verdict["verdict"] == "incomplete"

    full = judge(PassCriteria(
        probe_macro_accuracy=0.9, a_group_accuracy=0.8,
        pass_rate=0.7, b_group_pass_rate=0.6,
        structure_rate=0.99, decision_density=0.01, speed_drop=0.05,
    ))
    assert full["verdict"] == "pass"
