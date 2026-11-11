# -*- coding: utf-8 -*-
"""KV cache 推理核测试（T-0331）。

核心红线：同一序列，「全量重算的 logits」与「增量（prefill + 逐步 decode）算的
logits」必须在 ``atol=1e-4`` 内相等。RoPE 用绝对位置，缓存只追加、不重置位置。
"""

from __future__ import annotations

import pytest
import torch

torch = pytest.importorskip("torch", reason="KV cache 推理核需要 torch")

from lightprism.model import LightPrismModel, PrismConfig  # noqa: E402


def _make_model(seed: int = 0) -> LightPrismModel:
    torch.manual_seed(seed)
    cfg = PrismConfig(vocab_size=128, dim=64, n_layers=2, n_heads=4, max_seq=64)
    model = LightPrismModel(cfg)
    model.eval()
    return model


def _random_ids(seq_len: int, vocab: int = 128, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, seq_len), generator=g)


def test_incremental_prefill_matches_full():
    """整段 prefill（past_kv=None）的 logits 必须与全量 forward 一致。"""
    model = _make_model()
    ids = _random_ids(16)
    with torch.no_grad():
        full_logits, _ = model(ids)
        inc_logits, _ = model.incremental_forward(ids)
    assert torch.allclose(inc_logits, full_logits, atol=1e-4, rtol=1e-4)


def test_kv_cache_matches_full():
    """prefill 前半段 + 逐步 decode 后半段，逐位置对齐全量重算。"""
    model = _make_model()
    ids = _random_ids(16)
    split = 8

    with torch.no_grad():
        full_logits, _ = model(ids)

        prefill = ids[:, :split]
        inc_logits, past = model.incremental_forward(prefill)
        # prefill 段逐位置对齐
        assert torch.allclose(
            inc_logits, full_logits[:, :split], atol=1e-4, rtol=1e-4)

        for i in range(split, ids.shape[1]):
            step = ids[:, i:i + 1]
            inc_logits, past = model.incremental_forward(step, past)
            assert torch.allclose(
                inc_logits[:, -1, :], full_logits[:, i, :],
                atol=1e-4, rtol=1e-4,
            ), f"位置 {i} 增量与全量 logits 不一致"


def test_cache_shapes_grow_per_layer():
    """每层缓存按 decode 步追加 K/V，形状 [B, H, T_cache, head_dim]。"""
    model = _make_model()
    ids = _random_ids(8)
    with torch.no_grad():
        _, past = model.incremental_forward(ids)
    assert len(past) == model.cfg.n_layers
    for k, v in past:
        assert k.shape == (1, model.cfg.n_heads, 8, model.cfg.head_dim)
        assert v.shape == (1, model.cfg.n_heads, 8, model.cfg.head_dim)

    with torch.no_grad():
        _, past = model.incremental_forward(ids[:, -1:], past)
    for k, v in past:
        assert k.shape[2] == 9
        assert v.shape[2] == 9


def test_incremental_respects_max_seq():
    """缓存长度 + 新 token 超过 max_seq 必须显式报错，不能静默越界。"""
    model = _make_model()
    ids = _random_ids(64)
    with torch.no_grad():
        _, past = model.incremental_forward(ids)
    with pytest.raises(ValueError, match="max_seq"):
        model.incremental_forward(ids[:, :1], past)
