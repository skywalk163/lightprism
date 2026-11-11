# -*- coding: utf-8 -*-
"""LightPrism 模型：1B 级 decoder-only Transformer（pre-norm + QK-Norm）。

为什么是这套结构（对应方案文档 §8 的「必要条件」，不是调优技巧）：

* **QK-Norm**：注意力点积前对 Q/K 各自过 RMSNorm。Chameleon 的经验是多模态 token
  早期融合时没有 QK-Norm 会在约 20% 进度处发散；本方案异质性更低但容量更紧。
* **预归一化**：决策 token 与代码 token 共享同一条归一化路径，不允许分叉。
* **tie_embedding**：输入嵌入与输出头共享权重，64K 词表下省掉 0.1—0.13B 参数。
* **早期融合**：决策/验证 token 的嵌入从训练第一步就参与学习，不后期注入。

本模块只依赖 ``torch``；纯标准库路径不 import 它（见 ``tests/test_model.py`` 的 skip 逻辑）。
"""

from __future__ import annotations

from typing import List, Optional

from .model_cfg import CONFIG_1B, CONFIG_300M, PrismConfig, count_parameters

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PrismConfig",
    "LightPrismModel",
    "count_parameters",
    "CONFIG_300M",
    "CONFIG_1B",
]


class RMSNorm(nn.Module):
    """RMSNorm。用 fp32 累加，避免 bf16 下的数值抖动。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h.to(dtype)) * self.weight


def _rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]；cos/sin: [T, D/2]。"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * torch.cat((cos, cos), dim=-1) + rotated * torch.cat((sin, sin), dim=-1)


class Attention(nn.Module):
    """MHA + RoPE + 因果掩码 +（可选）QK-Norm。"""

    def __init__(self, cfg: PrismConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h, hd = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, t, h, hd).transpose(1, 2)   # [B,H,T,D]
        k = k.view(b, t, h, hd).transpose(1, 2)
        v = v.view(b, t, h, hd).transpose(1, 2)

        # QK-Norm：在点积**之前**归一，两个投影各一份，参数不共享
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.cfg.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.dropout(self.proj(out))


class SwiGLU(nn.Module):
    """SwiGLU FFN。隐层宽度按 8/3·d·mult 取整到 64 的倍数（与 Llama 同口径）。"""

    def __init__(self, cfg: PrismConfig) -> None:
        super().__init__()
        hidden = int(round(cfg.dim * cfg.ffn_mult * 2 / 3 / 64)) * 64
        hidden = max(hidden, 64)
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: PrismConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class LightPrismModel(nn.Module):
    """decoder-only 主体。``forward`` 同时给出 logits 与（可选）下一 token 损失。"""

    def __init__(self, cfg: PrismConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embedding:
            self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    # ---- 前向 ---------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        b, t = input_ids.shape
        if t > self.cfg.max_seq:
            raise ValueError(f"序列长度 {t} 超过 max_seq={self.cfg.max_seq}")

        cos, sin = _rope_cache(t, self.cfg.head_dim, self.cfg.rope_theta,
                               input_ids.device, self.embed.weight.dtype)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        hidden = self.norm_f(x)
        logits = self.lm_head(hidden)

        if targets is None:
            return logits, None

        # 下一 token 预测：位置 i 预测 i+1，targets 已由调用方错位
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction="none",
        ).view(b, t)
        if loss_weights is not None:
            per_token = per_token * loss_weights
        loss = per_token.sum() / loss_weights.sum() if loss_weights is not None \
            else per_token.mean()
        return logits, loss

    # ---- 供训练侧用 ---------------------------------------------------
    def param_groups(self, control_lr_mult: float = 2.0) -> List[Dict]:
        """AdamW 的 param_groups：嵌入单独一组，其余权重一组。

        方案文档 §8 第 4 条要求「决策/验证 token 的嵌入学习率可略高于代码 token，
        但不超过 2 倍」。**v0.1 的诚实边界**：torch 的 Embedding 是一整块参数张量，
        按行设置不同学习率需要把嵌入拆成两段并在每次 step 后拼回，属于训练器层面的
        改造，本骨架没有做。所以这里只把**整块嵌入**单独成组、倍数上限卡在 2.0，
        并在加载时用 ``assert_control_budget()`` 报复位偏差。

        真正按行分 lr 的实现排队到 v0.2，记在 CHANGELOG「未包含」。
        """
        if control_lr_mult > 2.0:
            raise ValueError(f"嵌入学习率倍数 {control_lr_mult} 超过硬上限 2.0（方案文档 §8.4）")
        embed = [self.embed.weight]
        rest = [p for n, p in self.named_parameters() if n != "embed.weight"]
        return [
            {"params": embed, "lr_mult": control_lr_mult, "name": "embed"},
            {"params": rest, "lr_mult": 1.0, "name": "weights"},
        ]

    def count_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        return total - self.embed.weight.numel() if non_embedding else total
