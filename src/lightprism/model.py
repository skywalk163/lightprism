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

from typing import List, Optional, Tuple

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
        rstd = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        out = h * rstd
        if dtype != torch.float32:
            out = out.to(dtype)
        return out * self.weight


def _rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype,
                start: int = 0):
    """生成位置 ``[start, start+seq_len)`` 的 RoPE 余弦 / 正弦表。

    ``start`` 用于增量解码：缓存里已经有 ``start`` 个 token，新进来的这一段
    必须用**绝对位置**编码，不能每次都从 0 开始（否则 RoPE 位置感错位）。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(start, start + seq_len, device=device).float()
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
        out, _ = self.forward_with_cache(x, cos, sin, past_kv=None)
        return out

    def forward_with_cache(self, x, cos, sin, past_kv=None):
        """前向并返回 ``(输出, 本层新 KV)``。

        ``past_kv`` 为 ``None``：整段序列一次性前向（训练 / prefill 用），因果掩码
        由 ``is_causal`` 给出。``past_kv`` 给定时把新 K/V 拼到缓存尾部：decode 步
        （``t=1``）对全部历史可见，不需要因果掩码；分块 prefill 则按绝对位置加显式
        因果掩码。
        """
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

        dropout_p = self.cfg.dropout if self.training else 0.0
        if past_kv is None:
            new_k, new_v = k, v
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True,
            )
        else:
            past_k, past_v = past_kv
            new_k = torch.cat([past_k, k], dim=2)
            new_v = torch.cat([past_v, v], dim=2)
            if t == 1:
                # decode 单步：唯一的 query 可以看到缓存里全部历史 + 当前位置
                out = F.scaled_dot_product_attention(
                    q, new_k, new_v, dropout_p=0.0, is_causal=False,
                )
            else:
                # 分块 prefill：query 的绝对下标起点是 past_k 的长度，按它裁因果掩码
                past_len = past_k.shape[2]
                total = new_k.shape[2]
                causal = torch.ones(t, total, dtype=torch.bool, device=x.device)
                causal = causal.tril(diagonal=past_len)
                out = F.scaled_dot_product_attention(
                    q, new_k, new_v, attn_mask=causal,
                    dropout_p=dropout_p,
                )

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.dropout(self.proj(out)), (new_k, new_v)


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

    def forward_with_cache(self, x, cos, sin, past_kv=None):
        attn_out, new_kv = self.attn.forward_with_cache(
            self.norm1(x), cos, sin, past_kv,
        )
        x = x + attn_out
        return x + self.ffn(self.norm2(x)), new_kv


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
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None
        self._init_weights()

    def _rope_tables(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """惰性构建一次全位置 RoPE 余弦/正弦表。

        decode 每步只按位置切片，避免原先每步都重建 ``inv_freq``/``outer``/
        ``cos``/``sin`` 一整套小张量（见 T-0340 提速复盘）。表随模型权重一起
        ``.to(device)``，dtype 与嵌入权重一致。
        """
        if self._rope_cos is None:
            hd = self.cfg.head_dim
            inv_freq = 1.0 / (self.cfg.rope_theta ** (
                torch.arange(0, hd, 2, dtype=torch.float32) / hd))
            pos = torch.arange(self.cfg.max_seq, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            self._rope_cos = freqs.cos().to(self.embed.weight.dtype)
            self._rope_sin = freqs.sin().to(self.embed.weight.dtype)
        return self._rope_cos, self._rope_sin

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

        cos_full, sin_full = self._rope_tables()
        cos, sin = cos_full[:t], sin_full[:t]
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

    # ---- 增量推理（KV cache）------------------------------------------
    def incremental_forward(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ):
        """带 KV cache 的前向，返回 ``(logits, new_kv)``。

        用法：

        1. prefilling：``past_kv=None``，一次喂入整段 prompt，拿回逐层 K/V 缓存；
        2. decode：每次只喂 1 个新 token，带上一步的 ``past_kv``，缓存按层追加。

        ``past_kv`` 是长度 ``n_layers`` 的列表，每层为 ``(k, v)``，形状
        ``[B, H, T_cache, head_dim]``。RoPE 用**绝对位置**：新段的位置从
        ``缓存长度`` 开始编号，不从 0 重置。

        数值口径必须与 :meth:`forward` 全量重算一致，由
        ``tests/test_kv_cache.py::test_kv_cache_matches_full`` 兜住。
        """
        b, t = input_ids.shape
        cache_len = 0
        if past_kv is not None:
            first_k = past_kv[0][0]
            cache_len = int(first_k.shape[2])
        if cache_len + t > self.cfg.max_seq:
            raise ValueError(
                f"序列长度 {cache_len + t} 超过 max_seq={self.cfg.max_seq}")

        cos_full, sin_full = self._rope_tables()
        cos = cos_full[cache_len:cache_len + t]
        sin = sin_full[cache_len:cache_len + t]
        x = self.embed(input_ids)
        new_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for i, block in enumerate(self.blocks):
            past = past_kv[i] if past_kv is not None else None
            x, kv = block.forward_with_cache(x, cos, sin, past)
            new_kv.append(kv)
        hidden = self.norm_f(x)
        logits = self.lm_head(hidden)
        return logits, new_kv

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
