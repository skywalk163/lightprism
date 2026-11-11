# -*- coding: utf-8 -*-
"""模型配置与参数量核算（**纯标准库，不 import torch**）。

单独成模块的理由：``lightprism config`` 子命令与 CI 都要在**没装 torch** 的机器上
跑通。第一版把 ``PrismConfig`` 放在 model.py 里，结果 CLI 一执行就 import torch ——
这个耦合必须切断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

__all__ = ["PrismConfig", "count_parameters", "CONFIG_300M", "CONFIG_1B", "from_json_file"]


@dataclass
class PrismConfig:
    """模型超参。默认值 = light-prism-300m（CPU 可跑）；1B 见 :data:`CONFIG_1B`。"""

    vocab_size: int = 65536
    dim: int = 1024
    n_layers: int = 20
    n_heads: int = 16
    ffn_mult: float = 4.0
    max_seq: int = 4096
    dropout: float = 0.0
    qk_norm: bool = True
    tie_embedding: bool = True
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError(f"dim={self.dim} 不能被 n_heads={self.n_heads} 整除")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size 必须为正")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


#: light-prism-300m：d=1024 / L=20 → 319M（含 67M 嵌入）
CONFIG_300M = PrismConfig()
#: light-prism-1b：d=1536 / L=30 → 950M（含 100M 嵌入）
CONFIG_1B = PrismConfig(dim=1536, n_layers=30, n_heads=12)


#: light-prism-300m：d=1024 / L=20 → 320M（含 67M 嵌入）
CONFIG_300M = PrismConfig()
#: light-prism-1b：d=1536 / L=30 → 950M（含 100M 嵌入），非嵌入 849M
CONFIG_1B = PrismConfig(dim=1536, n_layers=30, n_heads=12)


def count_parameters(cfg: PrismConfig, non_embedding: bool = False) -> int:
    """不建模型直接算参数量。

    逐项口径（与 :class:`lightprism.model.Block` 的实际结构一一对应）：

    - attention：``qkv`` 3·d² + ``proj`` d² = 4·d²
    - QK-Norm：q/k 各 ``head_dim`` 个缩放参数 = 2·head_dim
    - SwiGLU：``w1``/``w3`` 各 d·h + ``w2`` h·d = 3·d·h，h 由 dim 与 ffn_mult 推出
    - 两个 RMSNorm：各 d 个缩放参数 = 2·d
    - 末尾 RMSNorm：d（只一份，不在每层里）
    - 词嵌入 V·d；``tie_embedding`` 时输出头共享，不重复计数

    必须与 ``LightPrismModel.count_parameters()`` 相等，
    由 ``tests/test_model.py::test_formula_matches_actual`` 兜住。
    """
    d = cfg.dim
    h = max(int(round(d * cfg.ffn_mult * 2 / 3 / 64)) * 64, 64)
    per_layer = (
        4 * d * d            # attn: qkv + proj
        + 2 * cfg.head_dim   # QK-Norm
        + 3 * d * h          # SwiGLU
        + 2 * d              # norm1 + norm2
    )
    embed = cfg.vocab_size * d
    head = 0 if cfg.tie_embedding else cfg.vocab_size * d
    if non_embedding:
        return cfg.n_layers * per_layer + d + head
    return cfg.n_layers * per_layer + d + embed + head


def from_json_file(path: str) -> PrismConfig:
    """读 configs/*.json。允许 ``name`` / ``note`` 这类说明字段，直接忽略。"""
    import json
    from pathlib import Path

    raw: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {f for f in PrismConfig.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(raw) - allowed
    return PrismConfig(**{k: v for k, v in raw.items() if k in allowed}), unknown
