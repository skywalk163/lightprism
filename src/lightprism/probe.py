# -*- coding: utf-8 -*-
"""探针实验骨架：用 20% 训练成本回答「通不通」（方案文档 §7）。

三件事：

1. :class:`LinearProbe` —— 冻结主干、只训一个线性分类头，从决策 token 位置的
   隐藏状态读「该选哪个选项」；
2. :func:`evaluate_probe` —— 每个决策点报 top-1 准确率 + 宏平均；
3. :class:`PassCriteria` —— 方案 §7.3 的四条通过标准，逐条判定并给出结论。

**v0.1 的边界**：只实现 C 组（探针）。A 组（独立决策模型）与 B 组（纯生成）
需要已部署的外部系统，本仓无法自证，只能读取外部报告的指标填进来 ——
所以 :func:`judge` 接受外部指标作为输入，而不是假装能自己测出来。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tokens import CONTROL_TOKEN_MAX_RATIO, TokenSpace

__all__ = [
    "PASS_THRESHOLDS",
    "LinearProbe",
    "ProbeResult",
    "train_probe",
    "PassCriteria",
    "judge",
]

#: 方案 §7.3 的通过标准。
PASS_THRESHOLDS: Dict[str, float] = {
    "probe_accuracy_vs_A": 0.0,      # 探针宏平均准确率 ≥ A 组（差值为负即不通过）
    "pass_rate_vs_B": 0.0,           # 一次通过率 ≥ B 组
    "structure_rate": 0.95,          # 结构合法率 ≥ 95%
    "decision_density_max": CONTROL_TOKEN_MAX_RATIO,   # ≤ 2%
    "speed_drop_max": 0.15,          # 生成速度下降 ≤ 15%
}


@dataclass
class ProbeResult:
    """探针实验结果。"""

    per_point: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    macro_accuracy: float = 0.0
    n_train: int = 0
    n_test: int = 0
    params: int = 0
    #: 有独立测试集、因而被计入 macro_accuracy 的决策点数。为 0 时宏平均无意义。
    points_with_test_split: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "per_point": self.per_point,
            "macro_accuracy": round(self.macro_accuracy, 4),
            "macro_accuracy_is_meaningful": self.points_with_test_split > 0,
            "points_with_test_split": self.points_with_test_split,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "params": self.params,
        }


class LinearProbe:
    """每个决策点一个线性头。输入是隐藏状态，输出该点各选项的 logits。"""

    def __init__(self, hidden_size: int, space: Optional[TokenSpace] = None) -> None:
        import torch
        import torch.nn as nn

        self.torch = torch
        self.space = space or TokenSpace()
        self.heads = nn.ModuleDict({
            p.name: nn.Linear(hidden_size, len(p.options))
            for p in self.space.decision_points
        })

    def parameters(self):
        return self.heads.parameters()

    def forward(self, point: str, hidden):
        return self.heads[point](hidden)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _collect_hidden(model, tokenizer, samples, space: TokenSpace):
    """跑一遍模型，收「决策 token 位置的隐藏状态」与标签。

    取的是**决策 token 自己那一行**的隐藏状态（它前面已经看过任务与上文），
    不是下一个位置 —— 方案 §7.1 的探针定义是「从模型对决策 token 的隐藏状态中
    预测下一步应该选哪个」。
    """
    import torch

    from .data import build_sequence

    rows: List[Tuple[str, int, "torch.Tensor"]] = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            ids = build_sequence(sample, tokenizer, space)
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long)
            # 需要隐藏状态：直接跑前向拿 logits 前的表示
            hidden = _hidden_states(model, x)
            for pos, tid in enumerate(ids):
                point = tokenizer.point_of(tid)
                if point is None:
                    continue
                option_text = tokenizer._text_of(tid)
                option = option_text[len(f"<D:{point}="):-1]
                idx = space.point(point).options.index(option)
                rows.append((point, idx, hidden[pos]))
    return rows


def _hidden_states(model, input_ids):
    """取出最后一层 RMSNorm 之后的隐藏状态（与 lm_head 的输入一致）。"""
    import torch

    from .model import _rope_cache

    b, t = input_ids.shape
    cos, sin = _rope_cache(t, model.cfg.head_dim, model.cfg.rope_theta,
                           input_ids.device, model.embed.weight.dtype)
    x = model.embed(input_ids)
    for block in model.blocks:
        x = block(x, cos, sin)
    return model.norm_f(x)[0]


def train_probe(
    ckpt_path: str | Path,
    samples,
    tokenizer,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    test_ratio: float = 0.3,
    seed: int = 0,
    space: Optional[TokenSpace] = None,
) -> ProbeResult:
    """训练线性探针并返回每个决策点的准确率。"""
    import torch

    from .infer import load_generator

    space = space or TokenSpace()
    gen = load_generator(ckpt_path)
    model = gen.model
    for p in model.parameters():
        p.requires_grad_(False)

    rows = _collect_hidden(model, tokenizer, samples, space)
    if not rows:
        raise ValueError("样本里没有任何决策 token，无法训练探针")

    torch.manual_seed(seed)
    by_point: Dict[str, List[Tuple[int, Any]]] = {}
    for point, label, hidden in rows:
        by_point.setdefault(point, []).append((label, hidden))

    probe = LinearProbe(model.cfg.dim, space)
    result = ProbeResult(params=probe.count_parameters())
    accs: List[float] = []

    for point, items in sorted(by_point.items()):
        perm = torch.randperm(len(items))
        n_test = max(1, int(len(items) * test_ratio)) if len(items) > 2 else 0
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        if len(train_idx) == 0:
            continue

        X = torch.stack([items[i][1] for i in train_idx])
        y = torch.tensor([items[i][0] for i in train_idx])
        opt = torch.optim.AdamW(probe.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(probe.forward(point, X), y)
            loss.backward()
            opt.step()

        if n_test:
            Xt = torch.stack([items[i][1] for i in test_idx])
            yt = torch.tensor([items[i][0] for i in test_idx])
        else:
            Xt, yt = X, y
        with torch.no_grad():
            pred = probe.forward(point, Xt).argmax(-1)
        acc = float((pred == yt).float().mean())
        # n_test=0 时上面是用训练集自评的，准确率必然虚高。必须显式标出来，
        # 否则 1.0 会被当成泛化能力（骨架阶段样本只有 3 条，实测就会这样）。
        eval_split = "test" if n_test else "train(样本不足，未切分；此准确率不代表泛化)"
        result.per_point[point] = {
            "n_train": len(train_idx),
            "n_test": n_test,
            "accuracy": round(acc, 4),
            "eval_split": eval_split,
            "n_classes": len(space.point(point).options),
            # 类别数决定的随机基线，低于它说明探针什么也没学到
            "random_baseline": round(1 / len(space.point(point).options), 4),
        }
        # 宏平均只统计有独立测试集的点，避免被自评数字抬高
        if n_test:
            accs.append(acc)

    result.macro_accuracy = sum(accs) / len(accs) if accs else 0.0
    result.points_with_test_split = len(accs)
    result.n_train = sum(v["n_train"] for v in result.per_point.values())
    result.n_test = sum(v["n_test"] for v in result.per_point.values())
    return result


@dataclass
class PassCriteria:
    """四条标准的判定结果。"""

    probe_macro_accuracy: float = 0.0
    a_group_accuracy: Optional[float] = None
    pass_rate: Optional[float] = None
    b_group_pass_rate: Optional[float] = None
    structure_rate: Optional[float] = None
    decision_density: Optional[float] = None
    speed_drop: Optional[float] = None

    def checks(self) -> List[Dict[str, Any]]:
        """逐条判定。缺失的指标记 ``unknown``，**不当作通过**。"""
        out: List[Dict[str, Any]] = []

        def add(name: str, ok: Optional[bool], detail: str) -> None:
            out.append({"criterion": name,
                        "result": "unknown" if ok is None else ("pass" if ok else "fail"),
                        "detail": detail})

        if self.a_group_accuracy is None:
            add("探针准确率 ≥ A 组", None, "缺 A 组指标（需外部系统报告）")
        else:
            add("探针准确率 ≥ A 组",
                self.probe_macro_accuracy >= self.a_group_accuracy,
                f"C={self.probe_macro_accuracy:.3f} vs A={self.a_group_accuracy:.3f}")

        if self.pass_rate is None or self.b_group_pass_rate is None:
            add("一次通过率 ≥ B 组", None, "缺 C 或 B 组一次通过率")
        else:
            add("一次通过率 ≥ B 组", self.pass_rate >= self.b_group_pass_rate,
                f"C={self.pass_rate:.3f} vs B={self.b_group_pass_rate:.3f}")

        if self.structure_rate is None:
            add("结构合法率 ≥ 95%", None, "缺结构合法率")
        else:
            add("结构合法率 ≥ 95%", self.structure_rate >= PASS_THRESHOLDS["structure_rate"],
                f"{self.structure_rate:.3f}")

        ok_density = None
        if self.decision_density is not None:
            ok_density = self.decision_density <= PASS_THRESHOLDS["decision_density_max"]
        add("决策密度 ≤ 2%", ok_density,
            "缺决策密度" if self.decision_density is None else f"{self.decision_density:.4f}")

        ok_speed = None
        if self.speed_drop is not None:
            ok_speed = self.speed_drop <= PASS_THRESHOLDS["speed_drop_max"]
        add("生成速度下降 ≤ 15%", ok_speed,
            "缺速度数据" if self.speed_drop is None else f"{self.speed_drop:.3f}")
        return out


def judge(criteria: PassCriteria) -> Dict[str, Any]:
    """汇总结论。**任一条 unknown 都不算通过** —— 不能拿缺数据当好消息。"""
    checks = criteria.checks()
    fails = [c for c in checks if c["result"] == "fail"]
    unknowns = [c for c in checks if c["result"] == "unknown"]
    verdict = "pass" if not fails and not unknowns else ("fail" if fails else "incomplete")
    return {
        "verdict": verdict,
        "checks": checks,
        "next_step": {
            "pass": "进入阶段二「序列内协同」训练",
            "fail": "按 docs/探针实验设计.md §6 的固定顺序调参后重跑阶段一",
            "incomplete": "补齐缺失指标（尤其 A/B 组）后重新判定",
        }[verdict],
    }
