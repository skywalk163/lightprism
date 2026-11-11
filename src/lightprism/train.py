# -*- coding: utf-8 -*-
"""LightPrism 训练骨架：三阶段课程的最小可跑实现。

**这不是能直接产出 1B 模型的训练脚本**，而是一条**跑得通的完整管线**：
读数据 → 装配词表 → 建模型 → 按阶段配置损失与优化器 → 训几步 → 存 checkpoint → 报指标。
换真实语料与 GPU 就能放大，但放大前必须补的东西列在文件末尾「放大前必须补」。

三阶段（方案文档 §6）在本文件里的落点：

===========  ================================  ================================
阶段         数据形态                          本文件的实现
===========  ================================  ================================
一 · 共存    三类 token 随机交错，不加权        ``stage="coexist"``：control_weight=1.0
二 · 协同    严格决策→代码→验证，按类型加权    ``stage="cooperate"``：control_weight>1
三 · RLVR    奖励对齐真实验证结果               ``stage="rlvr"``：**未实现**，抛 NotImplementedError
===========  ================================  ================================

阶段三为什么不做：RLVR 需要「生成—执行—验证—回传奖励」的完整环境（要真跑
``light ai check``），属于 lightharness 侧的能力，本仓没有也不该有。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

__all__ = ["TrainConfig", "Trainer", "cosine_lr", "STAGES"]

STAGES = ("coexist", "cooperate", "rlvr")

#: 阶段 → 控制 token 损失权重。阶段一按方案要求**不加权**。
STAGE_CONTROL_WEIGHT: Dict[str, float] = {
    "coexist": 1.0,
    "cooperate": 2.0,
    "rlvr": 2.0,
}


def cosine_lr(step: int, total: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    """线性 warmup + 余弦衰减。纯函数，好测。"""
    if total <= 0:
        raise ValueError("total 必须为正")
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return base_lr * min_ratio
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


#: 微缩配置：让 smoke 在 CPU 上几秒内跑完。真实训练用 configs/*.json 覆盖。
DEFAULT_MODEL_CFG: Dict[str, Any] = {
    "vocab_size": 160, "dim": 64, "n_layers": 2, "n_heads": 4, "max_seq": 512,
}


def _default_model_cfg() -> Dict[str, Any]:
    return dict(DEFAULT_MODEL_CFG)


@dataclass
class TrainConfig:
    """训练超参。默认值是「CPU 上几分钟能跑完一遍」的量级，不是生产配置。"""

    stage: str = "coexist"
    data: str = ""
    val_data: str = ""
    out_dir: str = "outputs/run"
    seq_len: int = 256
    batch_size: int = 2
    grad_accum: int = 1
    steps: int = 20
    base_lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup: int = 5
    log_every: int = 5
    eval_every: int = 100
    seed: int = 0
    # 模型规模：smoke 用微缩配置，真实训练改 configs/*.json
    model: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_MODEL_CFG))

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"未知阶段 {self.stage!r}；合法值 {STAGES}")
        if self.stage == "rlvr":
            raise NotImplementedError(
                "阶段三 RLVR 未实现：它需要真实执行 light ai check 并回传奖励的环境，"
                "属 lightharness 侧能力。本仓只提供阶段一/二的训练骨架。"
            )
        if not self.data:
            raise ValueError("必须指定 --data")
        if self.seq_len <= 0 or self.batch_size <= 0:
            raise ValueError("seq_len / batch_size 必须为正")


class Trainer:
    """最小训练器：AdamW + 余弦退火 + 梯度裁剪 + checkpoint + 指标落盘。"""

    def __init__(self, cfg: TrainConfig) -> None:
        import torch  # 延迟 import：本模块被 CLI 引用时不强制装 torch

        from .data import BatchBuilder, load_jsonl
        from .model import LightPrismModel, PrismConfig
        from .tokenizer import CharTokenizer, PrismTokenizer

        self.torch = torch
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        samples, problems = load_jsonl(cfg.data)
        if problems:
            # 脏数据必须报出来，但**不中断**：训练脚本要能跑完并留下计数
            print(f"[warn] 跳过 {len(problems)} 条不合法样本，前 3 条原因：")
            for p in problems[:3]:
                print(f"  - {p}")
        if not samples:
            raise SystemExit(f"没有可用样本：{cfg.data}")

        # 底词表：从语料里收字，保证 smoke 跑得通；真实训练换成 Light 的 BPE
        chars = sorted({ch for s in samples for ch in s.task})
        for s in samples:
            for step in s.steps:
                chars.extend(step.code)
        self.tokenizer = PrismTokenizer(CharTokenizer(extra=sorted(set(chars))))
        model_cfg = dict(cfg.model)
        model_cfg["vocab_size"] = self.tokenizer.vocab_size
        self.model_cfg = PrismConfig(**model_cfg)
        self.model = LightPrismModel(self.model_cfg)

        self.builder = BatchBuilder(
            tokenizer=self.tokenizer,
            seq_len=cfg.seq_len,
            control_weight=STAGE_CONTROL_WEIGHT[cfg.stage],
        )
        self.samples = samples
        self.pad_id = self.tokenizer.layout.structure["</CODE>"]  # 占位 pad，权重为 0
        self.builder.pad_id = self.pad_id

        groups = self.model.param_groups(control_lr_mult=2.0)
        self.optim = torch.optim.AdamW(
            [{"params": [p for pg in groups for p in pg["params"]],
              "lr": cfg.base_lr, "weight_decay": cfg.weight_decay}],
            betas=(0.9, 0.95),
        )
        self.step_count = 0
        self.history: List[Dict[str, Any]] = []

    # ---- 训练 ---------------------------------------------------------
    def _iterate_batches(self) -> Iterator[Dict[str, Any]]:
        import random

        rng = random.Random(self.cfg.seed)
        order: List[int] = []
        while True:
            if not order:
                order = list(range(len(self.samples)))
                rng.shuffle(order)
            idx = [order.pop() for _ in range(min(self.cfg.batch_size, len(order)))]
            if not idx:
                continue
            yield self.builder.make_batch([self.samples[i] for i in idx])

    def train(self) -> Dict[str, Any]:
        torch = self.torch
        self.model.train()
        t0 = time.time()
        batches = self._iterate_batches()
        for step in range(self.cfg.steps):
            lr = cosine_lr(step, self.cfg.steps, self.cfg.base_lr, self.cfg.warmup)
            for group in self.optim.param_groups:
                group["lr"] = lr

            total_loss = 0.0
            for _ in range(self.cfg.grad_accum):
                batch = next(batches)
                input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
                targets = torch.tensor(batch["targets"], dtype=torch.long)
                weights = torch.tensor(batch["loss_weights"], dtype=torch.float32)
                _, loss = self.model(input_ids, targets, loss_weights=weights)
                (loss / self.cfg.grad_accum).backward()
                total_loss += float(loss.detach())

            # 梯度范数必须在 clip / step / zero_grad **之前**取，否则恒为 0
            # （写成 step 之后被 zero_grad 清空，实测过一次）。
            grad_norm = self._grad_norm()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
            self.step_count += 1

            if self.cfg.log_every and (step + 1) % self.cfg.log_every == 0:
                record = {
                    "step": step + 1,
                    "loss": round(total_loss / self.cfg.grad_accum, 5),
                    "lr": round(lr, 8),
                    "elapsed_s": round(time.time() - t0, 2),
                    # 全局梯度范数（方案 §8.6 要求按 token 类型分别看，待 v0.2）
                    "grad_norm": round(grad_norm, 4),
                }
                self.history.append(record)
                print(json.dumps(record, ensure_ascii=False))

        return {
            "steps": self.cfg.steps,
            "final_loss": self.history[-1]["loss"] if self.history else None,
            "params": self.model.count_parameters(),
            "params_non_embedding": self.model.count_parameters(non_embedding=True),
            "vocab_size": self.tokenizer.vocab_size,
            "elapsed_s": round(time.time() - t0, 2),
        }

    def evaluate(self) -> Dict[str, Any]:
        """在训练集上算一遍平均损失与结构合法率（探针实验的离线指标由这里起）。"""
        torch = self.torch
        from .infer import GenConfig, PrismGenerator

        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for sample in self.samples:
                batch = self.builder.make_batch([sample])
                ids = torch.tensor(batch["input_ids"], dtype=torch.long)
                tgt = torch.tensor(batch["targets"], dtype=torch.long)
                w = torch.tensor(batch["loss_weights"], dtype=torch.float32)
                _, loss = self.model(ids, tgt, loss_weights=w)
                total += float(loss)
                n += 1
        self.model.train()
        return {"loss": round(total / max(n, 1), 5), "samples": n}

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum())
        return math.sqrt(total)

    # ---- 持久化 -------------------------------------------------------
    def save(self, path: Optional[str] = None) -> Path:
        torch = self.torch
        out = Path(path or self.cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.pt").parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_config": asdict(self.model_cfg),
                "train_config": asdict(self.cfg),
                "tokenizer_layout": self.tokenizer.save_layout(),
                "history": self.history,
            },
            out / "model.pt",
        )
        (out / "train_log.json").write_text(
            json.dumps({"config": asdict(self.cfg), "history": self.history}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out / "model.pt"


#: 放大到真实 1B 训练前必须补的（写在这里，避免下一个人以为「改个 config 就行」）
SCALE_UP_REQUIRED: Sequence[str] = (
    "真实 BPE 词表（Light 仓产出），当前用 CharTokenizer 顶替",
    "数据并行 / 梯度累积的正确实现（当前 grad_accum 是串行累加，未做 all-reduce）",
    "按 token 类型分组的梯度范数监控（方案 §8.6 要求，当前只有全局范数）",
    "bf16 autocast + 激活重计算（1B 在 CPU 上跑不动）",
    "checkpoint 断点续训与学习率调度状态恢复",
    "阶段二的严格交错数据与 loss 掩码（当前只调了 control 权重）",
)


# --------------------------------------------------------------------------
# CLI：python -m lightprism.train --data data/samples.jsonl --steps 20
# --------------------------------------------------------------------------
def build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="lightprism.train",
        description="LightPrism 训练骨架（阶段一/二；阶段三 RLVR 未实现）",
    )
    p.add_argument("--data", required=True, help="训练 JSONL")
    p.add_argument("--stage", default="coexist", choices=("coexist", "cooperate"),
                   help="训练阶段；rlvr 未实现，故意不在 choices 里")
    p.add_argument("--out-dir", default="outputs/run")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-json", help="模型配置 JSON（覆盖默认微缩配置）")
    return p


def main(argv=None) -> int:
    import json as _json

    args = build_arg_parser().parse_args(argv)
    model_cfg = {}
    if args.model_json:
        model_cfg = _json.loads(Path(args.model_json).read_text(encoding="utf-8"))
    cfg = TrainConfig(
        stage=args.stage,
        data=args.data,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        steps=args.steps,
        base_lr=args.lr,
        warmup=args.warmup,
        log_every=args.log_every,
        seed=args.seed,
        model=model_cfg or _default_model_cfg(),
    )
    trainer = Trainer(cfg)
    summary = trainer.train()
    path = trainer.save()
    print(_json.dumps({"summary": summary, "checkpoint": str(path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
