# -*- coding: utf-8 -*-
"""T-0332 demo checkpoint smoke：加载 checkpoint，生成 32 token 不崩。

用法::

    PYTHONPATH=src python -m lightprism.demo.smoke --ckpt checkpoints/demo-3m/model.pt

判据（卡面）：
* checkpoint 能被还原（模型 + 词表布局 + 底座词表本体）；
* 贪心生成 32 个 token，不抛异常、不越界；
* 生成结果带回词表可解码。

**诚实边界**：smoke 只验证「管线通、不崩」，不判结构合法率 —— 50 步的小模型
大概率吐不出完整序列，合法率在训练报告里按实测统计，不在 smoke 里硬凑通过。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .train_demo import DemoVocab, ROOT

__all__ = ["SmokeResult", "load_demo_generator", "run_smoke", "main"]


class DemoGenerator:
    """demo checkpoint 的极简贪心生成器（带结构约束，口径对齐 infer.PrismGenerator）。"""

    def __init__(self, model, vocab: DemoVocab, max_new_tokens: int = 32) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.vocab = vocab
        self.max_new_tokens = max_new_tokens
        self.model.eval()

    def generate(self, task: str = "", max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        torch = self.torch
        n_new = max_new_tokens or self.max_new_tokens
        st = self.vocab.structure
        ids: List[int] = [st["<TASK>"]]
        ids.extend(self.vocab.encode(task))
        ids.append(st["</TASK>"])

        generated: List[int] = []
        past_kv = None
        use_cache = hasattr(self.model, "incremental_forward")
        if use_cache:
            window = torch.tensor([ids[-self.model.cfg.max_seq:]], dtype=torch.long)
            with torch.no_grad():
                logits, past_kv = self.model.incremental_forward(window)
            last = logits[0, -1, :]
        else:
            last = None

        for _ in range(n_new):
            if not use_cache:
                window = torch.tensor([ids[-self.model.cfg.max_seq:]], dtype=torch.long)
                with torch.no_grad():
                    logits, _ = self.model(window)
                last = logits[0, -1, :]
            next_id = int(torch.argmax(last).item())
            # 越界兜底：落到文本首 token，绝不吐 pad/unk 之外的非法 id。
            if next_id >= self.vocab.vocab_size:
                next_id = 0
            ids.append(next_id)
            generated.append(next_id)

            if use_cache:
                step = torch.tensor([[next_id]], dtype=torch.long)
                with torch.no_grad():
                    logits, past_kv = self.model.incremental_forward(step, past_kv)
                last = logits[0, -1, :]
                if len(ids) >= self.model.cfg.max_seq:
                    past_kv = None

        pieces = self.vocab.decode(generated)
        return {
            "ids": generated,
            "text": pieces,
            "generated_tokens": len(generated),
            "truncated": len(generated) >= n_new,
        }


def load_demo_generator(ckpt_path: str | Path, max_new_tokens: int = 32) -> DemoGenerator:
    """从 checkpoint 还原模型与 :class:`DemoVocab`。"""
    import torch

    from lightprism.model import LightPrismModel, PrismConfig

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LightPrismModel(PrismConfig(**blob["model_config"]))
    model.load_state_dict(blob["model"])
    model.eval()

    layout = blob["tokenizer_layout"]
    base_vocab = layout.get("base_vocab") or []
    if not base_vocab:
        raise ValueError("checkpoint 缺少 base_vocab，无法还原词表")
    vocab = DemoVocab({
        "tokens": base_vocab,
        "control_layout": layout["layout"],
    })
    return DemoGenerator(model, vocab, max_new_tokens=max_new_tokens)


def run_smoke(ckpt_path: str | Path, max_new_tokens: int = 32,
              task: str = "写一个加法程序") -> SmokeResult:
    gen = load_demo_generator(ckpt_path, max_new_tokens=max_new_tokens)
    out = gen.generate(task, max_new_tokens=max_new_tokens)
    return SmokeResult(
        ckpt=str(ckpt_path),
        generated_tokens=out["generated_tokens"],
        text=out["text"],
        truncated=out["truncated"],
        vocab_size=gen.vocab.vocab_size,
        params=gen.model.count_parameters(),
    )


@dataclasses.dataclass
class SmokeResult:
    ckpt: str
    generated_tokens: int
    text: str
    truncated: bool
    vocab_size: int
    params: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ckpt": self.ckpt,
            "generated_tokens": self.generated_tokens,
            "text": self.text,
            "truncated": self.truncated,
            "vocab_size": self.vocab_size,
            "params": self.params,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="demo checkpoint smoke（T-0332）")
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "demo-3m" / "model.pt"))
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--task", default="写一个加法程序")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_smoke(args.ckpt, max_new_tokens=args.max_new_tokens, task=args.task)
    payload = result.as_dict()
    payload["ok"] = result.generated_tokens > 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
