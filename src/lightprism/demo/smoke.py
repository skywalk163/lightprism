# -*- coding: utf-8 -*-
"""T-0332 demo checkpoint smoke：加载 checkpoint，生成 32 token 不崩。

用法::

    PYTHONPATH=src python -m lightprism.demo.smoke --ckpt checkpoints/demo-3m/model.pt

判据（卡面）：
* checkpoint 能被还原（模型 + 词表布局 + 底座词表本体）；
* 贪心生成 32 个 token，不抛异常、不越界；
* 生成结果带回词表可解码。

生成器自 T-0345 起走受约束路径（:mod:`lightprism.demo.infer_demo`）：结构状态机、
空槽 mask 与 KV cache 都复用 ``infer.PrismGenerator``。``scripts/demo.sh`` /
``make demo`` / ``lightprism chat`` 都通过本模块的 :func:`load_demo_generator`
进入同一条路径，不再各自实现裸 argmax。

**诚实边界**：smoke 只验证「管线通、不崩」；约束版结构合法率是**产品指标**，
由外部状态机强制，不能当模型能力证据。能力侧看无约束裸 argmax 的结构合法率
与 ``light ai check`` 的代码段合法率，见 ``reports/约束解码接线报告.md``。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .infer_demo import ConstrainedDemoGenerator, load_demo_generator
from .train_demo import ROOT

__all__ = [
    "DemoGenerator",
    "ConstrainedDemoGenerator",
    "load_demo_generator",
    "SmokeResult",
    "run_smoke",
    "main",
]

#: 兼容旧 import 名；实际实现是受约束生成器。
DemoGenerator = ConstrainedDemoGenerator


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
