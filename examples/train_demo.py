#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端示例：训练 4 步 → 生成一条序列 → 训探针 → 判定。

跑法（需要 torch）：
    PYTHONPATH=src python examples/train_demo.py

用 3 条样本训出来的模型只会重复高频片段，**输出不是能力证明**；
这个脚本的价值是证明「数据 → 词表 → 模型 → 生成 → 探针 → 判定」这条链路是通的。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lightprism.data import load_jsonl
from lightprism.infer import GenConfig, load_generator
from lightprism.probe import PassCriteria, judge, train_probe
from lightprism.train import TrainConfig, Trainer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "samples.jsonl"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        trainer = Trainer(TrainConfig(
            data=str(DATA), out_dir=str(out), steps=4,
            seq_len=128, batch_size=3, log_every=2, warmup=1,
        ))
        summary = trainer.train()
        ckpt = trainer.save()
        print("\n== 训练完成 ==")
        print(json.dumps(summary, ensure_ascii=False))

        gen = load_generator(ckpt, GenConfig(max_new_tokens=24))
        out_gen = gen.generate("检查磁盘使用率", decisions={"EDIT": "FR"})
        print("\n== 生成（未训练模型，仅验证结构约束生效）==")
        print("决策:", out_gen["decisions"])
        print("结构完整:", out_gen["structure_ok"],
              "| 是否被截断:", out_gen["truncated"])
        print("片段:", "".join(out_gen["text"])[:80])

        samples, _ = load_jsonl(DATA)
        result = train_probe(ckpt, samples, gen.tokenizer, epochs=30)
        print("\n== 探针 ==")
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        print("\n== 判定（缺 A/B 组指标 → incomplete，不是 pass）==")
        print(json.dumps(judge(PassCriteria(probe_macro_accuracy=result.macro_accuracy)),
                         ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
