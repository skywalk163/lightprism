#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LightPrism 推理性能基线 bench（T-0331）。

## 口径（写死在脚本里，不接受命令行覆盖）

* **单线程**：``torch.set_num_threads(1)``。decode 阶段多线程常因同步开销反而更慢，
  以单线程数字为准；多线程对比用 ``--mode multi`` 另跑一份分别报告。
* **贪心解码**：``argmax``，不做 temperature / top-k 采样。
* **prompt 长度 32**，预填（prefill）耗时不计入 decode 吞吐。
* **预热 10 个 token**：前 10 个 decode 步丢弃（含首次缓存命中、内存预热），
  吞吐只在剩余 token 上统计。
* 模型权重随机初始化：本脚本只测**推理路径速度**，不测生成质量。

只依赖 ``torch``，不引入 transformers / vllm 等重依赖。

用法::

    PYTHONPATH=src python scripts/bench.py --config configs/demo-3m.json --tokens 100
    PYTHONPATH=src python scripts/bench.py --config configs/demo-3m.json --mode multi
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import torch

# ---- 写死的基准口径（改这里等于改验收标准，需另开卡）-----------------------
PROMPT_LEN = 32          # 固定 prompt 长度
WARMUP_TOKENS = 10       # 丢弃前 N 个 decode 步作为预热
DEFAULT_TOKENS = 200     # 默认生成 token 数；验收可用 --tokens 调小快跑
# --------------------------------------------------------------------------


def peak_rss_mb() -> tuple[float, str]:
    """峰值常驻内存（MB）。

    Linux 读 ``/proc/self/status`` 的 ``VmHWM``；FreeBSD / 无 /proc 的平台退回
    ``resource.getrusage`` 的 ``ru_maxrss``。**Linux 与 FreeBSD 的 ``ru_maxrss``
    单位都是 KB**（已用 ``ps -o rss`` 交叉核对：FreeBSD 287660 ru ≈ 287756 KB），
    不要再按字节换算。
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM"):
                    return int(line.split()[1]) / 1024.0, "VmHWM(/proc)"
    except OSError:
        pass
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024.0, "ru_maxrss(KB)"


def build_model(config_path: str):
    from lightprism.model import LightPrismModel
    from lightprism.model_cfg import count_parameters, from_json_file

    cfg, _ = from_json_file(config_path)
    torch.manual_seed(0)
    model = LightPrismModel(cfg)
    model.eval()
    return model, cfg, count_parameters(cfg)


def run_decode(model, vocab_size: int, n_tokens: int, mode: str) -> dict:
    if mode == "single":
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(os.cpu_count() or 1)

    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, vocab_size, (1, PROMPT_LEN), generator=g)

    with torch.no_grad():
        logits, past_kv = model.incremental_forward(prompt)
    next_id = int(logits[0, -1].argmax().item())

    step_times: list[float] = []
    with torch.no_grad():
        for _ in range(n_tokens):
            step = torch.tensor([[next_id]], dtype=torch.long)
            t0 = time.perf_counter()
            logits, past_kv = model.incremental_forward(step, past_kv)
            step_times.append(time.perf_counter() - t0)
            next_id = int(logits[0, -1].argmax().item())

    timed = step_times[WARMUP_TOKENS:]
    if not timed:
        raise SystemExit(f"--tokens {n_tokens} 少于预热 {WARMUP_TOKENS}，无法统计")
    avg_s = sum(timed) / len(timed)
    tok_per_s = 1.0 / avg_s if avg_s > 0 else float("inf")
    return {
        "mode": mode,
        "threads": torch.get_num_threads(),
        "tokens_generated": n_tokens,
        "warmup_dropped": WARMUP_TOKENS,
        "timed_tokens": len(timed),
        "avg_ms_per_token": avg_s * 1000.0,
        "tok_per_s": tok_per_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LightPrism 推理性能基线 bench")
    parser.add_argument("--config", required=True, help="模型配置 JSON")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                        help=f"生成 token 数（默认 {DEFAULT_TOKENS}；预热 {WARMUP_TOKENS} 个不计入）")
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="single=单线程（验收口径，默认）；multi=多线程对比，另报一份")
    parser.add_argument("--json", action="store_true", help="只输出机器可读 JSON")
    args = parser.parse_args()

    model, cfg, params = build_model(args.config)
    result = run_decode(model, cfg.vocab_size, args.tokens, args.mode)
    rss_mb, rss_source = peak_rss_mb()

    summary = {
        "config": str(Path(args.config)),
        "model": {"dim": cfg.dim, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                  "vocab_size": cfg.vocab_size, "max_seq": cfg.max_seq},
        "params": params,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
        },
        "benchmark": result,
        "peak_rss_mb": round(rss_mb, 1),
        "rss_source": rss_source,
        "fixed_protocol": {
            "prompt_len": PROMPT_LEN,
            "warmup_tokens": WARMUP_TOKENS,
            "greedy": True,
            "prefill_excluded_from_throughput": True,
        },
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print("LightPrism 推理性能基线")
    print(f"  配置: {summary['config']}  d={cfg.dim} L={cfg.n_layers} "
          f"H={cfg.n_heads} V={cfg.vocab_size}  参数 {params:,}")
    print(f"  平台: {platform.system()} {platform.release()} {platform.machine()}  "
          f"torch {torch.__version__}  CPU {os.cpu_count()} 核")
    print(f"  口径: 单线程={result['mode'] == 'single'} 贪心  prompt={PROMPT_LEN}  "
          f"预热={WARMUP_TOKENS}  计时 token={result['timed_tokens']}")
    print(f"  速度: {result['tok_per_s']:.1f} tok/s  "
          f"({result['avg_ms_per_token']:.3f} ms/token, {result['threads']} 线程)")
    print(f"  峰值内存: {rss_mb:.1f} MB（{rss_source}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
