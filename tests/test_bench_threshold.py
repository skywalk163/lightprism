# -*- coding: utf-8 -*-
"""``bench.py`` ``--min-tok-s`` 阈值断言回归测试（T-0340）。

阈值语义（只做断言，不改 ``PROMPT_LEN`` / ``WARMUP_TOKENS`` / 单线程口径）：

* 默认 ``--min-tok-s 0``：不校验，行为与旧版一致，退出码 0；
* 实测 tok/s 低于阈值：退出码 1；达到 / 超过阈值：退出码 0；
* 测试用 monkeypatch 固定一个合成吞吐结果，不真实建模型 / 解码，
  避免把阈值逻辑测试绑到本机硬件性能上。

无 torch 的环境下各用例在函数内 skip（不整模块跳过，保证 pytest 收集到
用例、退出码为 0）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "scripts" / "bench.py"


def _load_bench():
    """惰性加载 bench 模块；torch 不可用时 skip 用例。"""
    try:
        spec = importlib.util.spec_from_file_location("lp_bench", BENCH)
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        return bench
    except ModuleNotFoundError as exc:
        pytest.skip(f"bench 依赖不可用：{exc}")


def _fake_result(tok_per_s: float) -> dict:
    return {
        "mode": "single",
        "threads": 1,
        "tokens_generated": 200,
        "warmup_dropped": 10,
        "timed_tokens": 190,
        "avg_ms_per_token": 1000.0 / tok_per_s,
        "tok_per_s": tok_per_s,
    }


def _run_main(monkeypatch, tok_per_s: float, min_tok_s: float | None = None) -> int:
    bench = _load_bench()
    from lightprism.model_cfg import PrismConfig
    fake_cfg = PrismConfig(vocab_size=128, dim=64, n_layers=2,
                           n_heads=4, max_seq=64)
    monkeypatch.setattr(bench, "build_model", lambda config: (None, fake_cfg, 0))
    monkeypatch.setattr(
        bench, "run_decode",
        lambda model, vocab, n, mode: _fake_result(tok_per_s),
    )
    argv = ["bench.py", "--config", "x.json", "--tokens", "20"]
    if min_tok_s is not None:
        argv += ["--min-tok-s", str(min_tok_s)]
    monkeypatch.setattr(sys, "argv", argv)
    return bench.main()


def test_default_threshold_zero_does_not_fail(monkeypatch):
    """默认阈值 0：即使速度很低也退出 0（行为不变）。"""
    assert _run_main(monkeypatch, tok_per_s=50.0) == 0


def test_below_threshold_exits_one(monkeypatch):
    """实测低于 --min-tok-s 必须退出 1。"""
    assert _run_main(monkeypatch, tok_per_s=77.8, min_tok_s=200.0) == 1


def test_above_threshold_exits_zero(monkeypatch):
    """实测达到阈值退出 0。"""
    assert _run_main(monkeypatch, tok_per_s=250.0, min_tok_s=200.0) == 0


def test_threshold_boundary_inclusive(monkeypatch):
    """恰好等于阈值视为达标（>= 判据）。"""
    assert _run_main(monkeypatch, tok_per_s=200.0, min_tok_s=200.0) == 0


def test_cli_default_threshold_is_zero():
    """CLI 默认值必须是 0（不校验），保证旧调用方行为不变。"""
    src = BENCH.read_text(encoding="utf-8")
    assert '"--min-tok-s"' in src
    assert "default=0.0" in src
