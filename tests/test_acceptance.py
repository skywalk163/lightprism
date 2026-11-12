# -*- coding: utf-8 -*-
"""验收脚本自身的回归测试（T-0334）。

卡面要求「用假 checkpoint 测逻辑」。这里覆盖：

* 三条阈值判据（V2 合法率 / V4 速度 / V5 内存）的边界；
* 生成结果里 ``<CODE>…</CODE>`` 的提取；
* ``light ai check --json`` 输出的解析（含前置日志、空输出）；
* checker 诊断的 error 级判定（不放过无 severity 字段的诊断）；
* checker 入口探测（``LIGHT_CHECK_CLI``）与一次真实子进程调用（假 checker）；
* V2 的抽样判定：全空输出必须 fail，完整代码 + 无诊断必须 pass；
* V7 在缺 Makefile 时必须 fail（判据入口缺失，不冒充通过）；
* 假 checkpoint：用 4096 词表 + 迷你模型存盘，能被 smoke 加载并生成。

需要 torch 的用例与 ``test_train_demo.py`` 同口径：``importorskip``。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "data" / "vocab_4096.json"
CONFIG_PATH = ROOT / "configs" / "demo-3m.json"


def _load_acceptance():
    spec = importlib.util.spec_from_file_location(
        "lp_acceptance", ROOT / "scripts" / "acceptance.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lp_acceptance"] = mod
    spec.loader.exec_module(mod)
    return mod


acceptance = _load_acceptance()


# --------------------------------------------------------------------------- #
# 1. 阈值判据
# --------------------------------------------------------------------------- #
def test_speed_judge_boundary():
    assert acceptance.judge_speed(acceptance.SPEED_TARGET) == "pass"
    assert acceptance.judge_speed(200.01) == "pass"
    assert acceptance.judge_speed(199.99) == "fail"
    assert acceptance.judge_speed(65.97) == "fail"


def test_memory_judge_boundary():
    assert acceptance.judge_memory(acceptance.MEMORY_LIMIT_MB) == "pass"
    assert acceptance.judge_memory(280.4) == "pass"
    assert acceptance.judge_memory(acceptance.MEMORY_LIMIT_MB + 0.1) == "fail"


def test_code_rate_judge_boundary():
    # 20 条里 16 条合法 = 80% → pass；15 条 = 75% → fail
    assert acceptance.judge_rate(16, 20) == "pass"
    assert acceptance.judge_rate(15, 20) == "fail"
    assert acceptance.judge_rate(0, 0) == "fail"
    assert acceptance.judge_rate(3, 4) == "fail"      # 75%，阈值是 80% 不是 3/4


def test_thresholds_are_hardcoded_from_plan():
    assert acceptance.SPEED_TARGET == 200.0
    assert acceptance.MEMORY_LIMIT_MB == 8192
    assert acceptance.CODE_SAMPLES == 20
    assert acceptance.CODE_PASS_RATE == 0.80


# --------------------------------------------------------------------------- #
# 2. 代码段提取
# --------------------------------------------------------------------------- #
def test_extract_code_with_markers():
    text = "<TASK>x</TASK>\n<CODE>\n设 x 为 1\n</CODE>\n<V:PASS>"
    assert acceptance.extract_code(text) == "设 x 为 1"


def test_extract_code_without_markers_returns_trimmed_text():
    assert acceptance.extract_code("  设 x 为 1  ") == "设 x 为 1"


def test_extract_code_whitespace_only_is_empty():
    assert acceptance.extract_code(" " * 64) == ""
    assert acceptance.extract_code("") == ""


def test_extract_code_multiline():
    text = "<CODE>\n设 a 为 1\n设 b 为 2\n</CODE>"
    assert acceptance.extract_code(text) == "设 a 为 1\n设 b 为 2"


# --------------------------------------------------------------------------- #
# 3. checker 输出解析与严重度
# --------------------------------------------------------------------------- #
def test_parse_check_output_plain_json():
    payload = {"ok": True, "diagnostics": []}
    assert acceptance.parse_check_output(json.dumps(payload)) == payload


def test_parse_check_output_tolerates_prefix_logs():
    text = "[1/2] 语法检查\n{\"ok\": false, \"diagnostics\": [{\"severity\": \"error\"}]}\n"
    parsed = acceptance.parse_check_output(text)
    assert parsed is not None and parsed["ok"] is False
    assert parsed["diagnostics"][0]["severity"] == "error"


def test_parse_check_output_empty_and_garbage():
    assert acceptance.parse_check_output("") is None
    assert acceptance.parse_check_output("not json at all") is None


def test_error_diagnostic_detection():
    assert acceptance._is_error_diag({"severity": "error"}) is True
    assert acceptance._is_error_diag({"level": "fatal"}) is True
    assert acceptance._is_error_diag({"severity": "warning"}) is False
    # 没有严重度字段时按 error 计（保守，不放宽）
    assert acceptance._is_error_diag({"code": "L1001"}) is True


def test_empty_program_checker_blind_spot_documented():
    """判据盲区必须写进脚本：空程序经 checker 可能是 ok，不能算 V2 合法。"""
    src = (ROOT / "scripts" / "acceptance.py").read_text(encoding="utf-8")
    assert "empty_program_control" in src
    assert "盲区" in src


# --------------------------------------------------------------------------- #
# 4. checker 入口与一次真实调用
# --------------------------------------------------------------------------- #
def _write_fake_checker(dirpath: Path, payload: dict, exit_code: int = 0) -> Path:
    cli = dirpath / "cli" / "light.py"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text(
        "# -*- coding: utf-8 -*-\n"
        "import json, sys\n"
        f"print(json.dumps({payload!r}))\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    return cli


def test_find_checker_honors_env(tmp_path, monkeypatch):
    cli = _write_fake_checker(tmp_path, {"ok": True, "diagnostics": []})
    monkeypatch.setenv("LIGHT_CHECK_CLI", str(cli))
    monkeypatch.delenv("LIGHT_REPO", raising=False)
    found = acceptance.find_light_checker()
    assert found is not None
    assert found["label"] == "env:LIGHT_CHECK_CLI"
    assert found["path"] == str(cli)


def test_run_light_check_reads_error_diagnostics(tmp_path):
    cli = _write_fake_checker(
        tmp_path, {"ok": False, "diagnostics": [{"code": "L1001", "severity": "error"}]},
        exit_code=1)
    checker = {"cmd": [sys.executable, str(cli)], "cwd": str(tmp_path),
               "repo": str(tmp_path), "label": "fake", "path": str(cli)}
    out = acceptance.run_light_check(checker, "设 x 为 1")
    assert out["parsed"] is True
    assert out["ok"] is False
    assert out["n_errors"] == 1
    assert out["n_diagnostics"] == 1


# --------------------------------------------------------------------------- #
# 5. V2 抽样判定（假 generator + 假 checker）
# --------------------------------------------------------------------------- #
class _FakeGen:
    def __init__(self, text: str):
        self.text = text
        self.max_new_tokens = 64

    def generate(self, _task, max_new_tokens=None):
        return {"text": self.text, "generated_tokens": max_new_tokens or self.max_new_tokens,
                "ids": [0]}


class _FakeCtx(acceptance.Ctx):
    def __init__(self, gen=None, measure=None, checker=None):
        super().__init__(ckpt=None)
        self._fake_gen = gen
        self._fake_measure = measure
        self._fake_checker = checker

    def generator(self, max_new_tokens=32):
        assert self._fake_gen is not None, "测试未提供假 generator"
        return self._fake_gen

    def decode_measure(self):
        return self._fake_measure

    def checker(self):
        return self._fake_checker


def _ok_checker(tmp_path: Path) -> dict:
    cli = _write_fake_checker(tmp_path, {"ok": True, "diagnostics": []})
    return {"cmd": [sys.executable, str(cli)], "cwd": str(tmp_path),
            "repo": str(tmp_path), "label": "fake-ok", "path": str(cli)}


def test_v2_all_whitespace_output_fails(tmp_path):
    gen = _FakeGen(" " * 64)
    ctx = _FakeCtx(gen=gen, checker=_ok_checker(tmp_path))
    res = acceptance.check_v2(ctx)
    assert res.verdict == "fail"
    assert res.metric["valid"] == 0
    assert res.metric["with_code_marker"] == 0
    # 盲区对照：空程序 checker 确实判 ok，但不得给 V2 加分
    assert res.metric["empty_program_control"]["ok"] is True


def test_v2_complete_code_passes(tmp_path):
    gen = _FakeGen("<CODE>\n设 x 为 1\n</CODE>")
    ctx = _FakeCtx(gen=gen, checker=_ok_checker(tmp_path))
    res = acceptance.check_v2(ctx)
    assert res.verdict == "pass"
    assert res.metric["valid"] == res.metric["samples"] == 20
    assert res.metric["with_code_marker"] == 20


def test_v2_missing_checker_is_unverified(tmp_path):
    ctx = _FakeCtx(gen=_FakeGen("<CODE>设 x 为 1</CODE>"), checker=None)
    res = acceptance.check_v2(ctx)
    assert res.verdict == "unverified"
    assert "light ai check" in res.detail


# --------------------------------------------------------------------------- #
# 6. V4 / V5 逻辑（假测量结果）
# --------------------------------------------------------------------------- #
def _measure(tok_per_s: float, peak_mb: float) -> dict:
    return {
        "tok_per_s": tok_per_s, "avg_ms_per_token": 1000.0 / tok_per_s,
        "timed_tokens": 190, "threads": 1, "params": 3443520,
        "peak_mb": peak_mb, "peak_source": "test",
        "elapsed_s": 1.0,
        "protocol": {"prompt_len": 32, "warmup_tokens": 10, "tokens": 200,
                     "mode": "single", "greedy": True, "prefill_excluded": True},
    }


def test_v4_fails_below_200_tok_s():
    res = acceptance.check_v4(_FakeCtx(measure=_measure(65.97, 280.4)))
    assert res.verdict == "fail"
    assert res.metric["tok_per_s"] == 65.97
    assert res.metric["gap_ratio"] < 1.0


def test_v4_passes_at_or_above_200_tok_s():
    res = acceptance.check_v4(_FakeCtx(measure=_measure(283.0, 280.4)))
    assert res.verdict == "pass"
    assert res.metric["tok_per_s"] == 283.0


def test_v5_passes_small_model_memory():
    res = acceptance.check_v5(_FakeCtx(measure=_measure(200.0, 280.4)))
    assert res.verdict == "pass"
    assert res.metric["peak_mb"] == 280.4


def test_v5_fails_over_8gb():
    res = acceptance.check_v5(_FakeCtx(measure=_measure(200.0, 9000.0)))
    assert res.verdict == "fail"


def test_v5_unavailable_metric_is_unverified_not_pass():
    """拿不到峰值口径时不能判 pass（0.0MB 不是「内存很小」）。"""
    m = _measure(200.0, 0.0)
    m["peak_source"] = "unavailable"
    res = acceptance.check_v5(_FakeCtx(measure=m))
    assert res.verdict == "unverified"


@pytest.mark.skipif(__import__("os").name != "nt", reason="仅 Windows 有峰值工作集")
def test_windows_peak_memory_is_available():
    mb, src = acceptance.peak_memory_mb()
    assert src == "PeakWorkingSetSize(Windows)"
    assert mb > 10.0


# --------------------------------------------------------------------------- #
# 7. V7：缺 Makefile 必须判 fail（入口缺失）
# --------------------------------------------------------------------------- #
def test_v7_missing_makefile_fails(tmp_path):
    ctx = acceptance.Ctx(ckpt=None, repo_root=tmp_path)
    try:
        res = acceptance.check_v7(ctx)
    finally:
        ctx.close()
    assert res.verdict == "fail"
    assert "Makefile" in res.detail


# --------------------------------------------------------------------------- #
# 8. 假 checkpoint 端到端（需要 torch）
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch", reason="checkpoint 用例需要 torch")


def _write_fake_checkpoint(tmp_path: Path) -> Path:
    from dataclasses import asdict

    from lightprism.demo.train_demo import load_demo_vocab
    from lightprism.model import LightPrismModel
    from lightprism.model_cfg import PrismConfig

    vocab = load_demo_vocab(VOCAB_PATH)
    cfg = PrismConfig(dim=32, n_layers=1, n_heads=2, max_seq=64,
                      vocab_size=vocab.vocab_size, ffn_mult=2.0,
                      qk_norm=True, tie_embedding=True)
    torch.manual_seed(0)
    model = LightPrismModel(cfg)
    ckpt = tmp_path / "fake.pt"
    torch.save({
        "model": model.state_dict(),
        "model_config": asdict(cfg),
        "tokenizer_layout": vocab.save_layout(),
        "train_config": {},
        "history": [],
        "dataset": {},
    }, ckpt)
    return ckpt


def test_fake_checkpoint_loads_and_generates(tmp_path):
    from lightprism.demo.smoke import load_demo_generator

    ckpt = _write_fake_checkpoint(tmp_path)
    gen = load_demo_generator(ckpt, max_new_tokens=4)
    out = gen.generate("写一个加法程序", max_new_tokens=4)
    assert out["generated_tokens"] == 4
    assert len(out["ids"]) == 4
    assert all(0 <= i < gen.vocab.vocab_size for i in out["ids"])
    assert isinstance(out["text"], str)


def test_acceptance_v1_uses_fake_checkpoint_for_substitute_evidence(tmp_path):
    """V1 判据入口不存在时：verdict 必须 fail，但假 checkpoint 能生成。"""
    ckpt = _write_fake_checkpoint(tmp_path)
    ctx = acceptance.Ctx(ckpt=ckpt, repo_root=tmp_path)
    try:
        res = acceptance.check_v1(ctx)
    finally:
        ctx.close()
    # tmp_path 不是仓库，chat/demo 入口不存在；判据必须 fail
    assert res.verdict == "fail"
    assert "入口未实现" in res.detail
    # 替代诊断：假 checkpoint 能加载并生成 32 token
    assert any("generated_tokens=32" in ev for ev in res.evidence)


def test_acceptance_main_exit_code_reflects_failures(tmp_path, monkeypatch):
    """--only V7 在空目录上必须返回 1（退出码反映通过与否）。"""
    monkeypatch.setattr(acceptance, "ROOT", tmp_path)
    rc = acceptance.main(["--only", "V7", "--json"])
    assert rc == 1


def test_acceptance_main_exit_code_zero_when_all_pass(monkeypatch):
    """用假结果替换检查函数，验证 --only 全 pass 时退出码为 0。"""
    fake = acceptance.CheckResult("V4", "速度 ≥ 200 tok/s", "judge", "pass")
    monkeypatch.setitem(acceptance.CHECKS, "V4", lambda _ctx: fake)
    rc = acceptance.main(["--only", "V4", "--json"])
    assert rc == 0
