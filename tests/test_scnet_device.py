# -*- coding: utf-8 -*-
"""SCNet 运行时设备层测试：NVIDIA CUDA / 曙光海光 DCU(HIP) / ROCm / CPU。

全部用例只喂 ``probe_env`` 形状的纯 dict，**不要求本机有 torch / 显卡**；
这样 CI 与无卡机器也能判定「设备与 AMP 选择」是否符合平台事实。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lightprism.device import (
    doctor_payload,
    format_doctor_text,
    info_from_env,
    plan_from_env,
    probe_env,
    summary_line,
)


def _env(**overrides):
    """一份「装了 torch、但看不到卡」的基线 env。"""
    env = {
        "python": "3.10.12",
        "platform": "Linux-x86_64",
        "torch_version": "2.4.0+cu121",
        "hip_version": "",
        "cuda_version": "12.1",
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "devices": [],
        "bf16_supported": False,
    }
    env.update(overrides)
    return env


def _nvidia(name="NVIDIA L20", major=8, minor=9, memory=47.5, bf16=True):
    return _env(
        cuda_available=True,
        device_count=1,
        devices=[{"index": 0, "name": name, "total_memory_gb": memory,
                  "major": major, "minor": minor}],
        bf16_supported=bf16,
    )


def _dcu(name="Hygon DCU K100-AI", hip="6.2.41133", bf16=False):
    return _env(
        torch_version="2.3.0+git",
        cuda_version="",
        hip_version=hip,
        cuda_available=True,
        device_count=1,
        devices=[{"index": 0, "name": name, "total_memory_gb": 63.6,
                  "major": 9, "minor": 0}],
        bf16_supported=bf16,
    )


# --------------------------------------------------------------------------
# 后端识别
# --------------------------------------------------------------------------
def test_nvidia_l20_is_cuda_and_auto_uses_bf16():
    plan = plan_from_env(_nvidia(), device="auto", amp="auto")
    assert plan.backend == "cuda"
    assert plan.device == "cuda:0"
    assert plan.device_type == "cuda"
    assert plan.amp == "bf16"
    assert plan.scaler is False
    assert "NVIDIA L20" in summary_line(plan)


def test_v100_reports_bf16_but_auto_must_pick_fp16():
    """V100（cc7.0）没有 bf16 Tensor Core：is_bf16_supported() 为真也不能用。"""
    env = _nvidia(name="Tesla V100-SXM2-16GB", major=7, minor=0,
                  memory=15.8, bf16=True)
    plan = plan_from_env(env, device="auto", amp="auto")
    assert plan.amp == "fp16"
    assert plan.scaler is True

    with pytest.raises(ValueError, match="bf16"):
        plan_from_env(env, device="auto", amp="bf16")


def test_dcu_is_detected_and_defaults_to_fp16_with_scaler():
    plan = plan_from_env(_dcu(), device="auto", amp="auto")
    assert plan.backend == "dcu"
    # DCU 的 torch 仍用 cuda 命名，不能写 hip
    assert plan.device == "cuda:0"
    assert plan.amp == "fp16"
    assert plan.scaler is True
    assert any("GradScaler" in n for n in plan.notes)
    assert any("HIP" in n for n in plan.info.notes)


def test_dcu_bf16_requires_support_and_is_marked_experimental():
    with pytest.raises(ValueError, match="bf16"):
        plan_from_env(_dcu(bf16=False), amp="bf16")

    plan = plan_from_env(_dcu(bf16=True), amp="bf16")
    assert plan.amp == "bf16"
    assert plan.scaler is False
    assert any("实验" in n for n in plan.notes)


def test_generic_amd_hip_is_rocm():
    env = _dcu(name="AMD Radeon RX 7900 XTX")
    info = info_from_env(env)
    assert info.backend == "rocm"
    assert plan_from_env(env).amp == "fp16"


def test_cpu_env_is_cpu_and_rejects_half_precision():
    plan = plan_from_env(_env(), device="auto", amp="auto")
    assert plan.backend == "cpu"
    assert plan.device == "cpu"
    assert plan.amp == "off"
    assert plan.scaler is False

    with pytest.raises(ValueError, match="半精度"):
        plan_from_env(_env(), device="cpu", amp="bf16")
    with pytest.raises(ValueError, match="加速卡"):
        plan_from_env(_env(), device="cuda")
    with pytest.raises(ValueError, match="半精度"):
        plan_from_env(_env(), device="cpu", amp="fp16")


def test_missing_torch_reports_missing_backend():
    env = {"python": "3.10.12", "platform": "Linux-x86_64",
           "torch_version": "", "import_error": "ModuleNotFoundError"}
    info = info_from_env(env)
    assert info.backend == "missing"
    assert any("未检测到可用的 torch" in n for n in info.notes)

    plan = plan_from_env(env, device="auto", amp="auto")
    assert plan.device == "cpu"
    assert plan.amp == "off"

    payload = doctor_payload_from_env(env)
    assert payload["ok"] is False


def doctor_payload_from_env(env):
    """用假 env 跑 doctor_payload（避免依赖本机是否装 torch）。"""
    import lightprism.device as device_mod

    original = device_mod.probe_env
    device_mod.probe_env = lambda torch_module=None: env
    try:
        return device_mod.doctor_payload()
    finally:
        device_mod.probe_env = original


# --------------------------------------------------------------------------
# 设备名解析
# --------------------------------------------------------------------------
def test_explicit_device_index_and_bounds():
    env = _nvidia()
    env["device_count"] = 2
    env["devices"].append({"index": 1, "name": "NVIDIA L20", "total_memory_gb": 47.5,
                           "major": 8, "minor": 9})
    assert plan_from_env(env, device="cuda:1").device == "cuda:1"
    with pytest.raises(ValueError, match="越界"):
        plan_from_env(env, device="cuda:2")
    with pytest.raises(ValueError, match="非负整数"):
        plan_from_env(env, device="cuda:x")
    with pytest.raises(ValueError, match="未知设备"):
        plan_from_env(env, device="tpu")


def test_dcu_alias_requires_hip():
    with pytest.raises(ValueError, match="HIP"):
        plan_from_env(_nvidia(), device="dcu")
    assert plan_from_env(_dcu(), device="dcu").device == "cuda:0"


def test_amp_off_and_aliases():
    plan = plan_from_env(_nvidia(), amp="off")
    assert plan.amp == "off" and plan.scaler is False
    assert plan_from_env(_nvidia(), amp="fp32").amp == "off"
    with pytest.raises(ValueError, match="未知 AMP"):
        plan_from_env(_nvidia(), amp="tf32")


def test_plan_dict_is_json_serializable():
    payload = plan_from_env(_dcu(), device="auto", amp="auto").to_dict(include_info=True)
    text = json.dumps(payload, ensure_ascii=False)
    assert "runtime" in payload
    assert "cuda:0" in text


# --------------------------------------------------------------------------
# probe_env / doctor
# --------------------------------------------------------------------------
def test_probe_env_reads_torch_without_real_torch():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_properties(index):
            return SimpleNamespace(
                name="Hygon DCU K100-AI", total_memory=64 * 1024 ** 3,
                major=9, minor=0)

        @staticmethod
        def is_bf16_supported():
            return False

    fake = SimpleNamespace(
        __version__="2.3.0+git",
        version=SimpleNamespace(hip="6.2.41133", cuda=None),
        cuda=FakeCuda,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    env = probe_env(fake)
    assert env["hip_version"] == "6.2.41133"
    assert env["device_count"] == 1
    assert env["devices"][0]["name"] == "Hygon DCU K100-AI"
    assert info_from_env(env).backend == "dcu"


def test_doctor_text_and_payload_shape():
    payload = doctor_payload_from_env(_nvidia())
    assert payload["ok"] is True
    assert payload["plan"]["device"] == "cuda:0"
    lines = format_doctor_text(payload)
    assert any("NVIDIA L20" in line for line in lines)
    assert any("选用 AMP" in line for line in lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_doctor_json_on_cpu_env(monkeypatch, capsys):
    import lightprism.device as device_mod
    from lightprism.cli import main

    monkeypatch.setattr(device_mod, "probe_env", lambda torch_module=None: _env())
    rc = main(["doctor", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert payload["runtime"]["backend"] == "cpu"
    assert payload["plan"]["device"] == "cpu"


def test_cli_doctor_fails_when_torch_missing(monkeypatch, capsys):
    import lightprism.device as device_mod
    from lightprism.cli import main

    env = {"python": "3.10.12", "platform": "Linux-x86_64",
           "torch_version": "", "import_error": "ModuleNotFoundError"}
    monkeypatch.setattr(device_mod, "probe_env", lambda torch_module=None: env)
    rc = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    assert payload["runtime"]["backend"] == "missing"
