# -*- coding: utf-8 -*-
"""LightPrism 运行时设备层：NVIDIA CUDA / 曙光海光 DCU（HIP）/ ROCm / CPU。

为什么需要单独一层
------------------
1. **同一个 ``torch.cuda``，两种硬件后端**。海光 DCU（曙光「异构加速卡」）的
   PyTorch 由 DTK 提供，走 HIP/ROCm，但对外仍暴露 ``torch.cuda.*`` API，
   ``torch.cuda.is_available()`` 返回 ``True``，设备字符串也必须写 ``cuda``。
   用 ``torch.cuda.is_available()`` 判断「是不是英伟达卡」在 SCNet 上会判错。
2. **AMP 不能用一张表套所有卡**。V100（cc7.0）没有 bf16 Tensor Core，
   ``torch.cuda.is_bf16_supported()`` 却可能返回 ``True``（软件可跑、不加速）；
   DCU 的 bf16 支持随型号 / DTK 版本变化，默认应走 fp16 + ``GradScaler``。
3. **平台脚本要能在没有 torch 的控制节点上跑诊断**。所以本模块只依赖标准库，
   ``torch`` 在 :func:`probe_env` 里延迟导入；全部判定逻辑集中在
   :func:`plan_from_env`，用假 env 就能在无卡机器上做单元测试。

口径
----
* ``backend``：``cuda``（英伟达）/ ``dcu``（海光 DCU，HIP）/ ``rocm``（其他 AMD）/
  ``mps`` / ``cpu`` / ``missing``（没装 torch）。
* ``device``：交给 ``torch.device`` 的字符串。DCU 也写 ``cuda`` 或 ``cuda:N``，
  不要写 ``hip``——torch 没有 ``hip`` device 类型。
* ``amp``：``bf16`` / ``fp16`` / ``off``。``fp16`` 一律同时启用 ``GradScaler``。
"""

from __future__ import annotations

import contextlib
import os
import platform as _platform
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "DeviceDesc",
    "RuntimeInfo",
    "RuntimePlan",
    "probe_env",
    "info_from_env",
    "plan_from_env",
    "resolve_plan",
    "autocast_context",
    "make_grad_scaler",
    "summary_line",
    "doctor_payload",
    "format_doctor_text",
    "GPU_BACKENDS",
    "AMP_KINDS",
]

#: 能当加速卡用的后端。``cuda`` 与 ``dcu`` 都通过 ``torch.cuda`` 访问。
GPU_BACKENDS = ("cuda", "dcu", "rocm")
AMP_KINDS = ("auto", "bf16", "fp16", "off")

#: DCU 设备名里常见的标识（不同 DTK 版本返回的字符串不一致，宁可多列几个）。
_DCU_NAME_HINTS = ("hygon", "dcu", "异构", "曙光", "sugon", "deep算", "深算")


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceDesc:
    """一张加速卡的最小描述。"""

    index: int
    name: str
    total_memory_gb: float = 0.0
    major: int = 0
    minor: int = 0


@dataclass
class RuntimeInfo:
    """当前运行时（torch + 硬件）的探测结果。"""

    backend: str = "missing"
    torch_version: str = ""
    hip_version: str = ""
    cuda_version: str = ""
    device_count: int = 0
    devices: List[DeviceDesc] = field(default_factory=list)
    bf16_supported: bool = False
    cuda_available: bool = False
    mps_available: bool = False
    python: str = ""
    platform: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["devices"] = [asdict(d) for d in self.devices]
        return data


@dataclass(frozen=True)
class RuntimePlan:
    """一次运行实际采用的设备 + 精度方案。"""

    backend: str
    device: str
    device_type: str
    amp: str
    scaler: bool
    dtype: str
    notes: Tuple[str, ...] = ()
    info: Optional[RuntimeInfo] = None

    def to_dict(self, *, include_info: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "backend": self.backend,
            "device": self.device,
            "device_type": self.device_type,
            "amp": self.amp,
            "scaler": self.scaler,
            "dtype": self.dtype,
            "notes": list(self.notes),
        }
        if include_info and self.info is not None:
            data["runtime"] = self.info.to_dict()
        return data


# --------------------------------------------------------------------------
# 探测
# --------------------------------------------------------------------------
def _safe(getter, default=None):
    try:
        return getter()
    except Exception:  # noqa: BLE001 - 驱动 / 后端异常不能把诊断脚本带崩
        return default


def probe_env(torch_module: Any = None) -> Dict[str, Any]:
    """把真实 torch 探成一个纯数据 dict（不传给判定逻辑任何对象）。

    ``torch_module`` 省略时自动 ``import torch``；没装 torch 返回
    ``{"torch_version": ""}``，而不是抛异常——SCNet 控制节点上没有 torch 也要能出诊断。
    """
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{_platform.system()}-{_platform.machine()}",
        "torch_version": "",
        "hip_version": "",
        "cuda_version": "",
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "devices": [],
        "bf16_supported": False,
    }

    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception as exc:  # noqa: BLE001
            env["import_error"] = f"{type(exc).__name__}: {exc}"
            return env

    env["torch_version"] = str(_safe(lambda: torch_module.__version__, "") or "")
    env["hip_version"] = str(_safe(lambda: torch_module.version.hip, "") or "")
    env["cuda_version"] = str(_safe(lambda: torch_module.version.cuda, "") or "")
    env["cuda_available"] = bool(_safe(lambda: torch_module.cuda.is_available(), False))
    env["mps_available"] = bool(_safe(
        lambda: torch_module.backends.mps.is_available(), False))

    if env["cuda_available"]:
        count = int(_safe(lambda: torch_module.cuda.device_count(), 0) or 0)
        env["device_count"] = count
        for i in range(count):
            props = _safe(lambda i=i: torch_module.cuda.get_device_properties(i))
            if props is None:
                continue
            env["devices"].append({
                "index": i,
                "name": str(_safe(lambda p=props: p.name, "") or ""),
                "total_memory_gb": round(
                    float(_safe(lambda p=props: p.total_memory, 0) or 0) / 1024 ** 3, 1),
                "major": int(_safe(lambda p=props: p.major, 0) or 0),
                "minor": int(_safe(lambda p=props: p.minor, 0) or 0),
            })
        env["bf16_supported"] = bool(_safe(
            lambda: torch_module.cuda.is_bf16_supported(), False))
    return env


def _classify_backend(env: Mapping[str, Any]) -> str:
    if not env.get("torch_version"):
        return "missing"
    if env.get("hip_version"):
        names = " ".join(str(d.get("name", "")) for d in env.get("devices", []))
        lowered = names.lower()
        if any(hint in lowered for hint in _DCU_NAME_HINTS):
            return "dcu"
        return "rocm"
    if env.get("cuda_available") and int(env.get("device_count", 0) or 0) > 0:
        return "cuda"
    if env.get("mps_available"):
        return "mps"
    return "cpu"


def info_from_env(env: Mapping[str, Any]) -> RuntimeInfo:
    """把 :func:`probe_env` 的 dict 转成 :class:`RuntimeInfo`（含判读提示）。"""
    backend = _classify_backend(env)
    devices = [
        DeviceDesc(
            index=int(d.get("index", i)),
            name=str(d.get("name", "")),
            total_memory_gb=float(d.get("total_memory_gb", 0.0) or 0.0),
            major=int(d.get("major", 0) or 0),
            minor=int(d.get("minor", 0) or 0),
        )
        for i, d in enumerate(env.get("devices", []) or [])
    ]
    info = RuntimeInfo(
        backend=backend,
        torch_version=str(env.get("torch_version", "") or ""),
        hip_version=str(env.get("hip_version", "") or ""),
        cuda_version=str(env.get("cuda_version", "") or ""),
        device_count=int(env.get("device_count", len(devices)) or 0),
        devices=devices,
        bf16_supported=bool(env.get("bf16_supported", False)),
        cuda_available=bool(env.get("cuda_available", False)),
        mps_available=bool(env.get("mps_available", False)),
        python=str(env.get("python", "") or ""),
        platform=str(env.get("platform", "") or ""),
    )

    if backend == "missing":
        err = env.get("import_error")
        info.notes.append(
            "未检测到可用的 torch"
            + (f"（{err}）" if err else "")
            + "：只能跑纯标准库命令；训练/推理请先安装 torch。"
        )
    elif backend in ("dcu", "rocm"):
        info.notes.append(
            f"检测到 HIP 运行时（torch.version.hip={info.hip_version or '未知'}）："
            "torch.cuda.* 实际走海光 DCU / ROCm 后端，device 字符串仍写 cuda。"
        )
        if backend == "rocm":
            info.notes.append(
                "设备名未含 DCU 标识，按通用 ROCm 处理；AMP 默认 fp16 + GradScaler。"
            )
    elif backend == "cuda":
        info.notes.append(
            f"检测到 NVIDIA CUDA 后端（torch.version.cuda={info.cuda_version or '未知'}）。"
        )
    elif backend == "cpu":
        info.notes.append(
            "torch 看不到加速卡：确认任务已分到 GPU/DCU 资源组、"
            "可见设备环境变量（CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES）未清空，"
            "且安装的不是 CPU 版 torch。"
        )

    if backend in GPU_BACKENDS and devices:
        dev = devices[0]
        if backend == "cuda":
            if (dev.major, dev.minor) >= (8, 0):
                info.notes.append(
                    f"{dev.name} cc={dev.major}.{dev.minor}（Ampere 及以上）："
                    "bf16 可用于 AMP。"
                )
            else:
                info.notes.append(
                    f"{dev.name} cc={dev.major}.{dev.minor}（Volta/Turing）："
                    "没有 bf16 Tensor Core，即使 is_bf16_supported() 为真也应使用 "
                    "fp16 + GradScaler。"
                )
        elif backend == "dcu":
            info.notes.append(
                f"DCU：{dev.name} {dev.total_memory_gb:.1f} GiB。默认 fp16 + GradScaler；"
                "bf16 是否可用随型号 / DTK 版本变化，需以实测为准。"
            )
    return info


# --------------------------------------------------------------------------
# 方案判定（纯函数，假 env 可测）
# --------------------------------------------------------------------------
def _gpu_device(info: RuntimeInfo, index: int, want: str) -> str:
    if info.backend not in GPU_BACKENDS:
        raise ValueError(
            f"请求设备 {want!r}，但当前 torch 后端是 {info.backend!r}："
            "请检查任务是否分到加速卡、可见设备变量与 torch 是否为对应版本"
            "（DCU 需要 DTK 提供的 torch，NVIDIA 需要 +cuXXX 轮子）。"
        )
    if info.device_count <= 0:
        raise ValueError(f"请求设备 {want!r}，但 torch.cuda.device_count()=0。")
    if index < 0 or index >= info.device_count:
        raise ValueError(
            f"设备序号 {index} 越界：当前只有 {info.device_count} 张卡"
            f"（合法范围 0..{info.device_count - 1}）。"
        )
    return f"cuda:{index}"


def resolve_device(request: str, info: RuntimeInfo) -> Tuple[str, List[str]]:
    """把用户写的设备名解析成 ``torch.device`` 可用的字符串。"""
    raw = (request or "auto").strip().lower()
    notes: List[str] = []

    if raw in ("auto", ""):
        if info.backend in GPU_BACKENDS and info.device_count > 0:
            return "cuda:0", notes
        if info.mps_available:
            return "mps", notes
        return "cpu", notes

    if raw in ("gpu", "accelerator", "accel"):
        return _gpu_device(info, 0, raw), notes

    if raw in ("dcu", "hip", "rocm"):
        if info.backend not in ("dcu", "rocm"):
            raise ValueError(
                f"请求 {raw!r}，但未检测到 HIP 运行时（torch.version.hip 为空）。"
                "SCNet 的 DCU 资源组需要 DTK 镜像自带的 torch。"
            )
        notes.append("已在 DCU/ROCm 上请求 cuda 设备（torch 的 HIP 后端沿用 cuda 命名）。")
        return _gpu_device(info, 0, raw), notes

    if raw.startswith("cuda"):
        parts = raw.split(":", 1)
        index = 0
        if len(parts) == 2:
            if not parts[1].isdigit():
                raise ValueError(f"设备 {request!r} 的序号不是非负整数。")
            index = int(parts[1])
        return _gpu_device(info, index, raw), notes

    if raw == "mps":
        if not info.mps_available:
            raise ValueError("请求 mps，但 torch.backends.mps.is_available() 为 False。")
        return "mps", notes

    if raw == "cpu":
        return "cpu", notes

    raise ValueError(
        f"未知设备 {request!r}；可用值：auto / cpu / cuda / cuda:N / gpu / dcu。"
    )


def _resolve_amp(request: str, info: RuntimeInfo, device: str) -> Tuple[str, bool, List[str]]:
    """返回 ``(amp, scaler, notes)``。"""
    raw = (request or "auto").strip().lower()
    notes: List[str] = []
    if raw in ("none", "fp32", "float32"):
        raw = "off"
    if raw not in AMP_KINDS:
        raise ValueError(f"未知 AMP 模式 {request!r}；可用值：{', '.join(AMP_KINDS)}。")

    if device.startswith(("cpu", "mps")):
        if raw in ("auto", "off"):
            return "off", False, notes
        raise ValueError(
            f"{device} 上不做半精度 AMP（--amp {raw}）。请用 --amp off；"
            "半精度只在 CUDA / DCU 上启用。"
        )

    dev = info.devices[0] if info.devices else None
    cc = (dev.major, dev.minor) if dev else (0, 0)

    if raw == "off":
        return "off", False, notes

    if raw == "bf16":
        if not info.bf16_supported:
            raise ValueError(
                "请求 bf16，但 torch.cuda.is_bf16_supported() 为 False；"
                "改用 --amp fp16 或 --amp off。"
            )
        if info.backend == "cuda" and cc < (8, 0):
            raise ValueError(
                f"请求 bf16，但设备 cc={cc[0]}.{cc[1]}（Volta/Turing）没有 bf16 "
                "Tensor Core；软件模拟只会更慢，请用 --amp fp16。"
            )
        if info.backend == "dcu":
            notes.append(
                "DCU 上显式启用 bf16：属实验路径，须与 fp16 基线对比 loss 曲线后再用于长训。"
            )
        return "bf16", False, notes

    if raw == "fp16":
        notes.append("fp16 已启用 GradScaler（防梯度下溢）。")
        return "fp16", True, notes

    # auto
    if info.backend == "cuda":
        if info.bf16_supported and cc >= (8, 0):
            return "bf16", False, notes
        notes.append(
            "auto：当前卡没有可用的 bf16 Tensor Core，回落 fp16 + GradScaler。"
        )
        return "fp16", True, notes

    if info.backend in ("dcu", "rocm"):
        notes.append(
            "auto：DCU/ROCm 默认 fp16 + GradScaler（bf16 需显式 --amp bf16 并自行验证）。"
        )
        return "fp16", True, notes

    # 理论上到不了这里（device 已排除 cpu/mps），保守回落
    return "off", False, notes


def plan_from_env(
    env: Mapping[str, Any],
    device: str = "auto",
    amp: str = "auto",
) -> RuntimePlan:
    """从 :func:`probe_env` 的结果算出运行方案。"""
    info = info_from_env(env)
    resolved, dev_notes = resolve_device(device, info)
    amp_kind, scaler, amp_notes = _resolve_amp(amp, info, resolved)
    device_type = resolved.split(":", 1)[0]
    dtype = {"bf16": "bfloat16", "fp16": "float16"}.get(amp_kind, "float32")
    return RuntimePlan(
        backend=info.backend,
        device=resolved,
        device_type=device_type,
        amp=amp_kind,
        scaler=scaler,
        dtype=dtype,
        notes=tuple(dev_notes + amp_notes),
        info=info,
    )


def resolve_plan(device: str = "auto", amp: str = "auto", torch_module: Any = None) -> RuntimePlan:
    """探测真实环境并给出方案（训练 / 推理入口的统一调用点）。"""
    return plan_from_env(probe_env(torch_module), device=device, amp=amp)


# --------------------------------------------------------------------------
# AMP 小工具（兼容 torch 1.13 / 2.x / 3.x）
# --------------------------------------------------------------------------
def autocast_context(torch_module: Any, plan: RuntimePlan):
    """返回 ``with`` 用的 autocast 上下文；不需要时返回 nullcontext。"""
    if plan.amp == "off" or plan.device_type == "cpu":
        return contextlib.nullcontext()
    dtype = torch_module.bfloat16 if plan.amp == "bf16" else torch_module.float16
    try:
        return torch_module.autocast(device_type=plan.device_type, dtype=dtype)
    except (AttributeError, TypeError):
        return torch_module.cuda.amp.autocast(dtype=dtype)


def make_grad_scaler(torch_module: Any, plan: RuntimePlan):
    """fp16 训练用的 GradScaler；不需要时返回 ``None``。"""
    if not plan.scaler:
        return None
    try:  # torch >= 2.4
        return torch_module.amp.GradScaler(plan.device_type, enabled=True)
    except (AttributeError, TypeError):
        try:  # torch 1.13 - 2.3
            return torch_module.cuda.amp.GradScaler(enabled=True)
        except AttributeError:
            return None


# --------------------------------------------------------------------------
# 诊断输出
# --------------------------------------------------------------------------
def _backend_label(backend: str) -> str:
    return {
        "cuda": "NVIDIA CUDA",
        "dcu": "曙光海光 DCU（HIP）",
        "rocm": "AMD ROCm（HIP）",
        "mps": "Apple MPS",
        "cpu": "CPU",
        "missing": "未检测到 torch",
    }.get(backend, backend)


def summary_line(plan: RuntimePlan) -> str:
    """一行中文摘要，写进训练日志头部。"""
    info = plan.info
    hw = ""
    if info is not None and info.devices:
        dev = info.devices[0]
        hw = f"{dev.name} {dev.total_memory_gb:.0f}GB"
        if plan.backend == "cuda":
            hw += f" cc={dev.major}.{dev.minor}"
    amp = plan.amp + (" + GradScaler" if plan.scaler else "")
    parts = [f"设备 {plan.device}", f"后端 {_backend_label(plan.backend)}"]
    if hw:
        parts.append(hw)
    parts.append(f"AMP {amp}")
    if info is not None and info.hip_version:
        parts.append(f"HIP {info.hip_version}")
    return " · ".join(parts)


def _visible_env() -> Dict[str, str]:
    keys = (
        "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
        "ROCM_PATH", "HIP_PATH", "DTK_ROOT", "LIGHTPRISM_DEVICE", "LIGHTPRISM_AMP",
        "PJM_JOBID", "SLURM_JOB_ID",
    )
    return {k: v for k, v in os.environ.items() if k in keys}


def doctor_payload(device: str = "auto", amp: str = "auto",
                   torch_module: Any = None) -> Dict[str, Any]:
    """``lightprism doctor`` 的机器可读结果。"""
    env = probe_env(torch_module)
    info = info_from_env(env)
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("torch", info.backend != "missing",
        f"torch {info.torch_version}" if info.torch_version else "未安装 torch")
    add("accelerator", info.backend in GPU_BACKENDS and info.device_count > 0,
        f"{_backend_label(info.backend)}，{info.device_count} 张卡")
    if info.backend in GPU_BACKENDS:
        build_ok = bool(info.hip_version) if info.backend == "dcu" else bool(info.cuda_version)
        add("build", build_ok,
            f"cuda={info.cuda_version or '-'} hip={info.hip_version or '-'}")
        add("bf16", info.bf16_supported,
            "is_bf16_supported()=%s" % info.bf16_supported)

    plan: Optional[RuntimePlan] = None
    plan_error = ""
    try:
        plan = plan_from_env(env, device=device, amp=amp)
    except ValueError as exc:
        plan_error = str(exc)

    return {
        "ok": plan is not None and info.backend != "missing",
        "request": {"device": device, "amp": amp},
        "runtime": info.to_dict(),
        "plan": plan.to_dict(include_info=True) if plan else None,
        "plan_error": plan_error,
        "checks": checks,
        "environment": _visible_env(),
        "notes": info.notes,
    }


def format_doctor_text(payload: Mapping[str, Any]) -> List[str]:
    """把 :func:`doctor_payload` 排成人读文本。"""
    rt = payload.get("runtime") or {}
    plan = payload.get("plan") or {}
    lines = ["LightPrism · 运行环境诊断", "=" * 52]
    lines.append(f"torch     : {rt.get('torch_version') or '未安装'}")
    lines.append(f"CUDA 版本 : {rt.get('cuda_version') or '-'}")
    lines.append(f"HIP 版本  : {rt.get('hip_version') or '-'}")
    lines.append(f"后端      : {_backend_label(str(rt.get('backend', 'missing')))}")
    lines.append(f"加速卡数  : {rt.get('device_count', 0)}")
    for dev in rt.get("devices") or []:
        lines.append(
            f"  [{dev.get('index')}] {dev.get('name')}  "
            f"{dev.get('total_memory_gb')} GiB  cc={dev.get('major')}.{dev.get('minor')}"
        )
    lines.append(f"bf16 支持 : {rt.get('bf16_supported')}")
    if plan:
        lines.append("")
        lines.append(f"选用设备  : {plan.get('device')}")
        lines.append(f"选用 AMP  : {plan.get('amp')}"
                     + ("（GradScaler 开）" if plan.get("scaler") else ""))
    elif payload.get("plan_error"):
        lines.append("")
        lines.append(f"方案失败  : {payload['plan_error']}")
    lines.append("")
    lines.append("检查项:")
    for check in payload.get("checks") or []:
        mark = "OK  " if check.get("ok") else "FAIL"
        lines.append(f"  [{mark}] {check.get('name')}: {check.get('detail')}")
    env = payload.get("environment") or {}
    if env:
        lines.append("")
        lines.append("相关环境变量:")
        for key, value in sorted(env.items()):
            lines.append(f"  {key}={value}")
    notes = payload.get("notes") or []
    if notes:
        lines.append("")
        lines.append("提示:")
        for note in notes:
            lines.append(f"  - {note}")
    return lines
