#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LightPrism Demo 验收脚本（T-0334）。

逐条执行 ``docs/Demo开发计划.md`` §2 的七条验收标准：

====  ==========================================  ==========================
编号   标准                                        本脚本能自动做什么
====  ==========================================  ==========================
V1     能加载模型并推理                           尝试计划书写的 ``lightprism chat``；缺失则退化为
                                                      demo smoke 的「加载 checkpoint + 生成」诊断
V2     输出语法合法                               生成 20 条代码样本 → ``light ai check --json``
V3     JSON 模式合法                              ``--format json``（缺失则诊断生成 JSON 的可解析率）
V4     速度 ≥ 200 tok/s                          复用 ``scripts/bench.py`` 的写死口径实测
V5     内存 ≤ 8GB                                 峰值常驻内存（POSIX: VmHWM → ru_maxrss；
                                                      Windows: PeakWorkingSetSize）
V6     目标机实测                                 仅记录 blocked（需要人工在真 5 代酷睿上跑）
V7     可复现（make demo ≤ 10 分钟）               检查 Makefile + ``make demo`` 干净 clone 计时
====  ==========================================  ==========================

设计原则：

* **不硬凑**。判据命令不存在就记 ``entry_missing``，不拿替代入口冒充通过；
  替代入口的输出只作 ``evidence``，单独标注「非判据」。
* **口径不覆盖**。V4/V5 直接 import ``scripts/bench.py`` 调用其 ``build_model`` /
  ``run_decode`` / ``PROMPT_LEN`` / ``WARMUP_TOKENS``，不复制一份计时代码。
* **退出码反映通过与否**：自动项 V1—V5、V7 全部 pass 才返回 0；任何 fail 返回 1。
  V6 是人工项，不计入退出码，但会写进结果与报告。
* Windows 上 ``resource`` 模块不存在；脚本注入一个只提供 ``getrusage`` 的桩，
  让 ``bench.py`` 能被导入（V4 不需要真实 RSS，V5 用 Windows 自己的峰值口径）。

用法::

    PYTHONPATH=src python scripts/acceptance.py --all
    PYTHONPATH=src python scripts/acceptance.py --all --json
    PYTHONPATH=src python scripts/acceptance.py --only V4,V5 --json

环境变量：

``LIGHT_CHECK_CLI``   ``light ai check`` 入口（light 仓 ``cli/light.py`` 路径或可执行名）。
                     不设时按常见位置/ PATH 探测。
``LIGHT_REPO``        light 仓根目录（用于给 checker 设 PYTHONPATH）。
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

# 验收阈值（写死，与 docs/Demo开发计划.md §2 一致；改这里等于改验收标准）
SPEED_TARGET = 200.0          # tok/s
MEMORY_LIMIT_MB = 8 * 1024    # 8GB
CODE_SAMPLES = 20             # V2 抽样条数
CODE_PASS_RATE = 0.80         # V2 合法率下限
V2_MAX_NEW_TOKENS = 64
V3_MAX_NEW_TOKENS = 64
MAKE_DEMO_LIMIT_S = 600.0     # V7：10 分钟


# --------------------------------------------------------------------------- #
# 结果模型
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    id: str
    title: str
    judge: str
    verdict: str                      # pass / fail / blocked / unverified
    detail: str = ""
    metric: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "judge": self.judge,
            "verdict": self.verdict,
            "detail": self.detail,
            "metric": self.metric,
            "evidence": self.evidence,
            "elapsed_s": round(self.elapsed_s, 3),
        }


class Ctx:
    """一次验收运行共享的上下文（checkpoint、缓存结果、临时目录）。"""

    def __init__(self, ckpt: Optional[Path], repo_root: Path = ROOT,
                 keep_tmp: bool = False) -> None:
        self.root = repo_root
        self.ckpt = ckpt
        self.keep_tmp = keep_tmp
        self.tmpdir = Path(tempfile.mkdtemp(prefix="lp-accept-"))
        self._gen = None
        self._bench = None
        self._decode: Optional[Dict[str, Any]] = None
        self._tasks_code: Optional[List[str]] = None
        self._tasks_json: Optional[List[str]] = None
        self._checker: Optional[Dict[str, Any]] = None

    # -- 生命周期 -------------------------------------------------------
    def close(self) -> None:
        if self.keep_tmp:
            return
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- 资源 -----------------------------------------------------------
    def generator(self, max_new_tokens: int = 32):
        if self._gen is None:
            if self.ckpt is None or not Path(self.ckpt).is_file():
                raise FileNotFoundError(
                    f"checkpoint 不存在: {self.ckpt}（先用 train_demo 训练或从目标机拷贝）")
            from lightprism.demo.smoke import load_demo_generator
            # 单线程既是 bench 口径，也避免小模型在 6/12 核上被线程同步拖慢
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:  # noqa: BLE001
                pass
            self._gen = load_demo_generator(self.ckpt, max_new_tokens=max_new_tokens)
        self._gen.max_new_tokens = max_new_tokens
        return self._gen

    def bench(self):
        """导入 ``scripts/bench.py``（Windows 上先注入 resource 桩）。"""
        if self._bench is not None:
            return self._bench
        if os.name == "nt":
            _install_windows_resource_stub()
        spec = importlib.util.spec_from_file_location(
            "lp_bench", self.root / "scripts" / "bench.py")
        if spec is None or spec.loader is None:
            raise ImportError("无法导入 scripts/bench.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._bench = mod
        return mod

    def decode_measure(self) -> Dict[str, Any]:
        """按 bench.py 写死口径跑一次 200 token 单线程 decode，同时取峰值内存。"""
        if self._decode is not None:
            return self._decode
        bench = self.bench()
        cfg_path = self.root / "configs" / "demo-3m.json"
        model, cfg, params = bench.build_model(str(cfg_path))
        t0 = time.perf_counter()
        result = bench.run_decode(model, cfg.vocab_size, bench.DEFAULT_TOKENS, "single")
        elapsed = time.perf_counter() - t0
        peak_mb, peak_source = peak_memory_mb()
        self._decode = {
            "tok_per_s": float(result["tok_per_s"]),
            "avg_ms_per_token": float(result["avg_ms_per_token"]),
            "timed_tokens": int(result["timed_tokens"]),
            "threads": int(result["threads"]),
            "params": int(params),
            "peak_mb": float(peak_mb),
            "peak_source": peak_source,
            "elapsed_s": elapsed,
            "protocol": {
                "prompt_len": int(bench.PROMPT_LEN),
                "warmup_tokens": int(bench.WARMUP_TOKENS),
                "tokens": int(bench.DEFAULT_TOKENS),
                "mode": "single",
                "greedy": True,
                "prefill_excluded": True,
            },
        }
        return self._decode

    def code_tasks(self, n: int = CODE_SAMPLES) -> List[str]:
        if self._tasks_code is None:
            self._tasks_code = _read_tasks(self.root / "data" / "sft" / "code_tasks.jsonl")
        return self._tasks_code[:n]

    def json_tasks(self, n: int = CODE_SAMPLES) -> List[str]:
        if self._tasks_json is None:
            self._tasks_json = _read_tasks(self.root / "data" / "sft" / "json_tasks.jsonl")
        return self._tasks_json[:n]

    def checker(self) -> Optional[Dict[str, Any]]:
        if self._checker is not None:
            return self._checker or None
        found = find_light_checker()
        self._checker = found or {}
        return found


def _read_tasks(path: Path) -> List[str]:
    tasks: List[str] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            task = str(obj.get("task") or "").strip()
            if task:
                tasks.append(task)
    return tasks


# --------------------------------------------------------------------------- #
# 平台相关的峰值内存
# --------------------------------------------------------------------------- #
def _install_windows_resource_stub() -> None:
    """Windows 没有 ``resource``；给 bench.py 一个能 import 的桩。

    V4 只用 bench 的计时逻辑；V5 不调用 ``bench.peak_rss_mb``，而是用本文件
    :func:`peak_memory_mb` 的 Windows 峰值口径，避免把桩值当成真实内存。
    桩**不能**回调 :func:`peak_memory_mb`：那里失败时会再 import resource，
    形成无限递归。
    """
    if "resource" in sys.modules:
        return
    mod = types.ModuleType("resource")
    mod.RUSAGE_SELF = 0

    def getrusage(_who: int):
        peak_kb = _windows_peak_mb()[0]
        return types.SimpleNamespace(ru_maxrss=int(peak_kb * 1024))

    mod.getrusage = getrusage
    sys.modules["resource"] = mod


class _WindowsPMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_peak_mb() -> Tuple[float, str]:
    """Windows 峰值工作集（MB）；失败返回 (0.0, "unavailable")，不递归回退。"""
    if os.name != "nt":
        return 0.0, "unavailable"
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        pmc = _WindowsPMC()
        pmc.cb = ctypes.sizeof(_WindowsPMC)
        fn = None
        for lib_name in ("psapi", "kernel32"):
            lib = getattr(ctypes.windll, lib_name, None)
            for name in ("GetProcessMemoryInfo", "K32GetProcessMemoryInfo"):
                candidate = getattr(lib, name, None) if lib is not None else None
                if candidate is not None:
                    fn = candidate
                    break
            if fn is not None:
                break
        if fn is not None:
            # 不设 argtypes 时 HANDLE 会被截成 32 位，调用返回 0（静默失败）
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_WindowsPMC),
                           ctypes.c_uint32]
            fn.restype = ctypes.c_int
            if fn(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / (1024.0 * 1024.0), \
                    "PeakWorkingSetSize(Windows)"
    except (AttributeError, OSError):
        pass
    return 0.0, "unavailable"


def peak_memory_mb() -> Tuple[float, str]:
    """进程峰值常驻内存（MB）与来源。

    POSIX 优先读 ``/proc/self/status`` 的 VmHWM（Linux）；FreeBSD 没有 /proc，
    退到 ``resource.getrusage`` 的 ru_maxrss（内核记账的峰值，不是 ps 瞬时 RSS）。
    Windows 用 GetProcessMemoryInfo 的 PeakWorkingSetSize。
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0, "VmHWM(/proc)"
    except OSError:
        pass
    if os.name == "nt":
        mb, src = _windows_peak_mb()
        if src != "unavailable":
            return mb, src
        return mb, src
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / 1024.0, "ru_maxrss(KB)"
    except (ImportError, AttributeError):
        return 0.0, "unavailable"


# --------------------------------------------------------------------------- #
# light ai check 入口探测
# --------------------------------------------------------------------------- #
def find_light_checker() -> Optional[Dict[str, Any]]:
    """返回 ``{"cmd": [...], "cwd": ..., "label": ...}`` 或 None。"""
    explicit = os.environ.get("LIGHT_CHECK_CLI", "").strip()
    candidates: List[Tuple[str, Path]] = []
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            candidates.append(("env:LIGHT_CHECK_CLI", p))
        else:
            exe = shutil.which(explicit)
            if exe:
                candidates.append(("env:LIGHT_CHECK_CLI", Path(exe)))
    light_repo = os.environ.get("LIGHT_REPO", "").strip()
    if light_repo:
        candidates.append(("env:LIGHT_REPO", Path(light_repo) / "cli" / "light.py"))
    exe = shutil.which("light")
    if exe:
        candidates.append(("PATH:light", Path(exe)))
    for sibling in ("light", "light-merge"):
        candidates.append((f"sibling:{sibling}",
                           ROOT.parent / sibling / "cli" / "light.py"))
        candidates.append((f"cwd-sibling:{sibling}",
                           Path.cwd() / sibling / "cli" / "light.py"))
    for known in ("/home/skywalk/dswork/light/cli/light.py",
                  "G:/dswork/duan-light-merge/light-merge/cli/light.py"):
        candidates.append(("known-path", Path(known)))

    seen: set[str] = set()
    for label, path in candidates:
        key = str(path.resolve()) if path.is_absolute() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            continue
        repo = path.parent.parent
        if path.suffix == ".py":
            cmd = [sys.executable, str(path)]
        else:
            cmd = [str(path)]
        return {"cmd": cmd, "cwd": str(repo), "label": label,
                "repo": str(repo), "path": str(path)}
    return None


def run_light_check(checker: Dict[str, Any], code: str,
                    timeout: float = 60.0) -> Dict[str, Any]:
    """把一段代码写临时文件，跑 ``light ai check --json``，返回解析后的结果。"""
    from lightprism.demo.train_demo import ROOT as _unused  # noqa: F401  (确保 src 在路径上)

    suffix = ".json" if _looks_json(code) else ".light"
    fd, path = tempfile.mkstemp(prefix="lp-check-", suffix=suffix)
    os.close(fd)
    p = Path(path)
    try:
        p.write_text(code, encoding="utf-8")
        env = dict(os.environ)
        extra = [str(Path(checker["repo"]) / "src"), checker["repo"]]
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "src")] + extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        t0 = time.perf_counter()
        proc = subprocess.run(
            list(checker["cmd"]) + ["ai", "check", "--json", str(p)],
            cwd=checker["cwd"], env=env, capture_output=True, text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        payload = parse_check_output(proc.stdout or "")
        diagnostics = list((payload or {}).get("diagnostics") or [])
        errors = [d for d in diagnostics if _is_error_diag(d)]
        ok = bool((payload or {}).get("ok"))
        return {
            "returncode": proc.returncode,
            "ok": ok,
            "diagnostics": diagnostics,
            "n_diagnostics": len(diagnostics),
            "n_errors": len(errors),
            "parsed": payload is not None,
            "elapsed_s": elapsed,
            "stderr_tail": (proc.stderr or "").strip().splitlines()[-3:],
        }
    finally:
        try:
            p.unlink()
        except OSError:
            pass


def judge_speed(tok_per_s: float) -> str:
    """V4 判据：单线程 decode 吞吐是否达标。"""
    return "pass" if float(tok_per_s) >= SPEED_TARGET else "fail"


def judge_memory(peak_mb: float) -> str:
    """V5 判据：峰值常驻内存是否在 8GB 内。"""
    return "pass" if float(peak_mb) <= MEMORY_LIMIT_MB else "fail"


def judge_rate(valid: int, total: int) -> str:
    """V2 判据：合法样本占比是否 ≥ 80%。"""
    if total <= 0:
        return "fail"
    return "pass" if valid / total >= CODE_PASS_RATE else "fail"


def parse_check_output(text: str) -> Optional[Dict[str, Any]]:
    """解析 ``light ai check --json`` 的 stdout；容忍前置日志。"""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _is_error_diag(d: Dict[str, Any]) -> bool:
    for key in ("severity", "level", "kind"):
        val = str(d.get(key) or "").lower()
        if val in ("error", "fatal", "err"):
            return True
        if val in ("warning", "warn", "info"):
            return False
    # 没有严重度字段时：出现任何诊断都按 error 计（保守，不放宽）
    return True


def _looks_json(text: str) -> bool:
    t = text.strip()
    return bool(t) and t[0] in "[{"


def extract_code(text: str) -> str:
    """提取 ``<CODE>…</CODE>`` 内容；没有控制标记就返回去掉首尾空白的全文。"""
    m = re.search(r"<CODE>(.*?)</CODE>", text or "", re.S)
    if m:
        return m.group(1).strip()
    return (text or "").strip()


# --------------------------------------------------------------------------- #
# 各条验收
# --------------------------------------------------------------------------- #
def check_v1(ctx: Ctx) -> CheckResult:
    """V1：能加载模型并推理 —— `lightprism chat --prompt "..."` 输出非空。"""
    t0 = time.perf_counter()
    res = CheckResult("V1", "能加载模型并推理",
                      'lightprism chat --prompt "写一个加法程序" 输出非空', "fail")
    # 1) 计划书写的判据入口
    for cmd, label in (
        ([sys.executable, "-m", "lightprism", "chat", "--prompt", "写一个加法程序"],
         "lightprism chat --prompt"),
        ([sys.executable, "-m", "lightprism.demo", "--prompt", "写一个加法程序"],
         "python -m lightprism.demo --prompt"),
    ):
        proc = _run(cmd, cwd=ctx.root, timeout=300)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        res.evidence.append(
            f"$ {label}\n  exit={proc.returncode}\n  stdout={out[:200]!r}\n"
            f"  stderr={err[-400:]!r}")
        if proc.returncode == 0 and out:
            res.verdict = "pass"
            res.detail = f"判据入口 {label} 返回非空输出"
            res.metric = {"command": label, "stdout_chars": len(out)}
            res.elapsed_s = time.perf_counter() - t0
            return res

    # 2) 入口缺失 → 能力侧只作诊断，不冒充通过
    res.detail = "判据入口未实现（lightprism chat / python -m lightprism.demo 均不存在）；" \
                 "Checkpoint 加载与生成能力见 evidence（非判据，不计入通过）"
    try:
        gen = ctx.generator(max_new_tokens=32)
        out = gen.generate("写一个加法程序", max_new_tokens=32)
        text = str(out.get("text") or "")
        res.evidence.append(
            "替代诊断 smoke.load_demo_generator(...).generate(task, 32)：" \
            f"generated_tokens={out.get('generated_tokens')}, "
            f"text_chars={len(text)}, text_repr={text[:80]!r}")
        res.metric = {
            "ckpt": str(ctx.ckpt),
            "generated_tokens": out.get("generated_tokens"),
            "text_chars": len(text),
            "unique_chars": len(set(text)),
        }
    except Exception as exc:  # noqa: BLE001
        res.evidence.append(f"替代诊断失败: {type(exc).__name__}: {exc}")
    res.verdict = "fail"
    res.elapsed_s = time.perf_counter() - t0
    return res


def check_v2(ctx: Ctx) -> CheckResult:
    """V2：20 条生成代码抽样，≥80% 经 light ai check 无 error 级诊断。"""
    t0 = time.perf_counter()
    res = CheckResult("V2", "输出语法合法",
                      "生成代码经 light ai check 无 error 级诊断（20 条抽样，≥80% 合法）",
                      "fail")
    checker = ctx.checker()
    if checker is None:
        res.verdict = "unverified"
        res.detail = "未找到 light ai check 入口（设 LIGHT_CHECK_CLI 指向 light 仓 cli/light.py）"
        res.elapsed_s = time.perf_counter() - t0
        return res
    res.metric["checker"] = checker["label"]
    res.metric["checker_path"] = checker["path"]

    try:
        gen = ctx.generator(max_new_tokens=V2_MAX_NEW_TOKENS)
    except Exception as exc:  # noqa: BLE001
        res.verdict = "unverified"
        res.detail = f"checkpoint 不可用: {type(exc).__name__}: {exc}"
        res.elapsed_s = time.perf_counter() - t0
        return res

    tasks = ctx.code_tasks(CODE_SAMPLES)
    if not tasks:
        res.verdict = "unverified"
        res.detail = "没有可用的 code_tasks.jsonl 抽样"
        res.elapsed_s = time.perf_counter() - t0
        return res

    n_valid = 0
    n_checked = 0
    n_with_code_marker = 0
    details: List[str] = []
    samples: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks, 1):
        out = gen.generate(task, max_new_tokens=V2_MAX_NEW_TOKENS)
        text = str(out.get("text") or "")
        if "<CODE>" in text and "</CODE>" in text:
            n_with_code_marker += 1
        code = extract_code(text)
        if not code:
            samples.append({"i": i, "task": task, "code_chars": 0,
                            "valid": False, "reason": "空输出（未生成任何代码字符）"})
            details.append(f"#{i} 空输出")
            continue
        chk = run_light_check(checker, code)
        n_checked += 1
        ok = bool(chk["parsed"]) and chk["ok"] and chk["n_errors"] == 0
        if ok:
            n_valid += 1
        samples.append({
            "i": i, "task": task, "code_chars": len(code), "valid": ok,
            "checker_exit": chk["returncode"], "n_diagnostics": chk["n_diagnostics"],
            "n_errors": chk["n_errors"], "code_head": code[:80],
        })
        details.append(f"#{i} {'pass' if ok else 'fail'} "
                       f"(chars={len(code)}, diag={chk['n_diagnostics']})")

    rate = n_valid / len(tasks) if tasks else 0.0
    res.metric.update({
        "samples": len(tasks),
        "checked": n_checked,
        "valid": n_valid,
        "pass_rate": round(rate, 4),
        "threshold": CODE_PASS_RATE,
        "with_code_marker": n_with_code_marker,
        "sample_detail": samples,
    })
    res.evidence.extend(details[:20])
    if judge_rate(n_valid, len(tasks)) == "pass":
        res.verdict = "pass"
        res.detail = f"合法率 {rate:.1%} ≥ {CODE_PASS_RATE:.0%}"
    else:
        res.verdict = "fail"
        res.detail = (f"合法率 {rate:.1%} < {CODE_PASS_RATE:.0%}；"
                      f"{len(tasks)} 条中 {n_with_code_marker} 条带 <CODE> 标记，"
                      f"{n_checked} 条有可检查代码")
    # 盲区对照：空程序 checker 也判 ok —— 只作证据，不给 V2 加分
    control = run_light_check(checker, " " * 64)
    res.evidence.append(
        "盲区对照：64 个空格经 light ai check --json → "
        f"ok={control['ok']}, diagnostics={control['n_diagnostics']}, "
        f"exit={control['returncode']}（说明判据本身对空程序是假绿）")
    res.metric["empty_program_control"] = {
        "ok": control["ok"], "n_diagnostics": control["n_diagnostics"],
        "returncode": control["returncode"],
    }
    res.elapsed_s = time.perf_counter() - t0
    return res


def check_v3(ctx: Ctx) -> CheckResult:
    """V3：--format json 输出可被 json.loads 解析。"""
    t0 = time.perf_counter()
    res = CheckResult("V3", "JSON 模式合法",
                      "lightprism chat --prompt '...' --format json 输出可被 json.loads 解析",
                      "fail")
    # 判据入口：--format json
    cmd = [sys.executable, "-m", "lightprism", "chat", "--prompt",
           "列出当前目录中的 py 文件及大小，输出 JSON", "--format", "json"]
    proc = _run(cmd, cwd=ctx.root, timeout=300)
    res.evidence.append(
        f"$ lightprism chat --prompt ... --format json\n  exit={proc.returncode}\n"
        f"  stderr={(proc.stderr or '').strip()[-400:]!r}")
    res.detail = "判据入口 --format json 未实现（chat 子命令不存在）"

    # 诊断：用 JSON 任务生成，抽 <CODE> 段尝试 json.loads（非判据）
    try:
        gen = ctx.generator(max_new_tokens=V3_MAX_NEW_TOKENS)
        tasks = ctx.json_tasks(CODE_SAMPLES)
        n_parse = 0
        details = []
        for i, task in enumerate(tasks, 1):
            out = gen.generate(task, max_new_tokens=V3_MAX_NEW_TOKENS)
            text = str(out.get("text") or "")
            code = extract_code(text)
            try:
                json.loads(code)
                n_parse += 1
                details.append(f"#{i} json.loads ok")
            except json.JSONDecodeError as exc:
                details.append(f"#{i} json.loads fail: {exc}")
        res.metric = {
            "diagnostic_samples": len(tasks),
            "json_parse_ok": n_parse,
            "json_parse_rate": round(n_parse / len(tasks), 4) if tasks else 0.0,
        }
        res.evidence.extend(details[:20])
        res.detail += f"；诊断：{len(tasks)} 条 JSON 任务生成中 {n_parse} 条可被 json.loads 解析（非判据）"
    except Exception as exc:  # noqa: BLE001
        res.evidence.append(f"诊断失败: {type(exc).__name__}: {exc}")

    if proc.returncode == 0 and (proc.stdout or "").strip():
        try:
            json.loads((proc.stdout or "").strip())
            res.verdict = "pass"
        except json.JSONDecodeError:
            res.verdict = "fail"
            res.detail = "判据入口可执行但输出不是合法 JSON"
    res.elapsed_s = time.perf_counter() - t0
    return res


def check_v4(ctx: Ctx) -> CheckResult:
    """V4：scripts/bench.py 单线程口径，200 token 平均 ≥ 200 tok/s。"""
    t0 = time.perf_counter()
    res = CheckResult("V4", "速度 ≥ 200 tok/s",
                      "scripts/bench.py 单线程、贪心、prompt 32、生成 200、丢弃前 10 预热",
                      "fail")
    try:
        m = ctx.decode_measure()
    except Exception as exc:  # noqa: BLE001
        res.detail = f"bench 不可用: {type(exc).__name__}: {exc}"
        res.evidence.append(res.detail)
        res.elapsed_s = time.perf_counter() - t0
        return res
    speed = m["tok_per_s"]
    res.metric = {
        "tok_per_s": round(speed, 2),
        "avg_ms_per_token": round(m["avg_ms_per_token"], 3),
        "timed_tokens": m["timed_tokens"],
        "threads": m["threads"],
        "target_tok_per_s": SPEED_TARGET,
        "gap_ratio": round(speed / SPEED_TARGET, 4),
        "protocol": m["protocol"],
        "platform": _platform_label(),
    }
    res.evidence.append(
        f"口径：single / greedy / prompt_len={m['protocol']['prompt_len']} / "
        f"生成 {m['protocol']['tokens']} / 丢弃 {m['protocol']['warmup_tokens']} 预热")
    res.evidence.append(
        f"实 measured {speed:.2f} tok/s（{m['avg_ms_per_token']:.3f} ms/token，"
        f"{m['threads']} 线程）")
    if judge_speed(speed) == "pass":
        res.verdict = "pass"
        res.detail = f"{speed:.1f} tok/s ≥ {SPEED_TARGET:.0f}"
    else:
        res.verdict = "fail"
        res.detail = (f"{speed:.1f} tok/s < {SPEED_TARGET:.0f}，"
                      f"只有目标的 {speed / SPEED_TARGET:.1%}，"
                      f"每 token 需再快 {SPEED_TARGET / speed:.2f}x")
    res.elapsed_s = time.perf_counter() - t0
    return res


def check_v5(ctx: Ctx) -> CheckResult:
    """V5：峰值常驻内存 ≤ 8GB。"""
    t0 = time.perf_counter()
    res = CheckResult("V5", "内存 ≤ 8GB", "峰值常驻内存（VmHWM/ru_maxrss/PeakWorkingSetSize）≤ 8GB",
                      "fail")
    try:
        m = ctx.decode_measure()
    except Exception as exc:  # noqa: BLE001
        res.detail = f"bench 不可用: {type(exc).__name__}: {exc}"
        res.evidence.append(res.detail)
        res.elapsed_s = time.perf_counter() - t0
        return res
    peak_mb = m["peak_mb"]
    res.metric = {
        "peak_mb": round(peak_mb, 1),
        "limit_mb": MEMORY_LIMIT_MB,
        "source": m["peak_source"],
        "platform": _platform_label(),
        "params": m["params"],
    }
    res.evidence.append(f"峰值来源：{m['peak_source']}，{peak_mb:.1f} MB")
    res.evidence.append("口径：模型 + KV cache + 运行时；非 ps 瞬时 RSS")
    if m["peak_source"] == "unavailable" or peak_mb <= 0:
        res.verdict = "unverified"
        res.detail = "本平台拿不到峰值内存口径（unavailable），不能判通过"
        res.elapsed_s = time.perf_counter() - t0
        return res
    if judge_memory(peak_mb) == "pass":
        res.verdict = "pass"
        res.detail = f"{peak_mb:.1f} MB ≤ {MEMORY_LIMIT_MB / 1024:.0f}GB"
    else:
        res.verdict = "fail"
        res.detail = f"{peak_mb:.1f} MB > {MEMORY_LIMIT_MB / 1024:.0f}GB"
    res.elapsed_s = time.perf_counter() - t0
    return res


def check_v6(_ctx: Ctx) -> CheckResult:
    """V6：目标机（第 5 代酷睿）实测 —— 人工项，本脚本只记录 blocked。"""
    return CheckResult(
        "V6", "目标机实测", "在真第 5 代酷睿（Broadwell，有 AVX2）机器上跑 bench.py 达标",
        "blocked",
        detail="blocked：无第 5 代酷睿机器可用；本机/验收机是 Xeon E5-2643 v2"
               "（2013 Ivy Bridge，无 AVX2），数字是保守下界，不能冒充目标机。",
        metric={"target_machine": "Intel 5th-gen Core (Broadwell, AVX2)",
                "measured_on": _platform_label()},
    )


def check_v7(ctx: Ctx) -> CheckResult:
    """V7：make demo 从干净 clone 到出结果 ≤ 10 分钟。"""
    t0 = time.perf_counter()
    res = CheckResult("V7", "可复现", "make demo 从干净 clone 到出结果 ≤ 10 分钟", "fail")
    makefile = ctx.root / "Makefile"
    if not makefile.is_file():
        res.detail = "入口缺失：仓库根没有 Makefile，`make demo` 无法执行"
        res.evidence.append("ls Makefile → 不存在")
        res.verdict = "fail"
        res.elapsed_s = time.perf_counter() - t0
        return res
    text = makefile.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"^demo\s*:", text, re.M):
        res.detail = "Makefile 存在但没有 demo 目标"
        res.verdict = "fail"
        res.elapsed_s = time.perf_counter() - t0
        return res
    if shutil.which("make") is None:
        res.detail = "make 命令不可用，无法执行 (Judge requires make)"
        res.verdict = "unverified"
        res.elapsed_s = time.perf_counter() - t0
        return res
    clone_dir = ctx.tmpdir / "clean-clone"
    t_clone = time.perf_counter()
    proc = _run(["git", "clone", "--quiet", str(ctx.root), str(clone_dir)],
                cwd=ctx.root, timeout=300)
    if proc.returncode != 0:
        res.detail = f"干净 clone 失败: {(proc.stderr or '')[-300:]}"
        res.elapsed_s = time.perf_counter() - t0
        return res
    clone_s = time.perf_counter() - t_clone
    t_make = time.perf_counter()
    proc = _run(["make", "demo"], cwd=clone_dir, timeout=MAKE_DEMO_LIMIT_S)
    make_s = time.perf_counter() - t_make
    total_s = clone_s + make_s
    res.metric = {"clone_s": round(clone_s, 2), "make_demo_s": round(make_s, 2),
                  "total_s": round(total_s, 2), "limit_s": MAKE_DEMO_LIMIT_S,
                  "exit": proc.returncode}
    res.evidence.append(f"$ git clone <repo> → {clone_s:.1f}s")
    res.evidence.append(f"$ make demo → exit={proc.returncode}, {make_s:.1f}s")
    res.evidence.append("stdout tail: " + (proc.stdout or "").strip()[-500:])
    if proc.returncode == 0 and total_s <= MAKE_DEMO_LIMIT_S:
        res.verdict = "pass"
        res.detail = f"总耗时 {total_s:.1f}s ≤ {MAKE_DEMO_LIMIT_S:.0f}s"
    else:
        res.verdict = "fail"
        res.detail = (f"make demo 失败或超时（exit={proc.returncode}, "
                      f"{total_s:.1f}s / 上限 {MAKE_DEMO_LIMIT_S:.0f}s）")
    res.elapsed_s = time.perf_counter() - t0
    return res


CHECKS = {"V1": check_v1, "V2": check_v2, "V3": check_v3,
          "V4": check_v4, "V5": check_v5, "V6": check_v6, "V7": check_v7}
AUTOMATED = ("V1", "V2", "V3", "V4", "V5", "V7")


# --------------------------------------------------------------------------- #
# 运行与输出
# --------------------------------------------------------------------------- #
def _run(cmd: Sequence[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        return subprocess.run(list(cmd), cwd=str(cwd), env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd, 124, stdout=(exc.stdout or ""), stderr=f"timeout {timeout}s")


def _platform_label() -> str:
    import platform
    return (f"{platform.system()} {platform.release()} {platform.machine()} / "
            f"Python {platform.python_version()}")


def default_ckpt(root: Path) -> Path:
    return root / "checkpoints" / "demo-3m" / "model.pt"


def run(ids: Iterable[str], ckpt: Optional[Path], keep_tmp: bool = False,
        repo_root: Path = ROOT) -> List[CheckResult]:
    ctx = Ctx(ckpt=ckpt, repo_root=repo_root, keep_tmp=keep_tmp)
    results: List[CheckResult] = []
    try:
        for cid in ids:
            fn = CHECKS.get(cid)
            if fn is None:
                continue
            try:
                results.append(fn(ctx))
            except Exception as exc:  # noqa: BLE001 - 单条失败不拖垮整轮
                results.append(CheckResult(cid, cid, "", "unverified",
                                           detail=f"{type(exc).__name__}: {exc}"))
    finally:
        ctx.close()
    return results


def print_text(results: Sequence[CheckResult]) -> None:
    print("=" * 72)
    print("LightPrism Demo 验收（T-0334）")
    print(f"平台：{_platform_label()}")
    print("=" * 72)
    for r in results:
        mark = {"pass": "PASS", "fail": "FAIL", "blocked": "BLOCKED",
                "unverified": "UNVERIFIED"}.get(r.verdict, r.verdict.upper())
        print(f"\n[{mark}] {r.id} {r.title}  ({r.elapsed_s:.2f}s)")
        print(f"  判据：{r.judge}")
        if r.metric:
            print(f"  指标：{json.dumps(r.metric, ensure_ascii=False)[:800]}")
        if r.detail:
            print(f"  结论：{r.detail}")
        for ev in r.evidence:
            for line in ev.splitlines():
                print(f"    {line}")

    auto = [r for r in results if r.id in AUTOMATED]
    n_pass = sum(1 for r in auto if r.verdict == "pass")
    n_fail = sum(1 for r in auto if r.verdict in ("fail", "unverified"))
    blocked = [r.id for r in results if r.verdict == "blocked"]
    print("\n" + "=" * 72)
    print(f"自动项 V1—V5、V7：{n_pass} 通过 / {n_fail} 未通过（共 {len(auto)}）")
    if blocked:
        print(f"人工项 blocked：{', '.join(blocked)}")
    print("=" * 72)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LightPrism Demo 验收（V1—V7）")
    parser.add_argument("--all", action="store_true",
                        help="跑自动项 V1—V5、V7（V6 记 blocked）")
    parser.add_argument("--only", help="只跑指定项，逗号分隔，如 V4,V5")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument("--ckpt", default=None,
                        help="demo checkpoint 路径（默认 checkpoints/demo-3m/model.pt）")
    parser.add_argument("--keep-tmp", action="store_true", help="保留临时目录排查")
    args = parser.parse_args(argv)

    if args.only:
        ids = [x.strip().upper() for x in args.only.split(",") if x.strip()]
    elif args.all:
        ids = list(AUTOMATED) + ["V6"]
    else:
        parser.error("需要 --all 或 --only")
    unknown = [x for x in ids if x not in CHECKS]
    if unknown:
        parser.error(f"未知验收项: {unknown}")

    # 确保 src 在 sys.path（即使没设 PYTHONPATH）
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    ckpt = Path(args.ckpt).expanduser() if args.ckpt else default_ckpt(ROOT)
    results = run(ids, ckpt, keep_tmp=args.keep_tmp)

    if args.json:
        payload = {
            "schema_version": 1,
            "task": "T-0334",
            "platform": _platform_label(),
            "repo": str(ROOT),
            "ckpt": str(ckpt),
            "results": [r.as_dict() for r in results],
            "summary": {
                "automated": [r.id for r in results if r.id in AUTOMATED],
                "passed": [r.id for r in results
                           if r.id in AUTOMATED and r.verdict == "pass"],
                "failed": [r.id for r in results
                           if r.id in AUTOMATED and r.verdict != "pass"],
                "blocked": [r.id for r in results if r.verdict == "blocked"],
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_text(results)

    auto = [r for r in results if r.id in AUTOMATED]
    ok = bool(auto) and all(r.verdict == "pass" for r in auto)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())