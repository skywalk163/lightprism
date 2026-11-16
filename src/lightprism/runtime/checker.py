# -*- coding: utf-8 -*-
"""对 ``light`` CLI 的**真实**调用封装：``ai check --json`` 与 ``run``。

两条硬约束
----------
1. ``<V:...>`` 的唯一来源是 ``light ai check --json --no-corpus --run`` 的
   原始 JSON payload；本模块把它原样带回来（``payload`` / ``stdout`` / ``cmd``
   / ``rc`` / ``cwd``），供决策日志逐条回溯。
2. ``--run`` 不可省：``reports/ERR样本覆盖.md`` 与 ``exporters/light_check.py``
   实测，不带 ``--run`` 时运行时错误（L3001 越界）会被判成 ``ok: true``。
   ``--no-corpus`` 也不可省：否则每次评估都往 light 的报错归档里写，污染数据源。

真跑输出另取 ``light run``：``ai check --json`` 的 payload **不含程序 stdout**
（``cli/light.py::_ai_check_json`` 只把 ``ran.stdout`` 拼进诊断文本，不返回），
所以「真跑输出与预期一致」必须单独跑一次并比对，不能拿 checker 的 ok 顶替。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

__all__ = [
    "CheckResult", "RunResult", "Runner", "LightRunner",
    "find_light_root", "CHECK_CMD_TEMPLATE", "RUN_CMD_TEMPLATE",
    "VERIFY_EVIDENCE_FIELDS", "verify_log",
]

CHECK_CMD_TEMPLATE = (
    "python -X utf8 -m cli.light ai check --json --no-corpus --run "
    "--timeout {timeout} <file>"
)
RUN_CMD_TEMPLATE = "python -X utf8 -m cli.light run <file>"

_COMMON_LIGHT_ROOTS = (
    r"C:\dswork\light",
    r"G:\dswork\duan-light-merge\light-merge",
    "/home/skywalk/dswork/light",
    "/home/workbuddy/dswork/light",
    "/usr/home/workbuddy/dswork/light",
)


def find_light_root(explicit: Optional[str] = None) -> Optional[Path]:
    """定位 ``light`` 仓根（含 ``cli/light.py``），本机 / FreeBSD 都能找到。

    顺序：显式参数 → 环境变量 ``LIGHT_ROOT`` / ``LIGHT_CLI_ROOT`` →
    从本文件向上逐级找 ``<p>/light/cli/light.py`` 与 ``<p>/cli/light.py``
    → 若干已知绝对路径。找不到返回 None（由调用方如实记 CHECKER_FAIL，不硬凑）。
    """
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    for key in ("LIGHT_ROOT", "LIGHT_CLI_ROOT"):
        value = os.environ.get(key)
        if value:
            candidates.append(value)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(str(parent / "light"))
        candidates.append(str(parent))
    candidates.extend(_COMMON_LIGHT_ROOTS)

    seen = set()
    for raw in candidates:
        try:
            p = Path(raw).expanduser()
        except (OSError, ValueError):
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if (p / "cli" / "light.py").is_file():
            return p
    return None


@dataclass
class CheckResult:
    payload: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    rc: Optional[int] = None
    cmd: List[str] = field(default_factory=list)
    cwd: str = ""
    error: str = ""
    elapsed_ms: int = 0


@dataclass
class RunResult:
    stdout: str = ""
    stderr: str = ""
    rc: Optional[int] = None
    cmd: List[str] = field(default_factory=list)
    cwd: str = ""
    error: str = ""
    elapsed_ms: int = 0


class Runner(Protocol):
    """引擎依赖的最小接口；测试注入假 runner 时不碰真实 light CLI。"""

    @property
    def cwd(self) -> str: ...

    def check(self, file_path: str) -> CheckResult: ...

    def run(self, file_path: str) -> RunResult: ...


class LightRunner:
    """真实调用 light CLI。所有调用都显式带 ``-X utf8``（Windows 必需）。"""

    def __init__(
        self,
        light_root: Optional[str] = None,
        *,
        python: str = "",
        timeout: float = 240.0,
        light_run_timeout: int = 60,
    ) -> None:
        root = find_light_root(light_root)
        self._root = root
        self._python = python or sys.executable
        self._timeout = float(timeout)
        self._light_run_timeout = int(light_run_timeout)
        self.cwd = str(root) if root else ""

    @property
    def light_root(self) -> Optional[Path]:
        return self._root

    def available(self) -> bool:
        return self._root is not None

    def _exec(self, cmd: List[str]) -> "subprocess.CompletedProcess[str] | str":
        if not self.available():
            return "light 仓不可用：找不到 cli/light.py（设置 LIGHT_ROOT 可显式指定）"
        try:
            return subprocess.run(
                cmd,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return f"子进程超时（{self._timeout}s）: {exc}"
        except OSError as exc:
            return f"无法启动子进程: {exc}"

    def check(self, file_path: str) -> CheckResult:
        cmd = [
            self._python, "-X", "utf8", "-m", "cli.light", "ai", "check",
            "--json", "--no-corpus", "--run",
            "--timeout", str(self._light_run_timeout), str(file_path),
        ]
        started = time.monotonic()
        result = CheckResult(cmd=cmd, cwd=self.cwd)
        proc = self._exec(cmd)
        result.elapsed_ms = int(round((time.monotonic() - started) * 1000))
        if isinstance(proc, str):
            result.error = proc
            return result
        result.rc = proc.returncode
        result.stdout = proc.stdout or ""
        result.stderr = proc.stderr or ""
        payload = _parse_json(result.stdout)
        if payload is None:
            result.error = (
                f"light ai check stdout 不是合法 JSON（rc={proc.returncode}）: "
                f"{result.stdout[:200]!r}"
            )
        else:
            result.payload = payload
        return result

    def run(self, file_path: str) -> RunResult:
        cmd = [self._python, "-X", "utf8", "-m", "cli.light", "run", str(file_path)]
        started = time.monotonic()
        result = RunResult(cmd=cmd, cwd=self.cwd)
        proc = self._exec(cmd)
        result.elapsed_ms = int(round((time.monotonic() - started) * 1000))
        if isinstance(proc, str):
            result.error = proc
            return result
        result.rc = proc.returncode
        result.stdout = proc.stdout or ""
        result.stderr = proc.stderr or ""
        return result


def _parse_json(stdout: str) -> Optional[Dict[str, Any]]:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# 日志自查入口：PYTHONPATH=src python -m lightprism.runtime.checker --verify <jsonl>
# --------------------------------------------------------------------------
#: 每条决策记录都必须带上的「真实调用证据」字段（不是模型生成的描述）。
VERIFY_EVIDENCE_FIELDS = ("checker_cmd", "checker_rc", "checker_payload")


def verify_log(path: str | Path, *, min_records: int = 0) -> Tuple[int, List[str], List[str]]:
    """逐条校验决策日志：记录结构合法 + 真实调用证据齐全。

    返回 ``(条数, 问题清单, 警告清单)``；问题清单为空即通过。
    """
    from .decisions import validate_record  # 延迟导入，避免顶层循环依赖

    p = Path(path)
    problems: List[str] = []
    warnings: List[str] = []
    count = 0
    seen_sha: Dict[str, str] = {}
    dup = 0
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        count += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"第{lineno}行: 不是合法 JSON: {exc}")
            continue
        tag = f"第{lineno}行({rec.get('task_id')}#{rec.get('attempt')})"
        for issue in validate_record(rec):
            problems.append(f"{tag}: {issue}")
        for field_name in VERIFY_EVIDENCE_FIELDS:
            if rec.get(field_name) in (None, "", [], {}):
                problems.append(f"{tag}: 缺少 {field_name}（真实调用证据）")
        if rec.get("checker_rc") is not None and not isinstance(rec.get("checker_rc"), int):
            warnings.append(f"{tag}: checker_rc 不是整数: {rec.get('checker_rc')!r}")
        sha = str(rec.get("source_sha256") or "")
        if sha:
            if sha in seen_sha:
                dup += 1
            else:
                seen_sha[sha] = tag
    if min_records and count < min_records:
        problems.append(f"只有 {count} 条记录，少于要求的 {min_records} 条")
    if count:
        rate = dup / count
        if rate > 0.20:
            problems.append(f"重复率 {rate:.2%} 超过 20%（{dup} 条重复源码）")
    return count, problems, warnings


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="决策日志自查（结构 + 真实调用证据）")
    ap.add_argument("--verify", metavar="JSONL", default="",
                    help="校验决策日志：每条都要有 checker_cmd/checker_rc/checker_payload")
    ap.add_argument("--min-records", type=int, default=0, help="期望的最小记录条数")
    args = ap.parse_args(argv)

    if not args.verify:
        ap.print_help()
        return 2
    path = Path(args.verify)
    if not path.is_file():
        print(f"[错误] 找不到日志 {path}", file=sys.stderr)
        return 2
    count, problems, warnings = verify_log(path, min_records=args.min_records)
    for w in warnings[:20]:
        print(f"[警告] {w}")
    if problems:
        print(f"[FAIL] {path}：{count} 条记录，{len(problems)} 处问题")
        for issue in problems[:20]:
            print(f"  - {issue}")
        if len(problems) > 20:
            print(f"  ... 还有 {len(problems) - 20} 处")
        return 1
    print(f"[OK] {path}：{count} 条记录，字段齐全，每条都带真实调用证据"
          f"（checker_cmd / checker_rc / checker_payload）")
    print("checker 后端：" + (find_light_root() or "未找到 light 仓").__str__())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
