# -*- coding: utf-8 -*-
"""T-0341 · 一键 demo 门面的「不依赖 checkpoint」回归测试。

背景
----
V7 判据是「从干净 clone 到 ``make demo`` 出结果 ≤ 10 分钟」。T-0334 判 V7 失败，
根因是当时 main 上没有 Makefile，``make demo`` 根本无法执行；T-0335 把 Makefile /
scripts/demo.sh / README / docs 合进 main 之后，需要一条**没有模型权重、没有 torch、
甚至没有 make 也能跑**的检查，把「入口还在、配方还能展开、脚本语法正确、
缺 checkpoint 的机器判据没被改坏」固化下来，避免又变成「没模型就测不了」。

本文件只做静态检查 / ``make -n`` dry-run / 缺 checkpoint 分支，不生成任何 token。
不改动 ``scripts/demo.sh`` 的生成口径，也不触碰 ``make bench`` 的口径。

真实 V7 的时间判据（干净 clone + checkpoint 上实跑 ``make demo``）在
192.168.0.88 (FreeBSD 15.1) 完成，命令与耗时见 ``docs/Demo使用.md`` §5.5。
本测试**不替代**那次实跑，更不以「跳过 make」的方式宣称 V7 通过：有 make 时
会真的调用 ``make -n demo``，没有 make 时只 skip 这一个用例，其余静态检查照跑，
并在 skip 原因里指向 §5.5 的真实 make 证据。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
DEMO_SH = ROOT / "scripts" / "demo.sh"
DEMO_SH_REL = "scripts/demo.sh"
DEFAULT_CKPT = "checkpoints/demo-3m/model.pt"


def _text(path: Path) -> str:
    """始终按 UTF-8 读：Windows 非 UTF-8 locale 下不能依赖 open() 的默认编码。"""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _run(argv: list[object], *, env: dict[str, str] | None = None,
         timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(a) for a in argv],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _find_make() -> str | None:
    for name in ("make", "gmake"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _find_bash() -> str | None:
    """Windows 上 Git for Windows 的 bash.exe 常常不在 PATH，补几个常见位置。"""
    found = shutil.which("bash")
    if found:
        return found
    candidates: list[Path] = [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Git" / "bin" / "bash.exe",
        Path("/usr/bin/bash"),
        Path("/bin/bash"),
    ]
    portable = Path.home() / ".workbuddy" / "binaries" / "PortableGit" / "versions"
    if portable.is_dir():
        candidates.extend(sorted(portable.glob("*/usr/bin/bash.exe")))
        candidates.extend(sorted(portable.glob("*/bin/bash.exe")))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _demo_env() -> dict[str, str]:
    """让 demo.sh 复用跑 pytest 的同一个解释器，避免 PATH 上没有 python3。"""
    env = os.environ.copy()
    # Windows 反斜杠路径交给 MSYS bash 时可能被当转义符，统一成 C:/... 形式
    env["PYTHON"] = Path(sys.executable).as_posix()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _absent_ckpt(tmp_path: Path) -> str:
    """一个肯定不存在的 checkpoint 路径；用 posix 形式传给 bash，避免反斜杠被吃掉。"""
    ckpt = tmp_path / "t0341-no-such-checkpoint" / "model.pt"
    assert not ckpt.exists(), "测试前置条件：该路径必须不存在"
    return ckpt.as_posix()


def _demo_recipe() -> tuple[dict[str, str], str]:
    """取出 Makefile 里的 ?= 变量与 demo 目标配方（不做 make 的变量求值，够用即可）。"""
    text = _text(MAKEFILE)
    variables = dict(re.findall(r"^([A-Z_]+)\s*\?=\s*(.*)$", text, re.M))
    match = re.search(r"^demo[ \t]*:[ \t]*\n((?:\t[^\n]*\n?)+)", text, re.M)
    assert match, "Makefile 里找不到 demo 目标的配方"
    recipe = "\n".join(line.lstrip("\t") for line in match.group(1).splitlines())
    return variables, recipe


def _expand(recipe: str, variables: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        return variables.get(m.group(1) or m.group(2), m.group(0))
    return re.sub(r"\$\(([A-Z_]+)\)|\$\{([A-Z_]+)\}", repl, recipe)


# ---------------------------------------------------------------------------
# 1. 入口还在：Makefile 在 main 上，且 demo 目标不把 checkpoint 当文件依赖
# ---------------------------------------------------------------------------

def test_makefile_declares_demo_target_without_checkpoint_prerequisite() -> None:
    assert MAKEFILE.is_file(), "main 上必须有 Makefile —— 这正是 T-0334 判 V7 失败的根因"
    text = _text(MAKEFILE)
    target = re.search(r"^demo[ \t]*:[^\n]*$", text, re.M)
    assert target, "Makefile 缺少 demo 目标"
    rhs = target.group(0).split(":", 1)[1].strip()
    assert rhs == "", (
        "demo 目标不能把 checkpoint（或其它文件）写成前置依赖，"
        f"否则干净 clone 上没模型就直接失败：{target.group(0)!r}"
    )
    assert re.search(r"^\.PHONY[ \t]*:.*\bdemo\b", text, re.M), "demo 应声明为 .PHONY"


def test_demo_recipe_statically_expands_to_demo_sh() -> None:
    """没有 make 也能验的等价语法/展开检查：配方必须转调 scripts/demo.sh。"""
    variables, recipe = _demo_recipe()
    expanded = _expand(recipe, variables)
    assert DEMO_SH_REL in expanded, f"demo 配方没有转调 demo.sh：{expanded!r}"
    assert "--ckpt" in expanded and "--format" in expanded, expanded
    assert variables.get("CKPT") == DEFAULT_CKPT, variables.get("CKPT")
    assert variables.get("FORMAT") in {"text", "json"}, variables.get("FORMAT")
    assert variables.get("PROMPT"), "PROMPT 默认值不能为空"
    assert DEFAULT_CKPT in expanded, expanded
    assert DEMO_SH.is_file(), "scripts/demo.sh 必须随 Makefile 一起在 main 上"


def test_generation_contract_is_untouched() -> None:
    """红线回归：不改 demo.sh 生成口径、不改 bench 口径、不让 demo 配方绕过脚本。"""
    _, recipe = _demo_recipe()
    assert "demo.sh" in recipe and "-m lightprism.demo" not in recipe
    assert "bench.py" not in recipe, "demo 目标不得夹带 bench 口径"
    script = _text(DEMO_SH)
    assert "set -euo pipefail" in script
    assert "MAX_NEW_TOKENS=32" in script, "生成 token 数默认值被改动"
    assert "load_demo_generator" in script, "生成器编排被改动"
    assert 'sys.exit(0 if payload["ok"] else 1)' in script, "生成失败必须非零退出"


# ---------------------------------------------------------------------------
# 2. 有 make 就真的跑 `make -n demo`；dry-run 不执行生成、不需要 checkpoint
# ---------------------------------------------------------------------------

def test_make_dry_run_demo_needs_no_checkpoint() -> None:
    make = _find_make()
    if make is None:
        pytest.skip(
            "本机 PATH 没有 make/gmake：配方静态展开已由 "
            "test_demo_recipe_statically_expands_to_demo_sh 覆盖；真实 make 判据与耗时 "
            "见 docs/Demo使用.md §5.5（192.168.0.88 / FreeBSD 15.1 实跑）"
        )
    # dry-run 只读 Makefile、展开变量，checkpoint 在不在都不影响结果
    proc = _run([make, "-n", "demo"], timeout=60)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"make -n demo 退出码异常\n{combined}"
    assert DEMO_SH_REL in combined, combined
    assert "--ckpt" in combined and "--format" in combined, combined
    assert "No rule to make target" not in combined, combined
    # dry-run 只打印命令：绝不能真的执行 demo.sh 或触发 checkpoint 分支
    assert "checkpoint_missing" not in combined, "make -n 不应执行生成脚本"
    assert "本次未执行生成" not in combined, "make -n 不应执行生成脚本"


# ---------------------------------------------------------------------------
# 3. demo.sh 本体：语法、帮助、参数校验、缺 checkpoint 的机器判据
# ---------------------------------------------------------------------------

def test_demo_sh_passes_bash_syntax_check() -> None:
    bash = _find_bash()
    if bash is None:
        pytest.skip("本机没有 bash，跳过脚本语法检查")
    proc = _run([bash, "-n", DEMO_SH_REL])
    assert proc.returncode == 0, f"bash -n 失败\n{proc.stdout}{proc.stderr}"


def test_missing_checkpoint_json_is_machine_readable(tmp_path: Path) -> None:
    """核心：没有模型权重时仍可验证，且机器判据明确区分「没跑」与「跑砸了」。"""
    bash = _find_bash()
    if bash is None:
        pytest.skip("本机没有 bash，跳过 demo.sh 运行检查")
    ckpt = _absent_ckpt(tmp_path)
    proc = _run(
        [bash, DEMO_SH_REL, "--ckpt", ckpt, "--format", "json", "T-0341 probe"],
        env=_demo_env(),
    )
    assert proc.returncode == 0, f"缺 checkpoint 应正常退出\n{proc.stdout}{proc.stderr}"
    raw = proc.stdout
    payload = json.loads(raw[raw.find("{"):])
    assert payload["ok"] is False, payload
    assert payload["error"] == "checkpoint_missing", payload
    assert payload["generated_tokens"] is None, payload
    assert payload["elapsed_s"] is None and payload["tok_per_s"] is None, payload
    assert payload["ckpt"] == ckpt, payload
    assert payload["light_ai_check"]["status"] == "skipped", payload


def test_missing_checkpoint_text_is_human_readable(tmp_path: Path) -> None:
    bash = _find_bash()
    if bash is None:
        pytest.skip("本机没有 bash，跳过 demo.sh 运行检查")
    ckpt = _absent_ckpt(tmp_path)
    proc = _run(
        [bash, DEMO_SH_REL, "--ckpt", ckpt, "--format", "text", "T-0341 probe"],
        env=_demo_env(),
    )
    assert proc.returncode == 0, f"缺 checkpoint 应正常退出\n{proc.stdout}{proc.stderr}"
    assert "未找到" in proc.stdout, proc.stdout
    assert "本次未执行生成" in proc.stdout, proc.stdout
    assert "Traceback" not in (proc.stdout + proc.stderr), "缺 checkpoint 不能抛 traceback"


def test_bad_format_is_rejected_without_checkpoint(tmp_path: Path) -> None:
    bash = _find_bash()
    if bash is None:
        pytest.skip("本机没有 bash，跳过 demo.sh 运行检查")
    ckpt = _absent_ckpt(tmp_path)
    proc = _run(
        [bash, DEMO_SH_REL, "--ckpt", ckpt, "--format", "bogus", "T-0341 probe"],
        env=_demo_env(),
    )
    assert proc.returncode == 64, f"非法 --format 应以 EX_USAGE(64) 拒绝\n{proc.stdout}{proc.stderr}"
    assert "text" in proc.stderr and "json" in proc.stderr, proc.stderr


def test_demo_sh_help_does_not_need_checkpoint() -> None:
    bash = _find_bash()
    if bash is None:
        pytest.skip("本机没有 bash，跳过 demo.sh 运行检查")
    proc = _run([bash, DEMO_SH_REL, "--help"], env=_demo_env())
    assert proc.returncode == 0, f"--help 应正常退出\n{proc.stdout}{proc.stderr}"
    assert "用法" in proc.stdout and "--ckpt" in proc.stdout, proc.stdout
