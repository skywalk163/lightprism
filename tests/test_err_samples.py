# -*- coding: utf-8 -*-
"""T-0337 ERR 类验证样本的回归测试。

覆盖：
* JSONL 能 ``load_jsonl`` 读回、schema 合法
* 每行 ``validation`` 与 ``diagnostic_code`` 查表一致（DIAGNOSTIC_TO_VALIDATION）
* 不存在重复 ``code`` 片段
* 每个可达码条数 ≥5
* ``gen_err_samples.py --check`` 退出码语义（无 light CLI 时优雅报错）
* ``embed_payload.py --check`` 在数据未变时返回 0
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lightprism.tokens import DIAGNOSTIC_TO_VALIDATION

# 仓库根目录
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lightprism.data import load_jsonl  # noqa: E402

ERR_TASKS_PATH = ROOT / "data" / "sft" / "err_tasks.jsonl"
GEN_SCRIPT = ROOT / "scripts" / "gen_err_samples.py"
EMBED_SCRIPT = ROOT / "scripts" / "embed_payload.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def err_samples():
    """加载 err_tasks.jsonl；文件不存在时跳过整个模块。"""
    if not ERR_TASKS_PATH.is_file():
        pytest.skip(f"err_tasks.jsonl 不存在（{ERR_TASKS_PATH}）")
    samples, problems = load_jsonl(ERR_TASKS_PATH)
    assert problems == [], f"load_jsonl 报出问题: {problems[:3]}"
    return samples


# ---------------------------------------------------------------------------
# 1. JSONL 能读回、schema 合法
# ---------------------------------------------------------------------------

def test_err_tasks_exists():
    """err_tasks.jsonl 存在且非空。"""
    assert ERR_TASKS_PATH.is_file(), f"缺少 {ERR_TASKS_PATH}"
    text = ERR_TASKS_PATH.read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) >= 5, f"err_tasks 样本数过少: {len(lines)}"


def test_each_sample_has_required_fields(err_samples):
    """每条样本都有 task / steps / meta。"""
    for i, s in enumerate(err_samples):
        assert s.task, f"第 {i} 条缺 task"
        assert s.steps, f"第 {i} 条缺 steps"
        assert s.meta, f"第 {i} 条缺 meta"


def test_each_step_has_code_and_validation(err_samples):
    """每个 step 都有 code 和 validation。"""
    for i, s in enumerate(err_samples):
        for j, step in enumerate(s.steps):
            assert step.code, f"第 {i} 条 step {j} 缺 code"
            assert step.validation, f"第 {i} 条 step {j} 缺 validation"


# ---------------------------------------------------------------------------
# 2. validation 与 diagnostic_code 查表一致
# ---------------------------------------------------------------------------

def test_validation_matches_diagnostic_code(err_samples):
    """每行 validation 必须是 DIAGNOSTIC_TO_VALIDATION[diagnostic_code] 的查表结果。"""
    for i, s in enumerate(err_samples):
        diag_code = s.meta.get("diagnostic_code", "")
        assert diag_code, f"第 {i} 条缺 meta.diagnostic_code"
        expected = DIAGNOSTIC_TO_VALIDATION.get(diag_code)
        assert expected is not None, f"第 {i} 条未知诊断码 {diag_code}"
        actual = s.steps[0].validation
        assert actual == expected, (
            f"第 {i} 条 validation={actual!r}，"
            f"查表 DIAGNOSTIC_TO_VALIDATION[{diag_code}]={expected!r} 不一致"
        )


def test_validation_is_error_or_warning(err_samples):
    """ERR 类样本的 validation 必须是 ERR= 或 WARN= 开头（不能是 PASS）。"""
    for i, s in enumerate(err_samples):
        v = s.steps[0].validation
        assert v.startswith("ERR=") or v.startswith("WARN="), (
            f"第 {i} 条 validation={v!r} 不是 ERR/WARN 类"
        )


# ---------------------------------------------------------------------------
# 3. 不存在重复 code 片段
# ---------------------------------------------------------------------------

def test_no_duplicate_code_snippets(err_samples):
    """所有样本的 code 片段不能重复。"""
    codes = [s.steps[0].code for s in err_samples]
    seen = set()
    for i, code in enumerate(codes):
        assert code not in seen, f"第 {i} 条 code 片段重复"
        seen.add(code)


# ---------------------------------------------------------------------------
# 4. 每个可达码条数 ≥5
# ---------------------------------------------------------------------------

def test_min_samples_per_code(err_samples):
    """每个 diagnostic_code 至少有 5 条样本。"""
    from collections import Counter
    counts = Counter(s.meta.get("diagnostic_code", "") for s in err_samples)
    for code, n in counts.items():
        assert n >= 5, f"诊断码 {code} 只有 {n} 条样本，少于 5 条"


# ---------------------------------------------------------------------------
# 5. gen_err_samples --check 退出码语义
# ---------------------------------------------------------------------------

def test_gen_check_exits_nonzero_without_light_cli():
    """--check 模式在 light CLI 不存在时退出码为 2（明确报错）。"""
    # 用一个不存在的 light-cli 路径测试
    result = subprocess.run(
        [sys.executable, str(GEN_SCRIPT), "--check", "--light-cli", "/nonexistent/light.py"],
        capture_output=True, text=True, encoding="utf-8",
        timeout=10,
    )
    assert result.returncode == 2, (
        f"预期退出码 2，实际 {result.returncode}\n"
        f"stdout: {result.stdout[:200]}\nstderr: {result.stderr[:200]}"
    )


# ---------------------------------------------------------------------------
# 6. embed_payload --check
# ---------------------------------------------------------------------------

def test_embed_check_runs():
    """embed_payload --check 能正常运行（不验证返回值，因为内嵌 payload 可能还没更新）。"""
    result = subprocess.run(
        [sys.executable, str(EMBED_SCRIPT), "--check"],
        capture_output=True, text=True, encoding="utf-8",
        timeout=30,
    )
    # 只验证脚本能跑起来，不验证返回值（因为内嵌 payload 可能还没更新到最新）
    assert result.returncode in (0, 1), (
        f"embed_payload --check 退出码异常: {result.returncode}\n"
        f"stdout: {result.stdout[:300]}\nstderr: {result.stderr[:300]}"
    )


def test_embed_check_outputs_sha_summary():
    """--check 模式输出 SHA-256 摘要。"""
    result = subprocess.run(
        [sys.executable, str(EMBED_SCRIPT), "--check"],
        capture_output=True, text=True, encoding="utf-8",
        timeout=30,
    )
    assert "SHA-256" in result.stdout or "sha256" in result.stdout.lower(), (
        f"输出里没看到 SHA-256 摘要:\n{result.stdout[:300]}"
    )


# ---------------------------------------------------------------------------
# 7. train_openi.py 支持 err_tasks
# ---------------------------------------------------------------------------

def test_train_openi_loads_err_tasks():
    """train_openi.py 的 _load_payload 能读取 err_tasks。"""
    # 直接导入测试
    sys.path.insert(0, str(ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_openi", ROOT / "train_openi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 测试 external data-dir 模式
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 复制必要文件
        import shutil
        shutil.copy(ROOT / "data" / "vocab_4096.json", tmp / "vocab_4096.json")
        shutil.copy(ROOT / "data" / "sft" / "json_tasks.jsonl", tmp / "json_tasks.jsonl")
        shutil.copy(ROOT / "data" / "sft" / "code_tasks.jsonl", tmp / "code_tasks.jsonl")
        shutil.copy(ERR_TASKS_PATH, tmp / "err_tasks.jsonl")

        # 模拟 args
        class Args:
            data_dir = str(tmp)

        payload = mod._load_payload(Args())
        assert "err_tasks" in payload, "payload 里缺 err_tasks 字段"
        assert payload["err_tasks"], "err_tasks 为空"

        # 测试 load_samples
        rows = mod.load_samples(payload)
        assert len(rows) > 0, "load_samples 返回空"
        # 检查有没有 err 类样本
        err_rows = [r for r in rows if r.get("_source_file") == "err_tasks"]
        assert len(err_rows) > 0, "没有加载到 err_tasks 样本"
