# -*- coding: utf-8 -*-
"""T-0333 数据配比与 JSON 输出任务集的回归测试。

覆盖卡片「完成标准」里的四条：条数范围、JSON 可解析、schema 兼容、不超 500，
外加 schema 前缀、代码类样本来源与 ``--validate`` CLI 本身。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lightprism.data import SOURCES, load_jsonl
from lightprism.demo.sft_build import (
    CATEGORIES,
    CODE_SEGMENT_RE,
    CODE_TASKS_PATH,
    FORMAT_PREFIX_RE,
    JSON_TASKS_PATH,
    MANIFEST_PATH,
    MAX_SAMPLES,
    MIN_SAMPLES,
    ROOT,
    SEQUENCES_PATH,
    SFT_SOURCE,
    render_training_sequence,
)


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def json_samples():
    assert JSON_TASKS_PATH.is_file(), f"缺少 {JSON_TASKS_PATH}"
    samples, problems = load_jsonl(JSON_TASKS_PATH)
    assert problems == [], f"load_jsonl 报出问题: {problems[:3]}"
    return samples


@pytest.fixture(scope="module")
def code_samples():
    assert CODE_TASKS_PATH.is_file(), f"缺少 {CODE_TASKS_PATH}"
    samples, problems = load_jsonl(CODE_TASKS_PATH)
    assert problems == [], f"load_jsonl 报出问题: {problems[:3]}"
    return samples


# ---- 条数 ---------------------------------------------------------------
def test_row_count_in_range():
    n = len(_lines(JSON_TASKS_PATH))
    assert MIN_SAMPLES <= n <= MAX_SAMPLES, f"JSON 任务集 {n} 条，不在 200—500 之间"


def test_never_over_500():
    n_json = len(_lines(JSON_TASKS_PATH))
    n_code = len(_lines(CODE_TASKS_PATH))
    assert n_json <= MAX_SAMPLES
    assert n_json + n_code <= MAX_SAMPLES, f"合计 {n_json + n_code} 超过 500"


# ---- schema 兼容 --------------------------------------------------------
def test_load_jsonl_keeps_every_row(json_samples):
    assert len(json_samples) == len(_lines(JSON_TASKS_PATH)), "有行被 load_jsonl 丢弃"


def test_samples_use_registered_source(json_samples):
    assert SFT_SOURCE in SOURCES
    for s in json_samples:
        assert s.source in SOURCES, f"source {s.source!r} 未登记"
        assert s.meta.get("source_kind") == "demo_json"


def test_samples_are_consumable_by_existing_sequence_parser(json_samples):
    """schema 兼容：每行都能走 Sample.to_sequence()（现有序列解析器）。"""
    for s in json_samples:
        seq = s.to_sequence()
        assert seq.task == s.task
        assert seq.validate() == [], f"序列不合法: {s.task[:30]}"


def test_code_samples_are_schema_compatible(code_samples):
    assert 50 <= len(code_samples) <= 100
    for s in code_samples:
        assert s.source in SOURCES
        assert s.meta.get("format") == "code"
        assert s.meta.get("origin", "").startswith("light/")
        assert s.steps and s.steps[0].code.strip()
        seq = s.to_sequence()
        assert seq.task == s.task


# ---- 类别配比 -----------------------------------------------------------
def test_every_category_present_and_bounded(json_samples):
    counts = {c: 0 for c in CATEGORIES}
    for s in json_samples:
        counts[s.meta["category"]] += 1
    for cat, n in counts.items():
        assert 30 <= n <= 80, f"类别 {cat} 只有 {n} 条（要求 30—80）"
    assert sum(counts.values()) == len(json_samples)


# ---- JSON 可解析 --------------------------------------------------------
def test_every_code_segment_is_parseable_json(json_samples):
    parsed = 0
    for s in json_samples:
        seq = render_training_sequence(s, "json")
        segments = CODE_SEGMENT_RE.findall(seq)
        assert len(segments) == 1, f"{s.task[:30]} 没有唯一的 <CODE> 段"
        obj = json.loads(segments[0].strip())
        assert isinstance(obj, dict), "JSON 答案必须是对象（模型要先输出 {）"
        parsed += 1
    assert parsed == len(json_samples)


def test_training_sequences_have_schema_prefix(json_samples):
    for s in json_samples:
        seq = render_training_sequence(s, "json")
        m = FORMAT_PREFIX_RE.match(seq)
        assert m and m.group(1) == "json", "缺 <FORMAT>json</FORMAT> schema 前缀"
        assert f"<TASK>{s.task}</TASK>" in seq
        assert "<V:PASS>" in seq
        # schema 前缀之后紧跟 <TASK>，再紧跟 <CODE>{ —— 让模型学会先输出 {
        assert seq.splitlines()[2].startswith("<CODE>{")


def test_prebuilt_sequences_file(json_samples):
    rows = [json.loads(line) for line in _lines(SEQUENCES_PATH)]
    json_rows = [r for r in rows if r["format"] == "json"]
    code_rows = [r for r in rows if r["format"] == "code"]
    assert len(json_rows) == len(json_samples)
    assert len(code_rows) >= 50
    assert len(rows) <= MAX_SAMPLES
    for r in json_rows:
        assert FORMAT_PREFIX_RE.match(r["sequence"])
        segments = CODE_SEGMENT_RE.findall(r["sequence"])
        assert len(segments) == 1
        assert isinstance(json.loads(segments[0].strip()), dict)
    for r in code_rows:
        assert r["sequence"].startswith("<FORMAT>code</FORMAT>\n")
        assert CODE_SEGMENT_RE.search(r["sequence"])


# ---- 代码类样本来源 -----------------------------------------------------
def test_code_samples_come_from_real_corpus_and_are_checked(code_samples):
    checked = 0
    for s in code_samples:
        chk = s.meta.get("light_ai_check") or {}
        assert chk.get("ok") is True, f"{s.meta.get('origin')} 没有通过的 light ai check"
        assert "ai check" in chk.get("command", "")
        checked += 1
        assert s.task.strip(), "代码样本缺任务描述"
    assert checked == len(code_samples)


def test_manifest_matches_deliverables(json_samples, code_samples):
    m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert m["counts"]["json_tasks"] == len(json_samples)
    assert m["counts"]["code_tasks"] == len(code_samples)
    assert m["counts"]["total"] == len(json_samples) + len(code_samples) <= MAX_SAMPLES
    assert m["json_loads"]["pass_rate"] == 1.0
    assert m["counts"]["by_category"] == {
        c: sum(1 for s in json_samples if s.meta["category"] == c) for c in CATEGORIES
    }


# ---- 验收命令本身 -------------------------------------------------------
def test_validate_cli_exits_zero():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "lightprism.demo.sft_build",
         "--validate", "data/sft/json_tasks.jsonl"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[PASS]" in proc.stdout
