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


# ---- T-0371：共享段标记渲染器 / unified 数据 / 词表追加 ----------------
PY_SRC = "x = 1\nprint(x)"
LIGHT_SRC = "设 x 为 1。\n印 x。"


def test_translation_training_and_inference_share_one_renderer():
    from lightprism.demo.sft_build import (
        SEGMENT_MODES,
        render_inference_translation,
        render_training_translation,
    )

    for mode in SEGMENT_MODES:
        train = render_training_translation("把 Python 翻译成光明", PY_SRC, LIGHT_SRC, mode=mode)
        prompt = render_inference_translation("把 Python 翻译成光明", PY_SRC, mode=mode)
        # 推理 prompt 必须恰好是训练序列的前缀：同一渲染核心，杜绝两边漂移。
        assert train.startswith(prompt), f"{mode} 训练/推理不是同一格式"
        assert PY_SRC in prompt
        if mode == "markers":
            assert prompt.endswith("<PY>\n" + PY_SRC + "\n")
            assert "<CODE>" not in train
        else:
            assert prompt.endswith("<CODE>\n" + PY_SRC + "\n</CODE>\n")


def test_renderer_source_contract_catches_one_sided_edit():
    """反向验证的源码级锚点：两个入口都必须走同一 `_render_translation_core`。

    如果有人把训练侧改成硬编码 `<CODE>`，这里会红——不需要真的改坏一次也能
    防止「训练改了、推理忘了」的单边漂移。
    """
    import inspect

    from lightprism.demo import sft_build

    train_src = inspect.getsource(sft_build.render_training_translation)
    infer_src = inspect.getsource(sft_build.render_inference_translation)
    assert "_render_translation_core" in train_src
    assert "_render_translation_core" in infer_src
    assert "include_target=True" in train_src
    assert "include_target=False" in infer_src


def test_training_inference_drift_would_be_visible():
    from lightprism.demo.sft_build import (
        render_inference_translation,
        render_training_translation,
    )

    marked_train = render_training_translation("t", PY_SRC, LIGHT_SRC, mode="markers")
    legacy_prompt = render_inference_translation("t", PY_SRC, mode="baseline")
    # 这正是「训练改 markers、推理还留在 baseline」的形态：前缀关系断裂。
    assert not marked_train.startswith(legacy_prompt)


def test_marked_format_extracts_target_and_has_teeth():
    from lightprism.demo.sft_build import (
        extract_light_segment,
        format_ok,
        render_training_translation,
        translation_mixed_segment_error,
    )

    train = render_training_translation("t", PY_SRC, LIGHT_SRC)
    assert extract_light_segment(train) == LIGHT_SRC.strip()
    assert format_ok(train)
    assert not translation_mixed_segment_error(train)
    # 空光明段：抽不到代码，判不合法
    assert not format_ok(render_training_translation("t", PY_SRC, ""))
    # 光明段里混进 Python 源码（混段错误的判别力）
    bad = train.replace(LIGHT_SRC, "def f():\n    return 1")
    assert translation_mixed_segment_error(bad)


def test_baseline_mode_keeps_two_code_blocks_for_ab():
    from lightprism.demo.sft_build import (
        extract_light_segment,
        format_ok,
        render_training_translation,
    )

    seq = render_training_translation("t", PY_SRC, LIGHT_SRC, mode="baseline")
    assert seq.count("<CODE>") == 2 and seq.count("</CODE>") == 2
    assert "<PY>" not in seq and "<LIGHT>" not in seq
    assert extract_light_segment(seq, mode="baseline") == LIGHT_SRC.strip()
    assert format_ok(seq, mode="baseline")


def test_unified_translation_rows_carry_marked_sequences():
    from lightprism.demo.sft_build import render_translation_record
    from lightprism.sequence import parse_sequence

    unified = ROOT / "data" / "unified"
    rows = [json.loads(line) for line in
            (unified / "translation_tasks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert len(rows) > 100
    checked = 0
    for rec in rows:
        marked = rec.get("sequence_marked")
        assert marked, f"缺 sequence_marked: {rec.get('sha256')}"
        assert marked == render_translation_record(rec, mode="markers")
        kind = str(rec.get("kind") or "pair")
        assert "<LIGHT>" in marked
        if kind == "pair":
            assert "<PY>" in marked and "<CODE>" not in marked
        else:
            assert "<PY>" not in marked and "<CODE>" not in marked
        body = marked.split("\n", 1)[1]
        parsed = parse_sequence(body)
        assert parsed.validate() == [], f"段序列不合法: {rec.get('sha256')}"
        checked += 1
    assert checked == len(rows)

    manifest = json.loads((unified / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vocab"]["size"] == 2902
    assert manifest["vocab"]["control_span"] == [2868, 2902]
    assert manifest["vocab"]["segment_markers"] == {"<PY>": 2900, "<LIGHT>": 2901}
    from lightprism.runtime.err_samples import _sha256 as crlf_sha256
    assert manifest["sha256"]["translation_tasks.jsonl"] == crlf_sha256(
        unified / "translation_tasks.jsonl")


def test_apply_segment_markers_is_idempotent_and_keeps_old_ids(tmp_path):
    import shutil

    from lightprism.demo.sft_build import apply_segment_markers

    old_vocab = ROOT / "data" / "vocab_4096.json"
    target = tmp_path / "vocab_4096.json"
    shutil.copy(old_vocab, target)
    before = json.loads(target.read_text(encoding="utf-8"))
    first = apply_segment_markers(target)
    assert first["changed"] is True
    assert first["ids"] == {"<PY>": 2900, "<LIGHT>": 2901}
    after = json.loads(target.read_text(encoding="utf-8"))
    assert after["tokens"][:len(before["tokens"])] == before["tokens"], "旧 token id 被改动"
    assert after["size"] == 2902
    assert after["control_layout"]["end"] == 2902
    assert after["tokens"][-2:] == ["<PY>", "<LIGHT>"]
    assert after["counts"]["control"] == before["counts"]["control"] + 2
    # 幂等：第二次不应重复追加
    second = apply_segment_markers(target)
    assert second["changed"] is False
    assert json.loads(target.read_text(encoding="utf-8"))["size"] == 2902
