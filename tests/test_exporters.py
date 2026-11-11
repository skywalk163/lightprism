# -*- coding: utf-8 -*-
"""样本导出器测试：用假 corpus + 假 checker，不依赖真实 light 仓与真实归档。

真实验证另有一组（``test_export_against_real_corpus``），标 ``skipif`` —— 只在
本机存在 ``~/.light/error_corpus`` 时跑，避免 CI 依赖用户家目录。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from lightprism.exporters import (
    LightChecker,
    SourceIndex,
    export_samples,
    read_corpus,
)
from lightprism.exporters.light_check import (
    DIAGNOSTIC_TO_FIX,
    USAGE_ERROR_CODES,
    _sha16,
    fault_to_fix_decision,
)
from lightprism.tokens import DIAGNOSTIC_TO_VALIDATION


# ---- 构造假 corpus ----------------------------------------------------
def _write_corpus(root: Path, records, month: str = "2026-11") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{month}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _record(code: str, source: str, message: str = "测试错误", ts: str = "2026-11-11T02:00:00+00:00"):
    return {
        "ts": ts,
        "last_seen": ts,
        "code": code,
        "message": message,
        "hint": "",
        "fixes": [],
        "source_snippet": source[:40],
        "source_hash": _sha16(source),
        "lang_level": "v3",
        "backend": "src",
        "light_version": "0.4.1",
        "hits": 1,
    }


class FakeChecker:
    """按源文件内容决定「现在是否通过」。"""

    def __init__(self, passing: set[str] | None = None, error: str = ""):
        self.passing = passing or set()
        self.error = error
        self.calls: list[str] = []

    def check_source(self, source: str) -> dict:
        self.calls.append(source)
        if self.error:
            return {"ok": False, "validation": "ERR=CHECKER_FAIL", "diagnostics": [],
                    "error": self.error, "mode": "syntax+run"}
        ok = source in self.passing
        return {"ok": ok, "validation": "PASS" if ok else "ERR=RUNTIME",
                "diagnostics": [], "error": "", "mode": "syntax+run"}


# ---- hash 与 corpus 读取 ----------------------------------------------
def test_sha16_matches_light_error_corpus_algorithm():
    # light 侧：先归一化换行，再 sha256 前 16 位
    source = "设 甲 为 1\r\n打印(甲)\n"
    expected = hashlib.sha256("设 甲 为 1\n打印(甲)\n".encode("utf-8")).hexdigest()[:16]
    assert _sha16(source) == expected


def test_read_corpus_dedupes_same_code_and_hash(tmp_path):
    src = "设 甲 为 [1,2]\n打印(甲[9])\n"
    rec = _record("L3001", src, ts="2026-11-01T00:00:00+00:00")
    other = dict(rec, ts="2026-11-02T00:00:00+00:00", last_seen="2026-11-02T00:00:00+00:00")
    _write_corpus(tmp_path, [rec])
    # 第二个月文件里同键再出现一次
    _write_corpus(tmp_path, [other], month="2026-12")

    records, problems = read_corpus(tmp_path)
    assert problems == []
    assert len(records) == 1                       # 同 (code, hash) 只留一条
    assert records[0].last_seen.startswith("2026-11-02")   # 取较新的 last_seen


def test_read_corpus_takes_hits_from_index_not_jsonl(tmp_path):
    src = "设 甲 为 1\n"
    _write_corpus(tmp_path, [_record("L2002", src)])       # JSONL 里 hits=1
    index = {"entries": {
        f"L2002\x1f{_sha16(src)}": {"code": "L2002", "hits": 7},
    }}
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")

    records, _ = read_corpus(tmp_path)
    # JSONL 的 hits 恒为首次值，真实次数只在 index 里
    assert records[0].hits == 7


def test_read_corpus_reports_bad_lines_without_raising(tmp_path):
    path = tmp_path / "2026-11.jsonl"
    path.write_text('not json\n{"code": "L2002", "source_hash": "x"}\n', encoding="utf-8")
    records, problems = read_corpus(tmp_path)
    assert len(records) == 1
    assert any("JSON 解析失败" in p for p in problems)


def test_read_corpus_missing_dir_is_reported(tmp_path):
    records, problems = read_corpus(tmp_path / "nope")
    assert records == []
    assert any("不存在" in p for p in problems)


# ---- SourceIndex ------------------------------------------------------
def test_source_index_finds_file_by_hash(tmp_path):
    src = "设 甲 为 1\n"
    (tmp_path / "a.light").write_text(src, encoding="utf-8")
    (tmp_path / "b.txt").write_text("无关内容", encoding="utf-8")

    index = SourceIndex().build([tmp_path])
    assert index.find(_sha16(src)) == tmp_path / "a.light"
    assert index.find("deadbeefdeadbeef") is None
    # 只扫 .light，不该把 .txt 算进去
    assert index.scanned == 1


def test_source_index_prefers_shortest_path_for_duplicates(tmp_path):
    src = "设 甲 为 1\n"
    short = tmp_path / "a.light"
    deep = tmp_path / "sub" / "deeper" / "a.light"
    short.write_text(src, encoding="utf-8")
    deep.parent.mkdir(parents=True)
    deep.write_text(src, encoding="utf-8")
    index = SourceIndex().build([tmp_path])
    assert index.find(_sha16(src)) == short


def test_source_index_skips_oversized_files(tmp_path):
    (tmp_path / "big.light").write_text("x" * 2000, encoding="utf-8")
    index = SourceIndex(max_file_bytes=100).build([tmp_path])
    assert index.scanned == 0
    assert index.skipped_large == 1


# ---- 导出主流程 -------------------------------------------------------
def test_export_requires_source_roots_and_counts_skips(tmp_path):
    src = "设 甲 为 [1,2]\n打印(甲[9])\n"
    _write_corpus(tmp_path / "corpus", [_record("L3001", src)])

    result = export_samples(corpus_dir=tmp_path / "corpus", source_roots=[])
    assert result.samples == []
    assert result.stats.skipped_no_source == 1
    assert result.stats.exported_failed == 0


def test_export_produces_failure_sample_not_repair_sample(tmp_path):
    src = "设 甲 为 [1,2]\n打印(甲[9])\n"
    _write_corpus(tmp_path / "corpus", [_record("L3001", src)])
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.light").write_text(src, encoding="utf-8")

    checker = FakeChecker(passing=set())          # 现跑仍失败
    result = export_samples(
        corpus_dir=tmp_path / "corpus",
        source_roots=[tmp_path / "src"],
        checker=checker,
    )
    assert result.stats.exported_failed == 1
    assert result.stats.exported_stale == 0
    sample = result.samples[0]
    # 只有一步，且**没有 PASS** —— 修复样本不在这里造
    assert len(sample.steps) == 1
    assert sample.steps[0].validation == "ERR=RUNTIME"
    assert all(s.validation != "PASS" for s in sample.steps)
    assert sample.meta["stale"] is False
    assert sample.meta["live_check"] == "still_failing"


def test_export_marks_stale_when_source_now_passes(tmp_path):
    """同 hash 却现跑通过 = 陈旧报错，标 stale 而不是造修复样本。"""
    src = "设 甲 为 [1,2]\n打印(甲[9])\n"
    _write_corpus(tmp_path / "corpus", [_record("L3001", src)])
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.light").write_text(src, encoding="utf-8")

    checker = FakeChecker(passing={src})          # 同源码，现在通过了
    result = export_samples(
        corpus_dir=tmp_path / "corpus",
        source_roots=[tmp_path / "src"],
        checker=checker,
    )
    assert result.stats.exported_stale == 1
    assert result.stats.exported_failed == 0
    sample = result.samples[0]
    assert sample.meta["stale"] is True
    assert sample.meta["live_check"] == "pass_now"
    # 关键：即便现跑通过，也**不**生成 PASS 步骤
    assert all(s.validation != "PASS" for s in sample.steps)


def test_export_skips_usage_errors_and_unknown_codes(tmp_path):
    src = "设 甲 为 1\n"
    _write_corpus(tmp_path / "corpus", [
        _record("L0000", src),           # 用法错误
        _record("L9999", src),           # 没有验证 token 映射
    ])
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.light").write_text(src, encoding="utf-8")

    result = export_samples(corpus_dir=tmp_path / "corpus", source_roots=[tmp_path / "src"])
    assert result.stats.skipped_usage_error == 1
    assert result.stats.skipped_unknown_code == 1
    assert result.samples == []
    assert any("L9999" in p for p in result.problems)


def test_export_distinguishes_checker_failure_from_real_error(tmp_path):
    """「没跑成」必须与「跑出错误」分开计，不能混成一个失败样本。"""
    src = "设 甲 为 1\n"
    _write_corpus(tmp_path / "corpus", [_record("L3001", src)])
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.light").write_text(src, encoding="utf-8")

    checker = FakeChecker(error="light 仓不可用")
    result = export_samples(
        corpus_dir=tmp_path / "corpus", source_roots=[tmp_path / "src"], checker=checker
    )
    assert result.stats.skipped_checker_failed == 1
    assert result.stats.exported_failed == 0
    assert result.samples == []


def test_export_respects_min_hits_and_limit(tmp_path):
    sources = [f"设 甲 为 {i}\n" for i in range(4)]
    _write_corpus(tmp_path / "corpus", [_record("L2002", s) for s in sources])
    (tmp_path / "src").mkdir()
    for i, s in enumerate(sources):
        (tmp_path / "src" / f"f{i}.light").write_text(s, encoding="utf-8")

    # min_hits=2：JSONL 全是 1 且没有 index.json，应全被过滤
    assert export_samples(corpus_dir=tmp_path / "corpus",
                          source_roots=[tmp_path / "src"], min_hits=2).samples == []
    # limit=2
    assert len(export_samples(corpus_dir=tmp_path / "corpus",
                              source_roots=[tmp_path / "src"], limit=2).samples) == 2


def test_export_without_checker_still_exports_and_says_so(tmp_path):
    src = "设 甲 为 [1,2]\n打印(甲[9])\n"
    _write_corpus(tmp_path / "corpus", [_record("L3001", src)])
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.light").write_text(src, encoding="utf-8")

    result = export_samples(corpus_dir=tmp_path / "corpus", source_roots=[tmp_path / "src"])
    assert result.stats.exported_failed == 1
    # 必须明确告知没有 PASS 来源
    assert any("没有 V:PASS 来源" in p for p in result.problems)


# ---- 映射表 -----------------------------------------------------------
def test_l3001_is_not_mapped_to_undefined_var():
    """回归：L3001 是索引越界，曾误映射到 ERR=UNDEFINED_VAR（语义错）。"""
    assert DIAGNOSTIC_TO_VALIDATION["L3001"] == "ERR=RUNTIME"
    assert DIAGNOSTIC_TO_VALIDATION["L3001"] != "ERR=UNDEFINED_VAR"


def test_l5001_is_not_checker_fail():
    """回归：L5001 是后端能力边界，不是验证器自身故障。"""
    assert DIAGNOSTIC_TO_VALIDATION["L5001"] == "ERR=RUNTIME"


def test_every_mapping_target_is_a_registered_validation_code():
    from lightprism.tokens import VALIDATION_CODES

    missing = {v for v in DIAGNOSTIC_TO_VALIDATION.values() if v not in VALIDATION_CODES}
    assert missing == set()


def test_every_fix_target_is_a_real_fix_option():
    from lightprism.tokens import TokenSpace

    options = set(TokenSpace().point("FIX").options)
    bad = {v for v in DIAGNOSTIC_TO_FIX.values() if v not in options}
    assert bad == set()


def test_unknown_code_has_no_fix_decision_rather_than_a_guess():
    assert fault_to_fix_decision("L9999") is None


def test_usage_error_codes_is_documented_set():
    assert USAGE_ERROR_CODES == {"L0000"}


# ---- LightChecker 的参数构造（不真跑）-------------------------------
def test_light_checker_rejects_non_integer_timeout(tmp_path):
    """回归：argparse 侧 --timeout 是 int，传 30.0 会让子进程 rc=2 且 stdout 为空。"""
    with pytest.raises(ValueError, match="整数"):
        LightChecker(tmp_path, run_timeout=30.5)


def test_light_checker_unavailable_is_an_error_not_a_pass(tmp_path):
    checker = LightChecker(tmp_path)               # 目录里没有 cli/light.py
    assert checker.available() is False
    out = checker.check_source("设 甲 为 1\n")
    assert out["ok"] is False
    assert out["error"]                        # 必须可区分于「跑出错误」
    assert out["validation"] == "ERR=CHECKER_FAIL"
    assert out["mode"] == "syntax+run"


# ---- 真实归档（仅本机有数据时跑）------------------------------------
_REAL_CORPUS = Path(os.environ.get(
    "LIGHT_ERROR_CORPUS_DIR", str(Path.home() / ".light" / "error_corpus")
))


@pytest.mark.skipif(not _REAL_CORPUS.is_dir(), reason="本机没有 light 报错归档")
def test_export_against_real_corpus_never_crashes():
    result = export_samples(corpus_dir=_REAL_CORPUS, source_roots=[])
    stats = result.stats.as_dict()
    # 没有 source_roots 时不可能导出样本，但必须把所有记录都归类计数
    assert result.samples == []
    assert stats["total_records"] > 0
    assert (stats["skipped_no_source"] + stats["skipped_usage_error"]
            + stats["skipped_unknown_code"] + stats["skipped_checker_failed"]
            ) == stats["total_records"]
