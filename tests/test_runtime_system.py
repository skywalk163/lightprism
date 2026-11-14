# -*- coding: utf-8 -*-
"""阶段 0 零模型系统验收测试（T-0346）。

用可注入的 FakeRunner 测**纪律**：验证码只来自真实 checker payload、
L0000 不冒充具体类型、空输出/截断/checker PASS 都不算通过、重试来自实际动作、
每条日志可回溯到一次 checker 调用、分母不剔除失败。

真实 ``light`` CLI 的端到端由 ``scripts/runtime_eval.py`` 执行；需要单测里也
跑真 CLI 时设 ``LIGHT_RUNTIME_INTEGRATION=1``（默认跳过，避免拖慢验收）。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from lightprism.runtime.checker import CheckResult, RunResult, find_light_root  # noqa: E402
from lightprism.runtime.decisions import (  # noqa: E402
    OBSERVABLE_VALIDATION_CODES, DecisionWriter, map_check_payload,
    validate_record, validation_token,
)
from lightprism.runtime.engine import (  # noqa: E402
    NO_TEMPLATE_ID, RuntimeEngine, TaskSpec, load_tasks, normalize_stdout,
)
from lightprism.runtime.retrieval import BM25Index, extract_slots  # noqa: E402
from lightprism.runtime.templates import (  # noqa: E402
    TEMPLATES, get_template, render,
)
from lightprism.tokens import DEFAULT_DECISION_POINTS, TokenSpace  # noqa: E402
import runtime_eval  # noqa: E402

TASKS_PATH = REPO / "data" / "runtime" / "tasks.jsonl"
DECISIONS_PATH = REPO / "data" / "runtime" / "decisions.jsonl"


def _diag(code: str, message: str = "x") -> Dict[str, Any]:
    return {"code": code, "severity": "error", "stage": "运行", "message": message}


def _payload(diags: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"schema_version": 1, "ok": not diags, "diagnostics": diags,
            "file": "x.light", "backend_hints": [], "duration_ms": 1}


class FakeRunner:
    """不碰真实 light CLI；check/run 的返回由测试显式给出。"""

    def __init__(
        self,
        check_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        run_fn: Optional[Callable[[str], Tuple[str, str, int]]] = None,
    ) -> None:
        self.cwd = "fake://light"
        self.light_root = None
        self.check_fn = check_fn or (lambda src: _payload([]))
        self.run_fn = run_fn or (lambda src: ("7\n", "", 0))
        self.check_calls: List[str] = []
        self.run_calls: List[str] = []

    def available(self) -> bool:
        return True

    def check(self, file_path: str) -> CheckResult:
        src = Path(file_path).read_text(encoding="utf-8")
        self.check_calls.append(src)
        payload = self.check_fn(src)
        return CheckResult(
            payload=payload, stdout=json.dumps(payload, ensure_ascii=False),
            rc=0 if payload.get("ok") else 1,
            cmd=["python", "-X", "utf8", "-m", "cli.light", "ai", "check",
                 "--json", "--no-corpus", "--run", "--timeout", "60", file_path],
            cwd=self.cwd, elapsed_ms=1,
        )

    def run(self, file_path: str) -> RunResult:
        src = Path(file_path).read_text(encoding="utf-8")
        self.run_calls.append(src)
        out, err, rc = self.run_fn(src)
        return RunResult(
            stdout=out, stderr=err, rc=rc,
            cmd=["python", "-X", "utf8", "-m", "cli.light", "run", file_path],
            cwd=self.cwd, elapsed_ms=1,
        )


def _engine(runner: FakeRunner, tmp_path: Path, **kw: Any) -> Tuple[RuntimeEngine, DecisionWriter]:
    writer = DecisionWriter(tmp_path / "decisions.jsonl")
    return RuntimeEngine(runner, writer, **kw), writer


def _task(text: str = "计算 3 加 4 的结果，输出数字。", expected: str = "7\n",
          task_id: str = "X01", cluster: str = "arith") -> TaskSpec:
    return TaskSpec(task_id=task_id, cluster=cluster, text=text, expected=expected)


# ---------------------------------------------------------------------------
# 1. 验证码映射：只认真实 checker 返回，L0000 不冒充具体类型
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload,err,want", [
    (_payload([]), "", "PASS"),
    (_payload([_diag("L3001", "列表索引越界")]), "", "ERR=RUNTIME"),
    (_payload([_diag("L0000", "意外的输入结束")]), "", "ERR=CHECKER_FAIL"),
    (_payload([_diag("L1001", "语法错误")]), "", "ERR=CHECKER_FAIL"),
    (_payload([_diag("L2002", "缺类型标注")]), "", "ERR=CHECKER_FAIL"),
    (_payload([_diag("L9999", "未登记码")]), "", "ERR=CHECKER_FAIL"),
    (None, "子进程超时", "ERR=CHECKER_FAIL"),
])
def test_validation_mapping(payload: Any, err: str, want: str) -> None:
    out = map_check_payload(payload, err)
    assert out.code == want
    assert out.code in OBSERVABLE_VALIDATION_CODES
    assert validation_token(out.code).startswith("<V:")


def test_l0000_never_masks_specific_errors() -> None:
    """卡上硬禁：不拿 L0000 冒充 SYNTAX / TYPE_MISMATCH 等具体码。"""
    out = map_check_payload(_payload([_diag("L0000")]))
    assert out.code == "ERR=CHECKER_FAIL"
    assert out.error_codes == ("L0000",)
    assert out.diagnostics[0]["code"] == "L0000"  # 原始码完整保留
    assert not any(x in out.code for x in ("SYNTAX", "TYPE", "MISSING", "UNDEFINED"))


def test_checker_ok_flag_is_not_the_verdict() -> None:
    """空文件也判 ok:true（Demo验收报告 §3.4），payload.ok 不参与映射。"""
    payload = {"ok": True, "diagnostics": []}
    assert map_check_payload(payload).code == "PASS"
    # 反过来：ok=true 但带 error 诊断，仍按诊断判
    payload2 = {"ok": True, "diagnostics": [_diag("L3001")]}
    assert map_check_payload(payload2).code == "ERR=RUNTIME"


# ---------------------------------------------------------------------------
# 2. 决策 token 规范：封闭 / 互斥 / 可解析
# ---------------------------------------------------------------------------
def test_decision_tokens_are_closed_and_parseable() -> None:
    space = TokenSpace()
    decisions = {"PLAN": "DIRECT", "EDIT": "FR", "SAFE": "L0", "CTX": "KEEP",
                 "RETRY": "YES", "FIX": "REFACTOR", "NEXT": "EDIT"}
    from lightprism.runtime.decisions import decision_tokens
    toks = decision_tokens(decisions)
    assert len(toks) == len(decisions)
    for tok in toks:
        parsed = space.parse_decision(tok)
        assert decisions[parsed.point] == parsed.option


def test_illegal_decision_option_rejected() -> None:
    from lightprism.runtime.decisions import decision_tokens
    from lightprism.tokens import ParseError
    with pytest.raises(ParseError):
        decision_tokens({"EDIT": "MADE_UP"})


# ---------------------------------------------------------------------------
# 3. 判据：空输出 / 输出不符 / 截断 / checker 失败都不算通过
# ---------------------------------------------------------------------------
def test_empty_stdout_is_failure_even_when_checker_passes(tmp_path: Path) -> None:
    runner = FakeRunner(run_fn=lambda src: ("", "", 0))
    engine, writer = _engine(runner, tmp_path, max_retries=0)
    out = engine.run_task(_task())
    assert out.final_ok is False
    assert "空输出" in out.failure_reason
    assert out.attempts[0].validation_code == "PASS"
    assert writer.records()[0]["final_ok"] is False


def test_checker_pass_but_wrong_stdout_is_failure(tmp_path: Path) -> None:
    runner = FakeRunner(run_fn=lambda src: ("8\n", "", 0))
    engine, _ = _engine(runner, tmp_path, max_retries=0)
    out = engine.run_task(_task(expected="7\n"))
    assert out.final_ok is False
    assert "不一致" in out.failure_reason


def test_truncated_stdout_is_failure(tmp_path: Path) -> None:
    runner = FakeRunner(run_fn=lambda src: ("a\n", "", 0))
    engine, _ = _engine(runner, tmp_path, max_retries=0)
    out = engine.run_task(_task(expected="a\nb\n"))
    assert out.final_ok is False
    assert "截断" in out.failure_reason


def test_checker_failure_is_not_success(tmp_path: Path) -> None:
    runner = FakeRunner(
        check_fn=lambda src: _payload([_diag("L0000")]),
        run_fn=lambda src: ("7\n", "", 0),
    )
    engine, _ = _engine(runner, tmp_path, max_retries=0)
    out = engine.run_task(_task())
    assert out.final_ok is False
    assert out.attempts[0].validation_code == "ERR=CHECKER_FAIL"


def test_normalize_stdout_keeps_content() -> None:
    assert normalize_stdout("7\r\n\r\n") == "7"
    assert normalize_stdout("总数=3\n总和=12\n") == "总数=3\n总和=12"
    assert normalize_stdout("") == ""


# ---------------------------------------------------------------------------
# 4. 重试：来自实际动作，日志逐 attempt 落盘
# ---------------------------------------------------------------------------
def test_runtime_oob_triggers_real_find_replace_retry(tmp_path: Path) -> None:
    text = "取列表 [10, 20, 30] 的第 5 个元素；若索引越界则输出最后一个元素 30。"
    runner = FakeRunner(
        check_fn=lambda src: _payload(
            [_diag("L3001", "列表索引越界")] if "甲[5]" in src else []),
        run_fn=lambda src: ("", "", 1) if "甲[5]" in src else ("30\n", "", 0),
    )
    engine, writer = _engine(runner, tmp_path, max_retries=2)
    out = engine.run_task(_task(text=text, expected="30\n", task_id="E01", cluster="edge"))

    assert out.final_ok is True
    assert out.retries == 1
    assert len(runner.check_calls) == 2 and len(runner.run_calls) == 2
    first, second = out.attempts
    assert first.validation_code == "ERR=RUNTIME"
    assert first.edit_mode == "RW"
    assert first.record.decisions["RETRY"] == "YES"
    assert second.validation_code == "PASS"
    assert second.edit_mode == "FR"
    assert second.record.decisions["FIX"] == "REFACTOR"
    assert second.record.decisions["RETRY"] == "NO"
    assert "甲[列表长度(甲) 减 1]" in second.source
    assert "find_replace" in second.record.retrieval["edit_action"]

    records = writer.records()
    assert len(records) == 2
    assert all(not validate_record(r) for r in records)
    assert all(r["checker_cmd"] and r["source_sha256"] for r in records)
    assert records[0]["checker_error_codes"] == ["L3001"]


def test_no_template_is_real_failure_and_logged_as_checker_diag(tmp_path: Path) -> None:
    text = "把这段音频转写成乐谱并渲染五线谱，输出乐谱摘要。"
    runner = FakeRunner(
        check_fn=lambda src: _payload([_diag("L0000")]) if "__" in src else _payload([]),
        run_fn=lambda src: ("", "", 1),
    )
    engine, writer = _engine(runner, tmp_path, max_retries=1)
    out = engine.run_task(_task(text=text, expected="乐谱摘要: C大调 4/4\n",
                                task_id="E02", cluster="edge"))
    assert out.final_ok is False
    assert out.retrieval["matched"] is False
    assert all(a.template_id == NO_TEMPLATE_ID for a in out.attempts)
    assert any(r["validation_code"] == "ERR=CHECKER_FAIL" for r in writer.records())


def test_each_log_record_is_traceable_to_a_real_checker_call(tmp_path: Path) -> None:
    runner = FakeRunner()
    engine, writer = _engine(runner, tmp_path, max_retries=0)
    engine.run_task(_task())
    rec = writer.records()[0]
    assert validate_record(rec) == []
    assert rec["checker_origin"].startswith("light ai check")
    assert "ai" in rec["checker_cmd"] and "check" in rec["checker_cmd"]
    assert "--json" in rec["checker_cmd"] and "--run" in rec["checker_cmd"]
    assert "--no-corpus" in rec["checker_cmd"]  # 不污染 light 报错归档
    assert isinstance(rec["checker_payload"], dict)
    assert rec["validation_token_text"] == "<V:PASS>"
    assert rec["decision_tokens"] and rec["decisions"]["EDIT"] == "RW"
    for key in ("task_id", "template_id", "decision_point", "decision_option",
                "validation_code", "retry", "checker_diagnostics", "final_ok", "elapsed_ms"):
        assert key in rec


# ---------------------------------------------------------------------------
# 5. 检索与模板：确定性、零模型、缺槽不编数值
# ---------------------------------------------------------------------------
def test_retrieval_is_deterministic_and_all_templates_reachable() -> None:
    idx = BM25Index()
    for d in runtime_eval.TASK_DEFS:
        row = next(r for r in runtime_eval.make_task_rows() if r["task_id"] == d["id"])
        hits = idx.search(row["text"], top_k=3)
        again = idx.search(row["text"], top_k=3)
        assert [h.template_id for h in hits] == [h.template_id for h in again]
        if d["tpl"] is not None:
            assert hits[0].template_id == d["tpl"], d["id"]
        else:
            assert not hits[0].trigger_hits, d["id"]  # 检索未命中是真实状态


def test_missing_slots_keep_placeholder_instead_of_inventing_values() -> None:
    t = get_template("arith_binop")
    slots, missing = extract_slots("计算两数之和，但没有给出数字", t)
    assert "A" in missing and "B" in missing
    assert "__A__" in render(t, slots)
    assert "__B__" in render(t, slots)


def test_rendered_sources_have_no_leftover_placeholders_for_task_set() -> None:
    idx = BM25Index()
    for row in runtime_eval.make_task_rows():
        hit = idx.search(row["text"], 1)[0]
        if not hit.trigger_hits:
            continue
        t = get_template(hit.template_id)
        slots, _ = extract_slots(row["text"], t)
        assert "__" not in render(t, slots), row["task_id"]


def test_templates_are_standard_library_only() -> None:
    assert len(TEMPLATES) >= 15
    assert all(t.snippet and t.keywords and t.triggers for t in TEMPLATES)


# ---------------------------------------------------------------------------
# 6. 任务集与统计口径（全分母，不剔除失败）
# ---------------------------------------------------------------------------
def test_task_set_schema_and_no_cheating_fields() -> None:
    tasks = load_tasks(TASKS_PATH)
    assert 20 <= len(tasks) <= 30
    clusters = Counter(t.cluster for t in tasks)
    assert len(clusters) >= 6
    for t in tasks:
        assert t.text.strip() and t.expected.strip()
        assert t.expected.endswith("\n") or "\n" in t.expected
    raw = [json.loads(x) for x in TASKS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    for row in raw:
        assert not ({"template_id", "slots", "expected_template", "source"} & set(row)), \
            "任务集不能携带模板/槽位/源码，否则系统会偷看答案"


def test_expected_is_independent_python_semantics() -> None:
    rows = {r["task_id"]: r for r in runtime_eval.make_task_rows()}
    assert rows["A01"]["expected"] == "7\n"
    assert rows["A04"]["expected"] == "5.0\n"      # light 的「除」返回 5.0
    assert rows["F02"]["expected"] == "120\n"
    assert rows["F03"]["expected"] == "15\n"       # 「从 1 到 5」取上界 5
    assert rows["C02"]["expected"] == "总数=3\n总和=12\n"
    assert rows["E01"]["expected"] == "30\n"


def test_summary_uses_full_denominator(tmp_path: Path) -> None:
    tasks = load_tasks(TASKS_PATH)
    runner = FakeRunner(run_fn=lambda src: ("与预期不符\n", "", 0))
    engine = RuntimeEngine(runner, max_retries=0)
    outcomes = [engine.run_task(t) for t in tasks]
    summary = runtime_eval.build_summary(outcomes, [], runner, 0.1)
    assert summary["total_tasks"] == len(tasks)
    assert summary["ok_tasks"] == 0
    assert summary["closed_loop_rate"] == 0.0
    assert summary["failed_tasks"] == len(tasks)
    assert len(summary["failures"]) == len(tasks)  # 失败样例逐条全贴
    assert sum(c["total"] for c in summary["by_cluster"].values()) == len(tasks)


def test_decision_log_from_real_run_if_present() -> None:
    if not DECISIONS_PATH.is_file():
        pytest.skip("真实决策日志尚未生成（先跑 runtime_eval.py）")
    rows = [json.loads(x) for x in DECISIONS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert rows, "决策日志不能为空"
    for rec in rows:
        assert validate_record(rec) == [], rec.get("task_id")
    codes = Counter(r["validation_code"] for r in rows)
    assert set(codes) <= set(OBSERVABLE_VALIDATION_CODES)


# ---------------------------------------------------------------------------
# 7. 真实 light CLI（显式开启；默认跳过以免拖慢 pytest）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("LIGHT_RUNTIME_INTEGRATION") != "1",
                    reason="设 LIGHT_RUNTIME_INTEGRATION=1 才跑真 CLI")
def test_real_light_cli_end_to_end(tmp_path: Path) -> None:
    from lightprism.runtime.checker import LightRunner
    assert find_light_root() is not None, "找不到 light 仓（可设 LIGHT_ROOT）"
    runner = LightRunner()
    engine, writer = _engine(runner, tmp_path, max_retries=0)
    out = engine.run_task(_task())
    assert out.final_ok is True, out.failure_reason
    rec = writer.records()[0]
    assert rec["validation_code"] == "PASS"
    assert rec["checker_payload"].get("ok") is True
    assert rec["stdout"] == "7\n"
    assert "ai" in rec["checker_cmd"]
