# -*- coding: utf-8 -*-
"""阶段 0 零模型确定性系统：运行编排 + 决策/验证日志自动导出。

流程（任务卡 §1 / §2 / §3）::

    任务文本 → 意图归一 / BM25+规则检索 → 选模板 / 填槽 → 确定性展开 .light
            → light ai check --json --no-corpus --run  （<V> 唯一来源）
            → light run 真跑并比对预期输出          （闭环判据）
            → 不过则换候选 / 带诊断反馈 find-replace 重试 1–2 次

判据**不是** ``checker exit 0``、不是 loss、不是 JSON 精确匹配：
``docs/Demo验收报告.md`` §3.4 实测空文件也判 ``ok:true``，所以必须
「check 通过 **且** 真跑输出与预期一致」，空输出 / 截断一律按失败计入全分母。

所有 ``<D:...>`` 都从本模块**实际执行的动作**映射，不编造；所有 ``<V:...>``
都来自 :func:`map_check_payload` 对真实 checker payload 的确定性映射。
"""
from __future__ import annotations

import json
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .checker import Runner
from .decisions import (
    DecisionRecord,
    DecisionWriter,
    decision_tokens,
    map_check_payload,
    source_sha256,
    validation_token,
)
from .retrieval import BM25Index, DEFAULT_MIN_SCORE, SearchHit, extract_slots
from .templates import Template, get_template, render

__all__ = [
    "TaskSpec", "AttemptOutcome", "TaskOutcome", "RuntimeEngine",
    "load_tasks", "normalize_stdout", "NO_TEMPLATE_ID", "RETRIEVAL_METHOD",
]

NO_TEMPLATE_ID = "NO_TEMPLATE"
RETRIEVAL_METHOD = "零模型：中文 bigram BM25 + trigger 关键词规则 + 正则槽位抽取"

#: 检索未命中时交给 checker 的占位骨架。故意保留 ``__槽__``：让 light 如实报
#: L0000（→ ERR=CHECKER_FAIL），而不是造一份「看起来能过」的空程序。
NO_TEMPLATE_SOURCE = "打印(__未匹配模板__)。\n"

_INDEX_OOB_RE = re.compile(r"(甲|数据|读数|用量)\[(\d+)\]")


def normalize_stdout(text: str) -> str:
    """统一换行、去行尾空白、去尾部空行。**不**做 JSON 结构匹配。"""
    lines = [ln.rstrip() for ln in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    cluster: str
    text: str
    expected: str
    note: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSpec":
        missing = [k for k in ("task_id", "cluster", "text", "expected") if not data.get(k)]
        if missing:
            raise ValueError(f"任务缺字段: {missing} -> {data}")
        return cls(
            task_id=str(data["task_id"]),
            cluster=str(data["cluster"]),
            text=str(data["text"]),
            expected=str(data["expected"]),
            note=str(data.get("note") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "cluster": self.cluster,
            "text": self.text, "expected": self.expected, "note": self.note,
        }


def load_tasks(path: str | Path) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    p = Path(path)
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            tasks.append(TaskSpec.from_dict(json.loads(line)))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{p}:{lineno}: {exc}") from exc
    return tasks


@dataclass
class AttemptOutcome:
    attempt: int
    template_id: str
    edit_mode: str
    source: str
    validation_code: str
    stdout: str
    final_ok: bool
    failure_reason: str
    record: DecisionRecord
    checker_error_codes: List[str] = field(default_factory=list)
    will_retry: bool = False


@dataclass
class TaskOutcome:
    task_id: str
    cluster: str
    text: str
    expected: str
    final_ok: bool
    attempts: List[AttemptOutcome] = field(default_factory=list)
    failure_reason: str = ""
    retrieval: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)


def _index_clamp(source: str) -> Tuple[str, str]:
    """诊断驱动的 find-replace：L3001 索引越界 → 取最后一个元素。

    这是**真实发生的编辑动作**，动作描述会写进日志的 ``edit_action``，
    对应的 ``<D:EDIT=FR>`` / ``<D:FIX=REFACTOR>`` 由此而来，不是标注出来的。
    """
    m = _INDEX_OOB_RE.search(source)
    if not m:
        return source, ""
    name = m.group(1)
    replacement = f"{name}[列表长度({name}) 减 1]"
    new = source[:m.start()] + replacement + source[m.end():]
    return new, f"find_replace: {m.group(0)} -> {replacement}（依据 L3001 索引越界诊断）"


class RuntimeEngine:
    """一次任务 = 1..N 次 attempt；每次 attempt 落一条决策日志。"""

    def __init__(
        self,
        runner: Runner,
        writer: Optional[DecisionWriter] = None,
        *,
        index: Optional[BM25Index] = None,
        max_retries: int = 2,
        min_score: float = DEFAULT_MIN_SCORE,
        run_id: str = "",
    ) -> None:
        self.runner = runner
        self.writer = writer
        self.index = index or BM25Index()
        self.max_retries = int(max_retries)
        self.min_score = float(min_score)
        self.run_id = run_id or uuid.uuid4().hex[:12]

    def run_task(self, task: TaskSpec) -> TaskOutcome:
        started = time.monotonic()
        hits = self.index.search(task.text, top_k=3)
        matched = bool(hits) and hits[0].score >= self.min_score and bool(hits[0].trigger_hits)
        candidate_ids = [h.template_id for h in hits[:3]] if matched else []
        retrieval = {
            "method": RETRIEVAL_METHOD,
            "top": [h.as_dict() for h in hits],
            "matched": matched,
            "candidates": candidate_ids,
            "min_score": self.min_score,
        }

        outcome = TaskOutcome(
            task_id=task.task_id, cluster=task.cluster, text=task.text,
            expected=task.expected, final_ok=False, retrieval=retrieval,
        )

        with tempfile.TemporaryDirectory(prefix="lightprism-runtime-") as tmp:
            file_path = str(Path(tmp) / "program.light")
            prev: Optional[AttemptOutcome] = None

            for attempt in range(self.max_retries + 1):
                attempt_started = time.monotonic()
                template_id, source, edit_mode, fix_mode, edit_action = self._build(
                    task, candidate_ids, attempt, prev, matched,
                )
                Path(file_path).write_text(source, encoding="utf-8")

                check = self.runner.check(file_path)
                mapped = map_check_payload(check.payload if not check.error else None, check.error)
                run = self.runner.run(file_path)

                ok, reason = self._judge(task, mapped.code, check.error, run)
                will_retry = (not ok) and attempt < self.max_retries
                decisions = self._decisions(task, attempt, edit_mode, fix_mode, ok, will_retry)

                record = DecisionRecord(
                    task_id=task.task_id,
                    template_id=template_id,
                    decision_point="EDIT",
                    decision_option=edit_mode,
                    validation_code=mapped.code,
                    retry=attempt,
                    checker_diagnostics=mapped.diagnostics,
                    final_ok=ok,
                    elapsed_ms=int(round((time.monotonic() - attempt_started) * 1000)),
                    decisions=decisions,
                    decision_tokens=decision_tokens(decisions),
                    validation_token_text=validation_token(mapped.code),
                    run_id=self.run_id,
                    attempt=attempt,
                    checker_origin=mapped.origin,
                    checker_cmd=check.cmd,
                    checker_cwd=check.cwd,
                    checker_rc=check.rc,
                    checker_payload=check.payload,
                    checker_stdout=check.stdout[:2000],
                    checker_error=check.error or mapped.checker_error,
                    checker_error_codes=list(mapped.error_codes),
                    checker_elapsed_ms=check.elapsed_ms,
                    run_cmd=run.cmd,
                    run_rc=run.rc,
                    stdout=run.stdout,
                    stderr=run.stderr[-500:],
                    expected_stdout=task.expected,
                    run_elapsed_ms=run.elapsed_ms,
                    cluster=task.cluster,
                    task_text=task.text,
                    source=source,
                    source_sha256=source_sha256(source),
                    retrieval={**retrieval, "edit_action": edit_action, "edit_mode": edit_mode},
                    failure_reason=reason,
                )
                if self.writer is not None:
                    self.writer.write(record)

                ao = AttemptOutcome(
                    attempt=attempt, template_id=template_id, edit_mode=edit_mode,
                    source=source, validation_code=mapped.code, stdout=run.stdout,
                    final_ok=ok, failure_reason=reason, record=record,
                    checker_error_codes=list(mapped.error_codes), will_retry=will_retry,
                )
                outcome.attempts.append(ao)
                if ok:
                    outcome.final_ok = True
                    outcome.failure_reason = ""
                    break
                outcome.failure_reason = reason
                prev = ao

        outcome.elapsed_ms = int(round((time.monotonic() - started) * 1000))
        return outcome

    # ---- 生成 ---------------------------------------------------------
    def _build(
        self,
        task: TaskSpec,
        candidate_ids: Sequence[str],
        attempt: int,
        prev: Optional[AttemptOutcome],
        matched: bool,
    ) -> Tuple[str, str, str, str, str]:
        """返回 ``(template_id, source, edit_mode, fix_mode, edit_action)``。"""
        if not matched:
            return NO_TEMPLATE_ID, NO_TEMPLATE_SOURCE, "RW", "", "检索未命中，交占位骨架给 checker 如实报错"

        if attempt == 0:
            template = get_template(candidate_ids[0])
            slots, missing = extract_slots(task.text, template)
            source = render(template, slots)
            action = "expand_snippet: %s（槽=%s，缺槽=%s）" % (template.id, sorted(slots), missing)
            return template.id, source, "RW", "", action

        # 重试 1：带诊断反馈的 find-replace（只对 L3001 索引越界这一真实诊断）
        if attempt == 1 and prev and "L3001" in prev.checker_error_codes:
            new_source, action = _index_clamp(prev.source)
            if action:
                return prev.template_id, new_source, "FR", "REFACTOR", action

        # 其它重试：换下一个检索候选，整文件重写
        idx = min(attempt, len(candidate_ids) - 1)
        template = get_template(candidate_ids[idx])
        slots, missing = extract_slots(task.text, template)
        source = render(template, slots)
        action = "switch_candidate: %s（槽=%s，缺槽=%s）" % (template.id, sorted(slots), missing)
        return template.id, source, "RW", "REFACTOR", action

    # ---- 判据 ---------------------------------------------------------
    @staticmethod
    def _judge(task: TaskSpec, code: str, check_error: str, run: Any) -> Tuple[bool, str]:
        if check_error:
            return False, f"checker 未跑成: {check_error[:160]}"
        if code != "PASS":
            return False, f"checker 返回 {code}"
        if run.error:
            return False, f"真跑未成功: {run.error[:160]}"
        if run.rc != 0:
            return False, f"真跑 rc={run.rc}: {run.stderr[-160:]}"
        if not (run.stdout or "").strip():
            # docs/Demo验收报告.md §3.4：空文件也判 ok，必须显式挡住假绿
            return False, "空输出（空程序也判 ok 的假绿防护）"
        got, want = normalize_stdout(run.stdout), normalize_stdout(task.expected)
        if len(got.split("\n")) < len(want.split("\n")):
            return False, f"输出截断：预期 {len(want.split(chr(10)))} 行，实得 {len(got.split(chr(10)))} 行"
        if got != want:
            return False, f"真跑输出与预期不一致: 实得 {got!r}，预期 {want!r}"
        return True, ""

    # ---- 决策映射（全部来自实际动作） ---------------------------------
    def _decisions(
        self,
        task: TaskSpec,
        attempt: int,
        edit_mode: str,
        fix_mode: str,
        ok: bool,
        will_retry: bool,
    ) -> Dict[str, str]:
        decisions: Dict[str, str] = {
            # 系统始终是「单模板直接展开」，没有真的做任务分解 → 不标 DECOMPOSE
            "PLAN": "DIRECT",
            "EDIT": edit_mode,
            # 全部动作是本地临时文件 + 只读模板，自动执行、无人工确认 → L0
            "SAFE": "L0",
            "CTX": "KEEP" if (attempt == 0 or edit_mode == "FR") else "FETCH",
            "RETRY": "YES" if will_retry else "NO",
            "NEXT": "TEST" if ok else ("EDIT" if will_retry else "ASK"),
        }
        if attempt > 0 and fix_mode:
            decisions["FIX"] = fix_mode
        return decisions
