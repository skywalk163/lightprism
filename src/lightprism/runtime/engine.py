# -*- coding: utf-8 -*-
"""阶段 0 零模型确定性系统：运行编排 + 决策/验证日志自动导出。

流程（任务卡 §1 / §2 / §3）::

    任务文本 → 意图归一 / BM25+规则检索 → 选模板 / 填槽 → 确定性展开 .light
            → light ai check --json --no-corpus --run  （<V> 唯一来源）
            → light run 真跑并比对预期输出          （闭环判据）
            → 不过则带诊断反馈修复重试 1–2 次

判据**不是** ``checker exit 0``、不是 loss、不是 JSON 精确匹配：
``docs/Demo验收报告.md`` §3.4 实测空文件也判 ``ok:true``，所以必须
「check 通过 **且** 真跑输出与预期一致」，空输出 / 截断一律按失败计入全分母。

所有 ``<D:...>`` 都从本模块**实际执行的动作**映射，不编造；所有 ``<V:...>``
都来自 :func:`map_check_payload` 对真实 checker payload 的确定性映射。

T-0370 起新增三类**真实动作**（标签跟着动作走，不是先写标签再编动作）：

* ``PLAN=DECOMPOSE``：任务文本含显式连接词（「，再」「，然后」…）且两段子文本
  各自命中带 trigger 的模板时，真的拆两段跑，子任务二的价格槽取子任务一的
  **真跑 stdout**（见 :mod:`lightprism.runtime.composite`）；
* ``CTX=DROP``：检索 top1 命中 trigger 但缺必需槽位时，**丢弃**该检索结果，
  改用下一个槽位齐备的候选（丢掉的模板不会产出任何 attempt 日志）；
* ``EDIT=PATCH``：诊断里带行号时做**行级补丁**（只改那一行，其余行保留）；
  诊断不带行号才退回全局子串 find-replace（``EDIT=FR``）。
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
from .composite import COMPOSITE_FAMILY, STAGE_INJECT, StagePlan, split_stages
from .decisions import (
    DecisionRecord,
    DecisionWriter,
    decision_tokens,
    map_check_payload,
    source_sha256,
    validation_token,
)
from .retrieval import BM25Index, DEFAULT_MIN_SCORE, SearchHit, extract_slots
from .splits import family_of, split_of
from .templates import Template, expected_of, get_template, render

__all__ = [
    "TaskSpec", "AttemptOutcome", "TaskOutcome", "RuntimeEngine",
    "load_tasks", "normalize_stdout", "NO_TEMPLATE_ID", "RETRIEVAL_METHOD",
]

NO_TEMPLATE_ID = "NO_TEMPLATE"
RETRIEVAL_METHOD = "零模型：中文 bigram BM25 + trigger 关键词规则 + 正则槽位抽取"

#: 检索未命中时交给 checker 的占位骨架。故意保留 ``__槽__``：让 light 如实报
#: L0000（→ ERR=CHECKER_FAIL），而不是造一份「看起来能过」的空程序。
NO_TEMPLATE_SOURCE = "打印(__未匹配模板__)。\n"

_INDEX_OOB_RE = re.compile(r"(甲|乙|数据|读数|用量|表|数)[\[［](\d+)[\]］]")


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
    stage: int = 1


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
    plan: str = "DIRECT"

    @property
    def retries(self) -> int:
        """真实发生过的重试次数（同一段内 attempt>0 的条数）。

        复合任务的第二段 attempt 从 0 开始，不算重试。
        """
        return sum(1 for a in self.attempts if a.attempt > 0)


def _clamp_text(text: str) -> Tuple[str, str]:
    """诊断驱动的索引边界改写：``甲[5]`` → ``甲[列表长度(甲) 减 1]``。

    这是**真实发生的编辑动作**，改写前后的片段会写进日志的 ``edit_action``，
    对应的 ``<D:EDIT=PATCH/FR>`` / ``<D:FIX=REFACTOR>`` 由此而来，不是标注出来的。
    """
    m = _INDEX_OOB_RE.search(text)
    if not m:
        return text, ""
    name = m.group(1)
    replacement = f"{name}[列表长度({name}) 减 1]"
    new = text[:m.start()] + replacement + text[m.end():]
    return new, f"{m.group(0)} -> {replacement}（依据 L3001 索引越界诊断）"


def _index_clamp(source: str) -> Tuple[str, str]:
    """整文件子串 find-replace（诊断不带行号时用）。"""
    new, note = _clamp_text(source)
    if not note:
        return source, ""
    return new, f"find_replace: {note}"


def _patch_line(source: str, line_no: int) -> Tuple[str, str]:
    """行级补丁：只替换诊断指出的那一行，其余行原样保留（EDIT=PATCH）。"""
    if line_no <= 0:
        return source, ""
    lines = source.splitlines(keepends=True)
    if line_no > len(lines):
        return source, ""
    new_line, note = _clamp_text(lines[line_no - 1])
    if not note:
        return source, ""
    lines[line_no - 1] = new_line
    return "".join(lines), f"line_patch: 第{line_no}行 {note}"


def _first_error_line(diagnostics: Sequence[Dict[str, Any]]) -> int:
    """取第一条 error 级诊断的行号（没有行号返回 0，绝不猜）。"""
    for d in diagnostics or []:
        if str(d.get("severity", "")).lower() != "error":
            continue
        try:
            line = int(d.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        if line > 0:
            return line
    return 0


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

    # ---- 任务级编排 ---------------------------------------------------
    def run_task(self, task: TaskSpec) -> TaskOutcome:
        started = time.monotonic()
        outcome = TaskOutcome(
            task_id=task.task_id, cluster=task.cluster, text=task.text,
            expected=task.expected, final_ok=False,
        )
        stage_plan = self._decompose_plan(task.text)
        with tempfile.TemporaryDirectory(prefix="lightprism-runtime-") as tmp:
            file_path = str(Path(tmp) / "program.light")
            if stage_plan is None:
                outcome.plan = "DIRECT"
                ok = self._run_segment(
                    task, task.text, task.expected, "DIRECT", None, outcome, file_path,
                )
                outcome.final_ok = ok
            else:
                outcome.plan = "DECOMPOSE"
                outcome.final_ok = self._run_composite(task, stage_plan, outcome, file_path)
        outcome.elapsed_ms = int(round((time.monotonic() - started) * 1000))
        return outcome

    def _decompose_plan(self, text: str) -> Optional[StagePlan]:
        """只有「文本可切两段 + 两段各自命中带 trigger 的不同模板」才拆。

        判据全部来自文本与检索结果，**不看** cluster 标签；达不到就退回 DIRECT。
        """
        plan = split_stages(text)
        if plan is None:
            return None
        hit_a = self.index.search(plan.text_a, top_k=1)
        hit_b = self.index.search(plan.text_b, top_k=1)
        if not hit_a or not hit_b:
            return None
        if not (hit_a[0].trigger_hits and hit_b[0].trigger_hits):
            return None
        if hit_a[0].template_id == hit_b[0].template_id:
            return None
        return plan

    def _run_composite(
        self, task: TaskSpec, plan: StagePlan, outcome: TaskOutcome, file_path: str,
    ) -> bool:
        """真两段：子任务一跑完 → stdout 注入子任务二 → 子任务二跑完。"""
        expected_a = self._expected_for_stage(plan.text_a)
        ok_a = self._run_segment(
            task, plan.text_a, expected_a, "DECOMPOSE", (1, 2, plan), outcome, file_path,
        )
        if not ok_a:
            outcome.failure_reason = "子任务一未过：未产生可用中间值，未进入子任务二"
            return False

        last_stdout = outcome.attempts[-1].stdout
        inject: Dict[str, object] = {}
        stage_b_tpl = self._top_template(plan.text_b)
        if stage_b_tpl is not None and stage_b_tpl.id in STAGE_INJECT:
            slot, mode = STAGE_INJECT[stage_b_tpl.id]
            value = self._stdout_value(last_stdout, mode)
            if value is None:
                outcome.failure_reason = (
                    f"子任务二需要子任务一的数值输出，实得 {last_stdout!r}，无法注入槽 {slot}"
                )
                return False
            inject[slot] = value
        ok_b = self._run_segment(
            task, plan.text_b, task.expected, "DECOMPOSE", (2, 2, plan), outcome, file_path,
            inject=inject,
        )
        if not ok_b:
            outcome.failure_reason = outcome.failure_reason or "子任务二未过"
        return ok_b

    @staticmethod
    def _stdout_value(stdout: str, mode: str) -> Optional[int]:
        text = normalize_stdout(stdout).strip()
        if not text:
            return None
        if mode == "stdout_int":
            try:
                return int(text.split("\n")[-1].strip())
            except ValueError:
                return None
        return None

    def _top_template(self, text: str) -> Optional[Template]:
        hits = self.index.search(text, top_k=1)
        if not hits or not hits[0].trigger_hits:
            return None
        return get_template(hits[0].template_id)

    def _expected_for_stage(self, text: str) -> str:
        """子任务一的预期值：模板的**独立 Python 语义**（不读 .light 代码）。"""
        tpl = self._top_template(text)
        if tpl is None or tpl.expected_fn is None:
            return ""
        slots, missing = extract_slots(text, tpl)
        if missing:
            return ""
        try:
            return expected_of(tpl, slots)
        except (KeyError, ValueError, ZeroDivisionError, IndexError):
            return ""

    # ---- 单段执行（直展任务 / 复合任务的每一段） -----------------------
    def _run_segment(
        self,
        task: TaskSpec,
        text: str,
        expected: str,
        plan: str,
        stage_meta: Optional[Tuple[int, int, StagePlan]],
        outcome: TaskOutcome,
        file_path: str,
        *,
        inject: Optional[Dict[str, object]] = None,
    ) -> bool:
        inject = dict(inject or {})
        hits = self.index.search(text, top_k=3)
        matched = bool(hits) and hits[0].score >= self.min_score and bool(hits[0].trigger_hits)
        candidates, dropped = self._select_candidates(text, hits, matched, inject)
        stage_index, stage_count, stage_plan = (stage_meta if stage_meta else (0, 0, None))
        retrieval = {
            "method": RETRIEVAL_METHOD,
            "top": [h.as_dict() for h in hits],
            "matched": matched,
            "candidates": candidates,
            "dropped_templates": self._dropped(hits, candidates),
            "min_score": self.min_score,
            "plan": plan,
            "stage": stage_index,
            "stage_count": stage_count,
            "stage_task": text if stage_meta else "",
            "stage_split": stage_plan.as_dict() if stage_plan else {},
            "injected_slots": {k: v for k, v in inject.items()},
        }
        if plan == "DIRECT" or stage_index == 1:
            outcome.retrieval = retrieval

        prev: Optional[AttemptOutcome] = None
        for attempt in range(self.max_retries + 1):
            attempt_started = time.monotonic()
            template_id, source, edit_mode, fix_mode, edit_action = self._build(
                text, candidates, attempt, prev, inject,
            )
            if template_id != NO_TEMPLATE_ID:
                retrieval_stage_family = family_of(template_id).name
            else:
                retrieval_stage_family = family_of(NO_TEMPLATE_ID).name

            Path(file_path).write_text(source, encoding="utf-8")
            check = self.runner.check(file_path)
            mapped = map_check_payload(check.payload if not check.error else None, check.error)
            run = self.runner.run(file_path)

            ok, reason = self._judge(expected, mapped.code, check.error, run)
            will_retry = (not ok) and attempt < self.max_retries
            decisions = self._decisions(attempt, edit_mode, fix_mode, ok, will_retry, plan, dropped)

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
                expected_stdout=expected,
                run_elapsed_ms=run.elapsed_ms,
                cluster=task.cluster,
                task_text=task.text,
                source=source,
                source_sha256=source_sha256(source),
                retrieval={
                    **retrieval,
                    "edit_action": edit_action,
                    "edit_mode": edit_mode,
                    "family": retrieval_stage_family,
                    "family_split": split_of(template_id),
                },
                failure_reason=reason,
            )
            if self.writer is not None:
                self.writer.write(record)

            ao = AttemptOutcome(
                attempt=attempt, template_id=template_id, edit_mode=edit_mode,
                source=source, validation_code=mapped.code, stdout=run.stdout,
                final_ok=ok, failure_reason=reason, record=record,
                checker_error_codes=list(mapped.error_codes), will_retry=will_retry,
                stage=stage_index,
            )
            outcome.attempts.append(ao)
            if ok:
                outcome.failure_reason = ""
                return True
            outcome.failure_reason = reason
            prev = ao
        return False

    # ---- 检索候选筛选（真实丢弃槽位不足的命中） ------------------------
    def _select_candidates(
        self,
        text: str,
        hits: Sequence[SearchHit],
        matched: bool,
        inject: Dict[str, object],
    ) -> Tuple[List[str], bool]:
        """按「槽位是否齐备」筛候选；top1 被丢且后面有可用的 ⇒ ``CTX=DROP``。

        全部候选都缺槽时**保留 top1**（不丢），让 checker 如实报错——这是
        「缺槽就编数值」的反面。
        """
        if not matched:
            return [], False
        usable: List[str] = []
        top1 = hits[0].template_id
        for h in hits:
            if h.score < self.min_score or not h.trigger_hits:
                continue
            tpl = get_template(h.template_id)
            slots, missing = extract_slots(text, tpl)
            missing = [m for m in missing if m not in inject]
            if missing:
                continue
            usable.append(h.template_id)
        if not usable:
            return [top1], False
        return usable, top1 not in usable

    def _dropped(self, hits: Sequence[SearchHit], candidates: Sequence[str]) -> List[str]:
        if not candidates:
            return []
        return [h.template_id for h in hits
                if h.trigger_hits and h.score >= self.min_score and h.template_id not in candidates]

    # ---- 生成 ---------------------------------------------------------
    def _build(
        self,
        text: str,
        candidates: Sequence[str],
        attempt: int,
        prev: Optional[AttemptOutcome],
        inject: Dict[str, object],
    ) -> Tuple[str, str, str, str, str]:
        """返回 ``(template_id, source, edit_mode, fix_mode, edit_action)``。"""
        if not candidates:
            return NO_TEMPLATE_ID, NO_TEMPLATE_SOURCE, "RW", "", "检索未命中，交占位骨架给 checker 如实报错"

        if attempt == 0:
            template = get_template(candidates[0])
            slots, missing = extract_slots(text, template)
            slots.update({k: v for k, v in inject.items() if k in template.slot_plan})
            missing = [m for m in missing if m not in inject]
            source = render(template, slots)
            action = "expand_snippet: %s（槽=%s，缺槽=%s）" % (template.id, sorted(slots), missing)
            return template.id, source, "RW", "", action

        # 重试：诊断驱动的真实修复（只对真实出现过的诊断码做动作）
        if prev is not None and "L3001" in prev.checker_error_codes:
            line = _first_error_line(prev.record.checker_diagnostics)
            if line:
                patched_source, patch_action = _patch_line(prev.source, line)
                if patch_action:
                    return prev.template_id, patched_source, "PATCH", "REFACTOR", patch_action
            fr_source, fr_action = _index_clamp(prev.source)
            if fr_action:
                return prev.template_id, fr_source, "FR", "REFACTOR", fr_action

        # 其它重试：换下一个检索候选，整文件重写
        idx = min(attempt, len(candidates) - 1)
        template = get_template(candidates[idx])
        slots, missing = extract_slots(text, template)
        slots.update({k: v for k, v in inject.items() if k in template.slot_plan})
        source = render(template, slots)
        action = "switch_candidate: %s（槽=%s，缺槽=%s）" % (template.id, sorted(slots), missing)
        return template.id, source, "RW", "REFACTOR", action

    # ---- 判据 ---------------------------------------------------------
    @staticmethod
    def _judge(expected: str, code: str, check_error: str, run: Any) -> Tuple[bool, str]:
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
        got, want = normalize_stdout(run.stdout), normalize_stdout(expected)
        if not want:
            return False, f"没有可比对的预期输出（子任务预期值算不出来）：实得 {got!r}"
        if len(got.split("\n")) < len(want.split("\n")):
            return False, f"输出截断：预期 {len(want.split(chr(10)))} 行，实得 {len(got.split(chr(10)))} 行"
        if got != want:
            return False, f"真跑输出与预期不一致: 实得 {got!r}，预期 {want!r}"
        return True, ""

    # ---- 决策映射（全部来自实际动作） ---------------------------------
    def _decisions(
        self,
        attempt: int,
        edit_mode: str,
        fix_mode: str,
        ok: bool,
        will_retry: bool,
        plan: str,
        dropped: bool,
    ) -> Dict[str, str]:
        if dropped and attempt == 0:
            ctx = "DROP"
        elif attempt == 0 or edit_mode in ("FR", "PATCH"):
            ctx = "KEEP"
        else:
            ctx = "FETCH"
        decisions: Dict[str, str] = {
            "PLAN": plan,
            "EDIT": edit_mode,
            # 全部动作是本地临时文件 + 只读模板，自动执行、无人工确认 → L0
            "SAFE": "L0",
            "CTX": ctx,
            "RETRY": "YES" if will_retry else "NO",
            "NEXT": "TEST" if ok else ("EDIT" if will_retry else "ASK"),
        }
        if attempt > 0 and fix_mode:
            decisions["FIX"] = fix_mode
        return decisions
