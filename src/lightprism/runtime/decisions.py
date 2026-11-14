# -*- coding: utf-8 -*-
"""决策 / 验证日志：schema、确定性映射、线程安全写入。

纪律（T-0346 任务卡 §2，对应方案一 §4.3 / §5）
------------------------------------------------
* ``<V:...>`` **只**来自 ``light ai check --json`` 的真实返回，不由模型自判、
  不由 ``light run`` 的退出码反推、不拿 checker ``exit 0`` 当通过。
* 只保留**可观测**的三类：``PASS`` / ``ERR=RUNTIME`` / ``ERR=CHECKER_FAIL``。
  ``reports/ERR样本覆盖.md`` 实测 8 类里 7 类不可达，绝大多数运行期错误回落
  ``L0000``。这里把 ``L0000`` 如实映射为 ``ERR=CHECKER_FAIL``（检查器未细分），
  **绝不**冒充 ``ERR=SYNTAX`` / ``ERR=TYPE_MISMATCH`` 等具体类型；原始诊断码
  完整保留在 ``checker_diagnostics`` 里，供后续细分后回溯。
* ``<D:...>`` 只从**实际发生的动作**映射。每个选项都过
  :class:`lightprism.tokens.TokenSpace` 校验（封闭性 / 互斥性），不编造。
* 每条记录都带一次真实 checker 调用的完整证据（命令、cwd、rc、原始 JSON
  payload、源码 hash），因此每条 ``validation_code`` 都可回溯到那一次调用。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..tokens import ParseError, TokenSpace, ValidationToken

__all__ = [
    "OBSERVABLE_VALIDATION_CODES",
    "RUNTIME_DIAGNOSTIC_CODES",
    "REQUIRED_FIELDS",
    "ValidationOutcome",
    "map_check_payload",
    "DecisionRecord",
    "DecisionWriter",
    "decision_tokens",
    "validation_token",
    "validate_record",
    "source_sha256",
]

#: 本卡允许出现的验证码。方案一 §5 的其它码在本机 light 版本下不可观测，
#: 一律不造（见 reports/ERR样本覆盖.md）。
OBSERVABLE_VALIDATION_CODES: Tuple[str, ...] = (
    "PASS",
    "ERR=RUNTIME",
    "ERR=CHECKER_FAIL",
)

#: 实测能稳定映射到「运行时错误」的诊断码。L3001=列表索引越界，
#: L5001=原生后端不支持（tokens.py 也把它归入 ERR=RUNTIME）。
RUNTIME_DIAGNOSTIC_CODES = frozenset({"L3001", "L5001"})

#: 任务卡要求的字段，一个都不能少。
REQUIRED_FIELDS: Tuple[str, ...] = (
    "task_id",
    "template_id",
    "decision_point",
    "decision_option",
    "validation_code",
    "retry",
    "checker_diagnostics",
    "final_ok",
    "elapsed_ms",
)

_TOKEN_SPACE = TokenSpace()


@dataclass(frozen=True)
class ValidationOutcome:
    """一次真实 ``light ai check`` 调用的确定性映射结果。"""

    code: str
    error_codes: Tuple[str, ...]
    diagnostics: List[Dict[str, Any]]
    checker_error: str
    origin: str = "light ai check --json --no-corpus --run"


def map_check_payload(
    payload: Optional[Mapping[str, Any]],
    checker_error: str = "",
    *,
    ok_flag: Optional[bool] = None,
) -> ValidationOutcome:
    """把 ``light ai check --json`` 的原始 payload 映射成唯一一个验证码。

    规则（确定性、无模型判断）：

    1. checker 没跑成（超时 / 仓缺失 / stdout 非 JSON）→ ``ERR=CHECKER_FAIL``；
    2. 没有 ``severity == "error"`` 的诊断 → ``PASS``；
    3. 第一条 error 诊断的码是 ``L3001``/``L5001`` → ``ERR=RUNTIME``；
    4. 其它任何码（含 ``L0000``、未登记码、空码）→ ``ERR=CHECKER_FAIL``，
       原始码留在 ``error_codes`` / ``diagnostics``，**不冒充具体类型**。

    注意：``payload["ok"]`` 只作为旁证记录，不参与判定 —— 空文件也判 ``ok=true``
    （``docs/Demo验收报告.md`` §3.4），拿它当通过会产生假绿。
    """
    if checker_error or payload is None:
        return ValidationOutcome(
            code="ERR=CHECKER_FAIL",
            error_codes=(),
            diagnostics=[],
            checker_error=checker_error or "没有拿到 checker payload",
        )

    diags = list(payload.get("diagnostics") or [])
    errors = [d for d in diags if str(d.get("severity", "")).lower() == "error"]
    codes = tuple(str(d.get("code") or "") for d in errors)
    if not errors:
        code = "PASS"
    elif codes[0] in RUNTIME_DIAGNOSTIC_CODES:
        code = "ERR=RUNTIME"
    else:
        code = "ERR=CHECKER_FAIL"
    return ValidationOutcome(
        code=code,
        error_codes=codes,
        diagnostics=diags,
        checker_error="",
    )


def validation_token(code: str) -> str:
    """验证码 → ``<V:...>``；不在可观测集合里直接报错，不兜底。"""
    if code not in OBSERVABLE_VALIDATION_CODES:
        raise ValueError(f"不可观测/未登记的验证码 {code!r}")
    return ValidationToken(code).text


def decision_tokens(decisions: Mapping[str, str]) -> List[str]:
    """决策字典 → 合法 ``<D:...>`` 列表；非法选项抛错而不是悄悄丢弃。"""
    out: List[str] = []
    for point, option in decisions.items():
        out.append(_TOKEN_SPACE.decision(point, option).text)
    return out


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest()


@dataclass
class DecisionRecord:
    """一次 attempt（生成 → check → 真跑比对）落一条。

    ``decision_point`` / ``decision_option`` 是该 attempt 的**主决策**（编辑策略），
    完整的决策集合在 ``decisions`` / ``decision_tokens``，报告按它统计分布。
    """

    # ---- 任务卡要求的字段 -------------------------------------------
    task_id: str
    template_id: str
    decision_point: str
    decision_option: str
    validation_code: str
    retry: int
    checker_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    final_ok: bool = False
    elapsed_ms: int = 0

    # ---- 决策 / 验证 token ------------------------------------------
    decisions: Dict[str, str] = field(default_factory=dict)
    decision_tokens: List[str] = field(default_factory=list)
    validation_token_text: str = ""

    # ---- 真实 checker 调用证据（可回溯） -----------------------------
    run_id: str = ""
    attempt: int = 0
    checker_origin: str = "light ai check --json --no-corpus --run"
    checker_cmd: List[str] = field(default_factory=list)
    checker_cwd: str = ""
    checker_rc: Optional[int] = None
    checker_payload: Dict[str, Any] = field(default_factory=dict)
    checker_stdout: str = ""
    checker_error: str = ""
    checker_error_codes: List[str] = field(default_factory=list)
    checker_elapsed_ms: int = 0

    # ---- 真跑输出比对证据 -------------------------------------------
    run_cmd: List[str] = field(default_factory=list)
    run_rc: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    expected_stdout: str = ""
    run_elapsed_ms: int = 0

    # ---- 上下文 ------------------------------------------------------
    cluster: str = ""
    task_text: str = ""
    source: str = ""
    source_sha256: str = ""
    retrieval: Dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_record(record: Mapping[str, Any]) -> List[str]:
    """校验一条日志：字段齐、验证码可观测、决策合法、checker 证据可回溯。"""
    issues: List[str] = []
    for key in REQUIRED_FIELDS:
        if key not in record:
            issues.append(f"缺字段 {key}")
    code = str(record.get("validation_code", ""))
    if code not in OBSERVABLE_VALIDATION_CODES:
        issues.append(f"验证码不在可观测集合: {code!r}")
    if not str(record.get("checker_origin", "")).startswith("light ai check"):
        issues.append("checker_origin 不是 light ai check")
    if not record.get("checker_cmd"):
        issues.append("没有 checker 命令，validation 无法回溯")
    if not record.get("source_sha256"):
        issues.append("没有源码 hash，validation 无法回溯")
    if not record.get("checker_error") and not isinstance(record.get("checker_payload"), dict):
        issues.append("checker_payload 缺失")
    decisions = record.get("decisions") or {}
    for point, option in decisions.items():
        try:
            _TOKEN_SPACE.decision(point, option)
        except ParseError as exc:
            issues.append(f"非法决策 {point}={option}: {exc}")
    return issues


class DecisionWriter:
    """线程安全地把每条 attempt 追加成一行 JSON。

    默认覆盖写（一次评估产出一份完整日志）；``append=True`` 用于跨次累积。
    """

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.count = 0
        mode = "a" if append else "w"
        with self.path.open(mode, encoding="utf-8"):
            pass

    def write(self, record: DecisionRecord | Mapping[str, Any]) -> None:
        data = record.to_dict() if isinstance(record, DecisionRecord) else dict(record)
        line = json.dumps(data, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self.count += 1

    def records(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.path.is_file():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
