# -*- coding: utf-8 -*-
"""LightPrism 统一序列：决策 → 代码 → 验证 的交错结构（纯标准库）。

序列形态（方案文档 §2.1）::

    <TASK>实现一个按文件大小排序并输出前10名的CLI工具</TASK>
    <D:PLAN=DECOMPOSE>
    <D:EDIT=NEW_FILE>
    <CODE>
    设 文件列表 为 列目录(".")。
    </CODE>
    <V:PASS>
    <D:NEXT=TEST>
    ...

本模块只做**结构与合法性**：解析、校验、统计。它不生成 token，也不调用模型；
生成侧由训练脚本负责，执行侧由 lightharness 负责（见 README「与 lightharness 的边界」）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple, Union

from .tokens import (
    CONTROL_TOKEN_MAX_RATIO,
    DecisionToken,
    ParseError,
    TokenSpace,
    ValidationToken,
)

__all__ = [
    "Task",
    "Code",
    "Segment",
    "ParsedSequence",
    "parse_sequence",
    "render_sequence",
    "decision_density",
]

#: 段类型的关键字。"NEW_FILE" 不在首批 DEFAULT_DECISION_POINTS 里 —— 方案文档
#: §2.1 的示例用了它，但 §4.2 的首批决策点表没有；本骨架按示例解析、按首批表
#: 校验，因此 NEW_FILE 会在 `validate()` 时被明确报错（**不静默吞掉**）。
_SEGMENT_RE = re.compile(
    r"<TASK>(?P<task>.*?)</TASK>"
    r"|<D:(?P<dpoint>[A-Z_]+)=(?P<doption>[A-Z0-9_]+)>"
    r"|<CODE>(?P<code>.*?)</CODE>"
    r"|<V:(?P<vcode>[A-Z]+(?:=[A-Z_]+)?)>",
    re.DOTALL,
)


@dataclass(frozen=True)
class Task:
    """``<TASK>...</TASK>``：自然语言任务描述（中文 BPE 侧）。"""

    text: str


@dataclass(frozen=True)
class Code:
    """``<CODE>...</CODE>``：光明代码片段（代码 token 侧）。"""

    text: str


@dataclass(frozen=True)
class Segment:
    """决策 / 验证 token 在序列里的位置包装，用于统计密度与做对齐检查。"""

    token: Union[DecisionToken, ValidationToken]
    start: int
    end: int


@dataclass
class ParsedSequence:
    """一次解析的结果：有序 token 流 + 原始任务 + 统计信息。"""

    items: List[Union[Task, Code, DecisionToken, ValidationToken]]
    raw: str
    space: TokenSpace = field(default_factory=TokenSpace)

    # ---- 视图 -------------------------------------------------------
    @property
    def decisions(self) -> List[DecisionToken]:
        return [i for i in self.items if isinstance(i, DecisionToken)]

    @property
    def validations(self) -> List[ValidationToken]:
        return [i for i in self.items if isinstance(i, ValidationToken)]

    @property
    def codes(self) -> List[Code]:
        return [i for i in self.items if isinstance(i, Code)]

    @property
    def task(self) -> str:
        for item in self.items:
            if isinstance(item, Task):
                return item.text
        return ""

    # ---- 校验 -------------------------------------------------------
    def validate(
        self,
        *,
        require_task: bool = True,
        forbid_consecutive_decisions: bool = False,
    ) -> List[str]:
        """返回**问题列表**（空列表 = 合法）。不复用异常，便于一次报全。

        硬性检查（默认开启）：
        1. 决策 token 必须在词表内（点与选项都登记过）；
        2. 每个 ``<V:...>`` 之前必须是一个 ``<CODE>`` 段 —— 验证只能针对代码，
           不能凭空出现（方案文档 §2.1、§5）。

        软性检查（``forbid_consecutive_decisions=True`` 才开）：
        3. 连续两个决策 token 之间没有代码。注意方案文档 §2.1 的示例本身就写了
           ``<D:PLAN=DECOMPOSE>`` 紧跟 ``<D:EDIT=NEW_FILE>`` —— "先规划再定编辑策略"
           是合法的复合决策，所以这条**默认不判**，只在阶段三做 RL 惩罚时按需打开。
        """
        problems: List[str] = []
        if require_task and not self.task:
            problems.append("缺少 <TASK>...</TASK> 任务段")

        for item in self.items:
            if isinstance(item, DecisionToken):
                try:
                    self.space.decision(item.point, item.option)
                except ParseError as exc:
                    problems.append(str(exc))

        prev: Union[Task, Code, DecisionToken, ValidationToken, None] = None
        for item in self.items:
            if isinstance(item, ValidationToken) and not isinstance(prev, Code):
                problems.append(f"验证 token {item.text} 前没有 <CODE> 段")
            if (
                forbid_consecutive_decisions
                and isinstance(item, DecisionToken)
                and isinstance(prev, DecisionToken)
            ):
                problems.append(f"连续决策 token（{prev.text} → {item.text}）之间没有代码")
            prev = item
        return problems

    # ---- 统计 -------------------------------------------------------
    def counts(self) -> dict:
        return {
            "task": 1 if self.task else 0,
            "decision": len(self.decisions),
            "code": len(self.codes),
            "validation": len(self.validations),
        }

    def render(self) -> str:
        return render_sequence(self.items)


def parse_sequence(text: str, space: TokenSpace | None = None) -> ParsedSequence:
    """把一段 LightPrism 序列解析成有序 token 流。

    严格模式：任何**形似**控制 token 但不符合语法的片段（例如 ``<D:EDIT=>``、
    或裸 ``<V:PAS>``）都会抛 :class:`ParseError`，而不是被当成代码文本吞掉。
    """
    space = space or TokenSpace()
    items: List[Union[Task, Code, DecisionToken, ValidationToken]] = []
    last = 0
    for m in _SEGMENT_RE.finditer(text):
        gap = text[last:m.start()]
        if gap.strip():
            raise ParseError(f"控制 token 之间出现裸文本（位置 {last}-{m.start()}）: {gap.strip()[:60]!r}")
        last = m.end()
        if m.group("task") is not None:
            items.append(Task(m.group("task").strip()))
        elif m.group("dpoint") is not None:
            # 走 TokenSpace 校验，保证解析出的 token 一定是词表内成员
            items.append(space.decision(m.group("dpoint"), m.group("doption")))
        elif m.group("code") is not None:
            items.append(Code(m.group("code").strip("\n")))
        else:
            items.append(ValidationToken(m.group("vcode")))
    tail = text[last:]
    if tail.strip():
        raise ParseError(f"序列尾部有裸露文本: {tail.strip()[:60]!r}")
    return ParsedSequence(items=items, raw=text, space=space)


def render_sequence(items: Iterable[Union[Task, Code, DecisionToken, ValidationToken]]) -> str:
    """把 token 流渲染回规范文本（往返一致，供 diff 与回归测试）。"""
    out: List[str] = []
    for item in items:
        if isinstance(item, Task):
            out.append(f"<TASK>{item.text}</TASK>")
        elif isinstance(item, Code):
            out.append(f"<CODE>\n{item.text}\n</CODE>")
        elif isinstance(item, (DecisionToken, ValidationToken)):
            out.append(item.text)
        else:  # pragma: no cover - 类型系统兜底
            raise TypeError(f"不认识的序列项: {type(item)!r}")
    return "\n".join(out) + "\n"


def decision_density(items: Sequence[object]) -> float:
    """决策 token 占全部序列项的比例，用于对照 :data:`CONTROL_TOKEN_MAX_RATIO`。

    注意这里是**按段计数**的粗粒度指标；训练侧的细粒度密度按子词 token 统计，
    两者的关系在探针实验报告里对齐（见 docs/探针实验设计.md）。
    """
    total = len(items)
    if not total:
        return 0.0
    n_decision = sum(1 for i in items if isinstance(i, DecisionToken))
    return n_decision / total


# 便于 `from lightprism.sequence import DECISION_DENSITY_OK`
def DECISION_DENSITY_OK(ratio: float) -> bool:  # noqa: N802 - 与常量同名便于阅读
    return ratio <= CONTROL_TOKEN_MAX_RATIO
