# -*- coding: utf-8 -*-
"""LightPrism 训练数据：schema、校验、批次构造（纯标准库）。

数据形态（对齐方案文档 §4.3 的监督信号来源）：

一份样本 = 一条完整序列，含:

* ``task``        自然语言任务（必填）
* ``steps``       有序步骤列表，每步是 ``{"decision": {点: 选项}, "code": "...", "validation": "PASS"}``
* ``source``      这条样本从哪导出的（``light_ai_check`` / ``harness_tool_log`` / ``human_edit``）
* ``meta``        自由字段（文件、任务 id 等）

导出规则（方案 §4.3）：

- find-replace 完成且随后 ````light ai check```` 通过 → ``EDIT=FR`` + ``V:PASS``
- 返回 ``MISSING_TYPE`` 且下次编辑加了类型标注并通过 → ``V:ERR=MISSING_TYPE`` + ``FIX=ADD_TYPE`` + ``V:PASS``
- 某次命令被人工确认拦截 → ``SAFE=L2``

**v0.1 只做「读—校验—拼 batch」**，不做从日志自动导出的解析器（那需要真实日志样本，
见 CHANGELOG「未包含」）。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .sequence import Code, ParsedSequence, Task, parse_sequence
from .tokens import (
    CONTROL_TOKEN_MAX_RATIO,
    DEFAULT_DECISION_POINTS,
    DecisionToken,
    ParseError,
    TokenSpace,
    ValidationToken,
    map_diagnostics,
)

__all__ = [
    "SOURCES",
    "Step",
    "Sample",
    "load_jsonl",
    "write_jsonl",
    "build_sequence",
    "BatchBuilder",
    "DatasetStats",
]

#: 允许的监督信号来源（方案 §4.3 的四个来源）。
SOURCES: Tuple[str, ...] = (
    "light_ai_check",
    "harness_tool_log",
    "human_edit",
    "task_tree",
)


@dataclass
class Step:
    """一步 = 决策集合 + 代码 + 验证结果。"""

    decisions: Dict[str, str] = field(default_factory=dict)
    code: str = ""
    validation: str = "PASS"
    validation_detail: str = ""

    def to_item_list(self, space: TokenSpace) -> List[object]:
        """按「决策 → 代码 → 验证」顺序展开成序列项。

        决策点顺序**按 :data:`DEFAULT_DECISION_POINTS` 的登记顺序**，而不是 dict 的
        插入顺序 —— 否则同一条逻辑序列在两次导出里会得到不同 id 顺序，训练目标抖动。
        """
        items: List[object] = []
        for point in space.decision_points:
            if point.name in self.decisions:
                items.append(space.decision(point.name, self.decisions[point.name]))
        if self.code:
            items.append(Code(self.code))
        if self.validation:
            items.append(ValidationToken(self.validation, self.validation_detail))
        return items


@dataclass
class Sample:
    """一条训练样本。"""

    task: str
    steps: List[Step] = field(default_factory=list)
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self, space: Optional[TokenSpace] = None) -> List[str]:
        """校验并返回问题列表。**不做任何静默修正。**"""
        space = space or TokenSpace()
        problems: List[str] = []
        if not self.task.strip():
            problems.append("task 为空")
        if self.source and self.source not in SOURCES:
            problems.append(f"source {self.source!r} 不在允许集合 {SOURCES}")
        if not self.steps:
            problems.append("steps 为空")
        for i, step in enumerate(self.steps):
            for name, option in step.decisions.items():
                try:
                    space.decision(name, option)
                except ParseError as exc:
                    problems.append(f"steps[{i}]: {exc}")
            if not step.code.strip():
                problems.append(f"steps[{i}]: code 为空（验证只能针对代码）")
            try:
                ValidationToken(step.validation)
            except ParseError as exc:
                problems.append(f"steps[{i}]: {exc}")
        return problems

    def to_sequence(self, space: Optional[TokenSpace] = None) -> ParsedSequence:
        space = space or TokenSpace()
        problems = self.validate(space)
        if problems:
            raise ParseError("样本不合法: " + "；".join(problems))
        items: List[object] = [Task(self.task)]
        for step in self.steps:
            items.extend(step.to_item_list(space))
        from .sequence import render_sequence
        return parse_sequence(render_sequence(items), space=space)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "source": self.source,
            "steps": [
                {
                    "decisions": dict(s.decisions),
                    "code": s.code,
                    "validation": s.validation,
                    **({"validation_detail": s.validation_detail} if s.validation_detail else {}),
                }
                for s in self.steps
            ],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sample":
        steps = []
        for raw in data.get("steps") or []:
            steps.append(
                Step(
                    decisions=dict(raw.get("decisions") or {}),
                    code=str(raw.get("code") or ""),
                    validation=str(raw.get("validation") or "PASS"),
                    validation_detail=str(raw.get("validation_detail") or ""),
                )
            )
        return cls(
            task=str(data.get("task") or ""),
            steps=steps,
            source=str(data.get("source") or ""),
            meta=dict(data.get("meta") or {}),
        )


# ---- JSONL 往返 --------------------------------------------------------
def load_jsonl(path: str | Path, space: Optional[TokenSpace] = None) -> Tuple[List[Sample], List[str]]:
    """读 JSONL，返回 ``(合法样本, 问题列表)``。

    坏的行走问题列表而**不抛异常**：真实日志里脏数据是常态，训练脚本需要
    「跳过并计数」而不是整批失败。但跳过多少必须报出来，不能静默吞。
    """
    space = space or TokenSpace()
    good: List[Sample] = []
    problems: List[str] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = Sample.from_dict(json.loads(line))
            except json.JSONDecodeError as exc:
                problems.append(f"{path}:{lineno}: JSON 解析失败 {exc}")
                continue
            issues = sample.validate(space)
            if issues:
                problems.append(f"{path}:{lineno}: " + "；".join(issues))
                continue
            good.append(sample)
    return good, problems


def write_jsonl(samples: Iterable[Sample], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


# ---- 批次构造 ----------------------------------------------------------
def build_sequence(sample: Sample, tokenizer, space: Optional[TokenSpace] = None) -> List[int]:
    """样本 → id 序列（含 ``<BOS>`` 语义：直接以 ``<TASK>`` 开头，不加额外 BOS）。"""
    return tokenizer.encode_sequence(sample.to_sequence(space))


@dataclass
class BatchBuilder:
    """定长打包：把多条样本拼成 ``[B, T]`` 的 id / target / 权重三件套。

    * ``targets`` 相对 ``input_ids`` 错位一位（位置 i 预测 i+1）；
    * ``loss_weights`` 按 token 类型加权：控制 token（决策/验证/结构）权重可调，
      默认 1.0——方案 §6 阶段一明确「不做特殊加权」，加权是阶段二的旋钮；
    * ``pad_id`` 的权重恒为 0，不参与损失。
    """

    tokenizer: object
    seq_len: int = 1024
    pad_id: int = 0
    control_weight: float = 1.0
    text_weight: float = 1.0
    space: TokenSpace = field(default_factory=TokenSpace)

    def _weights(self, ids: Sequence[int]) -> List[float]:
        kind_of = getattr(self.tokenizer, "kind_of")
        out = []
        for i in ids:
            out.append(self.text_weight if kind_of(i) == "text" else self.control_weight)
        return out

    def make_batch(self, samples: Sequence[Sample]) -> Dict[str, Any]:
        """返回 {input_ids, targets, loss_weights, lengths}，全部为 ``list[list[int]]``。

        不做 tensor 转换：本模块要保持纯标准库，转 tensor 是训练脚本的事。
        """
        if not samples:
            raise ValueError("空批次")
        rows_ids: List[List[int]] = []
        rows_tgt: List[List[int]] = []
        rows_w: List[List[float]] = []
        lengths: List[int] = []

        for sample in samples:
            ids = build_sequence(sample, self.tokenizer, self.space)
            ids = ids[: self.seq_len + 1]
            lengths.append(len(ids))
            weights = self._weights(ids)
            # 错位：输入去掉最后一个，目标去掉第一个
            inp, tgt = ids[:-1], ids[1:]
            w = weights[1:]
            pad = self.seq_len - len(inp)
            if pad > 0:
                inp = inp + [self.pad_id] * pad
                tgt = tgt + [self.pad_id] * pad
                w = w + [0.0] * pad
            rows_ids.append(inp)
            rows_tgt.append(tgt)
            rows_w.append(w)
        return {"input_ids": rows_ids, "targets": rows_tgt,
                "loss_weights": rows_w, "lengths": lengths}

    def pack_stream(self, samples: Sequence[Sample]) -> Iterator[List[int]]:
        """把多条样本首尾相接后切成定长块（阶段一的「随机交错」用法）。

        末块不足 ``seq_len`` 时**丢弃**：残块会让目标分布偏向长样本结尾。
        """
        buf: List[int] = []
        for sample in samples:
            buf.extend(build_sequence(sample, self.tokenizer, self.space))
        for start in range(0, len(buf) - self.seq_len + 1, self.seq_len):
            yield buf[start : start + self.seq_len]


@dataclass
class DatasetStats:
    """数据集体检结果，写进训练日志。"""

    samples: int = 0
    tokens: int = 0
    control_tokens: int = 0
    decision_tokens: int = 0
    validation_tokens: int = 0

    @property
    def decision_ratio(self) -> float:
        return self.decision_tokens / self.tokens if self.tokens else 0.0

    @property
    def control_ratio(self) -> float:
        return self.control_tokens / self.tokens if self.tokens else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "tokens": self.tokens,
            "control_tokens": self.control_tokens,
            "decision_tokens": self.decision_tokens,
            "validation_tokens": self.validation_tokens,
            "decision_ratio": round(self.decision_ratio, 5),
            "control_ratio": round(self.control_ratio, 5),
            # 方案 §8 第 5 条：决策 token 占比必须 ≤ 2%
            "decision_ratio_ok": self.decision_ratio <= CONTROL_TOKEN_MAX_RATIO,
        }


def compute_stats(samples: Sequence[Sample], tokenizer, space: Optional[TokenSpace] = None) -> DatasetStats:
    """统计真实 id 口径的 token 密度（而不是按序列段计数）。"""
    stats = DatasetStats()
    kind_of = getattr(tokenizer, "kind_of")
    for sample in samples:
        ids = build_sequence(sample, tokenizer, space)
        stats.samples += 1
        stats.tokens += len(ids)
        for i in ids:
            kind = kind_of(i)
            if kind == "decision":
                stats.decision_tokens += 1
                stats.control_tokens += 1
            elif kind == "validation":
                stats.validation_tokens += 1
                stats.control_tokens += 1
            elif kind in ("structure", "control"):
                stats.control_tokens += 1
    return stats
