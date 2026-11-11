# -*- coding: utf-8 -*-
"""LightPrism 词表预算核算与 Light 64K 词表适配层（纯标准库）。

本模块只做**预算与校验**，不构建真词表：真词表由训练侧（tokenizers /
sentencepiece）产出，这里负责回答三个问题 ——

1. 这套预算加起来是不是 64K？
2. 决策 / 验证 / 控制 token 有没有超占比（硬约束 ≤ 2%）？
3. L0 核心字有没有被 BPE 拆碎（"一字一 token"不可违反）？

对 Light 仓的依赖是**可选的**：能 import 到 ``src.keywords`` 就用真实关键字表，
import 不到就用 tokens.py 里的冻结表，并在报告里标注来源。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .tokens import (
    CONTROL_TOKEN_BUDGET,
    CONTROL_TOKEN_MAX_RATIO,
    LIGHT_L0_CORE,
    ParseError,
    TokenSpace,
)

__all__ = [
    "VOCAB_TOTAL",
    "DEFAULT_BUDGET",
    "BudgetSlot",
    "VocabBudget",
    "load_light_l0_core",
    "check_l0_atomicity",
]

#: Light 词表总预算。
VOCAB_TOTAL: int = 65536


@dataclass(frozen=True)
class BudgetSlot:
    """预算表里的一行。

    ``low``/``high`` 为闭区间；``exact`` 表示必须精确等于；
    ``recommended`` 是本版**建议落点**，八个槽位之和恰好等于 :data:`VOCAB_TOTAL`。
    """

    name: str
    low: int
    high: int
    note: str = ""
    exact: bool = False
    recommended: int = 0

    def contains(self, n: int) -> bool:
        if self.exact:
            return n == self.low
        return self.low <= n <= self.high


#: 方案文档 §3 的预算表（按 64K 总预算）。顺序即词表分段顺序：
#: 冻结区 → 语法区 → 库名区 → 中文 BPE → 代码余量 → 控制区 → 保留区。
DEFAULT_BUDGET: Tuple[BudgetSlot, ...] = (
    BudgetSlot("l0_core", 44, 44, "L0 核心字，一字一 token，永不被 BPE 拆分",
               exact=True, recommended=44),
    BudgetSlot("l1_l2_syntax", 4096, 8192, "L1/L2 高频语法组合（遍历/设/段/若…则…）",
               recommended=8192),
    BudgetSlot("stdlib_names", 6144, 10240, "标准库模块名/常用函数名/类型名",
               recommended=10240),
    BudgetSlot("chinese_bpe", 30720, 36864, "中文自然语言 BPE（指令/注释/错误说明）；"
               "取整余量落在这里", recommended=34804),
    BudgetSlot("code_reserve", 8192, 12288, "数字/运算符/标点/标识符片段",
               recommended=10240),
    BudgetSlot("decision", *CONTROL_TOKEN_BUDGET["decision"],
               "决策 token（硬上限 1%—2%）", recommended=320),
    BudgetSlot("validation", *CONTROL_TOKEN_BUDGET["validation"],
               "验证反馈 token", recommended=160),
    BudgetSlot("reserved", *CONTROL_TOKEN_BUDGET["reserved"],
               "后续决策点/实验 token/控制符", recommended=1536),
)

def _suggested(slot: BudgetSlot) -> int:
    """槽位建议落点。**不做运行时取中**：建议值是显式写死的，保证可复现。"""
    return slot.recommended or (slot.low if slot.exact else (slot.low + slot.high) // 2)


@dataclass
class VocabBudget:
    """64K 词表预算模型。

    ``alloc`` 用槽位名 → 条数覆盖默认建议值；未覆盖的槽位用建议落点。
    这样既能"按表核对"，也能"按真实统计核对"（训练侧统计出来后填进来）。
    """

    slots: Sequence[BudgetSlot] = DEFAULT_BUDGET
    alloc: Dict[str, int] = field(default_factory=dict)
    total: int = VOCAB_TOTAL

    def __post_init__(self) -> None:
        names = {s.name for s in self.slots}
        unknown = set(self.alloc) - names
        if unknown:
            raise ParseError(f"预算表里没有这些槽位: {sorted(unknown)}")

    def value_of(self, name: str) -> int:
        for slot in self.slots:
            if slot.name == name:
                return self.alloc.get(name, _suggested(slot))
        raise ParseError(f"未登记的预算槽位 {name!r}")

    def allocated(self) -> Dict[str, int]:
        return {slot.name: self.value_of(slot.name) for slot in self.slots}

    def total_allocated(self) -> int:
        return sum(self.allocated().values())

    def remaining(self) -> int:
        return self.total - self.total_allocated()

    def problems(self, space: Optional[TokenSpace] = None) -> List[str]:
        """预算合法性检查，返回问题列表（空 = 通过）。

        检查三件事：
        1. 每个槽位在区间内、冻结槽位精确匹配；
        2. 总额恰好等于 64K（不允许多也不允许少，少出来的额度必须显式进 reserved）；
        3. 决策 token 实际条目数与预算相容、且占比 ≤ 2%。
        """
        issues: List[str] = []
        for slot in self.slots:
            n = self.value_of(slot.name)
            if not slot.contains(n):
                rng = f"{slot.low}" if slot.exact else f"{slot.low}—{slot.high}"
                issues.append(f"槽位 {slot.name}={n} 超出预算 {rng}")
        total = self.total_allocated()
        if total != self.total:
            issues.append(f"预算合计 {total} ≠ 词表总量 {self.total}（差 {total - self.total:+d}）")

        if space is not None:
            n_decision = len(space.decision_tokens())
            n_validation = len(space.validation_tokens())
            if n_decision > self.value_of("decision"):
                issues.append(
                    f"已定义决策 token {n_decision} 个 > 预算 {self.value_of('decision')}"
                )
            if n_validation > self.value_of("validation"):
                issues.append(
                    f"已定义验证 token {n_validation} 个 > 预算 {self.value_of('validation')}"
                )
            ratio = n_decision / self.total
            if ratio > CONTROL_TOKEN_MAX_RATIO:
                issues.append(f"决策 token 占比 {ratio:.4%} > 硬上限 {CONTROL_TOKEN_MAX_RATIO:.0%}")
        return issues


def load_light_l0_core(light_root: Optional[str] = None) -> Tuple[Tuple[str, ...], str]:
    """尝试从 Light 仓读取 L0 核心字表；读不到就返回本项目冻结表。

    返回 ``(字表, 来源说明)``。**来源必须显式**，因为"用哪份表"是口径问题，
    不能让读者以为两种情况下拿到的是同一份权威数据。
    """
    try:
        if light_root:
            import sys

            sys.path.insert(0, light_root)
        keywords = importlib.import_module("src.keywords")
    except Exception as exc:  # noqa: BLE001 - 任何失败都退回冻结表，但记录原因
        return LIGHT_L0_CORE, f"内置冻结表（未能 import light 仓的 src.keywords: {exc.__class__.__name__}）"

    # light 仓的 keywords.py 明确声明「不再有 KEYWORDS_L0_CORE」，
    # 所以这里不假装能从代码里取到 L0 表；取到的是真关键字集合，仅作交叉校验。
    all_kw = getattr(keywords, "ALL_KEYWORDS", None)
    if all_kw is None:
        return LIGHT_L0_CORE, "内置冻结表（src.keywords 无 ALL_KEYWORDS）"
    missing = [ch for ch in LIGHT_L0_CORE if ch not in all_kw]
    note = f"内置冻结表（已与 src.keywords.ALL_KEYWORDS 交叉校验，缺失 {len(missing)} 字）"
    return LIGHT_L0_CORE, note


def check_l0_atomicity(tokens: Iterable[str], core: Sequence[str] = LIGHT_L0_CORE) -> List[str]:
    """检查 tokenizer 产物里 L0 核心字是否"一字一 token"。

    ``tokens`` 是某个 tokenizer 对样例代码切出来的 token 列表。任何 L0 字出现在
    **长度 > 1 的 token** 内部，都说明它被 BPE 拆碎或与邻字粘在一起 —— 这是
    方案文档 §3 的硬约束违反，返回问题列表。
    """
    core_set = set(core)
    issues: List[str] = []
    for tok in tokens:
        if len(tok) <= 1:
            continue
        for ch in core_set:
            if ch in tok:
                issues.append(f"L0 字 {ch!r} 被并入多字 token {tok!r}")
    return issues
