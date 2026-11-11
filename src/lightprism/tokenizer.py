# -*- coding: utf-8 -*-
"""LightPrism 词表装配：Light 64K 预算 + 控制 token 预留区。

**分工边界**：本模块不训练 BPE。Light 的中文 BPE / 语法组合 / 标准库名由 Light 仓侧
的语料统计产出（方案 §11「词表规范 v0.1」），这里只做三件事：

1. 按 :data:`lightprism.vocab.DEFAULT_BUDGET` 校验各段条数；
2. 把控制 token（``<D:…>`` / ``<V:…>`` / 结构标记）放进制表末尾的预留区，**一段连续 id**，
   方便训练侧按 id 区间判断 token 类型、做损失加权；
3. 提供 ``encode_sequence`` / ``decode_sequence``：把 :class:`ParsedSequence` 与
   ``input_ids`` 互转，控制 token 永远占 1 个 id（绝不进 BPE）。

不依赖 ``transformers``：需要一个 ``encode(str) -> List[int]`` / ``decode`` 的底座即可，
测试里用 :class:`CharTokenizer` 顶替。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .sequence import Code, ParsedSequence, Task
from .tokens import DEFAULT_DECISION_POINTS, VALIDATION_CODES, DecisionToken, ParseError, TokenSpace, ValidationToken
from .vocab import DEFAULT_BUDGET, VOCAB_TOTAL, VocabBudget

__all__ = [
    "CONTROL_PREFIXES",
    "ControlTokenLayout",
    "CharTokenizer",
    "PrismTokenizer",
]

#: 结构标记（不含决策点/验证码本身）。顺序即 id 顺序，改动必须升 schema 版本。
STRUCTURE_MARKERS: Tuple[str, ...] = (
    "<TASK>", "</TASK>", "<CODE>", "</CODE>",
)


@dataclass
class ControlTokenLayout:
    """控制 token 在词表里的**连续区间**布局。

    ``start`` 到 ``end``（不含）之间的 id 全部是控制 token，顺序为：
    结构标记 → 全部决策 token（按决策点顺序、点内选项顺序）→ 全部验证 token。
    """

    start: int
    structure: Dict[str, int] = field(default_factory=dict)
    decisions: Dict[str, int] = field(default_factory=dict)
    validations: Dict[str, int] = field(default_factory=dict)
    end: int = 0

    def finalize(self) -> "ControlTokenLayout":
        """装好三张表之后必须调用：把 ``end`` 定死为最后一张表的末尾。

        ``__post_init__`` 不能算这个数——三张表是构造后才填的，那时全是空 dict。
        这个坑在写第一版时踩过：``end`` 恒为 ``start``，区间变成空集。
        """
        ids = list(self.structure.values()) + list(self.decisions.values()) \
            + list(self.validations.values())
        if not ids:
            self.end = self.start
            return self
        self.end = max(ids) + 1
        # 区间必须连续：ids 恰好是 start..end-1 的一个排列
        if sorted(ids) != list(range(self.start, self.end)):
            raise ParseError("控制 token id 区间不连续，词表装配有空洞")
        return self

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def id_range(self) -> range:
        return range(self.start, self.end)

    def is_control(self, token_id: int) -> bool:
        return self.start <= token_id < self.end

    def kind_of(self, token_id: int) -> str:
        """返回 ``structure`` / ``decision`` / ``validation`` / ``text``。"""
        if self.structure and token_id in self.structure.values():
            return "structure"
        if token_id in self.decisions.values():
            return "decision"
        if token_id in self.validations.values():
            return "validation"
        if self.is_control(token_id):
            return "control"
        return "text"

    def to_dict(self) -> Dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "structure": self.structure,
            "decisions": self.decisions,
            "validations": self.validations,
        }


class CharTokenizer:
    """测试/离线用的极简底座：32K 个常用汉字 + ASCII。

    真实训练用 Light 仓产出的 BPE；这里只是让 ``PrismTokenizer`` 的装配逻辑
    在没有 ``tokenizers`` 的机器上也能被测试。**不要**拿它训练。
    """

    def __init__(self, extra: Iterable[str] = ()) -> None:
        self.tokens: List[str] = [chr(i) for i in range(32, 127)]
        self.index: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.unk_id = self.add("<unk>")
        for ch in extra:
            self.add(ch)

    def add(self, tok: str) -> int:
        if tok in self.index:
            return self.index[tok]
        self.index[tok] = len(self.tokens)
        self.tokens.append(tok)
        return self.index[tok]

    def encode(self, text: str) -> List[int]:
        return [self.index.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.tokens[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)


class PrismTokenizer:
    """把底座 tokenizer 与 LightPrism 控制 token 拼成一个词表。

    构造后 ``vocab_size`` 一定 ≤ ``VOCAB_TOTAL``；控制 token 落在**末尾连续区间**，
    其余空间留给经 :class:`VocabBudget` 校验过的文本侧 token。
    """

    def __init__(
        self,
        base: object,
        space: Optional[TokenSpace] = None,
        budget: Optional[VocabBudget] = None,
        total: int = VOCAB_TOTAL,
    ) -> None:
        self.base = base
        self.space = space or TokenSpace()
        self.budget = budget or VocabBudget()
        self.total = total

        issues = self.budget.problems(self.space)
        if issues:
            raise ParseError("预算不合法，拒绝装配词表：" + "；".join(issues))

        base_size = int(getattr(base, "vocab_size", 0))
        if base_size <= 0:
            raise ParseError("底座 tokenizer 必须暴露 vocab_size")

        self.layout = ControlTokenLayout(start=base_size)
        self.layout.structure["<TASK>"] = self.layout.start
        self.layout.structure["</TASK>"] = self.layout.start + 1
        self.layout.structure["<CODE>"] = self.layout.start + 2
        self.layout.structure["</CODE>"] = self.layout.start + 3
        cursor = self.layout.start + len(STRUCTURE_MARKERS)

        for tok in self.space.decision_tokens():
            self.layout.decisions[tok.text] = cursor
            cursor += 1
        for code in VALIDATION_CODES:
            text = f"<V:{code}>"
            self.layout.validations[text] = cursor
            cursor += 1
        self.layout.finalize()

        if self.layout.end > total:
            raise ParseError(
                f"控制 token 区间 [{self.layout.start}, {self.layout.end}) 超出词表 {total}"
            )
        decision_quota = self.budget.value_of("decision")
        if self.layout.size > decision_quota + self.budget.value_of("validation") + 64:
            raise ParseError(
                f"控制 token {self.layout.size} 个超出预算（决策 {decision_quota} + 验证 "
                f"{self.budget.value_of('validation')} + 结构 4）"
            )

    # ---- 基本属性 -----------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """已装配的**有效**词表大小 = 控制区间末尾。

        注意它小于 ``total``：文本侧还有预留空间（标准库名、语法组合），
        由 Light 仓的语料统计逐步填满，不是一次到位的 65536。
        """
        return self.layout.end

    @property
    def control_ids(self) -> List[int]:
        return list(self.layout.id_range)

    def kind_of(self, token_id: int) -> str:
        return self.layout.kind_of(token_id)

    def is_control(self, token_id: int) -> bool:
        return self.layout.is_control(token_id)

    # ---- 编解码 -------------------------------------------------------
    def encode_text(self, text: str) -> List[int]:
        ids = list(self.base.encode(text))
        if any(self.layout.is_control(i) for i in ids):
            raise ParseError("底座 tokenizer 返回了落在控制区间的 id，词表装配有冲突")
        return ids

    def decode_text(self, ids: Sequence[int]) -> str:
        return self.base.decode([i for i in ids if not self.layout.is_control(i)])

    def encode_sequence(self, parsed: ParsedSequence) -> List[int]:
        """``ParsedSequence`` → id 序列。控制 token 各占 1 个 id。"""
        out: List[int] = []
        for item in parsed.items:
            if isinstance(item, Task):
                out.append(self.layout.structure["<TASK>"])
                out.extend(self.encode_text(item.text))
                out.append(self.layout.structure["</TASK>"])
            elif isinstance(item, Code):
                out.append(self.layout.structure["<CODE>"])
                out.extend(self.encode_text("\n" + item.text + "\n"))
                out.append(self.layout.structure["</CODE>"])
            elif isinstance(item, DecisionToken):
                out.append(self._control_id(item.text))
            elif isinstance(item, ValidationToken):
                out.append(self._control_id(item.text))
            else:  # pragma: no cover
                raise TypeError(f"不认识的序列项 {type(item)!r}")
        return out

    def decode_tokens(self, ids: Sequence[int]) -> List[str]:
        """调试用：把 id 序列还原成「控制 token 文本 / 文本片段」的列表。"""
        out: List[str] = []
        for i in ids:
            if self.is_control(i):
                out.append(self._text_of(i))
            else:
                out.append(self.base.decode([i]))
        return out

    def _control_id(self, text: str) -> int:
        for table in (self.layout.structure, self.layout.decisions, self.layout.validations):
            if text in table:
                return table[text]
        raise ParseError(f"控制 token {text!r} 不在词表里")

    def _text_of(self, token_id: int) -> str:
        for table in (self.layout.structure, self.layout.decisions, self.layout.validations):
            for text, i in table.items():
                if i == token_id:
                    return text
        raise ParseError(f"id {token_id} 不在控制区间")

    # ---- 约束解码用 ---------------------------------------------------
    def allowed_next(self, history: Sequence[int]) -> Optional[List[int]]:
        """给约束解码用的**结构状态机**：返回下一个位置允许的控制 token id；None = 不约束。

        为什么必须看完整历史而不是紧邻一个 id：未经训练的小模型会连续吐同一个
        ``</TASK>`` 或 ``<D:EDIT=FR>`` 几十次，只看 ``last_id`` 的约束完全拦不住
        （这一版实测踩过两次）。状态机按 ``<TASK>`` / ``</TASK>`` / ``<CODE>`` /
        ``</CODE>`` / ``<V:…>`` 的出现重构当前状态：

        ===================  =================================================
        当前状态              下一个控制 token 允许范围
        ===================  =================================================
        还没有 ``<TASK>``      只能 ``<TASK>``
        ``<TASK>`` 里          只能 ``</TASK>``（其余是自由文本）
        ``</TASK>`` 之后       ``<D:…>``（同一步不同点）或 ``<CODE>``
        ``<D:…>`` 之后        同上（同一步不同点）或 ``<CODE>``
        ``<CODE>`` 里          只能 ``</CODE>``
        ``</CODE>`` 之后       只能 ``<V:…>``
        ``<V:…>`` 之后         ``<D:…>`` 或 ``<CODE>``（进入下一步）
        ===================  =================================================

        自由文本位置返回 None（不约束），由模型自己决定何时吐控制 token。
        """
        st = self.layout.structure
        in_task = in_code = False
        task_done = False
        last_control: Optional[int] = None
        step_points: set = set()

        for i in history:
            if i == st["<TASK>"]:
                in_task, last_control = True, i
            elif i == st["</TASK>"]:
                in_task, task_done, last_control = False, True, i
            elif i == st["<CODE>"]:
                in_code, last_control = True, i
            elif i == st["</CODE>"]:
                in_code, last_control = False, i
            elif i in self.layout.decisions.values():
                step_points.add(self.point_of(i))
                last_control = i
            elif i in self.layout.validations.values():
                last_control, step_points = i, set()

        all_decisions = sorted(self.layout.decisions.values())
        v_ids = sorted(self.layout.validations.values())

        if in_task:
            return [st["</TASK>"]]
        if in_code:
            return [st["</CODE>"]]
        if last_control is None:
            return [st["<TASK>"]]
        if last_control == st["<TASK>"]:
            return [st["</TASK>"]]
        if last_control == st["</CODE>"]:
            return v_ids
        if last_control == st["<CODE>"]:
            return [st["</CODE>"]]
        # </TASK> / <D:…> / <V:…> 之后
        free_decisions = [i for i in all_decisions if self.point_of(i) not in step_points]
        return sorted(free_decisions + [st["<CODE>"]])

    def allowed_after(self, token_id: int) -> Optional[List[int]]:
        """兼容入口：只看一个 id 的粗约束。**优先用** :meth:`allowed_next`。"""
        st = self.layout.structure
        if token_id == st.get("</CODE>"):
            return sorted(self.layout.validations.values())
        if token_id == st.get("<TASK>"):
            return sorted(self.layout.decisions.values())
        return None

    def point_of(self, token_id: int) -> Optional[str]:
        """决策 token → 决策点名；非决策 token 返回 None。"""
        for text, i in self.layout.decisions.items():
            if i == token_id:
                return text[len("<D:"):].split("=", 1)[0]  # "<D:EDIT=FR>" → "EDIT"
        return None

    def save_layout(self) -> Dict[str, object]:
        """checkpoint 里的分词器快照。

        **必须存底座词表本体**（``base_vocab``），不只是大小：上一版只存
        ``base_vocab_size``，恢复时用占位 token 凑数，解码出来全是 ``<padNNN>``，
        实测把结构合法率误判成 0。真实 BPE 场景下这一项换成词表文件路径 + 哈希。
        """
        return {
            "total": self.total,
            "base_vocab_size": int(getattr(self.base, "vocab_size", 0)),
            "base_vocab": list(getattr(self.base, "tokens", [])),
            "base_kind": type(self.base).__name__,
            "layout": self.layout.to_dict(),
            "schema_version": self.space.schema_version,
        }
