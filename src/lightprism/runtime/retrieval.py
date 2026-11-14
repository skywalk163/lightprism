# -*- coding: utf-8 -*-
"""零模型意图归一 / 检索 / 槽位抽取（纯标准库，无任何模型参与）。

方式（报告中如实标注的口径）
----------------------------
1. **意图归一**：全角转半角、去标点、中文按字切 bigram、数字/ASCII 按词切；
2. **检索**：模板 ``keywords`` 文档上的 BM25（k1=1.5, b=0.75）+
   ``triggers`` 精确命中加权（规则部分，可解释）；
3. **填槽**：正则从**任务文本**里抽数字 / 列表字面量 / 引号文本 / 运算符 /
   分隔符 / 索引 / 阈值。抽不到就留空槽，让 checker 如实报错，不编数值。

没有任何词向量、没有 LLM、没有外部索引。同一输入永远得到同一结果。
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .templates import TEMPLATES, Template

__all__ = [
    "normalize", "tokenize", "SearchHit", "BM25Index", "extract_slots",
    "DEFAULT_MIN_SCORE", "OP_WORDS", "SEP_WORDS",
]

DEFAULT_MIN_SCORE = 1.0
_K1 = 1.5
_B = 0.75
_TRIGGER_BONUS = 2.0

OP_WORDS: Tuple[Tuple[str, str], ...] = (
    ("乘以", "乘"), ("除以", "除"), ("加上", "加"), ("减去", "减"),
    ("相乘", "乘"), ("相除", "除"), ("相加", "加"), ("相减", "减"),
    ("加", "加"), ("减", "减"), ("乘", "乘"), ("除", "除"),
)
SEP_WORDS: Tuple[Tuple[str, str], ...] = (
    ("逗号", ","), ("分号", ";"), ("空格", " "), ("竖线", "|"), ("制表符", "\t"),
)

_FULLWIDTH = {ord("０") + i: ord("0") + i for i in range(10)}
_FULLWIDTH.update({ord("Ａ") + i: ord("A") + i for i in range(26)})
_FULLWIDTH.update({ord("ａ") + i: ord("a") + i for i in range(26)})

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[a-z0-9_]+")
_NUM_RE = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?")
_LIST_RE = re.compile(r"\[[^\[\]]*\]")
_QUOTE_RE = re.compile(r"[「『\"']([^」』\"']*)[」』\"']")
_INDEX_RE = re.compile(r"第\s*(\d+)\s*个")
_RANGE_END_RE = re.compile(r"[到至]\s*(\d+)")
_THRESHOLD_RE = re.compile(r"阈值\s*(\d+)")
_STATE_RE = re.compile(r"状态\s*[「『\"']([^」』\"']+)")


def normalize(text: str) -> str:
    """全角转半角 + NFKC + 小写，保留中文与数字。"""
    return unicodedata.normalize("NFKC", text).translate(_FULLWIDTH).lower()


def tokenize(text: str) -> List[str]:
    """中文按 bigram（单字也保留），数字/ASCII 按词。"""
    t = normalize(text)
    tokens: List[str] = []
    tokens.extend(_WORD.findall(t))
    for chunk in _CJK.findall(t):
        tokens.extend(chunk)
        tokens.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return tokens


@dataclass(frozen=True)
class SearchHit:
    template_id: str
    score: float
    bm25: float
    trigger_bonus: float
    trigger_hits: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "score": round(self.score, 4),
            "bm25": round(self.bm25, 4),
            "trigger_bonus": round(self.trigger_bonus, 4),
            "trigger_hits": list(self.trigger_hits),
        }


class BM25Index:
    """模板关键词文档上的确定性 BM25 + trigger 加权。"""

    def __init__(self, templates: Sequence[Template] = TEMPLATES) -> None:
        self.templates = tuple(templates)
        self.docs: Dict[str, List[str]] = {
            t.id: tokenize(" ".join(t.keywords)) for t in self.templates
        }
        self.doc_len: Dict[str, int] = {
            tid: len(toks) for tid, toks in self.docs.items()
        }
        self.avgdl = (sum(self.doc_len.values()) / len(self.doc_len)) if self.doc_len else 1.0
        df: Dict[str, int] = {}
        for toks in self.docs.values():
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        n = len(self.docs)
        self.idf: Dict[str, float] = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def _bm25(self, query: Sequence[str], tid: str) -> float:
        doc = self.docs[tid]
        if not doc:
            return 0.0
        tf: Dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        dl = self.doc_len[tid]
        score = 0.0
        for term in query:
            freq = tf.get(term)
            if not freq:
                continue
            denom = freq + _K1 * (1 - _B + _B * dl / self.avgdl)
            score += self.idf.get(term, 0.0) * freq * (_K1 + 1) / denom
        return score

    def search(self, text: str, top_k: int = 3) -> List[SearchHit]:
        query = tokenize(text)
        raw = normalize(text)
        hits: List[SearchHit] = []
        for t in self.templates:
            matched = tuple(trig for trig in t.triggers if trig and normalize(trig) in raw)
            bonus = min(len(matched), 3) * _TRIGGER_BONUS
            bm = self._bm25(query, t.id)
            hits.append(SearchHit(t.id, bm + bonus, bm, bonus, matched))
        hits.sort(key=lambda h: (-h.score, h.template_id))
        return hits[:top_k]


def _context(text: str) -> Dict[str, Any]:
    norm = normalize(text)
    quoted = [m.group(1).strip() for m in _QUOTE_RE.finditer(text)]
    return {
        "raw": text,
        "norm": norm,
        "numbers": _NUM_RE.findall(norm),
        "lists": [m.group(0) for m in _LIST_RE.finditer(norm)],
        "quoted": quoted,
    }


def _apply_rule(rule: str, arg: object, ctx: Dict[str, Any]) -> object:
    if rule == "const":
        return arg
    if rule == "numbers":
        idx = int(arg)
        seq = ctx["numbers"]
        return seq[idx] if -len(seq) <= idx < len(seq) else None
    if rule == "lists":
        idx = int(arg)
        seq = ctx["lists"]
        return seq[idx] if -len(seq) <= idx < len(seq) else None
    if rule == "quoted":
        idx = int(arg)
        seq = ctx["quoted"]
        return seq[idx] if -len(seq) <= idx < len(seq) else None
    if rule == "op":
        for word, canonical in OP_WORDS:
            if word in ctx["norm"]:
                return canonical
        return str(arg)
    if rule == "sep":
        for word, sep in SEP_WORDS:
            if word in ctx["norm"]:
                return sep
        return str(arg)
    if rule == "affix":
        return "开头" if any(w in ctx["norm"] for w in ("前缀", "开头", "起始")) else "结尾"
    if rule == "index":
        m = _INDEX_RE.search(ctx["norm"])
        if m:
            return m.group(1)
        return ctx["numbers"][-1] if ctx["numbers"] else None
    if rule == "range_end":
        # 「累加从 1 到 5」要的是上界 5，不是起始值 1。
        m = _RANGE_END_RE.search(ctx["norm"])
        if m:
            return m.group(1)
        return ctx["numbers"][-1] if ctx["numbers"] else None
    if rule == "threshold":
        m = _THRESHOLD_RE.search(ctx["norm"])
        if m:
            return m.group(1)
        return ctx["numbers"][-1] if ctx["numbers"] else None
    if rule == "state":
        m = _STATE_RE.search(ctx["norm"])
        if m:
            return m.group(1)
        return ctx["quoted"][1] if len(ctx["quoted"]) > 1 else "run"
    raise ValueError(f"未知槽位抽取规则 {rule!r}")


def extract_slots(text: str, template: Template) -> Tuple[Dict[str, object], List[str]]:
    """按模板 ``slot_plan`` 从任务文本抽槽。返回 ``(槽, 缺失槽名列表)``。"""
    ctx = _context(text)
    slots: Dict[str, object] = {}
    missing: List[str] = []
    for name, (rule, arg) in template.slot_plan.items():
        value = _apply_rule(str(rule), arg, ctx)
        if value is None or (isinstance(value, str) and value == "" and rule != "const"):
            missing.append(name)
        else:
            slots[name] = value
    return slots, missing
