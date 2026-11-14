# -*- coding: utf-8 -*-
"""T-0345：把约束解码接到 demo 推理路径，并给出「产品 / 能力」两列口径。

## 为什么要单独一层

``src/lightprism/infer.py`` 的 :class:`PrismGenerator` 已经实现了结构状态机
（``allowed_next``）、-inf 掩码（``_apply_constraint``）和 KV cache；但
``src/lightprism/demo/smoke.py`` 之前走的是裸 ``torch.argmax(last)``，一点没接上，
于是 demo 端 20 条抽样全是空格、0 条带 ``<CODE>``。

本模块只做适配，**不改** ``infer.py`` 的公共行为，也不动
``PrismTokenizer.allowed_next`` 的状态定义（全项目共享）：

* :class:`DemoTokenizerAdapter` 把 :class:`~lightprism.demo.train_demo.DemoVocab`
  适配成 ``PrismGenerator`` 认识的分词器接口，状态机直接复用
  :meth:`PrismTokenizer.allowed_next`；
* :class:`ConstrainedDemoGenerator` 复用 ``PrismGenerator._apply_constraint`` 做
  控制 token 白名单掩码，并额外做两件 demo 特有的事：
  1. **空槽永久 mask**：``data/vocab_4096.json`` 的 ``size=2900`` 之后、
     除 PAD/UNK 外的槽位在 logits 上恒置 ``-inf``；
  2. **结构推进**：状态要求「下一个必须是控制 token」时（``</TASK>`` 之后、
     决策/验证 token 之后），把自由文本也一起掩掉；``<CODE>`` 体内保留自由
     文本，但在生成预算只剩 2 步时强制 ``</CODE>``，保证序列能闭合。

  第 2 点是**产品路径的骨架保证**，不是模型能力：约束版结构合法率必然接近
  100%，按 ``train_openi.py`` 文件头的口径，属于「假绿」，不能填进能力表。

## 两列口径（不要合并）

* ``constrained``：带成对 ``<CODE>…</CODE>`` 且以 ``<V:…>`` 收尾 → **产品指标**；
* ``unconstrained``：关掉状态机、裸 argmax → **能力指标**；
* 代码段内合法率：抽出 ``<CODE>`` 内容交给 ``light ai check --json``，空输出
  判不合法，**全分母**，不剔除截断。

``light ai check`` 对空程序判 ``ok``（``docs/Demo验收报告.md`` §3.4 的已知盲区），
所以 :func:`evaluate_code_segments` 在调用 checker 之前就显式判空，不把空输出
计进合法分子。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..infer import GenConfig, PrismGenerator
from ..tokenizer import ControlTokenLayout, PrismTokenizer
from .train_demo import DemoVocab

__all__ = [
    "ConstrainedDemoGenerator",
    "DemoTokenizerAdapter",
    "extract_code",
    "structure_legal",
    "summarize_structure",
    "evaluate_code_segments",
    "evaluate_two_columns",
    "load_demo_generator",
]

_CODE_RE = re.compile(r"<CODE>(.*?)</CODE>", re.S)
_VALIDATION_RE = re.compile(r"<V:[^>]+>")


class _BaseShim:
    """``PrismGenerator`` 只需要 ``base.encode``；把 ``DemoVocab`` 包一层。"""

    def __init__(self, vocab: DemoVocab) -> None:
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        return self.vocab.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self.vocab.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.vocab.size

    @property
    def tokens(self) -> List[str]:
        return self.vocab.tokens


class DemoTokenizerAdapter:
    """让 :class:`DemoVocab` 满足 ``PrismGenerator`` 的分词器契约。

    状态机**不重写**：``allowed_next`` 直接指向
    :meth:`PrismTokenizer.allowed_next`，所以 demo 端与 ``infer.py`` 的结构定义
    永远同一份。
    """

    #: 复用全局状态机（只依赖 ``self.layout`` 与 ``self.point_of``）。
    allowed_next = PrismTokenizer.allowed_next

    def __init__(self, vocab: DemoVocab) -> None:
        self.vocab = vocab
        self.base = _BaseShim(vocab)

        layout = ControlTokenLayout(start=vocab.control_start)
        layout.structure = dict(vocab.structure)
        layout.decisions = dict(vocab.decisions)
        layout.validations = dict(vocab.validations)
        self.layout = layout.finalize()
        self.structure = self.layout.structure

    # ---- 状态机依赖的两个小接口 ---------------------------------------
    def point_of(self, token_id: int) -> Optional[str]:
        for text, i in self.layout.decisions.items():
            if i == token_id:
                return text[len("<D:"):].split("=", 1)[0]
        return None

    def kind_of(self, token_id: int) -> str:
        return self.layout.kind_of(token_id)

    def _text_of(self, token_id: int) -> str:
        for table in (self.layout.structure, self.layout.decisions,
                      self.layout.validations):
            for text, i in table.items():
                if i == token_id:
                    return text
        raise KeyError(f"id {token_id} 不在控制区间")

    def decode_tokens(self, ids: Sequence[int]) -> List[str]:
        pieces: List[str] = []
        for i in ids:
            if self.layout.is_control(i):
                pieces.append(self._text_of(i))
            elif 0 <= i < self.vocab.size:
                pieces.append(self.vocab.tokens[i])
            else:  # UNK / PAD / 空槽：解码时直接丢掉
                pieces.append("")
        return pieces


class _DemoCore(PrismGenerator):
    """在 ``PrismGenerator`` 上叠加「空槽 mask + 结构推进」，不改父类。"""

    def __init__(self, model, tokenizer: DemoTokenizerAdapter,
                 cfg: GenConfig, *, vocab: DemoVocab,
                 constrain_structure: bool, mask_empty_slots: bool,
                 force_structure: bool) -> None:
        super().__init__(model, tokenizer, cfg)
        self.vocab = vocab
        self.constrain_structure = constrain_structure
        self.mask_empty_slots = mask_empty_slots
        self.force_structure = force_structure
        self.prompt_len = 0
        self.empty_slot_ids: List[int] = [
            i for i in range(vocab.size, vocab.vocab_size)
            if i not in (vocab.unk_id, vocab.pad_id)
        ]

    # ---- 掩码 ---------------------------------------------------------
    def _apply_constraint(self, logits, history: Sequence[int]):
        import torch

        out = logits
        if self.constrain_structure:
            # 父类：控制 token 白名单（复用，不复制状态机）
            out = super()._apply_constraint(out, history)

        if self.mask_empty_slots and self.empty_slot_ids:
            out = out.clone()
            out[torch.tensor(self.empty_slot_ids, dtype=torch.long)] = float("-inf")

        if self.constrain_structure and self.force_structure:
            out = self._force_structural_progress(out, history)
        return out

    def _force_structural_progress(self, logits, history: Sequence[int]):
        """状态要求控制 token 时，连自由文本也一起掩掉；代码体尾部强制闭合。"""
        import torch

        st = self.structure
        in_task = in_code = False
        for token_id in history:
            if token_id == st["<TASK>"]:
                in_task = True
            elif token_id == st["</TASK>"]:
                in_task = False
            elif token_id == st["<CODE>"]:
                in_code = True
            elif token_id == st["</CODE>"]:
                in_code = False

        if in_task:
            # 任务文本本身是自由文本，不强制
            return logits

        if in_code:
            used = len(history) - self.prompt_len
            # 只剩 2 步：1 步 </CODE> + 1 步 <V:…>，必须闭合
            if used >= self.cfg.max_new_tokens - 2:
                return _only_allow(logits, torch, [st["</CODE>"]])
            return logits

        allowed = self.tokenizer.allowed_next(history)
        if not allowed:
            return logits
        return _only_allow(logits, torch, allowed)


def _only_allow(logits, torch, allowed: Sequence[int]):
    """只保留 ``allowed`` 的原始 logits，其余置 -inf。"""
    ids = sorted(set(int(i) for i in allowed))
    if not ids:
        return logits
    out = torch.full_like(logits, float("-inf"))
    idx = torch.tensor(ids, dtype=torch.long)
    out[idx] = logits[idx]
    return out


class ConstrainedDemoGenerator:
    """与旧 ``smoke.DemoGenerator`` 接口一致，但走受约束路径。

    参数
    ----
    constrain_structure
        是否接状态机（默认 True）。False 时是「裸 argmax」能力对照。
    mask_empty_slots
        是否把 ``size`` 之后的空槽（PAD/UNK 除外）永久置 -inf。
    force_structure
        是否在必须出控制 token 的状态强制推进（默认跟随 ``constrain_structure``）。
    """

    def __init__(self, model, vocab: DemoVocab, max_new_tokens: int = 32, *,
                 constrain_structure: bool = True,
                 mask_empty_slots: bool = True,
                 force_structure: Optional[bool] = None) -> None:
        self.model = model
        self.vocab = vocab
        self.max_new_tokens = max_new_tokens
        self.constrain_structure = constrain_structure
        self.mask_empty_slots = mask_empty_slots
        self.force_structure = (constrain_structure if force_structure is None
                                else force_structure)
        self.tokenizer = DemoTokenizerAdapter(vocab)
        self.core = _DemoCore(
            model, self.tokenizer, GenConfig(max_new_tokens=max_new_tokens),
            vocab=vocab,
            constrain_structure=constrain_structure,
            mask_empty_slots=mask_empty_slots,
            force_structure=self.force_structure,
        )

    def generate(self, task: str = "",
                 max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        n_new = int(max_new_tokens or self.max_new_tokens)
        if n_new <= 0:
            raise ValueError(f"max_new_tokens 必须为正，收到 {n_new}")
        self.core.cfg.max_new_tokens = n_new
        task = task or ""
        # prompt = <TASK> + 任务文本 + </TASK>
        self.core.prompt_len = 1 + len(self.vocab.encode(task)) + 1

        out = self.core.generate(task)
        ids = list(out["ids"])
        text = str(out.get("seq_text") or "")
        return {
            "ids": ids,
            "text": text,
            "generated_tokens": len(ids),
            "truncated": bool(out.get("truncated")),
            "raw_ids": list(out.get("raw_ids") or []),
            "structure_ok": bool(out.get("structure_ok")),
            "structure_problems": list(out.get("structure_problems") or []),
            "constrain_structure": self.constrain_structure,
            "mask_empty_slots": self.mask_empty_slots,
            "force_structure": self.force_structure,
        }


def load_demo_generator(ckpt_path: str | Path, max_new_tokens: int = 32,
                        **kwargs: Any) -> ConstrainedDemoGenerator:
    """从 checkpoint 还原模型 + ``DemoVocab``，返回受约束生成器。"""
    import torch

    from lightprism.model import LightPrismModel, PrismConfig

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LightPrismModel(PrismConfig(**blob["model_config"]))
    model.load_state_dict(blob["model"])
    model.eval()

    layout = blob["tokenizer_layout"]
    base_vocab = layout.get("base_vocab") or []
    if not base_vocab:
        raise ValueError("checkpoint 缺少 base_vocab，无法还原词表")
    vocab = DemoVocab({"tokens": base_vocab, "control_layout": layout["layout"]})
    return ConstrainedDemoGenerator(model, vocab, max_new_tokens=max_new_tokens,
                                    **kwargs)


# --------------------------------------------------------------------------
# 评测口径
# --------------------------------------------------------------------------
def _pairs(text: str) -> bool:
    return (text.count("<CODE>") == 1 and text.count("</CODE>") == 1
            and text.index("<CODE>") < text.index("</CODE>"))


def structure_legal(text: str) -> bool:
    """结构是否合法：成对 ``<CODE>…</CODE>`` + 以 ``<V:…>`` 收尾。

    与 ``train_openi.check_structure(expect_validation=True)`` 同口径：
    生成段里不允许再出现 ``<TASK>``/``</TASK>``，CODE 标记恰好一开一闭，
    且最后一个控制 token 是验证 token。
    """
    t = text or ""
    if "<TASK>" in t or "</TASK>" in t:
        return False
    if not _pairs(t):
        return False
    tail = t[t.index("</CODE>") + len("</CODE>"):]
    return bool(_VALIDATION_RE.search(tail))


def extract_code(text: str) -> str:
    """只抽 ``<CODE>…</CODE>`` 的内容；没有标记就是空（不是全文）。"""
    m = _CODE_RE.search(text or "")
    return m.group(1).strip() if m else ""


def summarize_structure(texts: Sequence[str]) -> Dict[str, Any]:
    """结构合法率（产品列 / 能力列都用它，区别在生成时是否接状态机）。"""
    texts = list(texts)
    legal = sum(1 for t in texts if structure_legal(t))
    with_pair = sum(1 for t in texts if _pairs(t))
    with_v = sum(1 for t in texts if _VALIDATION_RE.search(t))
    return {
        "samples": len(texts),
        "legal": legal,
        "rate": (legal / len(texts)) if texts else 0.0,
        "with_code_pair": with_pair,
        "with_validation": with_v,
    }


def evaluate_code_segments(texts: Sequence[str],
                           checker: Callable[[str], bool]) -> Dict[str, Any]:
    """代码段内合法率：全分母，空输出直接判不合法（不交给 checker）。

    ``light ai check`` 对空程序返回 ``ok=true``（验收报告 §3.4），所以这里必须
    先判空，否则空输出会被误计成合法。
    """
    texts = list(texts)
    empty = valid = checked = 0
    for text in texts:
        code = extract_code(text)
        if not code:
            empty += 1
            continue
        checked += 1
        if checker(code):
            valid += 1
    return {
        "samples": len(texts),
        "empty": empty,
        "checked": checked,
        "valid": valid,
        "rate": (valid / len(texts)) if texts else 0.0,
    }


def evaluate_two_columns(constrained_texts: Sequence[str],
                         unconstrained_texts: Sequence[str],
                         checker: Callable[[str], bool]) -> Dict[str, Any]:
    """把三列数字分开算，**不合并不代换**。

    * ``product_structure``：约束版结构合法率（产品指标，必然接近 100%）；
    * ``capability_structure``：无约束结构合法率（能力指标）；
    * ``capability_code``：``<CODE>`` 内容的 checker 合法率（能力指标）。
    """
    return {
        "product_structure": summarize_structure(constrained_texts),
        "capability_structure": summarize_structure(unconstrained_texts),
        "capability_code": evaluate_code_segments(constrained_texts, checker),
    }
