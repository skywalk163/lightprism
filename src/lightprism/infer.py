# -*- coding: utf-8 -*-
"""LightPrism 推理骨架：贪心 / 采样 + 结构约束 + 验证 token 后置复核。

三条与训练侧一致的红线（方案文档 §5）：:

1. 模型生成的 ``<V:…>`` **不作为**验证结论；部署时必须真跑 ``light ai check``
   复核，本模块的 :meth:`PrismGenerator.verify_with_checker` 是这个挂点。
2. ``</CODE>`` 之后强制只能生成 ``<V:…>``（:meth:`PrismTokenizer.allowed_after`），
   否则模型会漂移出结构。
3. 生成结束条件是「出现 ``<V:…>`` 且没有后续决策」，不由模型自由决定。

本模块不依赖 ``light`` 仓：验证器通过可调用对象注入，便于测试。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "GenConfig",
    "PrismGenerator",
    "CheckerResult",
    "load_generator",
]


@dataclass
class GenConfig:
    """生成超参。"""

    max_new_tokens: int = 256
    temperature: float = 0.0        # 0 = 贪心
    top_k: int = 0                  # 0 = 不截断
    stop_after_validation: bool = True
    #: 生成到出现某类控制 token 就停：``""`` 不停 / ``"validation"`` / ``"code_end"``。
    #: 骨架阶段用来产出「结构完整但可能没写完」的样例，探针实验需要它。
    stop_at: str = ""
    forbid_control_inside_code: bool = True


@dataclass
class CheckerResult:
    """验证器（``light ai check --json``）的极简结果。"""

    ok: bool
    validation: str                 # 映射后的 <V:…> 码，如 "PASS" / "ERR=SYNTAX"
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_check_json(cls, payload: Dict[str, Any]) -> "CheckerResult":
        from .tokens import map_diagnostics

        token = map_diagnostics(payload.get("diagnostics") or [])
        return cls(ok=bool(payload.get("ok")), validation=token.code,
                   diagnostics=list(payload.get("diagnostics") or []))


class PrismGenerator:
    """把模型 + 分词器 + 约束解码包成 ``generate(task) -> str``。"""

    def __init__(self, model, tokenizer, cfg: Optional[GenConfig] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or GenConfig()
        self.layout = tokenizer.layout
        self.structure = self.layout.structure
        self.validation_ids = sorted(self.layout.validations.values())
        self.all_control_ids = list(self.layout.id_range)

    # ---- 生成 ---------------------------------------------------------
    def _score_to_ids(self, logits) -> int:
        """按 temperature / top_k 选一个 id。temperature=0 走 argmax。"""
        import torch

        if self.cfg.temperature <= 0:
            return int(torch.argmax(logits).item())
        logits = logits / self.cfg.temperature
        if self.cfg.top_k and self.cfg.top_k > 0:
            k = min(self.cfg.top_k, logits.numel())
            thresh = torch.topk(logits, k).values[-1]
            logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits.float(), dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _apply_constraint(self, logits, history: Sequence[int]):
        """结构约束：用 ``allowed_next`` 状态机给出的白名单做 -inf 掩码。

        白名单里同时包含「控制 token」和「自由文本」两种可能时（例如 ``</TASK>``
        之后既可以出 ``<D:…>`` 也可以直接进 ``<CODE>``），本方法**只禁**不在白名单
        里的控制 token，文本 token 一律放行 —— 否则模型永远无法进入代码体。
        """
        import torch

        allowed = self.tokenizer.allowed_next(history)
        if allowed is None:
            return logits

        allow = set(allowed)
        mask = torch.zeros_like(logits)
        blocked = [i for i in self.all_control_ids if i not in allow]
        if not blocked:
            return logits
        mask[torch.tensor(blocked, dtype=torch.long)] = float("-inf")
        return logits + mask

    def generate(self, task: str, *, decisions: Optional[Dict[str, str]] = None,
                 max_code_tokens: int = 64) -> Dict[str, Any]:
        """生成一条序列。返回 ``{"ids": [...], "text": "...", "decisions": [...]}``。

        ``decisions`` 允许外部**钉死**某些决策（对应 lightharness 的人工确认节点）：
        钉死的决策 token 直接注入，不耗生成步数。
        """
        import torch

        self.model.eval()
        ids: List[int] = [self.structure["<TASK>"]]
        ids.extend(list(self.tokenizer.base.encode(task)))
        ids.append(self.structure["</TASK>"])

        generated: List[int] = []
        decisions_out: List[str] = []
        pinned = dict(decisions or {})

        # 先注入钉死的决策
        for point, option in pinned.items():
            text = f"<D:{point}={option}>"
            tid = self.layout.decisions[text]
            ids.append(tid)
            generated.append(tid)
            decisions_out.append(text)

        # 优先走 KV cache：prefill 一次整段 prompt，之后每步只喂 1 个新 token。
        # 模型没有 ``incremental_forward`` 时退回原来的全量重算路径。
        use_cache = hasattr(self.model, "incremental_forward")
        past_kv = None
        if use_cache:
            window = torch.tensor([ids[-self.model.cfg.max_seq:]], dtype=torch.long)
            with torch.no_grad():
                logits, past_kv = self.model.incremental_forward(window)
            last = logits[0, -1, :]
        else:
            last = None

        for _ in range(self.cfg.max_new_tokens):
            if not use_cache:
                window = torch.tensor([ids[-self.model.cfg.max_seq:]], dtype=torch.long)
                with torch.no_grad():
                    logits, _ = self.model(window)
                last = logits[0, -1, :]
            next_id = self._score_to_ids(self._apply_constraint(last, ids))
            ids.append(next_id)
            generated.append(next_id)

            if self.tokenizer.kind_of(next_id) == "decision":
                decisions_out.append(self.tokenizer._text_of(next_id))
            stopped = False
            if self.cfg.stop_after_validation and self.tokenizer.kind_of(next_id) == "validation":
                stopped = True
            if self.cfg.stop_at == "code_end" and next_id == self.structure["</CODE>"]:
                stopped = True
            if stopped:
                break

            if use_cache:
                step = torch.tensor([[next_id]], dtype=torch.long)
                with torch.no_grad():
                    logits, past_kv = self.model.incremental_forward(step, past_kv)
                last = logits[0, -1, :]
                # 缓存长度顶到上限就丢弃缓存，下一步退回全量重算，避免越界
                if len(ids) >= self.model.cfg.max_seq:
                    past_kv = None
        pieces = self.tokenizer.decode_tokens(generated)
        structure_ok, structure_problems = self._check_structure(pieces)
        # 区分「模型吐错」和「我们把它截断了」：截断导致的尾部裸露文本不该算模型的锅
        truncated = len(generated) >= self.cfg.max_new_tokens
        return {
            "ids": generated,
            "text": pieces,
            "decisions": decisions_out,
            "raw_ids": ids,
            "structure_ok": structure_ok,
            "structure_problems": structure_problems,
            "truncated": truncated,
            # 计入合法率的唯一口径：结构完整 **且** 不是被截断的
            "counts_toward_structure_rate": structure_ok and not truncated,
            "seq_text": "".join(pieces),
        }

    def _check_structure(self, pieces: Sequence[str]) -> Tuple[bool, List[str]]:
        """把生成的片段拼回序列文本并用 ``parse_sequence`` 复核。

        未经训练的模型会停在自由文本里（合法但无意义）或吐不出 ``</CODE>``。
        把结果显式报出来，而不是让调用方以为「没抛异常就是合法」—— 探针实验里
        结构合法率就是靠这个字段统计的。
        """
        from .sequence import parse_sequence

        # 生成以 <TASK> 开头是调用方拼进去的，这里补回去才能过 parse_sequence
        text = "<TASK></TASK>" + "".join(pieces)
        try:
            parsed = parse_sequence(text)
        except Exception as exc:  # noqa: BLE001 - ParseError 与拼装异常都要报出来
            return False, [str(exc)]
        problems = parsed.validate(require_task=True)
        return (not problems), problems

    # ---- 后置复核 -----------------------------------------------------
    @staticmethod
    def verify_with_checker(text: str, checker: Callable[[str], CheckerResult]) -> CheckerResult:
        """真跑验证器。

        **模型自己生成的 ``<V:…>`` 不能被信任**：这里把它剥掉，只拿代码段去验证。
        这是方案 §5「推理时 lightharness 仍需实际调用 light ai check 做最终确认」的落点。
        """
        code = _extract_code(text)
        return checker(code)


def _extract_code(items: Sequence[str]) -> str:
    """从生成的 token 文本列表里抽出 ``<CODE>`` 与 ``</CODE>`` 之间的内容。"""
    out: List[str] = []
    inside = False
    for item in items:
        if item == "<CODE>":
            inside = True
            continue
        if item == "</CODE>":
            inside = False
            continue
        if inside:
            out.append(item)
    return "".join(out)


def load_generator(ckpt_path: str | Path, cfg: Optional[GenConfig] = None) -> PrismGenerator:
    """从 checkpoint 还原模型与分词器。

    **已知限制**：checkpoint 里存的是 tokenizer 的 layout（id 区间），不是底座 BPE 本体。
    还原底座需要 Light 仓的词表文件；本骨架用 ``CharTokenizer`` 重建与服务端一致的字表，
    仅够 smoke 用。真实部署必须加载同一份 BPE，否则 id 会错位。
    """
    import torch

    from .model import LightPrismModel, PrismConfig
    from .tokenizer import CharTokenizer, PrismTokenizer

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LightPrismModel(PrismConfig(**blob["model_config"]))
    model.load_state_dict(blob["model"])
    model.eval()

    layout = blob["tokenizer_layout"]
    base_vocab = layout.get("base_vocab") or []
    if not base_vocab:
        raise ValueError(
            "checkpoint 缺少 base_vocab，无法还原分词器。"
            "用旧版 train.save() 存的 checkpoint 不兼容，请重训。"
        )
    base = CharTokenizer()
    # 按 checkpoint 里的顺序重建：CharTokenizer 默认已含 ASCII + <unk>，
    # 逐字补齐到完全一致即可（顺序由 add() 保证是追加语义）
    for tok_text in base_vocab:
        base.add(tok_text)
    if base.vocab_size != int(layout["base_vocab_size"]):
        raise ValueError(
            f"底座词表还原失败：期望 {layout['base_vocab_size']}，实得 {base.vocab_size}"
        )
    tok = PrismTokenizer(base)
    return PrismGenerator(model, tok, cfg)
