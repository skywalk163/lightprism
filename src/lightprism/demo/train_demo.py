# -*- coding: utf-8 -*-
"""T-0332：训练 demo checkpoint（约 3.4M 参数）。

## 这张卡解决什么

T-0330 产出了 4096 实词表（``data/vocab_4096.json``），T-0331 把模型核
（``src/lightprism/model.py`` + ``configs/demo-3m.json``，d=192 / L=6 / H=6）
固定下来。本模块把两者接到同一条训练管线上，在 3MB 光明语料 + T-0333 的
SFT 任务集上训 50 步，落一个能被 smoke 加载、生成不崩的 3.4M 参数 checkpoint。

## 数据口径（可复现，不编造）

训练语料分两路，都来自仓内既有产物，不调用模型批量生成：

1. **代码语料（自回归语言建模）**：``data/corpus/index.jsonl`` 里 ``kept=true``
   的 392 个上游 ``.light`` 文件。原始文件按索引的 ``src_path`` 从 light 仓读回
   （``--light-root``，默认 ``G:/dswork/duan-light-merge/light-merge``）；
   若原始文件不在场（只交付了索引），则退回 ``data/corpus/*.txt``（被
   ``.gitignore`` 挡住的本地产物）。按**文件**做 5% 验证集切分，切分种子固定，
   保证 Windows / FreeBSD 两次跑出来的 train/val 划分一致。
2. **SFT 任务集（prompt → 代码 / JSON）**：T-0333 产出的
   ``data/sft/json_tasks.jsonl`` + ``code_tasks.jsonl``，渲染成
   ``<TASK>…</TASK> <CODE>…</CODE> <V:PASS>`` 序列。

序列先按样本渲染成完整文本，再用**贪心最长匹配**分词（词表里既有单字也有
2—4 字组合），最后首尾相接切成 ``max_seq=512`` 的定长块；残块丢弃。
控制 token 各占 1 个 id，直接复用词表 JSON 里固化的 ``control_layout``。

## 用法

```
PYTHONPATH=src python -m lightprism.demo.train_demo \
    --config configs/demo-3m.json --steps 50
PYTHONPATH=src python -m lightprism.demo.smoke --ckpt checkpoints/demo-3m/model.pt
```

## 诚实边界

* 50 步 / 3MB 语料只够让模型记住语法骨架，**不承诺生成语义正确的程序**；
  结构合法率、过拟合起始步都如实写进训练报告，不硬凑。
* 本模块不训练 BPE，分词是词表清单上的贪心最长匹配；UNK 落到词表末尾的
  预留槽位（``vocab_size`` 之内，不越界）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

#: 仓库根：src/lightprism/demo/train_demo.py → parents[3]
ROOT = Path(__file__).resolve().parents[3]
VOCAB_PATH = ROOT / "data" / "vocab_4096.json"
CORPUS_INDEX = ROOT / "data" / "corpus" / "index.jsonl"
CORPUS_DIR = ROOT / "data" / "corpus"
SFT_JSON = ROOT / "data" / "sft" / "json_tasks.jsonl"
SFT_CODE = ROOT / "data" / "sft" / "code_tasks.jsonl"
DEFAULT_OUT = ROOT / "checkpoints" / "demo-3m"

#: 上游 light 仓默认位置（词表报告 §2.1 的扫描根）。可用 --light-root 覆盖。
#: 顺序：环境变量 LIGHT_ROOT → Windows 本机路径 → FreeBSD 测试机路径。
DEFAULT_LIGHT_ROOT = Path(
    os.environ.get("LIGHT_ROOT")
    or "G:/dswork/duan-light-merge/light-merge"
    or "/home/skywalk/dswork/light"
)

#: 验证集占比（按文件切，不按 token 切）。
VAL_RATIO = 0.05
#: 切分种子固定：跨平台复现同一份 train/val。
SPLIT_SEED = 20261112

__all__ = [
    "DemoVocab",
    "load_demo_vocab",
    "read_corpus_files",
    "render_sft_samples",
    "DemoTrainConfig",
    "DemoTrainer",
    "build_arg_parser",
    "main",
]


# --------------------------------------------------------------------------
# 词表装配
# --------------------------------------------------------------------------
class DemoVocab:
    """词表 JSON → 贪心最长匹配分词器 + 控制 token 布局。

    与 :class:`lightprism.tokenizer.PrismTokenizer` 的布局口径一致：
    文本 token 在 ``[0, control_start)``，控制 token 在 ``[control_start, size)``。
    这里不依赖 ``CharTokenizer``，直接用词表 JSON 里固化的 ``tokens`` 清单。
    """

    def __init__(self, vocab_json: Dict[str, Any]) -> None:
        self.tokens: List[str] = list(vocab_json["tokens"])
        self.size: int = len(self.tokens)
        self.index: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        layout = dict(vocab_json["control_layout"])
        self.control_start: int = int(layout["start"])
        self.control_end: int = int(layout["end"])
        self.structure: Dict[str, int] = dict(layout["structure"])
        self.decisions: Dict[str, int] = dict(layout["decisions"])
        self.validations: Dict[str, int] = dict(layout["validations"])
        #: UNK / pad 槽位：控制区间之后、4096 词表预算之内。
        self.unk_id: int = self.control_end
        self.pad_id: int = self.control_end + 1
        self.max_token_id: int = self.pad_id
        # 多字 token 按长度降序，贪心匹配时先试长的。
        self._multi: List[Tuple[str, int]] = sorted(
            ((t, i) for t, i in self.index.items() if len(t) > 1),
            key=lambda kv: -len(kv[0]),
        )
        self._max_multi_len = max((len(t) for t, _ in self._multi), default=0)

    @property
    def vocab_size(self) -> int:
        """模型嵌入层尺寸：预留 UNK/pad 槽位后对齐到 4096。"""
        return max(self.max_token_id + 1, 4096)

    # ---- 编解码 ---------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """贪心最长匹配；未覆盖字符落到 :attr:`unk_id`。"""
        out: List[int] = []
        i, n = 0, len(text)
        while i < n:
            matched: Optional[str] = None
            upper = min(self._max_multi_len, n - i)
            for k in range(upper, 1, -1):
                piece = text[i:i + k]
                if piece in self.index:
                    matched = piece
                    break
            if matched is None:
                ch = text[i]
                if ch in self.index:
                    out.append(self.index[ch])
                else:
                    out.append(self.unk_id)
                i += 1
            else:
                out.append(self.index[matched])
                i += len(matched)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        pieces: List[str] = []
        for i in ids:
            if i == self.unk_id or i == self.pad_id:
                continue
            if 0 <= i < self.size:
                pieces.append(self.tokens[i])
        return "".join(pieces)

    def kind_of(self, token_id: int) -> str:
        if token_id in self.structure.values():
            return "structure"
        if token_id in self.decisions.values():
            return "decision"
        if token_id in self.validations.values():
            return "validation"
        if self.control_start <= token_id < self.control_end:
            return "control"
        return "text"

    def save_layout(self) -> Dict[str, Any]:
        """checkpoint 里的分词器快照（含底座词表本体，恢复时可重建）。"""
        return {
            "total": self.vocab_size,
            "base_vocab_size": self.size,
            "base_vocab": list(self.tokens),
            "base_kind": "DemoVocab",
            "unk_id": self.unk_id,
            "pad_id": self.pad_id,
            "layout": {
                "start": self.control_start,
                "end": self.control_end,
                "structure": self.structure,
                "decisions": self.decisions,
                "validations": self.validations,
            },
        }


def load_demo_vocab(path: str | Path = VOCAB_PATH) -> DemoVocab:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"缺少词表产物: {p}")
    return DemoVocab(json.loads(p.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# 语料读取
# --------------------------------------------------------------------------
def _read_text(path: Path) -> str:
    """UTF-8 容错读取 + 行尾归一化（与 vocab_build 同口径）。"""
    text = path.read_bytes().decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_corpus_files(
    light_root: Path,
    corpus_dir: Path = CORPUS_DIR,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """按索引读回 kept 语料，返回 ``[(src_path, text), ...]`` 与跳过原因。

    优先从上游 light 仓按 ``src_path`` 读原始文件；上游不在场时退回
    ``corpus_dir`` 下的 ``.txt`` 副本（T-0330 本地产物，不入库）。
    若两个位置都不在场，回退到 light 仓的常见挂载路径，避免硬编码单一盘符。
    """
    records = [json.loads(line) for line in CORPUS_INDEX.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [r for r in records if r.get("kept")]
    roots: List[Path] = [Path(light_root)]
    for fallback in ("/home/skywalk/dswork/light", "G:/dswork/duan-light-merge/light-merge"):
        fb = Path(fallback)
        if fb not in roots:
            roots.append(fb)

    def _open(rel: str) -> Optional[Path]:
        for root in roots:
            cand = root / rel
            if cand.is_file():
                return cand
        # 本地产物/快照目录：data/corpus/flat/<rel>（与上游仓同目录结构，仅换扩展名）
        flat = corpus_dir / "flat" / rel
        if flat.is_file():
            return flat
        txt_rel = str(Path(rel).with_suffix(".txt"))
        local_txt = corpus_dir / txt_rel
        if local_txt.is_file():
            return local_txt
        return None

    out: List[Tuple[str, str]] = []
    skipped: List[str] = []
    for rec in kept:
        rel = str(rec["src_path"]).replace("\\", "/")
        cand = _open(rel)
        if cand is None:
            skipped.append(rel)
            continue
        out.append((rel, _read_text(cand)))
    return out, skipped


def _render_code_sequence(task: str, code: str, validation: str = "PASS") -> str:
    """把一条 SFT 样本渲染成标准 ``<TASK>/<CODE>/<V:…>`` 序列文本。"""
    val = validation or "PASS"
    return (
        f"<TASK>{task.strip()}</TASK>\n"
        f"<CODE>\n{code.strip()}\n</CODE>\n"
        f"<V:{val}>\n"
    )


def render_sft_samples() -> List[Tuple[str, str]]:
    """读 T-0333 的两份 JSONL，返回 ``(category, sequence_text)`` 列表。

    先把 dict 记录还原成 :class:`lightprism.data.Sample`，再复用
    :mod:`lightprism.demo.sft_build` 的渲染口径；坏行计数但不中断。
    """
    from lightprism.data import Sample

    from .sft_build import CODE_TASKS_PATH, JSON_TASKS_PATH, render_training_sequence

    rows: List[Tuple[str, str]] = []
    for path, default_fmt in ((JSON_TASKS_PATH, "json"), (CODE_TASKS_PATH, "code")):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sample = Sample.from_dict(rec)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            meta = rec.get("meta") or {}
            fmt = str(meta.get("format") or default_fmt)
            seq = render_training_sequence(sample, fmt)
            rows.append((str(meta.get("category") or ""), seq))
    return rows


# --------------------------------------------------------------------------
# 切分与打包
# --------------------------------------------------------------------------
def split_corpus(
    files: Sequence[Tuple[str, str]], val_ratio: float, seed: int
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """按文件稳定切分：排序后用固定种子抽样验证集，保证可复现。"""
    ordered = sorted(files, key=lambda kv: kv[0])
    rng = random.Random(seed)
    idx = list(range(len(ordered)))
    rng.shuffle(idx)
    n_val = max(1, round(len(ordered) * val_ratio)) if ordered else 0
    val_idx = set(idx[:n_val])
    train = [ordered[i] for i in range(len(ordered)) if i not in val_idx]
    val = [ordered[i] for i in sorted(val_idx)]
    return train, val


def pack_stream(
    sequences: Sequence[List[int]], seq_len: int, pad_id: int
) -> Tuple[List[List[int]], List[List[int]], List[List[float]]]:
    """多条 id 序列首尾相接，切成定长块；返回 (input_ids, targets, weights)。

    * 末块不足 ``seq_len`` 丢弃（残块偏向长样本结尾，见 ``data.py::pack_stream``）；
    * targets 相对 input 错位一位；pad 位权重 0，不进损失。
    """
    buf: List[int] = []
    for ids in sequences:
        buf.extend(ids)
    inp_rows: List[List[int]] = []
    tgt_rows: List[List[int]] = []
    w_rows: List[List[float]] = []
    for start in range(0, len(buf) - seq_len, seq_len):
        block = buf[start:start + seq_len + 1]
        if len(block) < seq_len + 1:
            break
        inp_rows.append(block[:-1])
        tgt_rows.append(block[1:])
        w_rows.append([1.0] * seq_len)
    return inp_rows, tgt_rows, w_rows


# --------------------------------------------------------------------------
# 训练配置与训练器
# --------------------------------------------------------------------------
@dataclass
class DemoTrainConfig:
    """demo 训练超参。默认值对齐 ``configs/demo-3m.json`` + 50 步。"""

    config: str = ""
    steps: int = 50
    seq_len: int = 512
    batch_size: int = 8
    grad_accum: int = 1
    base_lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup: int = 3          # warmup 5%：50 步 × 5% ≈ 3 步
    log_every: int = 5
    eval_every: int = 25
    seed: int = SPLIT_SEED
    light_root: str = os.environ.get("LIGHT_ROOT") or str(DEFAULT_LIGHT_ROOT)
    out_dir: str = str(DEFAULT_OUT)
    model: Dict[str, Any] = field(default_factory=dict)


def _load_model_config(config_path: Path) -> Dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {"vocab_size", "dim", "n_layers", "n_heads", "ffn_mult",
               "max_seq", "dropout", "qk_norm", "tie_embedding", "rope_theta"}
    return {k: v for k, v in raw.items() if k in allowed}


class DemoTrainer:
    """在词表 + 模型核上跑最小闭环：数据 → 训练 → 验证 → 存 checkpoint。"""

    def __init__(self, cfg: DemoTrainConfig) -> None:
        import torch

        from lightprism.model import LightPrismModel, PrismConfig

        self.torch = torch
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        self.vocab = load_demo_vocab(VOCAB_PATH)
        model_cfg = dict(cfg.model)
        model_cfg["vocab_size"] = self.vocab.vocab_size
        self.model_cfg = PrismConfig(**model_cfg)
        self.model = LightPrismModel(self.model_cfg)

        self.train_blocks: List[List[int]] = []
        self.val_blocks: List[List[int]] = []
        self.train_meta: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.step_count = 0
        self._prepared = False

    # ---- 数据装配 -------------------------------------------------------
    def prepare_data(self) -> Dict[str, Any]:
        """读语料 + SFT，分词、切分、打包。返回数据集统计。"""
        cfg = self.cfg
        files, skipped = read_corpus_files(Path(cfg.light_root))
        train_files, val_files = split_corpus(files, VAL_RATIO, cfg.seed)

        sft = render_sft_samples()

        def encode_corpus(rows: Sequence[Tuple[str, str]]) -> List[List[int]]:
            seqs: List[List[int]] = []
            for _, text in rows:
                ids = self.vocab.encode(text)
                if ids:
                    seqs.append(ids)
            return seqs

        train_seqs = encode_corpus(train_files)
        val_seqs = encode_corpus(val_files)
        # SFT 序列全部进训练集（小批量，提示对齐需要反复见）；验证集只用代码语料。
        sft_seqs = [self.vocab.encode(seq) for _, seq in sft]

        train_in, train_tgt, train_w = pack_stream(
            train_seqs + sft_seqs, cfg.seq_len, self.vocab.pad_id)
        val_in, val_tgt, val_w = pack_stream(val_seqs, cfg.seq_len, self.vocab.pad_id)
        self.train_blocks = train_in
        self.val_blocks = val_in

        train_tokens = sum(len(b) for b in train_in)
        val_tokens = sum(len(b) for b in val_in)
        self.train_meta = {
            "corpus_files": len(files),
            "corpus_files_skipped": len(skipped),
            "skipped_paths": skipped[:10],
            "train_files": len(train_files),
            "val_files": len(val_files),
            "sft_samples": len(sft),
            "train_blocks": len(train_in),
            "val_blocks": len(val_in),
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "seq_len": cfg.seq_len,
            "val_ratio": VAL_RATIO,
            "split_seed": cfg.seed,
        }
        self._prepared = True
        return self.train_meta

    def _batches(self) -> Iterator[Dict[str, List[List[int]]]]:
        import random

        rng = random.Random(self.cfg.seed)
        n = len(self.train_blocks)
        order: List[int] = []
        while True:
            if not order:
                order = list(range(n))
                rng.shuffle(order)
            idx = [order.pop() for _ in range(min(self.cfg.batch_size, len(order)))]
            if not idx:
                continue
            yield {
                "input_ids": [self.train_blocks[i] for i in idx],
                "targets": [self._target(self.train_blocks[i]) for i in idx],
                "loss_weights": [[1.0] * self.cfg.seq_len for _ in idx],
            }

    @staticmethod
    def _target(block: List[int]) -> List[int]:
        return block[1:] + [0]

    # ---- 训练 -----------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        torch = self.torch
        if not self._prepared:
            self.prepare_data()

        groups = self.model.param_groups(control_lr_mult=2.0)
        self.optim = torch.optim.AdamW(
            [{"params": [p for pg in groups for p in pg["params"]],
              "lr": self.cfg.base_lr, "weight_decay": self.cfg.weight_decay}],
            betas=(0.9, 0.95),
        )

        self.model.train()
        t0 = time.time()
        batches = self._batches()
        best_val: Optional[float] = None

        for step in range(self.cfg.steps):
            lr = cosine_lr(step, self.cfg.steps, self.cfg.base_lr, self.cfg.warmup)
            for group in self.optim.param_groups:
                group["lr"] = lr

            total_loss = 0.0
            for _ in range(self.cfg.grad_accum):
                batch = next(batches)
                input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
                targets = torch.tensor(batch["targets"], dtype=torch.long)
                weights = torch.tensor(batch["loss_weights"], dtype=torch.float32)
                _, loss = self.model(input_ids, targets, loss_weights=weights)
                (loss / self.cfg.grad_accum).backward()
                total_loss += float(loss.detach())

            grad_norm = self._grad_norm()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
            self.step_count += 1

            record: Dict[str, Any] = {
                "step": step + 1,
                "loss": round(total_loss / self.cfg.grad_accum, 5),
                "lr": round(lr, 8),
                "elapsed_s": round(time.time() - t0, 2),
                "grad_norm": round(grad_norm, 4),
            }

            if self.cfg.eval_every and (step + 1) % self.cfg.eval_every == 0:
                val = self.evaluate()
                record["val_loss"] = val["loss"]
                best_val = val["loss"] if best_val is None else min(best_val, val["loss"])

            if self.cfg.log_every and (step + 1) % self.cfg.log_every == 0:
                self.history.append(record)
                print(json.dumps(record, ensure_ascii=False))

        summary = {
            "steps": self.cfg.steps,
            "final_loss": self.history[-1]["loss"] if self.history else None,
            "best_val_loss": best_val,
            "params": self.model.count_parameters(),
            "params_non_embedding": self.model.count_parameters(non_embedding=True),
            "vocab_size": self.vocab.vocab_size,
            "elapsed_s": round(time.time() - t0, 2),
            "dataset": self.train_meta,
        }
        return summary

    def evaluate(self) -> Dict[str, Any]:
        torch = self.torch
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for block in self.val_blocks:
                ids = torch.tensor([block], dtype=torch.long)
                tgt = torch.tensor([self._target(block)], dtype=torch.long)
                w = torch.tensor([[1.0] * len(block)], dtype=torch.float32)
                _, loss = self.model(ids, tgt, loss_weights=w)
                total += float(loss)
                n += 1
        self.model.train()
        return {"loss": round(total / max(n, 1), 5), "blocks": n}

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum())
        return math.sqrt(total)

    # ---- 持久化 ---------------------------------------------------------
    def save(self, path: Optional[str] = None) -> Path:
        torch = self.torch
        out = Path(path or self.cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_config": asdict(self.model_cfg),
                "train_config": asdict(self.cfg),
                "tokenizer_layout": self.vocab.save_layout(),
                "history": self.history,
                "dataset": self.train_meta,
            },
            out / "model.pt",
        )
        (out / "train_log.json").write_text(
            json.dumps({"config": asdict(self.cfg), "history": self.history,
                        "dataset": self.train_meta}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out / "model.pt"


def cosine_lr(step: int, total: int, base_lr: float, warmup: int,
              min_ratio: float = 0.1) -> float:
    """线性 warmup + 余弦衰减（与 ``train.py`` 同口径）。"""
    if total <= 0:
        raise ValueError("total 必须为正")
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return base_lr * min_ratio
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LightPrism demo checkpoint 训练（T-0332）")
    p.add_argument("--config", default="configs/demo-3m.json", help="模型配置 JSON")
    p.add_argument("--data", help="预留：自定义训练 JSONL（默认走语料 + SFT）")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--light-root", default=str(DEFAULT_LIGHT_ROOT))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    model_cfg = _load_model_config(config_path) if config_path.is_file() else {}

    cfg = DemoTrainConfig(
        config=str(config_path),
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        base_lr=args.lr,
        warmup=args.warmup,
        log_every=args.log_every,
        eval_every=args.eval_every,
        seed=args.seed,
        light_root=args.light_root,
        out_dir=args.out_dir,
        model=model_cfg,
    )
    trainer = DemoTrainer(cfg)
    summary = trainer.train()
    ckpt = trainer.save()
    print(json.dumps({"summary": summary, "checkpoint": str(ckpt)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
