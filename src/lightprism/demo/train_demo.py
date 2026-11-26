# -*- coding: utf-8 -*-
"""T-0332 / T-0347：demo 3.4M checkpoint 训练管线。

## 这张卡解决什么

T-0330 产出 4096 实词表，T-0331 固定模型核（d=192 / L=6 / H=6），T-0333 产出
SFT 任务集。T-0332 把三者接成 50 步的最小闭环；T-0347 在它之上做三件事：

1. **统一数据目录** ``data/unified/``：词表、SFT、决策日志、对照语料、代码语料
   快照都放一处，训练与 ``train_openi.py`` 都从这里读。目录里有 ``manifest.json``
   记录计数、去重与 ``light ai check`` 通过率。
2. **合并真实对照语料与真标签**：``light/tools/ai_copilot/sft_dataset.jsonl``
   （Python↔光明平行对 / 光明工程体）每条都过一次 ``light ai check`` 取真值背书；
   决策 / 验证标签只用 ``data/runtime/decisions.jsonl``（T-0346）的真实运行日志。
3. **跑够步数**：lr schedule 按总步数重设（不再让 50 步余弦把大半训练压在低 lr），
   每步记 train loss，每 N 步记 val loss，并给出过拟合起始步。

## 数据口径（可复现，不编造）

* **代码语料**：``data/corpus`` 索引里 ``kept=true`` 的上游 ``.light`` 文件；优先读
  ``data/unified/corpus.jsonl`` 快照（跨机一致），没有快照再按 ``--light-root`` 读原文件。
* **SFT 任务集**：``json_tasks.jsonl`` + ``code_tasks.jsonl``，沿用
  ``<FORMAT>json|code</FORMAT> <TASK>…</TASK> <CODE>…</CODE> <V:PASS>`` 渲染。
* **翻译对照**：``<FORMAT>translate</FORMAT> <TASK>…</TASK> <CODE>Python</CODE>
  <CODE>光明</CODE> <V:PASS>``；工程体用 ``<FORMAT>light</FORMAT>``。
* **运行时决策**：``<FORMAT>runtime</FORMAT> <TASK>…</TASK>`` + 真实
  ``<D:…>`` 逐行 + ``<CODE>源码</CODE>`` + 真实 ``<V:…>``。

丢失权重（loss weight）：文本 token=1.0，结构 / 决策 / 验证 token 默认 **3.0**
（``--control-weight``）。SFT 与翻译样本按 ``--sft-repeat`` / ``--translate-repeat``
重复进流，真实决策序列按 ``--decision-repeat`` 重复，把控制 token 占比抬到 1% 量级。
报告里同时给 raw（去重后）与 weighted（进流后）两个口径。

## 用法

```
PYTHONPATH=src python -m lightprism.demo.train_demo --build-unified
PYTHONPATH=src python -m lightprism.demo.train_demo \
    --config configs/demo-3m.json --steps 3000 --smoke
```

## 诚实边界

* 决策 / 验证类别只保留日志里真出现过的组合，不硬凑 7 决策点 / 8 验证码。
* 小模型 / CPU 训练只承诺「决策 token 可学、翻译格式对齐」，不承诺端到端创作。
* ``--build-unified`` 调 ``light ai check`` 取真值；上游不在场时只跳过对照语料，
  绝不伪造通过记录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

#: 仓库根：src/lightprism/demo/train_demo.py → parents[3]
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
UNIFIED_DIR = DATA_DIR / "unified"
VOCAB_PATH = DATA_DIR / "vocab_4096.json"
CORPUS_INDEX = DATA_DIR / "corpus" / "index.jsonl"
CORPUS_DIR = DATA_DIR / "corpus"
SFT_JSON = DATA_DIR / "sft" / "json_tasks.jsonl"
SFT_CODE = DATA_DIR / "sft" / "code_tasks.jsonl"
SFT_ERR = DATA_DIR / "sft" / "err_tasks.jsonl"
DECISIONS_LOG = DATA_DIR / "runtime" / "decisions.jsonl"
DEFAULT_OUT = ROOT / "checkpoints" / "demo-3m"

#: 上游 light 仓默认位置。可用 --light-root / LIGHT_ROOT 覆盖。
DEFAULT_LIGHT_ROOT = Path(
    os.environ.get("LIGHT_ROOT")
    or str(ROOT.parent / "light")
)

#: 对照语料在 light 仓里的相对路径。
LIGHT_SFT_DATASET = Path("tools/ai_copilot/sft_dataset.jsonl")
LIGHT_CORPUS_HARNESS = Path("tools/ai_copilot/corpus_harness.jsonl")

#: 验证集占比（按文件切，不按 token 切）。
VAL_RATIO = 0.05
#: 切分种子固定：跨平台复现同一份 train/val。
SPLIT_SEED = 20261112

__all__ = [
    "DemoVocab",
    "load_demo_vocab",
    "read_corpus_files",
    "render_sft_samples",
    "render_translation_samples",
    "render_decision_samples",
    "render_unified_samples",
    "build_unified",
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
        #: 结构 / 决策 / 验证三类控制 token 的 id 集合（用于损失加权）。
        self.control_ids: frozenset[int] = frozenset(
            set(self.structure.values())
            | set(self.decisions.values())
            | set(self.validations.values())
        )
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


def resolve_vocab_path(data_dir: Optional[str | Path] = None) -> Path:
    """优先 ``data/unified/vocab_4096.json``，退回 ``data/vocab_4096.json``。"""
    if data_dir:
        cand = Path(data_dir) / "vocab_4096.json"
        if cand.is_file():
            return cand
    if (UNIFIED_DIR / "vocab_4096.json").is_file():
        return UNIFIED_DIR / "vocab_4096.json"
    return VOCAB_PATH


# --------------------------------------------------------------------------
# 语料读取
# --------------------------------------------------------------------------
def _read_text(path: Path) -> str:
    """UTF-8 容错读取 + 行尾归一化（与 vocab_build 同口径）。"""
    text = path.read_bytes().decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _light_roots(light_root: Optional[str | Path]) -> List[Path]:
    roots: List[Path] = []
    for cand in (
        Path(light_root) if light_root else None,
        DEFAULT_LIGHT_ROOT,
        ROOT.parent / "light",
        Path.home() / "dswork" / "light",
        Path("/home/skywalk/dswork/light"),
        Path("G:/dswork/duan-light-merge/light-merge"),
    ):
        if cand is not None and cand not in roots:
            roots.append(cand)
    return roots


def _resolve_light_root(light_root: Optional[str | Path] = None) -> Path:
    """在候选路径里挑第一个真的含 light 仓的目录。"""
    roots = _light_roots(light_root)
    for root in roots:
        if (root / LIGHT_SFT_DATASET).is_file() or (root / "cli" / "light.py").is_file():
            return root
    return roots[0]


def read_corpus_files(
    light_root: Path,
    corpus_dir: Path = CORPUS_DIR,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """按索引读回 kept 语料，返回 ``[(src_path, text), ...]`` 与跳过原因。

    优先从上游 light 仓按 ``src_path`` 读原始文件；上游不在场时退回
    ``corpus_dir`` 下的 ``flat/`` 与 ``.txt`` 副本。
    """
    if not CORPUS_INDEX.is_file():
        return [], []
    records = [json.loads(line) for line in CORPUS_INDEX.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [r for r in records if r.get("kept")]
    roots = _light_roots(light_root)

    def _open(rel: str) -> Optional[Path]:
        for root in roots:
            cand = root / rel
            if cand.is_file():
                return cand
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


def load_corpus(
    data_dir: Optional[Path] = None,
    light_root: Optional[str | Path] = None,
) -> Tuple[List[Tuple[str, str]], List[str], str]:
    """读代码语料：优先 ``data/unified/corpus.jsonl`` 快照，否则按索引读原文件。

    返回 ``(rows, skipped, source)``，``source`` 标明实际口径。
    """
    snapshot = (Path(data_dir) / "corpus.jsonl") if data_dir else None
    if snapshot is not None and snapshot.is_file():
        rows: List[Tuple[str, str]] = []
        for line in snapshot.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append((str(rec["src_path"]), str(rec["text"])))
        return rows, [], "unified_snapshot"
    rows, skipped = read_corpus_files(_resolve_light_root(light_root))
    return rows, skipped, "light_root"


# --------------------------------------------------------------------------
# 序列渲染
# --------------------------------------------------------------------------
def _render_code_sequence(task: str, code: str, validation: str = "PASS") -> str:
    """把一条 SFT 样本渲染成标准 ``<TASK>/<CODE>/<V:…>`` 序列文本。"""
    val = validation or "PASS"
    return (
        f"<TASK>{task.strip()}</TASK>\n"
        f"<CODE>\n{code.strip()}\n</CODE>\n"
        f"<V:{val}>\n"
    )


def _sft_paths(data_dir: Optional[Path] = None) -> List[Tuple[Path, str]]:
    base = Path(data_dir) if data_dir and (Path(data_dir) / "json_tasks.jsonl").is_file() else DATA_DIR / "sft"
    return [(base / "json_tasks.jsonl", "json"), (base / "code_tasks.jsonl", "code")]


def render_sft_samples(data_dir: Optional[str | Path] = None) -> List[Tuple[str, str]]:
    """读 SFT 的两份 JSONL，返回 ``(category, sequence_text)`` 列表。"""
    from lightprism.data import Sample

    from .sft_build import render_training_sequence

    rows: List[Tuple[str, str]] = []
    for path, default_fmt in _sft_paths(Path(data_dir) if data_dir else None):
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


def render_translation_sequence(rec: Dict[str, Any]) -> str:
    """渲染一条对照语料：``translate``（Python→光明）或 ``light``（工程体）。"""
    kind = str(rec.get("kind") or "pair")
    task = str(rec.get("instruction") or rec.get("note") or rec.get("category") or "").strip()
    if kind == "pair":
        py = str(rec.get("input") or "").strip()
        light = str(rec.get("output") or "").strip()
        return (
            "<FORMAT>translate</FORMAT>\n"
            f"<TASK>{task}</TASK>\n"
            f"<CODE>\n{py}\n</CODE>\n"
            f"<CODE>\n{light}\n</CODE>\n"
            "<V:PASS>\n"
        )
    light = str(rec.get("light") or rec.get("input") or rec.get("output") or "").strip()
    return (
        "<FORMAT>light</FORMAT>\n"
        f"<TASK>{task}</TASK>\n"
        f"<CODE>\n{light}\n</CODE>\n"
        "<V:PASS>\n"
    )


def render_translation_samples(data_dir: Optional[str | Path] = None) -> List[Tuple[str, str]]:
    """读 ``data/unified/translation_tasks.jsonl``，返回 ``(kind, sequence)``。"""
    path = Path(data_dir) / "translation_tasks.jsonl" if data_dir else UNIFIED_DIR / "translation_tasks.jsonl"
    if not path.is_file():
        return []
    rows: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append((str(rec.get("kind") or "pair"), render_translation_sequence(rec)))
    return rows


def render_decision_sequence(rec: Dict[str, Any]) -> str:
    """渲染一条真实运行时决策日志（<D:…> 只用日志里真出现的 token）。"""
    tokens = [str(t) for t in (rec.get("decision_tokens") or [])]
    task = str(rec.get("task_text") or rec.get("task_id") or "").strip()
    source = str(rec.get("source") or "").strip()
    val = str(rec.get("validation_token_text") or "").strip()
    if not val:
        code = str(rec.get("validation_code") or "PASS")
        val = f"<V:{code}>"
    body = "\n".join(tokens)
    return (
        "<FORMAT>runtime</FORMAT>\n"
        f"<TASK>{task}</TASK>\n"
        f"{body}\n"
        f"<CODE>\n{source}\n</CODE>\n"
        f"{val}\n"
    )


def render_decision_samples(data_dir: Optional[str | Path] = None) -> List[Tuple[str, str]]:
    """读 ``decisions.jsonl``（T-0346 真实日志），返回 ``(kind, sequence)``。"""
    path = Path(data_dir) / "decisions.jsonl" if data_dir else DECISIONS_LOG
    if not path.is_file():
        return []
    rows: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not rec.get("decision_tokens"):
            continue
        rows.append(("decision", render_decision_sequence(rec)))
    return rows


def render_err_sequences(data_dir: Optional[str | Path] = None) -> List[Tuple[str, str]]:
    """把 ``err_tasks.jsonl`` 渲染成「决策 → 代码 → 验证」序列（真值来自构造期日志）。"""
    path = Path(data_dir) / "err_tasks.jsonl" if data_dir else SFT_ERR
    if not path.is_file():
        return []
    rows: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            step = (rec.get("steps") or [])[0]
        except (json.JSONDecodeError, IndexError, TypeError):
            continue
        decisions = step.get("decisions") or {}
        tokens = [f"<D:{k}={v}>" for k, v in sorted(decisions.items())]
        val = str(step.get("validation") or "PASS")
        text = (
            "<FORMAT>err</FORMAT>\n"
            f"<TASK>{str(rec.get('task') or '').strip()}</TASK>\n"
            f"{chr(10).join(tokens)}\n"
            f"<CODE>\n{str(step.get('code') or '').strip()}\n</CODE>\n"
            f"<V:{val}>\n"
        )
        rows.append(("err", text))
    return rows


def render_unified_samples(data_dir: Optional[str | Path] = None) -> List[Tuple[str, str]]:
    """汇总所有序列来源，返回 ``(kind, sequence_text)``。"""
    rows: List[Tuple[str, str]] = []
    rows.extend((("sft", seq) for _, seq in render_sft_samples(data_dir)))
    rows.extend(render_translation_samples(data_dir))
    rows.extend(render_decision_samples(data_dir))
    rows.extend(render_err_sequences(data_dir))
    return rows


# --------------------------------------------------------------------------
# 统一数据目录构建
# --------------------------------------------------------------------------
def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha(payload: bytes) -> str:
    """内容 sha256：先 CRLF→LF 归一，跨平台检出算同一个值。

    只用于 manifest 的 ``sha256`` 字段；上游候选去重 / 缓存键仍用
    :func:`_sha256_bytes` 的原始字节，避免改缓存键导致全量重跑 light check。
    """
    return hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()


def _count_lines(path: Path) -> int:
    """非空行数；文件不存在返回 0。"""
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    """宽容读取 JSONL（坏行跳过），只用于 manifest 统计。"""
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_jsonl_stable(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """同 :func:`_write_jsonl`，但内容（CRLF 归一后）没变就不动文件。

    重建统一目录时 corpus / translation 的「内容」通常不变，直接重写只会产生
    几千行「全删全加」的行尾噪声 diff。内容不变即保留原文件字节。
    """
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.is_file():
        old = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if old == text.replace("\r\n", "\n"):
            return sum(1 for line in text.splitlines() if line.strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sum(1 for line in text.splitlines() if line.strip())


def _light_ai_check(light_root: Path, code: str, timeout: int = 30) -> Dict[str, Any]:
    """跑一次 ``light ai check --json --no-corpus``（语法级），返回真值证据。

    只做语法检查（不 ``--run``）：1640 条样本若逐条运行，单机要数小时；
    语法检查 1–2 s/条，且 ``ok`` 语义与运行时 checker 前置一致。
    """
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".light", delete=False, encoding="utf-8") as fh:
            fh.write(code.rstrip("\n") + "\n")
            tmp = Path(fh.name)
        proc = subprocess.run(
            [sys.executable, "-m", "cli.light", "ai", "check", "--json",
             "--no-corpus", "--timeout", str(timeout), str(tmp)],
            cwd=str(light_root), capture_output=True, text=True, timeout=timeout + 30,
        )
        payload: Dict[str, Any] = {}
        for line in reversed((proc.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = {}
                break
        return {
            "ok": bool(payload.get("ok")),
            "mode": "syntax",
            "diagnostics": payload.get("diagnostics") or [],
            "duration_ms": payload.get("duration_ms"),
            "rc": proc.returncode,
        }
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - 环境缺 light
        return {"ok": False, "mode": "syntax", "error": repr(exc)}
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def _collect_translation_rows(light_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """读对照语料并去重（按光明代码 sha256），返回候选行与计数。"""
    stats = {"sft_rows": 0, "harness_rows": 0, "pairs": 0, "light_only": 0,
             "dup_dropped": 0, "missing": 0}
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add(rec: Dict[str, Any], code: str, kind: str) -> None:
        digest = _sha256_bytes(code.encode("utf-8"))
        if digest in seen:
            stats["dup_dropped"] += 1
            return
        seen.add(digest)
        rec["kind"] = kind
        rec["sha256"] = digest
        rows.append(rec)
        stats["pairs" if kind == "pair" else "light_only"] += 1

    sft_path = light_root / LIGHT_SFT_DATASET
    if sft_path.is_file():
        for line in sft_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["sft_rows"] += 1
            out = str(rec.get("output") or "").strip()
            if out:
                _add({"source": "sft_dataset", "instruction": rec.get("instruction"),
                      "input": rec.get("input"), "output": out,
                      "category": rec.get("category")}, out, "pair")
            elif str(rec.get("input") or rec.get("light") or "").strip():
                body = str(rec.get("input") or rec.get("light")).strip()
                _add({"source": "sft_dataset", "note": rec.get("instruction"),
                      "light": body, "category": rec.get("category")}, body, "light_only")
            else:
                stats["missing"] += 1
    else:
        stats["missing"] += 1

    harness_path = light_root / LIGHT_CORPUS_HARNESS
    if harness_path.is_file():
        for line in harness_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["harness_rows"] += 1
            body = str(rec.get("light") or "").strip()
            if not body:
                stats["missing"] += 1
                continue
            _add({"source": "corpus_harness", "note": rec.get("note"),
                  "light": body, "category": rec.get("category"),
                  "upstream": rec.get("source")}, body, "light_only")
    return rows, stats


def build_unified(
    light_root: Optional[str | Path] = None,
    out_dir: Path = UNIFIED_DIR,
    jobs: int = 8,
    skip_check: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """生成 ``data/unified/``：词表 / SFT / 决策 / 对照语料 / 代码语料快照 + manifest。"""
    say = progress or (lambda msg: print(msg, flush=True))
    from .sft_build import apply_segment_markers, render_translation_record
    out_dir.mkdir(parents=True, exist_ok=True)
    root = _resolve_light_root(light_root)

    # 1) 词表与 SFT / 决策日志快照。
    #    ``keep`` 的两项（vocab / err_tasks）在 unified 侧被后续卡补强过
    #    （T-0371 加了 <PY>/<LIGHT>，T-0370 把 err 补到 25），而 data/ 下的
    #    源还是旧基线。直接覆盖等于把别人的成果回滚，因此**目标已存在就保留**，
    #    只有缺失时才从源重建（词表重建后再补段标记，幂等）。
    copies = [
        (VOCAB_PATH, out_dir / "vocab_4096.json", "keep"),
        (SFT_JSON, out_dir / "json_tasks.jsonl", "copy"),
        (SFT_CODE, out_dir / "code_tasks.jsonl", "copy"),
        (SFT_ERR, out_dir / "err_tasks.jsonl", "keep"),
        (DECISIONS_LOG, out_dir / "decisions.jsonl", "copy"),
    ]
    file_hashes: Dict[str, str] = {}
    for src, dst, mode in copies:
        if mode == "keep" and dst.is_file():
            file_hashes[dst.name] = _content_sha(dst.read_bytes())
            continue
        if not src.is_file():
            continue
        payload = src.read_bytes()
        dst.write_bytes(payload)
        if dst.name == "vocab_4096.json":
            apply_segment_markers(dst)  # 缺失时重建：补 T-0371 的段标记
            payload = dst.read_bytes()
        file_hashes[dst.name] = _content_sha(payload)
    say(f"[unified] 拷贝 {len(file_hashes)} 个既有产物 → {out_dir}")

    # 2) 代码语料快照（跨机一致；shot 内不带 checkpoint）。
    corpus_rows, skipped, corpus_source = load_corpus(None, root)
    if corpus_rows:
        corpus_records = [{"src_path": rel, "text": text} for rel, text in corpus_rows]
        n = _write_jsonl_stable(out_dir / "corpus.jsonl", corpus_records)
        say(f"[unified] 代码语料快照 {n} 文件（source={corpus_source}，跳过 {len(skipped)}）")

    # 3) 对照语料 + light ai check 真值背书（带缓存，可中断续跑）。
    candidates, cand_stats = _collect_translation_rows(root)
    cache_path = out_dir / ".light_check_cache.json"
    cache: Dict[str, Any] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    checked = 0
    if candidates and not skip_check:
        todo = [rec for rec in candidates if rec["sha256"] not in cache]
        for rec in candidates:
            if rec["sha256"] in cache:
                rec["light_check"] = cache[rec["sha256"]]

        def _run(rec: Dict[str, Any]) -> Dict[str, Any]:
            code = rec.get("output") if rec.get("kind") == "pair" else rec.get("light")
            rec["light_check"] = _light_ai_check(root, str(code or ""))
            return rec

        if todo:
            say(f"[unified] light ai check {len(todo)} 条（{max(1, jobs)} 并发，语法级）…")
            t0 = time.time()
            done = 0
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                for rec in pool.map(_run, todo):
                    cache[rec["sha256"]] = rec["light_check"]
                    done += 1
                    if done % 200 == 0:
                        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                        say(f"[unified] ai check {done}/{len(todo)}，已用 {time.time() - t0:.0f}s")
            checked = len(todo)
            say(f"[unified] ai check 完成，{checked} 条，用时 {time.time() - t0:.0f}s")
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        else:
            say(f"[unified] 复用 light ai check 缓存 {len(candidates)} 条")
    elif candidates:
        for rec in candidates:
            rec["light_check"] = {"ok": None, "mode": "skipped"}
    passed = candidates if skip_check else [
        r for r in candidates if (r.get("light_check") or {}).get("ok") is True]
    # 确定性顺序：重跑 `--build-unified` 必须字节稳定，否则同一份数据每次都
    # 产生 1394 行「全删全加」的噪声 diff（并发收集顺序不稳定导致）。
    passed = sorted(passed, key=lambda r: (
        str(r.get("kind") or ""), str(r.get("sha256") or ""),
        str(r.get("instruction") or "")))
    # 段标记序列（<PY> → <LIGHT>）是 T-0371 的训练格式，本文件是唯一事实源：
    # 每行必须自带 sequence_marked，否则下游与 test_sft_data 会判「缺段格式」。
    for rec in passed:
        rec["sequence_marked"] = render_translation_record(rec, mode="markers")
    n_trans = _write_jsonl_stable(out_dir / "translation_tasks.jsonl", passed)
    file_hashes["translation_tasks.jsonl"] = _content_sha(
        (out_dir / "translation_tasks.jsonl").read_bytes())

    by_kind: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    for rec in passed:
        by_kind[rec["kind"]] = by_kind.get(rec["kind"], 0) + 1
        cat = str(rec.get("category") or "")
        by_category[cat] = by_category.get(cat, 0) + 1

    # manifest 的计数与哈希一律从落盘文件复算。硬编码（旧版 errors=8、vocab=2900）
    # 会在上游补数后立刻说谎：T-0370 把 err 补到 25、T-0371 把词表补到 2902 之后，
    # 重跑一次就会把这两处「改回旧值」，并把 unified 里的新产物覆盖掉。
    # 保留上一版 manifest 里由其它卡补写的字段（T-0371 的 sequence_formats 等），
    # 本函数只覆盖自己掌握真值的段，避免一次「重建」把别人的注脚冲掉。
    manifest_path = out_dir / "manifest.json"
    prev_manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            prev_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev_manifest = {}
    prev_trans_build = dict(prev_manifest.get("translation_build") or {})
    prev_trans_build.update(cand_stats)

    vocab_file = out_dir / "vocab_4096.json"
    vocab_doc = (json.loads(vocab_file.read_text(encoding="utf-8"))
                 if vocab_file.is_file() else {})
    layout = vocab_doc.get("control_layout") or {}
    err_rows = _read_jsonl_rows(out_dir / "err_tasks.jsonl")
    err_split = {"train": 0, "test": 0}
    for row in err_rows:
        key = str((row.get("meta") or {}).get("split") or "train")
        err_split[key] = err_split.get(key, 0) + 1

    runtime_dir = DATA_DIR / "runtime"
    runtime_decisions = runtime_dir / "decisions.jsonl"
    runtime_tasks = runtime_dir / "tasks.jsonl"
    unified_decisions = out_dir / "decisions.jsonl"
    dec_lines = _count_lines(unified_decisions)
    rt_lines = _count_lines(runtime_decisions)
    dec_sha = _content_sha(unified_decisions.read_bytes()) \
        if unified_decisions.is_file() else None
    rt_sha = _content_sha(runtime_decisions.read_bytes()) \
        if runtime_decisions.is_file() else None
    aligned = bool(dec_sha and rt_sha and dec_sha == rt_sha)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "task": "T-0493",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "vocab": {
            "file": "vocab_4096.json",
            "size": vocab_doc.get("size"),
            "control_span": [layout.get("start"), layout.get("end")],
            "segment_markers": dict((vocab_doc.get("segment_markers") or {}).get("ids") or {}),
            "note": "demo/train_openi 共用本副本；train_openi 侧 2902 口径见其 manifest",
        },
        "counts": {
            "json_tasks": _count_lines(out_dir / "json_tasks.jsonl"),
            "code_tasks": _count_lines(out_dir / "code_tasks.jsonl"),
            "err_tasks": len(err_rows),
            "err_tasks_split": err_split,
            "decisions": len(render_decision_samples(out_dir)),
            "corpus_files": len(corpus_rows),
            "translation_rows": n_trans,
            "translation_by_kind": by_kind,
            "translation_candidates": len(candidates),
            "translation_dropped_by_check": len(candidates) - len(passed),
        },
        "translation_build": prev_trans_build,
        "light_ai_check": {
            "command": "python -m cli.light ai check --json --no-corpus --timeout 30 <file>",
            "mode": "syntax",
            "checked": checked,
            "passed": len(passed),
            "pass_rate": round(len(passed) / checked, 4) if checked else None,
        },
        "corpus_source": corpus_source,
        "corpus_skipped": len(skipped),
        "runtime": {
            "tasks": {
                "file": "data/runtime/tasks.jsonl",
                "lines": _count_lines(runtime_tasks),
                "sha256": _content_sha(runtime_tasks.read_bytes())
                if runtime_tasks.is_file() else None,
            },
            "decisions": {
                "file": "data/runtime/decisions.jsonl",
                "lines": rt_lines,
                "sha256": rt_sha,
            },
            "note": "runtime 决策日志是唯一事实源；unified 副本由本函数做字节级快照",
        },
        "decisions_snapshot": {
            "file": "data/unified/decisions.jsonl",
            "lines": dec_lines,
            "sha256": dec_sha,
            "note": "已与 runtime 对齐（T-0493 重建）" if aligned
                    else "未与 runtime 对齐：需重跑 --build-unified",
        },
        "sha256": file_hashes,
    }
    manifest = {**prev_manifest, **manifest}
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    say(f"[unified] 完成：translation {n_trans} 条，manifest 已写")
    return manifest


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


def encode_with_weights(
    vocab: DemoVocab, text: str, control_weight: float = 3.0
) -> Tuple[List[int], List[float]]:
    """编码一条序列，控制 token（结构/决策/验证）按 ``control_weight`` 加权。"""
    ids = vocab.encode(text)
    weights = [control_weight if i in vocab.control_ids else 1.0 for i in ids]
    return ids, weights


def pack_stream(
    sequences: Sequence[Any],
    seq_len: int,
    pad_id: int,
    control_ids: Optional[Iterable[int]] = None,
    control_weight: float = 1.0,
) -> Tuple[List[List[int]], List[List[int]], List[List[float]]]:
    """多条 id 序列首尾相接，切成定长块；返回 (input_ids, targets, weights)。

    ``sequences`` 的元素可以是 ``List[int]``，也可以是 ``(ids, weights)`` 元组
    （后者保留控制 token 权重）。``control_ids`` + ``control_weight`` 用于
    ``List[int]`` 形式的补加权（缺省不加权）。

    * 末块不足 ``seq_len`` 丢弃（残块偏向长样本结尾，见 ``data.py::pack_stream``）；
    * targets 相对 input 错位一位；pad 位权重 0，不进损失。
    """
    ctrl = set(control_ids or ())
    buf: List[int] = []
    wbuf: List[float] = []
    for seq in sequences:
        if isinstance(seq, tuple):
            ids, weights = seq
        else:
            ids = list(seq)
            weights = [control_weight if i in ctrl else 1.0 for i in ids]
        buf.extend(ids)
        wbuf.extend(weights)

    inp_rows: List[List[int]] = []
    tgt_rows: List[List[int]] = []
    w_rows: List[List[float]] = []
    for start in range(0, len(buf) - seq_len, seq_len):
        block = buf[start:start + seq_len + 1]
        if len(block) < seq_len + 1:
            break
        inp_rows.append(block[:-1])
        tgt_rows.append(block[1:])
        w_rows.append([1.0 if v == pad_id else w for v, w in zip(block[1:], wbuf[start + 1:start + seq_len + 1])])
    return inp_rows, tgt_rows, w_rows


# --------------------------------------------------------------------------
# 训练配置与训练器
# --------------------------------------------------------------------------
@dataclass
class DemoTrainConfig:
    """demo 训练超参。默认值对齐 ``configs/demo-3m.json``。"""

    config: str = ""
    steps: int = 50
    seq_len: int = 512
    batch_size: int = 8
    grad_accum: int = 1
    base_lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup: int = 0            # 0 = 按 steps 自动取 2%（最少 3 步）
    log_every: int = 50
    eval_every: int = 100
    seed: int = SPLIT_SEED
    light_root: str = os.environ.get("LIGHT_ROOT") or str(DEFAULT_LIGHT_ROOT)
    data_dir: str = ""
    out_dir: str = str(DEFAULT_OUT)
    control_weight: float = 3.0
    sft_repeat: int = 2
    translate_repeat: int = 1
    decision_repeat: int = 30
    err_repeat: int = 5
    corpus_limit: int = 0      # 0 = 全量；测试用
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

        self.vocab = load_demo_vocab(resolve_vocab_path(cfg.data_dir or None))
        model_cfg = dict(cfg.model)
        model_cfg["vocab_size"] = self.vocab.vocab_size
        self.model_cfg = PrismConfig(**model_cfg)
        self.model = LightPrismModel(self.model_cfg)

        self.train_blocks: List[List[int]] = []
        self.train_weights: List[List[float]] = []
        self.val_blocks: List[List[int]] = []
        self.train_meta: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.step_count = 0
        self._prepared = False

    # ---- 数据装配 -------------------------------------------------------
    def _repeats(self) -> Dict[str, int]:
        cfg = self.cfg
        return {
            "sft": max(1, cfg.sft_repeat),
            "pair": max(1, cfg.translate_repeat),
            "light_only": max(1, cfg.translate_repeat),
            "decision": max(0, cfg.decision_repeat),
            "err": max(1, cfg.err_repeat),
        }

    def prepare_data(self) -> Dict[str, Any]:
        """读统一目录 + 代码语料，分词、加权、切分、打包。返回数据集统计。"""
        cfg = self.cfg
        data_dir = Path(cfg.data_dir) if cfg.data_dir else None
        if data_dir is None and UNIFIED_DIR.is_dir():
            data_dir = UNIFIED_DIR
        files, skipped, corpus_source = load_corpus(data_dir, cfg.light_root)
        if cfg.corpus_limit:
            files = files[: cfg.corpus_limit]
        train_files, val_files = split_corpus(files, VAL_RATIO, cfg.seed)

        # 所有非语料序列：按 kind 重复进流；权重按控制 token 加权。
        samples = render_unified_samples(data_dir)
        reps = self._repeats()
        weighted: List[Tuple[List[int], List[float]]] = []
        kind_counts: Dict[str, int] = {}
        token_stats = {"tokens": 0, "control": 0, "decision": 0,
                       "validation": 0, "unk": 0, "repeats": {}}
        raw_stats = {"tokens": 0, "control": 0, "decision": 0, "validation": 0}
        for kind, text in samples:
            ids, weights = encode_with_weights(self.vocab, text, cfg.control_weight)
            if not ids:
                continue
            raw_stats["tokens"] += len(ids)
            for tid in ids:
                k = self.vocab.kind_of(tid)
                if k in ("structure", "validation", "decision"):
                    raw_stats["control"] += 1
                if k == "decision":
                    raw_stats["decision"] += 1
                if k == "validation":
                    raw_stats["validation"] += 1
            rep = reps.get(kind, 1)
            if rep <= 0:
                continue
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            token_stats["repeats"][kind] = token_stats["repeats"].get(kind, 0) + rep
            token_stats["tokens"] += len(ids) * rep
            for tid in ids:
                k = self.vocab.kind_of(tid)
                if k in ("structure", "validation", "decision"):
                    token_stats["control"] += rep
                if k == "decision":
                    token_stats["decision"] += rep
                if k == "validation":
                    token_stats["validation"] += rep
                if tid == self.vocab.unk_id:
                    token_stats["unk"] += rep
            weighted.extend([(ids, weights)] * rep)

        def _encode_files(rows: Sequence[Tuple[str, str]]) -> List[Tuple[List[int], List[float]]]:
            out: List[Tuple[List[int], List[float]]] = []
            for _, text in rows:
                ids = self.vocab.encode(text)
                if ids:
                    out.append((ids, [1.0] * len(ids)))
            return out

        train_seqs = _encode_files(train_files)
        val_seqs = _encode_files(val_files)
        # 语料缺失时用非语料序列的尾部做验证，避免 val_blocks=0 导致 val loss 恒 0。
        if not val_seqs:
            val_seqs, weighted = weighted[-32:], weighted[:-32] or weighted[:1]

        train_in, train_tgt, train_w = pack_stream(train_seqs + weighted, cfg.seq_len, self.vocab.pad_id)
        val_in, val_tgt, val_w = pack_stream(val_seqs, cfg.seq_len, self.vocab.pad_id)
        self.train_blocks = train_in
        self.train_weights = train_w
        self.val_blocks = val_in

        corpus_tokens = sum(len(ids) for ids, _ in train_seqs)
        total_tokens = sum(len(ids) for ids, _ in train_seqs) + sum(len(ids) for ids, _ in weighted)
        total = max(total_tokens, 1)
        raw_total = max(corpus_tokens + raw_stats["tokens"], 1)
        self.train_meta = {
            "data_dir": str(data_dir) if data_dir else str(DATA_DIR / "sft"),
            "corpus_source": corpus_source,
            "corpus_files": len(files),
            "corpus_files_skipped": len(skipped),
            "skipped_paths": skipped[:10],
            "train_files": len(train_files),
            "val_files": len(val_files),
            "sample_counts": kind_counts,
            "sample_repeats": token_stats["repeats"],
            "corpus_tokens": corpus_tokens,
            "sample_tokens": total_tokens - corpus_tokens,
            "train_blocks": len(train_in),
            "val_blocks": len(val_in),
            "train_tokens": sum(len(b) for b in train_in),
            "val_tokens": sum(len(b) for b in val_in),
            "control_tokens": token_stats["control"],
            "decision_tokens": token_stats["decision"],
            "validation_tokens": token_stats["validation"],
            "unk_tokens": token_stats["unk"],
            "control_ratio": round(token_stats["control"] / total, 6),
            "decision_ratio": round(token_stats["decision"] / total, 6),
            "unk_ratio": round(token_stats["unk"] / total, 6),
            "control_weight": cfg.control_weight,
            "raw_sample_tokens": raw_stats["tokens"],
            "raw_total_tokens": raw_total,
            "raw_control_tokens": raw_stats["control"],
            "raw_decision_tokens": raw_stats["decision"],
            "raw_validation_tokens": raw_stats["validation"],
            "raw_control_ratio": round(raw_stats["control"] / raw_total, 6),
            "raw_decision_ratio": round(raw_stats["decision"] / raw_total, 6),
            "seq_len": cfg.seq_len,
            "val_ratio": VAL_RATIO,
            "split_seed": cfg.seed,
        }
        self._prepared = True
        return self.train_meta

    def _batches(self) -> Iterator[Dict[str, List[List[Any]]]]:
        rng = random.Random(self.cfg.seed)
        n = len(self.train_blocks)
        order: List[int] = []
        while True:
            if not order:
                order = list(range(n))
                rng.shuffle(order)
            take = min(self.cfg.batch_size, len(order))
            idx = [order.pop() for _ in range(take)]
            if not idx:
                continue
            yield {
                "input_ids": [self.train_blocks[i] for i in idx],
                "targets": [self._target(self.train_blocks[i]) for i in idx],
                "loss_weights": [self.train_weights[i] for i in idx],
            }

    @staticmethod
    def _target(block: List[int]) -> List[int]:
        return block[1:] + [0]

    # ---- 训练 -----------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        torch = self.torch
        if not self._prepared:
            self.prepare_data()

        cfg = self.cfg
        warmup = cfg.warmup if cfg.warmup > 0 else max(3, cfg.steps // 50)
        groups = self.model.param_groups(control_lr_mult=2.0)
        self.optim = torch.optim.AdamW(
            [{"params": [p for pg in groups for p in pg["params"]],
              "lr": cfg.base_lr, "weight_decay": cfg.weight_decay}],
            betas=(0.9, 0.95),
        )

        self.model.train()
        t0 = time.time()
        batches = self._batches()
        best_val: Optional[float] = None
        best_step = 0
        val_at_best = 0.0
        val_history: List[Dict[str, Any]] = []
        overfit_step: Optional[int] = None

        for step in range(cfg.steps):
            lr = cosine_lr(step, cfg.steps, cfg.base_lr, warmup)
            for group in self.optim.param_groups:
                group["lr"] = lr

            total_loss = 0.0
            for _ in range(cfg.grad_accum):
                batch = next(batches)
                input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
                targets = torch.tensor(batch["targets"], dtype=torch.long)
                weights = torch.tensor(batch["loss_weights"], dtype=torch.float32)
                _, loss = self.model(input_ids, targets, loss_weights=weights)
                (loss / cfg.grad_accum).backward()
                total_loss += float(loss.detach())

            grad_norm = self._grad_norm()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
            self.step_count += 1

            record: Dict[str, Any] = {
                "step": step + 1,
                "loss": round(total_loss / cfg.grad_accum, 5),
                "lr": round(lr, 8),
                "elapsed_s": round(time.time() - t0, 2),
                "grad_norm": round(grad_norm, 4),
            }

            if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
                val = self.evaluate()
                record["val_loss"] = val["loss"]
                val_history.append({"step": step + 1, "val_loss": val["loss"]})
                if best_val is None or val["loss"] < best_val:
                    best_val = val["loss"]
                    best_step = step + 1
                    val_at_best = val["loss"]
                elif overfit_step is None and val["loss"] > val_at_best * 1.01:
                    # 相对历史最优回升 >1%：记为过拟合起点（只记第一次）。
                    overfit_step = step + 1

            self.history.append(record)
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                print(json.dumps(record, ensure_ascii=False), flush=True)

        summary = {
            "steps": cfg.steps,
            "final_loss": self.history[-1]["loss"] if self.history else None,
            "first_loss": self.history[0]["loss"] if self.history else None,
            "best_val_loss": best_val,
            "best_val_step": best_step if best_val is not None else None,
            "overfit_step": overfit_step,
            "val_history": val_history,
            "tokens_seen": cfg.steps * cfg.batch_size * cfg.seq_len,
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
    p = argparse.ArgumentParser(description="LightPrism demo checkpoint 训练（T-0332 / T-0347）")
    p.add_argument("--config", default="configs/demo-3m.json", help="模型配置 JSON")
    p.add_argument("--data", help="预留：自定义训练 JSONL（默认走统一目录）")
    p.add_argument("--data-dir", default="", help="统一数据目录（默认 data/unified）")
    p.add_argument("--build-unified", action="store_true", help="重建 data/unified 后再训练")
    p.add_argument("--build-only", action="store_true", help="只重建 data/unified，不训练")
    p.add_argument("--check-jobs", type=int, default=8, help="light ai check 并发数")
    p.add_argument("--skip-light-check", action="store_true", help="跳过 light ai check（不推荐）")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=0, help="0 = 自动 2%（最少 3 步）")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--control-weight", type=float, default=3.0)
    p.add_argument("--sft-repeat", type=int, default=2)
    p.add_argument("--translate-repeat", type=int, default=1)
    p.add_argument("--decision-repeat", type=int, default=30)
    p.add_argument("--err-repeat", type=int, default=5)
    p.add_argument("--corpus-limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--light-root", default=str(DEFAULT_LIGHT_ROOT))
    p.add_argument("--smoke", action="store_true", help="训练后跑 32 token 冒烟生成")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)

    if args.build_unified:
        build_unified(light_root=args.light_root, jobs=args.check_jobs,
                      skip_check=args.skip_light_check)
        if args.build_only:
            return 0

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
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        control_weight=args.control_weight,
        sft_repeat=args.sft_repeat,
        translate_repeat=args.translate_repeat,
        decision_repeat=args.decision_repeat,
        err_repeat=args.err_repeat,
        corpus_limit=args.corpus_limit,
        model=model_cfg,
    )
    trainer = DemoTrainer(cfg)
    summary = trainer.train()
    ckpt = trainer.save()
    result: Dict[str, Any] = {"summary": summary, "checkpoint": str(ckpt)}
    if args.smoke:
        from .smoke import run_smoke

        result["smoke"] = run_smoke(ckpt, max_new_tokens=32).as_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
