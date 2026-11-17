# -*- coding: utf-8 -*-
"""Demo 4096 实词表：语料清洗、词表构建与 ``--check`` 校验（纯标准库）。

设计口径（``docs/Demo开发计划.md`` §3、任务卡 T-0330）
------------------------------------------------------
64K 是目标形态，demo 只用 4096：``65536×192=12.6M`` 的嵌入层比整个 demo 模型还大，
3MB 语料也填不满。本模块**不训练 BPE**，只从 light 仓的 ``.light`` 语料统计出
token 清单，并满足四条硬约束：

1. 总数 ≤ 4096，且**留 ≥15% 余量**（不把词表塞满）；
2. L0 44 字各占 1 个 token，``check_l0_atomicity()`` 通过 —— 因此任何多字 token
   都**不能含 L0 核心字**（构建时强制过滤）；
3. 控制 token 由 :class:`~lightprism.tokens.TokenSpace` 生成，顺序与 id 布局
   和 :class:`~lightprism.tokenizer.ControlTokenLayout` 完全一致，不另造一套；
4. 字符级 UNK 率 < 1%（对全部唯一 md5 语料文件统计，口径见 ``corpus_stats``）。

交付环境没有原始语料（``data/corpus/*.txt`` 被 gitignore），所以 ``--check`` 是
**自包含**的：字符频次固化在词表 JSON 的 ``corpus_stats.char_freq`` 里，来源与
文件数可追溯；若原始语料在场，会额外实时重算 kept 子集做交叉参考。

用法::

    PYTHONPATH=src python -m lightprism.demo.vocab_build --build \
        --light-root /path/to/light --output data/vocab_4096.json
    PYTHONPATH=src python -m lightprism.demo.vocab_build --check data/vocab_4096.json

64K 实词表（T-0373）走同一模块的 ``--build-64k`` / ``--check-64k`` 分支：
按 ``src/lightprism/vocab.py::DEFAULT_BUDGET`` 的八个槽位逐槽填满 65536 条，
产出 ``data/vocab_64k.json`` + ``data/vocab_64k.manifest.json``（逐槽 sha256 与预算对账）::

    PYTHONPATH=src python -m lightprism.demo.vocab_build --build-64k \
        --light-root /path/to/light --output data/vocab_64k.json
    PYTHONPATH=src python -m lightprism.demo.vocab_build --check-64k data/vocab_64k.json

``--train-smoke`` 用 64K 词表跑 20 步真训练 + 生成 + ``light ai check``，
``--baseline-config configs/demo-3m.json`` 同时跑一份同 seeds/steps 的 4096 对照。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..tokens import LIGHT_L0_CORE, VALIDATION_CODES, TokenSpace
from ..tokenizer import STRUCTURE_MARKERS, ControlTokenLayout
from ..vocab import (
    DEFAULT_BUDGET,
    SEGMENT_MARKERS,
    SEGMENT_TOKEN_IDS,
    VocabBudget,
    check_l0_atomicity,
    segment_token_ids,
    slot_offsets,
)

# ---- 预算与判据 ---------------------------------------------------------
#: demo 词表硬上限（嵌入层尺寸）。
VOCAB_LIMIT: int = 4096
#: 至少留 15% 余量给后续新增 token，不把 4096 塞满。
HEADROOM_RATIO: float = 0.15
#: 实际允许的最大条数（4096 × 0.85 = 3481）。
MAX_SIZE: int = int(VOCAB_LIMIT * (1 - HEADROOM_RATIO))
#: 字符级 UNK 率硬上限。
UNK_LIMIT: float = 0.01
#: 单字 token 按频次累计覆盖到 99.8% 即停（留足 UNK 余量，不硬凑字符表）。
CHAR_COVERAGE_TARGET: float = 0.998

LAYER_NAMES: Tuple[str, ...] = ("l0", "syntax", "stdlib", "ascii", "control")
SCHEMA_VERSION = "0.1"

# 各层配额上限：只从语料里真实出现过的候选中取，取不满就少放，不硬凑。
_QUOTA_MULTI = 512          # syntax 层的高频多字组合
_QUOTA_STDLIB = 768         # stdlib 层的模块名/导入名
_QUOTA_ASCII_FRAGMENT = 900  # ascii 层的标识符片段
_MAX_SINGLE = 1600          # 通用单字 token 的安全上限

_L0_SET = frozenset(LIGHT_L0_CORE)
_ASCII_SINGLES: Tuple[str, ...] = tuple(chr(i) for i in range(32, 127))  # 95 个可打印 ASCII
_ASCII_SET = frozenset(_ASCII_SINGLES)

# ---- 语料清洗规则 -------------------------------------------------------
#: 路径任一段命中即剔除：测试目录、一次性/草稿目录、自举工程目录。
#: 注意按**目录段**匹配而不是文件名，才能保住 ``examples/test_hello.light`` 这类示例。
_DROP_DIR_NAMES = frozenset({
    "tests", "test", "lighttests", "archive",
    "bootstrap",
    "testproj2", "test_project", "test_project_v2",
    "_r159_scratch", "_scratch_r167t4",
    "demo", "demos", "playground", "tmp", "scratch",
})


def _bootstrap_reason(rel: str) -> str:
    """``bootstrap/`` 的细分剔除理由（报告 §bootstrap 抽检依据的代码化）。

    抽检 5 个文件的结论：``bootstrap_merged.light`` / ``bootstrap_v3.light`` 头部
    明确写「由 *_merge_bootstrap.py / merge_bootstrap.py 自动生成」；``level6`` 是
    由 level5 自动转换（无空格分词）且出现 UTF-8 双重编码乱码；``release/`` 是
    stdlib 的发布拷贝；``test_*`` / ``test_modules/`` 是自举工程的测试。其余
    level3/4/5 等是手写阶段源码，但同一份编译器有多个阶段副本、语义高度重复，
    按「宁可少收、不收脏语料」整目录剔除。
    """
    name = rel.rsplit("/", 1)[-1]
    if "/release/" in rel:
        return "bootstrap/release/ 发布拷贝（与 stdlib 内容重复，机器生成）"
    if "merged" in name or name.endswith("_v3.light") or "backup" in name:
        return "bootstrap/ 自动生成/备份产物（文件头标注由脚本生成）"
    if name.startswith("bootstrap_level6"):
        return "bootstrap/ 自动转换产物（由 level5 转换，含编码损坏）"
    if name.startswith("test") or "/test_modules/" in rel:
        return "bootstrap/ 自举工程测试文件"
    return "bootstrap/ 自举阶段源码（多版本重复副本，整目录剔除）"


def drop_reason(rel: str) -> str:
    """返回剔除理由；空串表示保留。规则可复现、按目录段判定。"""
    parts = rel.replace("\\", "/").split("/")
    dirs = parts[:-1]
    stem = Path(parts[-1]).stem
    if "bootstrap" in dirs:
        return _bootstrap_reason(rel)
    for d in dirs:
        dl = d.lower()
        if dl in _DROP_DIR_NAMES:
            return f"目录 {d}/ 属测试/一次性/自举产物"
        if dl.startswith("_scratch") or (dl.startswith("_r") and "scratch" in dl):
            return f"目录 {d}/ 属一次性草稿"
    # 顶层散落的 test_/demo_ 文件（examples/ 等子目录下的示例不受影响）
    if len(parts) == 1 and (stem.startswith("test") or stem.startswith("demo")):
        return "顶层一次性测试/演示文件"
    return ""


# ---- 底层工具 -----------------------------------------------------------
def _git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", check=False)
    return p.stdout if p.returncode == 0 else ""


def _list_light_files(root: Path) -> List[str]:
    """列出 ``light/**/*.light`` 的相对路径（tracked + 未跟踪），按字典序。

    优先走 git 索引（大仓上比递归 os.walk 快一个数量级）；git 不可用时退回
    os.walk，跳过 ``.git``。
    """
    tracked = [x for x in _git(root, "ls-files", "*.light").splitlines() if x]
    others = [x for x in _git(
        root, "ls-files", "--others", "--exclude-standard", "*.light").splitlines() if x]
    rels = sorted({x.replace("\\", "/") for x in tracked + others})
    if rels:
        return rels
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in filenames:
            if fn.endswith(".light"):
                out.append(str((Path(dirpath) / fn).relative_to(root)).replace("\\", "/"))
    return sorted(out)


def _read_text(path: Path) -> str:
    """UTF-8 容错读取 + 行尾归一化（CRLF/LF 不影响跨平台统计）。"""
    text = path.read_bytes().decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def unk_rate(char_freq: Mapping[str, int], tokens: Iterable[str]) -> Tuple[float, int, int]:
    """字符级 UNK 率 = 未被任何**单字** token 覆盖的字符数 / 总字符数。

    这是保守上界：多字 token 只会进一步降低真实分词 UNK，不会抬高它。
    判据因此对「删掉单字 token」有判别力（见 ``tests/test_demo_vocab.py``），
    而不是恒返回 0 的假绿。

    返回 ``(rate, unk_chars, total_chars)``；空语料返回 0.0。
    """
    covered = {t for t in tokens if len(t) == 1}
    total = sum(int(n) for n in char_freq.values())
    if total <= 0:
        return 0.0, 0, 0
    unk = sum(int(n) for ch, n in char_freq.items() if ch not in covered)
    return unk / total, unk, total


def _sorted_freq(counter: Mapping[str, int]) -> List[Tuple[str, int]]:
    """频次降序 + 字符字典序，保证同频次时结果可复现。"""
    return sorted(counter.items(), key=lambda kv: (-int(kv[1]), kv[0]))


def _segments(text: str) -> Iterable[str]:
    """把文本切成不含 L0 字/控制字符/空白的连续片段，供多字组合统计。

    L0 字必须单独成 token，因此含 L0 字的位置直接断开，绝不生成跨它的组合。
    """
    buf: List[str] = []
    for ch in text:
        if ch in _L0_SET or ord(ch) < 33 or ch.isspace():
            if len(buf) >= 2:
                yield "".join(buf)
            buf = []
        else:
            buf.append(ch)
    if len(buf) >= 2:
        yield "".join(buf)


def _ngram_candidates(texts: Sequence[str]) -> List[str]:
    """从语料统计高频 2—4 字组合，按「覆盖收益」排序（不含 L0 字、非纯 ASCII）。"""
    cnt: Counter = Counter()
    for text in texts:
        for seg in _segments(text):
            if seg.isascii():
                continue  # 纯 ASCII 片段归 ascii 层，整段跳过（C 级判断）
            size = len(seg)
            for i in range(size - 1):
                for n in (2, 3, 4):
                    if i + n > size:
                        break
                    cnt[seg[i:i + n]] += 1
    picked = [(s, n) for s, n in cnt.items() if n >= 3]
    picked.sort(key=lambda kv: (-kv[1] * (len(kv[0]) - 1), -kv[1], kv[0]))
    return [s for s, _ in picked]


def _ascii_fragment_candidates(texts: Sequence[str]) -> List[str]:
    cnt: Counter = Counter()
    pat = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,15}")
    for text in texts:
        for m in pat.finditer(text):
            cnt[m.group(0)] += 1
    picked = [(s, n) for s, n in cnt.items() if n >= 4 and len(s) >= 2]
    picked.sort(key=lambda kv: (-kv[1], kv[0]))
    return [s for s, _ in picked]


def _stdlib_candidates(rels: Sequence[str], texts: Sequence[str]) -> List[str]:
    """标准库模块名（``stdlib/**/*.light`` 文件名）+ 语料里 ``从 X 导入`` 的导入名。"""
    # 计数在扫描时一次完成：早年用 ``"\n".join(texts).count(name)`` 对每个候选
    # 全量扫一遍，1.9MB 语料 × 几千候选直接把构建拖过 120 秒（实测 SIGTERM）。
    cnt: Counter = Counter()
    for rel in rels:
        if rel.startswith("stdlib/"):
            cnt[Path(rel).stem] += 10  # 真实模块名优先
    for text in texts:
        for m in re.finditer(r"从\s*([^\s,，。:：\n]+)\s*导入", text):
            cnt[m.group(1)] += 1
    out: Counter = Counter()
    for name, n in cnt.items():
        if not (1 <= len(name) <= 16):
            continue
        if any(ord(c) < 33 or c.isspace() for c in name):
            continue
        if len(name) > 1 and any(c in _L0_SET for c in name):
            continue  # 多字 token 含 L0 字会破坏原子性
        out[name] = n
    return [s for s, _ in _sorted_freq(out)]


def _control_tokens() -> Tuple[List[str], List[str], List[str]]:
    """结构标记 / 决策 token / 验证 token，全部由既有能力生成。"""
    space = TokenSpace()
    structure = list(STRUCTURE_MARKERS)
    decisions = [t.text for t in space.decision_tokens()]
    validations = [f"<V:{code}>" for code in VALIDATION_CODES]
    return structure, decisions, validations


def _build_layout(start: int) -> ControlTokenLayout:
    """按 ``PrismTokenizer`` 同一套顺序构造连续控制区间并校验连续性。"""
    structure, decisions, validations = _control_tokens()
    layout = ControlTokenLayout(start=start)
    layout.structure = {t: start + i for i, t in enumerate(structure)}
    cursor = start + len(structure)
    layout.decisions = {t: cursor + i for i, t in enumerate(decisions)}
    cursor += len(decisions)
    layout.validations = {t: cursor + i for i, t in enumerate(validations)}
    return layout.finalize()


# ---- 构建 ---------------------------------------------------------------
def build_vocab(light_root: Path, output: Path, corpus_dir: Path,
                verbose: bool = True) -> Dict[str, object]:
    """扫描 → 清洗 → 统计 → 组装词表，写 ``corpus/``、词表 JSON，返回统计字典。"""
    light_root = Path(light_root).expanduser().resolve()
    output = Path(output)
    corpus_dir = Path(corpus_dir)

    rels = _list_light_files(light_root)
    if not rels:
        raise RuntimeError(f"{light_root} 下没有找到 .light 文件（--light-root 指对了吗？）")

    records: List[Dict[str, object]] = []
    unique_texts: Dict[str, str] = {}   # md5 -> 文本（每个内容只统计一次）
    kept_texts: List[str] = []
    for rel in rels:
        raw = (light_root / rel).read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        rec: Dict[str, object] = {
            "src_path": rel, "bytes": len(raw), "md5": md5,
            "kept": True, "reason_if_dropped": "",
        }
        records.append(rec)
        if md5 not in unique_texts:
            unique_texts[md5] = _read_text(light_root / rel)

    # md5 去重优先，再做目录规则剔除；index 里两类 dropped 都留理由，可追溯。
    first_by_md5: Dict[str, str] = {}
    for rec in records:
        rel, md5 = str(rec["src_path"]), str(rec["md5"])
        if md5 in first_by_md5:
            rec["kept"] = False
            rec["reason_if_dropped"] = f"md5 重复（保留 {first_by_md5[md5]}）"
            continue
        first_by_md5[md5] = rel
        reason = drop_reason(rel)
        rec["kept"] = not reason
        rec["reason_if_dropped"] = reason
        if not reason:
            kept_texts.append(unique_texts[md5])

    # 字符频次：全部唯一 md5 文件（严格口径，含被剔除文件，见报告）。
    char_freq: Counter = Counter()
    for text in unique_texts.values():
        char_freq.update(text)
    total_chars = sum(char_freq.values())

    # -- 写语料产物 ------------------------------------------------------
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        if not rec["kept"]:
            continue
        rel = str(rec["src_path"])
        dst = corpus_dir / str(Path(rel).with_suffix(".txt"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(unique_texts[str(rec["md5"])], encoding="utf-8")
    index_path = corpus_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (corpus_dir / ".gitignore").write_text(
        "# 原始语料来自 light 仓（版权归原仓、体积大），不入库。\n"
        "# 只提交 index.jsonl 与本 .gitignore；--check 所需的字符频次固化在词表 JSON 里。\n"
        "*.txt\n", encoding="utf-8")

    # -- 组装各层 --------------------------------------------------------
    l0 = list(LIGHT_L0_CORE)                       # 44，一字一 token，顺序即 id 顺序
    structure, decisions, validations = _control_tokens()
    control = structure + decisions + validations  # 4 + 19 + 9 = 32
    ascii_singles = [c for c in _ASCII_SINGLES if char_freq.get(c, 0) > 0]

    # 通用单字：按频次累计覆盖到 CHAR_COVERAGE_TARGET 即停，不硬凑字符表。
    base_covered = sum(char_freq.get(c, 0) for c in _L0_SET)
    base_covered += sum(char_freq.get(c, 0) for c in ascii_singles)
    singles: List[str] = []
    added = 0
    for ch, n in _sorted_freq(char_freq):
        if len(singles) >= _MAX_SINGLE:
            break
        if (base_covered + added) / max(total_chars, 1) >= CHAR_COVERAGE_TARGET:
            break
        if ch in _L0_SET or ch in _ASCII_SET or len(ch) != 1:
            continue
        singles.append(ch)
        added += n

    multi = _ngram_candidates(kept_texts)[:_QUOTA_MULTI]
    stdlib = _stdlib_candidates(rels, kept_texts)[:_QUOTA_STDLIB]
    ascii_frag = _ascii_fragment_candidates(kept_texts)[:_QUOTA_ASCII_FRAGMENT]

    # 统一去重 + 预算裁剪（l0 优先，control 末尾连续区）。
    seen = set()
    layers: Dict[str, List[str]] = {"l0": [], "syntax": [], "stdlib": [],
                                    "ascii": [], "control": []}

    def add(layer: str, items: Sequence[str]) -> None:
        for tok in items:
            if tok in seen or not tok:
                continue
            if len(tok) > 1 and any(c in _L0_SET for c in tok):
                continue  # 兜底：绝不把 L0 字并进多字 token
            if sum(len(v) for v in layers.values()) + len(control) > MAX_SIZE:
                return
            seen.add(tok)
            layers[layer].append(tok)

    add("l0", l0)
    add("syntax", singles + multi)
    add("stdlib", stdlib)
    add("ascii", ascii_singles + ascii_frag)
    # control 单独追加到最后，保证 id 区间连续
    start = sum(len(v) for v in layers.values())
    for tok in control:
        if tok not in seen:
            seen.add(tok)
            layers["control"].append(tok)

    tokens: List[str] = []
    kinds: List[str] = []
    for layer in LAYER_NAMES:
        tokens.extend(layers[layer])
        kinds.extend([layer] * len(layers[layer]))
    layout = _build_layout(start)

    data: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "demo-4096",
        "limit": VOCAB_LIMIT,
        "size": len(tokens),
        "headroom": VOCAB_LIMIT - len(tokens),
        "headroom_ratio": round((VOCAB_LIMIT - len(tokens)) / VOCAB_LIMIT, 6),
        "layer_names": list(LAYER_NAMES),
        "tokens": tokens,
        "kinds": kinds,
        # 卡面字面要求 ``kind`` 描述分层；这里给层→token 列表，供下游直接消费。
        "layers": layers,
        "counts": {name: len(layers[name]) for name in LAYER_NAMES},
        "control_layout": {
            **layout.to_dict(),
            "schema_version": TokenSpace().schema_version,
            "order": "structure -> decision -> validation",
        },
        "corpus_stats": {
            "source": "light/**/*.light",
            "light_root": str(light_root),
            "files_scanned": len(rels),
            "files_unique_md5": len(unique_texts),
            "files_kept": len(kept_texts),
            "files_dropped": len(records) - len(kept_texts),
            "total_chars": total_chars,
            "char_freq": dict(_sorted_freq(char_freq)),
            "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        },
        "notes": [
            "单字 token 覆盖率取 99.8% 即停；UNK 率是保守上界（只算单字覆盖）。",
            "控制 token 由 TokenSpace/VALIDATION_CODES/STRUCTURE_MARKERS 生成，不另造。",
            "多字 token 全部过滤掉 L0 核心字，保证 check_l0_atomicity 通过。",
            "不硬凑 4096：各层只取语料中真实出现且达到频次阈值的候选。",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    rate, unk, total = unk_rate(data["corpus_stats"]["char_freq"], tokens)
    stats: Dict[str, object] = {
        "files_scanned": len(rels), "files_unique_md5": len(unique_texts),
        "files_kept": len(kept_texts), "files_dropped": len(records) - len(kept_texts),
        "total_chars": total_chars, "size": len(tokens),
        "counts": data["counts"], "unk_rate": rate, "unk_chars": unk,
        "headroom": VOCAB_LIMIT - len(tokens), "output": str(output),
        "index": str(index_path),
        "records": records,
    }
    if verbose:
        print(f"[build] 扫描 {len(rels)} 个 .light；唯一 md5 {len(unique_texts)}；"
              f"保留 {len(kept_texts)}；剔除 {len(records) - len(kept_texts)}")
        print(f"[build] 字符 {total_chars}，词表 {len(tokens)} 条，余量 "
              f"{VOCAB_LIMIT - len(tokens)} ({(VOCAB_LIMIT - len(tokens)) / VOCAB_LIMIT:.2%})")
        print(f"[build] 各层 {data['counts']}")
        print(f"[build] UNK 率 {rate:.4%} ({unk}/{total})")
        print(f"[build] 写出 {output} 与 {index_path}")
    return stats


# ---- 校验 ---------------------------------------------------------------
def check_vocab(path: Path, corpus_dir: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """校验词表，返回 ``(是否全绿, 逐条报告)``。五项判据见模块 docstring。"""
    p = Path(path)
    lines: List[str] = []
    problems: List[str] = []

    def ok(label: str, detail: str = "") -> None:
        lines.append(f"[ OK ] {label}" + (f" — {detail}" if detail else ""))

    def bad(label: str, detail: str) -> None:
        lines.append(f"[FAIL] {label} — {detail}")
        problems.append(f"{label}: {detail}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, [f"[FAIL] 读取 {p} 失败 — {exc}"]

    tokens = list(data.get("tokens") or [])
    kinds = list(data.get("kinds") or [])
    counts = dict(data.get("counts") or {})
    stats = dict(data.get("corpus_stats") or {})
    size = len(tokens)

    # 1. 总数 ≤ 4096（外加「留 15% 余量」，卡面明确不要塞满）
    if size <= VOCAB_LIMIT:
        ok("总数 ≤ 4096", f"size={size}, limit={VOCAB_LIMIT}")
    else:
        bad("总数 ≤ 4096", f"size={size} > {VOCAB_LIMIT}")
    if (VOCAB_LIMIT - size) / VOCAB_LIMIT >= HEADROOM_RATIO:
        ok("留白 ≥ 15%", f"headroom={VOCAB_LIMIT - size} "
                         f"({(VOCAB_LIMIT - size) / VOCAB_LIMIT:.2%})")
    else:
        bad("留白 ≥ 15%", f"headroom={VOCAB_LIMIT - size}，不足 15%")

    # 2. L0：44 字各占 1 个 token + check_l0_atomicity
    missing = [c for c in LIGHT_L0_CORE if c not in tokens]
    dup_l0 = [c for c in LIGHT_L0_CORE if tokens.count(c) != 1]
    atomic = check_l0_atomicity(tokens)
    if len(LIGHT_L0_CORE) == 44 and not missing and not dup_l0 and not atomic:
        ok("L0 44 字各占 1 token / 原子性", "check_l0_atomicity 通过")
    else:
        bad("L0 44 字各占 1 token / 原子性",
            f"缺 {missing[:5]}，重复 {dup_l0[:5]}，原子性问题 {atomic[:3]}")

    # 3. 控制 token 与 TokenSpace 逐条对齐（数量 + 文本 + id 连续）
    structure, decisions, validations = _control_tokens()
    expected_control = structure + decisions + validations
    cl = dict(data.get("control_layout") or {})
    start = int(cl.get("start", -1))
    actual_control = tokens[start:] if start >= 0 else []
    ctl_ok = True
    ctl_detail: List[str] = []
    if actual_control != expected_control:
        ctl_ok = False
        ctl_detail.append(
            f"文本/顺序不一致（实际 {len(actual_control)} 条，期望 {len(expected_control)} 条）")
    if not (len(structure) == 4 and len(decisions) == 19
            and len(validations) == len(VALIDATION_CODES)):
        ctl_ok = False
        ctl_detail.append("结构/决策/验证数量异常")
    try:
        ref = _build_layout(start)
        if (start < 0 or cl.get("structure") != ref.structure
                or cl.get("decisions") != ref.decisions
                or cl.get("validations") != ref.validations
                or int(cl.get("end", -1)) != ref.end):
            ctl_ok = False
            ctl_detail.append("id 布局与 ControlTokenLayout 不一致")
        if start + len(expected_control) != size:
            ctl_ok = False
            ctl_detail.append(f"控制区不是词表末尾连续段（start={start}, size={size}）")
    except Exception as exc:  # noqa: BLE001 - 布局异常要变成可诊断的 FAIL
        ctl_ok = False
        ctl_detail.append(f"布局校验异常 {exc!r}")
    if ctl_ok:
        ok("控制 token 与 TokenSpace 对齐",
           f"4 结构 + {len(decisions)} 决策 + {len(validations)} 验证 = "
           f"{len(expected_control)}，id [{start},{size}) 连续")
    else:
        bad("控制 token 与 TokenSpace 对齐", "；".join(ctl_detail))

    # 4. 无重复 token + 分层元数据自洽
    dups = [t for t, n in Counter(tokens).items() if n > 1]
    if dups:
        bad("无重复 token", f"重复 {len(dups)} 个：{dups[:5]}")
    else:
        ok("无重复 token", f"{size} 条全唯一")
    if len(kinds) == size and all(counts.get(k, 0) == kinds.count(k) for k in LAYER_NAMES) \
            and sum(counts.values()) == size:
        ok("分层元数据自洽", str(counts))
    else:
        bad("分层元数据自洽", f"kinds={len(kinds)}, counts={counts}, size={size}")

    # 5. UNK 率 < 1%（字符频次固化在词表里，交付环境无需原始语料）
    char_freq = dict(stats.get("char_freq") or {})
    if not char_freq:
        bad("UNK 率 < 1%", "词表缺少 corpus_stats.char_freq，无法核算")
    else:
        rate, unk, total = unk_rate(char_freq, tokens)
        if rate < UNK_LIMIT:
            ok("UNK 率 < 1%",
               f"{rate:.4%} ({unk}/{total} 字符，口径：{stats.get('files_unique_md5', '?')} "
               f"个唯一 md5 语料文件的字符频次，保守单字覆盖上界)")
        else:
            bad("UNK 率 < 1%", f"{rate:.4%} ({unk}/{total}) ≥ 1%")
        lines.append(f"[INFO] 语料：扫描 {stats.get('files_scanned', '?')} / "
                     f"唯一 md5 {stats.get('files_unique_md5', '?')} / "
                     f"保留 {stats.get('files_kept', '?')} / "
                     f"剔除 {stats.get('files_dropped', '?')}")

        # 原始语料在场时，额外实时重算 kept 子集做交叉参考（不参与判据）
        if corpus_dir and Path(corpus_dir).exists():
            live: Counter = Counter()
            n_files = 0
            for txt in sorted(Path(corpus_dir).rglob("*.txt")):
                live.update(_read_text(txt))
                n_files += 1
            if n_files and sum(live.values()):
                live_rate, live_unk, live_total = unk_rate(live, tokens)
                lines.append(f"[INFO] 实时 kept 语料交叉参考：{n_files} 个 .txt，"
                             f"UNK {live_rate:.4%} ({live_unk}/{live_total})")

    lines.append("")
    lines.append("结论：" + ("全绿" if not problems else f"{len(problems)} 项未通过"))
    return not problems, lines


# ==========================================================================
# 64K 实词表（T-0373）：按 DEFAULT_BUDGET 逐槽构建
# ==========================================================================
#
# 设计口径（与 4096 版**同一条流水线**，不另写一套统计/清洗逻辑）：
#
# * 预算唯一事实源仍是 ``src/lightprism/vocab.py::DEFAULT_BUDGET`` 的
#   ``recommended``；本模块只负责把 65536 个槽位**填满且可追溯**。
# * 槽位顺序即 id 顺序：l0_core → l1_l2_syntax → stdlib_names → chinese_bpe →
#   code_reserve → reserved → decision → validation。
# * 已定义控制 token（4 结构 + 19 决策 + 9 验证 = 32）必须落在词表**末尾连续区间**
#   —— 这是 ``ControlTokenLayout.finalize()`` 与 ``PrismTokenizer`` 的硬要求。
#   decision/validation 两个槽的**预留 id**（301 / 151）因此排在控制块**之前**，
#   槽位在 manifest 里用「多段 range」如实登记，不假装连续。
# * 段标记 ``<PY>``/``<LIGHT>``（T-0371，64K 侧 T-0375 补建）从 reserved 槽头部切出，
#   id 由 ``vocab.slot_offsets()["reserved"]`` 推导（64000/64001）。由于 reserved 自有段
#   物理上排在 decision/validation 预留之前（起点 63520），段标记落在自有段内偏移
#   ``64000-63520=480`` 处；装配完再断言一次「tokens[声明 id] == 段标记」，错位直接失败。
# * 词表最后两位固定 ``<UNK>`` / ``<PAD>``（计入 reserved 槽）。这样
#   ``DemoVocab.vocab_size == 65536``：模型 V 与词表条数一致，不留 65538 的暗坑。
VOCAB64_TOTAL: int = 65536
VOCAB64_SCHEMA_VERSION = "1.0"
VOCAB64_SLOT_ORDER: Tuple[str, ...] = (
    "l0_core", "l1_l2_syntax", "stdlib_names", "chinese_bpe",
    "code_reserve", "reserved", "decision", "validation",
)
SPECIAL_UNK = "<UNK>"
SPECIAL_PAD = "<PAD>"
#: reserved 槽里**除 UNK/PAD 之外**的特殊控制符（它们不是 BPE 产物，是显式预留的控制位）。
RESERVED_SPECIALS: Tuple[str, ...] = (
    "<MASK>", "<SEP>", "<BOS>", "<EOS>", "<SYS>", "<USER>", "<ASSISTANT>", "<FORMAT>",
)
#: 代码槽的运算符/标点组合（人工白名单，按长度与常用度排；来源是 Light 语法参考）。
CODE_OPERATORS: Tuple[str, ...] = (
    "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "->", "=>", "::", "//", "**",
    "&&", "||", "..", "()", "[]", "{}", "[]=", "(())", "++", "--", "??", "?.",
    ":", ",", ";", ".", "=", "+", "-", "*", "/", "%", "<", ">", "&", "|", "^", "~",
    "!", "?", "#", "@", "$", "\\", "_", "(", ")", "[", "]", "{", "}",
)
#: 词表总条数的硬上限（与 budget.total 一致）。
_ASCII_PRINTABLE = tuple(chr(i) for i in range(33, 127))  # 94 个非空白可打印 ASCII
_WS_RE = re.compile(r"\s")


def slot_quotas() -> Dict[str, int]:
    """按 ``DEFAULT_BUDGET.recommended`` 取八个槽位的落点（不取区间中点）。"""
    budget = VocabBudget()
    return {slot.name: budget.value_of(slot.name) for slot in DEFAULT_BUDGET}


def _slot_sha(tokens: Sequence[str]) -> str:
    """槽位 sha256：``sha256("\\n".join(tokens) + "\\n")``，UTF-8、LF。

    规则写进 manifest 的 ``sha256.rule``，下游可逐槽复算；
    不按 JSON 序列化算，避免 indent / 转义差异把哈希变成平台指纹。
    """
    return hashlib.sha256(("\n".join(tokens) + "\n").encode("utf-8")).hexdigest()


def _is_all_han(tok: str) -> bool:
    return bool(tok) and all("\u3400" <= ch <= "\u9fff" for ch in tok)


def _ranked_ngrams(texts: Sequence[str], max_n: int, min_freq: int,
                   all_han: bool = False) -> List[str]:
    """语料 n-gram 候选，按「覆盖收益」排序（与 4096 版 ``_ngram_candidates`` 同口径）。

    ``_segments()`` 在 L0 字处断开，因此候选**天然不含 L0 字**；
    ``all_han=True`` 时只保留纯汉字 n-gram（中文 BPE 槽专用）。
    """
    cnt: Counter = Counter()
    for text in texts:
        for seg in _segments(text):
            if seg.isascii():
                continue
            size = len(seg)
            for i in range(size - 1):
                for n in range(2, max_n + 1):
                    if i + n > size:
                        break
                    piece = seg[i:i + n]
                    if all_han and not _is_all_han(piece):
                        continue
                    cnt[piece] += 1
    picked = [(s, n) for s, n in cnt.items() if n >= min_freq]
    picked.sort(key=lambda kv: (-kv[1] * (len(kv[0]) - 1), -kv[1], len(kv[0]), kv[0]))
    return [s for s, _ in picked]


def _ascii_identifiers(texts: Sequence[str]) -> List[str]:
    cnt: Counter = Counter()
    pat = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,24}")
    for text in texts:
        for m in pat.finditer(text):
            cnt[m.group(0)] += 1
    picked = [(s, n) for s, n in cnt.items() if n >= 2]
    picked.sort(key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    return [s for s, _ in picked]


def _numeric_literals(texts: Sequence[str]) -> List[str]:
    cnt: Counter = Counter()
    pat = re.compile(r"(?<![\w.])-?\d{1,12}(?:\.\d{1,6})?")
    for text in texts:
        for m in pat.finditer(text):
            cnt[m.group(0)] += 1
    picked = [(s, n) for s, n in cnt.items() if n >= 2]
    picked.sort(key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    return [s for s, _ in picked]


def _take(pool: Iterable[str], need: int, used: set,
          allow_blanks: bool = False) -> List[str]:
    """从候选池里取 ``need`` 个**合格且未用过**的 token；取不满就返回实际数量。

    合格 = 非空、无空白、长度 ≤ 24、多字 token 不含 L0 核心字（L0 原子性硬约束）。
    ``allow_blanks`` 只给 code 槽的单字档用：空格/换行本身是合法单字 token
    （4096 表里就有，缺了它们 ``encode`` 会把整段空白映射成 UNK）。
    """
    out: List[str] = []
    if need <= 0:
        return out
    for tok in pool:
        if len(out) >= need:
            break
        if not tok or len(tok) > 24:
            continue
        if tok in used:
            continue
        if _WS_RE.search(tok) and not (allow_blanks and len(tok) == 1):
            continue
        if len(tok) > 1 and any(c in _L0_SET for c in tok):
            continue
        used.add(tok)
        out.append(tok)
    return out


def _load_corpus_snapshot(data_dir: Path) -> Tuple[List[str], Dict[str, object]]:
    """读 ``data/unified/corpus.jsonl``（kept 语料快照，跨机一致）。

    64K 版**不重跑**清洗/写 ``data/corpus``：那是 4096 卡的产物，且清洗规则已在
    ``index.jsonl`` 里固化。这里只消费快照，保证「同一份语料」可比。
    """
    snap = Path(data_dir) / "unified" / "corpus.jsonl"
    if not snap.is_file():
        raise RuntimeError(
            f"缺少语料快照 {snap}；先跑 `PYTHONPATH=src python3 -m "
            "lightprism.demo.train_demo --build-unified`"
        )
    texts: List[str] = []
    for line in snap.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        body = str(rec.get("text") or "")
        if body:
            texts.append(body)
    stats: Dict[str, object] = {
        "source": str(snap),
        "files": len(texts),
        "chars": sum(len(t) for t in texts),
        "sha256": hashlib.sha256(snap.read_bytes()).hexdigest(),
        "note": "data/unified/corpus.jsonl（kept=true 快照，392 个 .light）",
    }
    return texts, stats


def _comment_texts(code_texts: Sequence[str]) -> List[str]:
    """从代码语料抽 ``#`` 注释行，作为中文自然语言语料的一部分。"""
    out: List[str] = []
    for text in code_texts:
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                body = stripped.lstrip("#").strip()
                if body:
                    out.append(body)
    return out


def _task_texts(data_dir: Path) -> List[str]:
    """SFT / 翻译 / 决策样本里的中文任务描述（自然语言，不进 BPE 统计会缺语料）。"""
    out: List[str] = []
    for name in ("json_tasks.jsonl", "code_tasks.jsonl",
                 "translation_tasks.jsonl", "err_tasks.jsonl"):
        path = Path(data_dir) / "unified" / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("task", "task_text", "instruction", "note", "prompt", "category"):
                val = rec.get(key)
                if isinstance(val, str) and val.strip():
                    out.append(val.strip())
    return out


def _load_repo_texts(light_root: Optional[Path]) -> List[str]:
    """读 light 仓全部 ``*.light`` 源文本（stdlib 名池与 code 池共用一次扫描）。"""
    texts: List[str] = []
    if not (light_root and Path(light_root).is_dir()):
        return texts
    for path in sorted(Path(light_root).rglob("*.light")):
        try:
            texts.append(_read_text(path))
        except OSError:
            continue
    return texts


def _stdlib_pools(light_root: Optional[Path],
                  corpus_texts: Sequence[str],
                  repo_texts: Optional[Sequence[str]] = None,
                  ) -> Tuple[List[Tuple[str, Iterable[str]]], Dict[str, object]]:
    """stdlib 槽候选：模块名 → 导出名 → 定义名 → 导入名 → ASCII 名 → 全限定名。

    「导出名」按 ``stdlib/**/*.light`` 里 ``导出 A B C。`` 的真实声明抽取；
    「全限定名」是 Light 的模块成员写法 ``模块.名字``（语法参考：``自.版本``），
    只在前面几类不够填满 10240 时才用到，并在 manifest 里单列计数。
    """
    modules: List[str] = []
    exports: List[str] = []
    defined: List[str] = []
    py_defs: List[str] = []
    zh_ids: List[str] = []
    ascii_ids: List[str] = []
    repo_ids: List[str] = []
    qualified: List[str] = []
    meta: Dict[str, object] = {"light_root": str(light_root) if light_root else "",
                               "files": 0, "exports": 0, "qualified": 0,
                               "py_defs": 0, "light_defs": 0,
                               "repo_identifiers": 0}

    exp_pat = re.compile(r"^\s*导出\s+(.+?)。", re.M)
    zh_pat = re.compile(r"[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9_]{1,15}")
    def_pat = re.compile(
        r"^\s*(?:(?P<kw1>段落|段|类|方法)\s+(?P<n1>[^\s(（]+)"
        r"|(?P<n2>[\u4e00-\u9fffA-Za-z_][^\s(（:：,，。]{0,20})\s+(?P<kw2>段|类|方法))",
        re.M)
    py_pat = re.compile(r"^\s*(?:def|class)\s+([^\s(:]+)", re.M)
    if light_root and Path(light_root).is_dir():
        std_dirs = ("stdlib/", "stdlib_v3/")
        files = sorted(p for p in Path(light_root).rglob("*.light")
                       if str(p.relative_to(light_root)).replace("\\", "/")
                       .startswith(std_dirs))
        meta["files"] = len(files)
        exp_by_module: Dict[str, List[str]] = {}
        for path in files:
            rel = str(path.relative_to(light_root)).replace("\\", "/")
            stem = Path(rel).stem
            modules.append(stem)
            try:
                txt = _read_text(path)
            except OSError:
                continue
            for m in exp_pat.finditer(txt):
                for name in re.split(r"\s+", m.group(1).strip()):
                    if name:
                        exports.append(name)
                        exp_by_module.setdefault(stem, []).append(name)
            for m in re.finditer(r"《([^》]{1,20})》", txt):
                defined.append(m.group(1).strip())
            for m in def_pat.finditer(txt):
                name = (m.group("n1") or m.group("n2") or "").strip()
                if name:
                    defined.append(name)
            for m in zh_pat.finditer(txt):
                zh_ids.append(m.group(0))
            for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{1,24}", txt):
                ascii_ids.append(m.group(0))
        # Python 壳（stdlib/*.py）同样是公开 API 面：def/class 名照收。
        py_files = sorted(p for p in Path(light_root).rglob("*.py")
                          if str(p.relative_to(light_root)).replace("\\", "/")
                          .startswith(std_dirs))
        for path in py_files:
            try:
                txt = _read_text(path)
            except OSError:
                continue
            for m in py_pat.finditer(txt):
                py_defs.append(m.group(1).strip())
        # 全限定名：模块 → 自己的导出，按 (模块, 导出) 字典序，长度 ≤ 20。
        for stem in sorted(exp_by_module):
            for name in sorted(set(exp_by_module[stem])):
                if len(stem) + len(name) + 1 <= 20:
                    qualified.append(f"{stem}.{name}")

    # 语料里的 `从 X 导入 Y`：真实调用面，优先级高于 ASCII 噪声。
    imports: List[str] = []
    imp_pat = re.compile(r"从\s*([^\s,，。:：\n]+)\s*导入\s*([^\n。]+)")
    for text in corpus_texts:
        for m in imp_pat.finditer(text):
            if m.group(1):
                imports.append(m.group(1))
            for name in re.split(r"[\s,，、]+", m.group(2)):
                if name:
                    imports.append(name)

    # 仓库内其它库模块的标识符名（blocks_pkg/contrib/...）：stdlib 名池的
    # 真实容量不足以填满 recommended 的 10240，缺口只能用**真实存在的名字**
    # 补，不用 f1 注释碎片或占位符硬凑；manifest 里单列来源，报告如实披露。
    if repo_texts:
        repo_ids = _ascii_identifiers(repo_texts)
        repo_ids = [s for s, _ in _sorted_freq(Counter(repo_ids))]

    exports = [s for s, _ in _sorted_freq(Counter(exports))]
    defined = [s for s, _ in _sorted_freq(Counter(defined))]
    py_defs = [s for s, _ in _sorted_freq(Counter(py_defs))]
    ascii_ids = [s for s, _ in _sorted_freq(Counter(ascii_ids))]
    zh_freq = Counter(zh_ids)
    zh_hot = [s for s, _ in _sorted_freq(zh_freq) if zh_freq[s] >= 2]

    meta["exports"] = len(set(exports))
    meta["modules"] = len(set(modules))
    meta["qualified"] = len(qualified)
    meta["py_defs"] = len(set(py_defs))
    meta["light_defs"] = len(set(defined))
    meta["zh_identifiers_f2"] = len(zh_hot)
    meta["repo_identifiers"] = len(repo_ids)

    pools: List[Tuple[str, Iterable[str]]] = [
        ("stdlib_modules", modules),
        ("stdlib_exports", exports),
        ("stdlib_py_defs", py_defs),
        ("stdlib_light_defs", defined),
        ("stdlib_imports", imports),
        ("stdlib_zh_ids_f2", zh_hot),
        ("stdlib_ascii", ascii_ids),
        ("stdlib_qualified", qualified),
        ("stdlib_repo_identifiers", repo_ids),
    ]
    return pools, meta


def _code_pools(light_root: Optional[Path],
                code_texts: Sequence[str],
                char_freq: Mapping[str, int],
                repo_texts: Optional[Sequence[str]] = None,
                ) -> List[Tuple[str, Iterable[str]]]:
    """code_reserve 槽候选：ASCII 单字 → 标识符 → 数字字面量 → 运算符 → 整数/十六进制。

    ``code_reserve`` 内部按 ``docs/词表预算.md`` §五.2 的待定项做了**子分档**：
    单字符 / 标识符片段 / 数字 / 运算符各占一块，manifest 里逐档计数。
    """
    idents = _ascii_identifiers(code_texts)
    nums = _numeric_literals(code_texts)
    if repo_texts is None:
        repo_texts = _load_repo_texts(light_root)
    if repo_texts:
        idents = idents + _ascii_identifiers(repo_texts)
        nums = nums + _numeric_literals(repo_texts)
    idents = [s for s, _ in _sorted_freq(Counter(idents))]
    nums = [s for s, _ in _sorted_freq(Counter(nums))]

    # 空白字符必须先占位：``DemoVocab.encode`` 逐字符走最长匹配，语料里的
    # 空格/换行占 29% 字符量，缺了它们整段文本会被映射成 UNK（4096 表里
    # '\n' 与 ' ' 也在，id 44 / 1923，这里保持同一行为）。
    whitespace = [ch for ch, _ in _sorted_freq(char_freq) if ch.isspace()]
    singles = whitespace + [ch for ch in _ASCII_PRINTABLE if char_freq.get(ch, 0) > 0]
    singles += [ch for ch in _ASCII_PRINTABLE if ch not in singles]

    # ASCII 标点二连（语料里真实出现且 freq≥2）：`->` `::` `»,` 之外的组合交给
    # 统计，比只放一份手写运算符表更能覆盖真实代码标点。
    bigram_freq: Counter = Counter()
    for text in list(code_texts) + list(repo_texts):
        for i in range(len(text) - 1):
            pair = text[i:i + 2]
            if all(ch in _ASCII_PRINTABLE for ch in pair) and not pair.isalnum():
                bigram_freq[pair] += 1
    bigrams = [p for p, n in _sorted_freq(bigram_freq) if n >= 2]

    def integers() -> Iterable[str]:
        for i in range(0, 10000):
            yield str(i)

    def hexes() -> Iterable[str]:
        for i in range(0, 256):
            yield f"0x{i:02X}"

    return [
        ("code_ascii_single", singles),
        ("code_identifiers", idents),
        ("code_numbers", nums),
        ("code_operators", CODE_OPERATORS),
        ("code_punct_bigrams", bigrams),
        ("code_hex", hexes()),
        ("code_integers", integers()),
    ]


def build_vocab_64k(light_root: Optional[Path], output: Path, manifest_path: Path,
                    data_dir: Optional[Path] = None,
                    verbose: bool = True) -> Dict[str, object]:
    """按 ``DEFAULT_BUDGET`` 八槽构建 64K 实词表，写词表 JSON + manifest。"""
    quotas = slot_quotas()
    total = sum(quotas.values())
    if total != VOCAB64_TOTAL:  # 预算表本身必须自洽，否则立刻炸，不写半成品
        raise RuntimeError(f"DEFAULT_BUDGET 合计 {total} ≠ {VOCAB64_TOTAL}")

    root = Path(__file__).resolve().parents[3]
    data_dir = Path(data_dir) if data_dir else root / "data"
    output = Path(output)
    manifest_path = Path(manifest_path)

    code_texts, corpus_stats = _load_corpus_snapshot(data_dir)
    char_freq: Counter = Counter()
    for text in code_texts:
        char_freq.update(text)

    structure = list(STRUCTURE_MARKERS)
    segment = list(SEGMENT_MARKERS)
    space = TokenSpace()
    decisions = [t.text for t in space.decision_tokens()]
    validations = [f"<V:{code}>" for code in VALIDATION_CODES]
    # 段标记（T-0371）从 reserved 槽头部切出，不在 decision/validation 额度里另占行：
    # reserved 1536 = 段标记 2 + 自有特殊符/占位 1528 + 结构标记 4 + UNK/PAD 2。
    reserved_own_quota = (quotas["reserved"] - len(segment)
                          - len(structure) - 2)              # 2 = <UNK>/<PAD>
    decision_reserve_quota = quotas["decision"] - len(decisions)
    validation_reserve_quota = quotas["validation"] - len(validations)
    if reserved_own_quota < len(RESERVED_SPECIALS) or decision_reserve_quota < 0 \
            or validation_reserve_quota < 0:
        raise RuntimeError("预算切分不足以容纳控制 token，请先核对 DEFAULT_BUDGET")

    used: set = set()
    slots: Dict[str, List[str]] = {}
    provenance: Dict[str, Dict[str, int]] = {}

    def fill(name: str, pools: Sequence[Tuple[str, object]], quota: int,
             allow_blanks: bool = False) -> None:
        """按池顺序取满 ``quota``；**取不满直接失败**，不 quietly 空着。"""
        got: List[str] = []
        prov: Dict[str, int] = {}
        for source, pool in pools:
            if len(got) >= quota:
                break
            items = pool() if callable(pool) else pool
            picked = _take(items, quota - len(got), used, allow_blanks)
            if picked:
                got.extend(picked)
                prov[source] = len(picked)
        if len(got) != quota:
            raise RuntimeError(
                f"64K 槽位 {name} 候选不足：{len(got)}/{quota}（差 {quota - len(got)}）")
        slots[name] = got
        provenance[name] = prov

    # ---- 1. L0 冻结区 ---------------------------------------------------
    fill("l0_core", [("frozen_l0", list(LIGHT_L0_CORE))], quotas["l0_core"])

    # ---- 2. L1/L2 语法组合（代码语料，纯汉字 n-gram） ---------------------
    # 只收纯汉字组合：跨标点/ASCII 的片段（如 "(拼音表"）不是语法组合，
    # 收进来只会让 BPE 在标点边界上错误粘连。
    fill("l1_l2_syntax", [
        ("corpus_code_ngram_f3_n4_han",
         lambda: _ranked_ngrams(code_texts, max_n=4, min_freq=3, all_han=True)),
        ("corpus_code_ngram_f2_n5_han",
         lambda: _ranked_ngrams(code_texts, max_n=5, min_freq=2, all_han=True)),
    ], quotas["l1_l2_syntax"])

    # ---- 3. stdlib 名字 -------------------------------------------------
    repo_texts = _load_repo_texts(light_root)
    std_pools, std_meta = _stdlib_pools(light_root, code_texts, repo_texts)
    fill("stdlib_names", std_pools, quotas["stdlib_names"])

    # ---- 4. 中文 BPE（单字底座 + 自然语言高频组合） ----------------------
    # 单字底座含全部非 ASCII 单字（汉字 + 中文标点），否则 UNK 率会被
    # 「，。：、」这类高频标点顶到 29%（实测过一次，见报告 §四）。
    non_ascii_singles = [ch for ch, _ in _sorted_freq(char_freq)
                         if len(ch) == 1 and not ch.isascii() and ch not in _L0_SET
                         and not ch.isspace()]
    ngram_corpus = _comment_texts(code_texts) + _task_texts(data_dir)

    fill("chinese_bpe", [
        ("char_singles", non_ascii_singles),
        ("nl_ngram_f2_n6", lambda: _ranked_ngrams(ngram_corpus, max_n=6, min_freq=2, all_han=True)),
        ("corpus_ngram_f2_n6", lambda: _ranked_ngrams(code_texts, max_n=6, min_freq=2, all_han=True)),
        ("nl_ngram_f1_n8", lambda: _ranked_ngrams(ngram_corpus, max_n=8, min_freq=1, all_han=True)),
        ("corpus_ngram_f1_n8", lambda: _ranked_ngrams(code_texts, max_n=8, min_freq=1, all_han=True)),
    ], quotas["chinese_bpe"])

    # ---- 5. 代码余量 ---------------------------------------------------
    fill("code_reserve", _code_pools(light_root, code_texts, char_freq, repo_texts),
         quotas["code_reserve"], allow_blanks=True)

    # ---- 6. 保留区（除 UNK/PAD 外的特殊符 + 占位 id + 段标记） ------------
    # 段标记的 id 唯一事实源是 ``vocab.slot_offsets()["reserved"]``（默认 64000），
    # 不是本模块的装配顺序。本模块为了让已定义控制块落在**末尾连续区间**，把
    # reserved 自有段排在 decision/validation 预留 id 之前，自有段的物理起点
    # （默认 63520）因此早于 ``slot_offsets()["reserved"]``；段标记必须落在预算
    # 口径的 reserved 头部，即自有段内部偏移 ``segment_offset`` 处。
    budget64 = VocabBudget(alloc=quotas)
    offsets64 = slot_offsets(budget64)
    reserved_own_start = offsets64["code_reserve"] + quotas["code_reserve"]
    segment_offset = offsets64["reserved"] - reserved_own_start
    own_plain = list(RESERVED_SPECIALS) + [
        f"<RSV:{i:04d}>" for i in range(1, reserved_own_quota - len(RESERVED_SPECIALS) + 1)
    ]
    if not 0 <= segment_offset <= len(own_plain):
        raise RuntimeError(
            f"段标记落点 {segment_offset} 不在 reserved 自有段 [0,{len(own_plain)}] 内，"
            "请先核对 DEFAULT_BUDGET 的槽位顺序")
    reserved_own = own_plain[:segment_offset] + segment + own_plain[segment_offset:]
    fill("reserved", [("reserved_specials_placeholders", reserved_own)], len(reserved_own))

    # ---- 7/8. 决策 / 验证的预留 id（排在控制块之前） ----------------------
    decision_rsv = [f"<D:RESERVED_{i:04d}>" for i in range(1, decision_reserve_quota + 1)]
    validation_rsv = [f"<V:RESERVED_{i:03d}>" for i in range(1, validation_reserve_quota + 1)]

    # ---- 装配：文本槽 → 预留 id → 末尾连续控制块 → UNK/PAD ---------------
    tokens: List[str] = []
    ranges: Dict[str, List[List[int]]] = {}

    def append_block(label: str, items: Sequence[str]) -> None:
        start = len(tokens)
        if start and not items:
            return
        tokens.extend(items)
        ranges.setdefault(label, []).append([start, len(tokens)])

    for name in ("l0_core", "l1_l2_syntax", "stdlib_names", "chinese_bpe", "code_reserve"):
        append_block(name, slots[name])
    append_block("reserved_own", slots["reserved"])
    append_block("decision_reserved", decision_rsv)
    append_block("validation_reserved", validation_rsv)
    append_block("structure", structure)
    append_block("decision_defined", decisions)
    append_block("validation_defined", validations)
    append_block("special_tail", [SPECIAL_UNK, SPECIAL_PAD])

    control_start = ranges["structure"][0][0]
    layout = ControlTokenLayout(start=control_start)
    layout.structure = {t: control_start + i for i, t in enumerate(structure)}
    cursor = control_start + len(structure)
    layout.decisions = {t: cursor + i for i, t in enumerate(decisions)}
    cursor += len(decisions)
    layout.validations = {t: cursor + i for i, t in enumerate(validations)}
    layout.finalize()

    unk_id = len(tokens) - 2
    pad_id = len(tokens) - 1
    if layout.end != unk_id:
        raise RuntimeError(f"控制块末尾 {layout.end} ≠ UNK id {unk_id}")
    if tokens[unk_id] != SPECIAL_UNK or tokens[pad_id] != SPECIAL_PAD:
        raise RuntimeError("词表最后两位必须是 <UNK>/<PAD>")
    if len(tokens) != VOCAB64_TOTAL:
        raise RuntimeError(f"词表条数 {len(tokens)} ≠ {VOCAB64_TOTAL}")

    # 段标记必须真的落在 vocab.py 声明的 id 上（两处事实源不一致就直接失败）
    seg_ids = segment_token_ids(budget64)
    if dict(SEGMENT_TOKEN_IDS) != seg_ids:
        raise RuntimeError(
            f"SEGMENT_TOKEN_IDS={dict(SEGMENT_TOKEN_IDS)} 与 slot_offsets 推导 "
            f"{seg_ids} 不一致")
    for tok, want in seg_ids.items():
        if want >= len(tokens) or tokens[want] != tok:
            got = tokens.index(tok) if tok in tokens else "缺失"
            raise RuntimeError(f"段标记 {tok} 应在 id {want}，实际 {got}")

    slot_ranges = {
        "l0_core": ranges["l0_core"],
        "l1_l2_syntax": ranges["l1_l2_syntax"],
        "stdlib_names": ranges["stdlib_names"],
        "chinese_bpe": ranges["chinese_bpe"],
        "code_reserve": ranges["code_reserve"],
        "reserved": ranges["reserved_own"] + ranges["structure"] + ranges["special_tail"],
        "decision": ranges["decision_reserved"] + ranges["decision_defined"],
        "validation": ranges["validation_reserved"] + ranges["validation_defined"],
    }
    slot_tokens = {name: [tokens[i] for a, b in rs for i in range(a, b)]
                   for name, rs in slot_ranges.items()}
    slot_counts = {name: len(v) for name, v in slot_tokens.items()}

    # ---- 自检（不通过就不落盘） -----------------------------------------
    checks = _audit_64k(tokens, slot_counts, quotas, layout, unk_id, pad_id)
    if checks:
        raise RuntimeError("64K 自检未通过：" + "；".join(checks))

    budget_problems = VocabBudget(alloc=slot_counts).problems(TokenSpace())

    vocab_payload: Dict[str, object] = {
        "schema_version": VOCAB64_SCHEMA_VERSION,
        "kind": "light-64k",
        "limit": VOCAB64_TOTAL,
        "size": len(tokens),
        "slot_names": list(VOCAB64_SLOT_ORDER),
        "slot_ranges": slot_ranges,
        "slot_counts": slot_counts,
        "tokens": tokens,
        "control_layout": {
            **layout.to_dict(),
            "schema_version": space.schema_version,
            "order": "structure -> decision -> validation",
            "defined_structure": len(structure),
            "defined_decisions": len(decisions),
            "defined_validations": len(validations),
        },
        "budget": {
            "source": "src/lightprism/vocab.py::DEFAULT_BUDGET(推荐落点)",
            "alloc": quotas,
            "total": VOCAB64_TOTAL,
            "sum": sum(quotas.values()),
            "vocab_problems": list(VocabBudget(alloc=quotas).problems(TokenSpace())),
            "slot_problems": list(budget_problems),
        },
        "special_ids": {"unk": unk_id, "pad": pad_id, "embedding_size": len(tokens)},
        "segment_markers": {
            "tokens": list(SEGMENT_MARKERS),
            "ids": dict(seg_ids),
            "slot": "reserved",
            "note": "T-0371 段标记：从 reserved 槽头部切出，id 由 "
                    "vocab.slot_offsets()['reserved'] 推导，不另占预算行",
        },
        "corpus_stats": {
            **corpus_stats,
            "total_chars": sum(char_freq.values()),
            "char_freq": dict(_sorted_freq(char_freq)),
        },
        "selection": {
            "l1_l2_syntax": {"max_n": 4, "min_freq": 3, "source": "data/unified/corpus.jsonl"},
            "chinese_bpe": {"max_n": 8, "min_freq": 1, "all_han": True,
                            "sources": ["corpus.jsonl", "代码 # 注释", "SFT/翻译任务描述"]},
            "stdlib_names": std_meta,
            "code_reserve": {"sub_tiers": ["ascii_single", "identifiers", "numbers",
                                          "operators", "integers_0_9999", "hex_0x00_0xFF"]},
        },
        "notes": [
            "槽位顺序即 id 顺序；decision/validation 的预留 id 排在控制块之前，"
            "保证已定义控制 token 在词表末尾严格连续（ControlTokenLayout.finalize 的硬要求）。",
            "<UNK>/<PAD> 固定占最后两位（计入 reserved 槽），使 DemoVocab.vocab_size == 65536。",
            "多字 token 一律不含 L0 核心字；L0 44 字各占 1 token 且顺序固定。",
            "不硬凑：每个槽位先取语料/标准库里的真实候选，不足才用显式命名空间占位"
            "（<D:RESERVED_…> / <V:RESERVED_…> / <RSV:…>），并在 manifest 的 sources 里逐档计数。",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(vocab_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    slot_manifest: Dict[str, object] = {}
    for name in VOCAB64_SLOT_ORDER:
        slot_manifest[name] = {
            "quota": quotas[name],
            "count": slot_counts[name],
            "ranges": slot_ranges[name],
            "sha256": _slot_sha(slot_tokens[name]),
            "sources": provenance.get(name, {}),
        }
    slot_manifest["reserved"]["charged"] = {
        "segment_markers": len(segment),
        "own_placeholders_and_specials": reserved_own_quota,
        "structure_markers": len(structure),
        "unk_pad": 2,
    }
    slot_manifest["decision"]["charged"] = {
        "defined": len(decisions), "reserved_ids": decision_reserve_quota,
    }
    slot_manifest["validation"]["charged"] = {
        "defined": len(validations), "reserved_ids": validation_reserve_quota,
    }

    manifest: Dict[str, object] = {
        "schema_version": VOCAB64_SCHEMA_VERSION,
        "kind": "light-64k-manifest",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "PYTHONPATH=src python3 -m lightprism.demo.vocab_build --build-64k",
        "task": "T-0373",
        "vocab_file": output.name,
        "size": len(tokens),
        "limit": VOCAB64_TOTAL,
        "sha256": {
            "rule": "sha256 of '\\n'.join(tokens)+'\\n', UTF-8, LF",
            "tokens": _slot_sha(tokens),
            "file_bytes": hashlib.sha256(output.read_bytes()).hexdigest(),
        },
        "budget": {
            "source": "src/lightprism/vocab.py::DEFAULT_BUDGET(推荐落点)",
            "alloc": quotas,
            "sum": sum(quotas.values()),
            "total": VOCAB64_TOTAL,
            "problems": list(VocabBudget(alloc=slot_counts).problems(TokenSpace())),
        },
        "slots": slot_manifest,
        "control_layout": vocab_payload["control_layout"],
        "special_ids": vocab_payload["special_ids"],
        "segment_markers": vocab_payload["segment_markers"],
        "corpus_stats": {k: v for k, v in vocab_payload["corpus_stats"].items()
                         if k != "char_freq"},
        "selection": vocab_payload["selection"],
        "light_root": str(light_root) if light_root else "",
        "data_dir": str(data_dir),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    stats: Dict[str, object] = {
        "size": len(tokens),
        "slot_counts": slot_counts,
        "slot_quota": quotas,
        "output": str(output),
        "manifest": str(manifest_path),
        "corpus_files": corpus_stats.get("files"),
        "corpus_chars": corpus_stats.get("chars"),
        "sources": provenance,
        "stdlib": std_meta,
    }
    if verbose:
        print(f"[build64k] 语料 {corpus_stats.get('files')} 个文件 / "
              f"{corpus_stats.get('chars')} 字符")
        for name in VOCAB64_SLOT_ORDER:
            print(f"[build64k] {name:<14} {slot_counts[name]:>6} / {quotas[name]:>6}  "
                  f"{provenance.get(name, {})}")
        print(f"[build64k] 合计 {len(tokens)}（控制块 [{control_start},{layout.end})，"
              f"UNK={unk_id} PAD={pad_id}）")
        print(f"[build64k] 写出 {output} 与 {manifest_path}")
    return stats


def _audit_64k(tokens: Sequence[str], slot_counts: Mapping[str, int],
               quotas: Mapping[str, int], layout: ControlTokenLayout,
               unk_id: int, pad_id: int) -> List[str]:
    """64K 词表的内部自检：预算 / L0 / 控制块 / 唯一性 / 槽位计数。"""
    issues: List[str] = []
    if len(tokens) != VOCAB64_TOTAL:
        issues.append(f"总数 {len(tokens)} ≠ {VOCAB64_TOTAL}")
    if len(set(tokens)) != len(tokens):
        dups = [t for t, n in Counter(tokens).items() if n > 1]
        issues.append(f"重复 token {len(dups)} 个：{dups[:5]}")
    for name in VOCAB64_SLOT_ORDER:
        if slot_counts.get(name) != quotas.get(name):
            issues.append(f"槽位 {name} 实际 {slot_counts.get(name)} ≠ 预算 {quotas.get(name)}")
    missing = [c for c in LIGHT_L0_CORE if tokens.count(c) != 1]
    if len(LIGHT_L0_CORE) != 44 or missing:
        issues.append(f"L0 44 字未各占 1 token：{missing[:5]}")
    atomic = check_l0_atomicity(tokens)
    if atomic:
        issues.append(f"L0 原子性：{atomic[:3]}")
    space = TokenSpace()
    expected = (list(STRUCTURE_MARKERS) + [t.text for t in space.decision_tokens()]
                + [f"<V:{c}>" for c in VALIDATION_CODES])
    if list(tokens[layout.start:layout.end]) != expected:
        issues.append("末尾控制块与 TokenSpace/STRUCTURE_MARKERS 不一致")
    ids = (list(layout.structure.values()) + list(layout.decisions.values())
           + list(layout.validations.values()))
    if sorted(ids) != list(range(layout.start, layout.end)):
        issues.append("控制 id 区间有空洞")
    if layout.end != unk_id or tokens[unk_id] != SPECIAL_UNK or tokens[pad_id] != SPECIAL_PAD:
        issues.append("<UNK>/<PAD> 不在词表末尾两位")
    seg_ids = segment_token_ids(VocabBudget(alloc=dict(slot_counts)))
    for tok, want in seg_ids.items():
        got = tokens.index(tok) if tok in tokens else -1
        if got != want:
            issues.append(f"段标记 {tok} 应在 id {want}，实际 {got if got >= 0 else '缺失'}")
    if any(len(t) > 1 and any(c in _L0_SET for c in t) for t in tokens):
        issues.append("有多字 token 含 L0 核心字")
    return issues


def check_vocab_64k(vocab_path: Path, manifest_path: Optional[Path] = None,
                    data_dir: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """校验 64K 词表与 manifest，五项硬判据 + 逐槽对账 + 哈希复算。"""
    p = Path(vocab_path)
    m = Path(manifest_path) if manifest_path else None
    lines: List[str] = []
    problems: List[str] = []

    def ok(label: str, detail: str = "") -> None:
        lines.append(f"[ OK ] {label}" + (f" — {detail}" if detail else ""))

    def bad(label: str, detail: str) -> None:
        lines.append(f"[FAIL] {label} — {detail}")
        problems.append(f"{label}: {detail}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, [f"[FAIL] 读取 {p} 失败 — {exc}"]

    tokens = list(data.get("tokens") or [])
    counts = dict(data.get("slot_counts") or {})
    quotas = slot_quotas()
    layout = dict(data.get("control_layout") or {})
    ranges = dict(data.get("slot_ranges") or {})

    # 1. 总数与预算
    if len(tokens) == VOCAB64_TOTAL and data.get("size") == VOCAB64_TOTAL:
        ok("总数 = 65536", f"size={len(tokens)}")
    else:
        bad("总数 = 65536", f"size={len(tokens)}, 字段 size={data.get('size')}")
    budget_problems = VocabBudget(alloc=counts).problems(TokenSpace())
    if not budget_problems:
        ok("VocabBudget.problems() 为空",
           f"合计 {sum(counts.values())}, 决策占比 "
           f"{len(TokenSpace().decision_tokens()) / VOCAB64_TOTAL:.2%} ≤ 2%")
    else:
        bad("VocabBudget.problems() 为空", "；".join(budget_problems))

    # 2. 逐槽对账（quota / count / 区间覆盖）
    per_slot_bad: List[str] = []
    for name in VOCAB64_SLOT_ORDER:
        n = counts.get(name)
        rs = ranges.get(name) or []
        covered = sum(b - a for a, b in rs)
        if n != quotas.get(name):
            per_slot_bad.append(f"{name}: count={n} ≠ quota={quotas.get(name)}")
        if covered != n:
            per_slot_bad.append(f"{name}: 区间合计 {covered} ≠ count {n}")
    if not per_slot_bad:
        ok("八槽位逐项对账", f"合计 {sum(counts.values())}，区间全覆盖、无重叠空洞")
    else:
        bad("八槽位逐项对账", "；".join(per_slot_bad))
    all_ids = [i for rs in ranges.values() for a, b in rs for i in range(a, b)]
    if sorted(all_ids) == list(range(len(tokens))):
        ok("槽位区间覆盖全词表", f"65536 个 id 恰好被 8 个槽覆盖一次")
    else:
        bad("槽位区间覆盖全词表", f"覆盖 {len(set(all_ids))} 个 id / 应有 {len(tokens)}")

    # 3. L0 原子性
    missing = [c for c in LIGHT_L0_CORE if tokens.count(c) != 1]
    atomic = check_l0_atomicity(tokens)
    if len(LIGHT_L0_CORE) == 44 and not missing and not atomic:
        ok("L0 44 字各占 1 token / 原子性", "check_l0_atomicity 通过")
    else:
        bad("L0 44 字各占 1 token / 原子性",
            f"缺 {missing[:5]}，原子性问题 {atomic[:3]}")

    # 4. 唯一性
    dups = [t for t, n in Counter(tokens).items() if n > 1]
    if dups:
        bad("无重复 token", f"重复 {len(dups)} 个：{dups[:5]}")
    else:
        ok("无重复 token", f"{len(tokens)} 条全唯一")

    # 5. 控制块与 UNK/PAD
    space = TokenSpace()
    expected = (list(STRUCTURE_MARKERS) + [t.text for t in space.decision_tokens()]
                + [f"<V:{c}>" for c in VALIDATION_CODES])
    start, end = int(layout.get("start", -1)), int(layout.get("end", -1))
    actual = tokens[start:end] if start >= 0 else []
    if actual == expected and end == len(tokens) - 2:
        ok("末尾控制块与 TokenSpace 对齐",
           f"4 结构 + {len(space.decision_tokens())} 决策 + {len(VALIDATION_CODES)} 验证 = "
           f"{len(expected)}，id [{start},{end})；UNK={end} PAD={len(tokens) - 1}")
    else:
        bad("末尾控制块与 TokenSpace 对齐",
            f"实际 {len(actual)} 条，end={end}，词表 {len(tokens)}")
    ids = (list((layout.get("structure") or {}).values())
           + list((layout.get("decisions") or {}).values())
           + list((layout.get("validations") or {}).values()))
    if sorted(ids) == list(range(start, end)):
        ok("控制 id 区间连续", f"[{start},{end}) 无空洞")
    else:
        bad("控制 id 区间连续", f"ids 与 [{start},{end}) 不一致")

    # 6. 段标记（T-0371 在 64K 表的落地，T-0375 补建）
    # 两条硬判据：① 实际 id 必须等于 vocab.py 的 SEGMENT_TOKEN_IDS；
    # ② 预算口径的 reserved 头部（slot_offsets["reserved"]）必须是段标记。
    budget64 = VocabBudget(alloc=counts)
    declared = dict(SEGMENT_TOKEN_IDS)
    derived = segment_token_ids(budget64)
    if declared != derived:
        bad("SEGMENT_TOKEN_IDS 与 slot_offsets 推导一致",
            f"SEGMENT_TOKEN_IDS={declared}，推导={derived}")
    seg_bad: List[str] = []
    for tok, want in declared.items():
        got = tokens.index(tok) if tok in tokens else -1
        if got != want:
            seg_bad.append(f"{tok}: 期望 id {want}，实际 {got if got >= 0 else '缺失'}")
    if not seg_bad:
        ok("段标记 id = SEGMENT_TOKEN_IDS", f"{declared}")
    else:
        bad("段标记 id = SEGMENT_TOKEN_IDS", "；".join(seg_bad))
    head_start = slot_offsets(budget64)["reserved"]
    head = tokens[head_start:head_start + len(SEGMENT_MARKERS)]
    if head == list(SEGMENT_MARKERS):
        ok("reserved 头部 = 段标记",
           f"tokens[{head_start}:{head_start + len(SEGMENT_MARKERS)}] = {head}")
    else:
        bad("reserved 头部 = 段标记",
            f"tokens[{head_start}:{head_start + len(SEGMENT_MARKERS)}] = {head}")

    # 7. manifest 对账 + 哈希复算
    if m is None:
        lines.append("[INFO] 未提供 manifest，跳过对账")
    elif not m.is_file():
        bad("manifest 存在", f"缺 {m}")
    else:
        man = json.loads(m.read_text(encoding="utf-8"))
        man_bad: List[str] = []
        if man.get("size") != len(tokens):
            man_bad.append(f"manifest.size={man.get('size')} ≠ {len(tokens)}")
        if list((man.get("budget") or {}).get("problems") or []):
            man_bad.append("manifest.budget.problems 非空")
        if (man.get("sha256") or {}).get("tokens") != _slot_sha(tokens):
            man_bad.append("manifest.sha256.tokens 与词表复算不一致")
        man_slots = dict(man.get("slots") or {})
        for name in VOCAB64_SLOT_ORDER:
            entry = dict(man_slots.get(name) or {})
            slot_tokens = [tokens[i] for a, b in (ranges.get(name) or []) for i in range(a, b)]
            if entry.get("count") != counts.get(name):
                man_bad.append(f"{name}: manifest count={entry.get('count')} ≠ {counts.get(name)}")
            if entry.get("quota") != quotas.get(name):
                man_bad.append(f"{name}: manifest quota={entry.get('quota')} ≠ {quotas.get(name)}")
            if entry.get("sha256") != _slot_sha(slot_tokens):
                man_bad.append(f"{name}: 槽 sha256 复算不一致")
        if man_bad:
            bad("manifest 逐项对账", "；".join(man_bad[:8]))
        else:
            ok("manifest 逐项对账", "size / 八槽 count+quota / 每槽 sha256 / budget.problems 全部一致")

    # 7. 字符级 UNK（自包含，口径写清楚）
    # 空格/换行是**分词分隔符**（``DemoVocab.encode`` 直接按空白切分），不含 token；
    # 与 4096 侧同一口径：UNK 只在「可见字符」上统计，否则会被空格顶到 29%。
    stats = dict(data.get("corpus_stats") or {})
    freq_all = dict(stats.get("char_freq") or {})
    if not freq_all:
        bad("字符级 UNK 可核算", "缺少 corpus_stats.char_freq")
    else:
        blanks = sum(int(n) for ch, n in freq_all.items() if ch.isspace())
        freq = {ch: n for ch, n in freq_all.items() if not ch.isspace()}
        rate, unk, total_chars = unk_rate(freq, tokens)
        ok("字符级 UNK", f"{rate:.4%} ({unk}/{total_chars} 可见字符；"
                         f"已剔除空白 {blanks} 个，kept 语料快照口径)")

    lines.append("")
    lines.append("结论：" + ("全绿" if not problems else f"{len(problems)} 项未通过"))
    return not problems, lines


def coverage_compare(vocab_paths: Sequence[Path], data_dir: Optional[Path] = None) -> Dict[str, object]:
    """同一份语料上对比多个词表的覆盖率 / token 密度（贪心最长匹配口径）。

    与训练侧 ``DemoVocab.encode`` 用同一套算法，避免「报告一套、训练一套」。
    """
    from .train_demo import DemoVocab

    root = Path(__file__).resolve().parents[3]
    texts, stats = _load_corpus_snapshot(Path(data_dir) if data_dir else root / "data")
    out: Dict[str, object] = {"corpus": {k: v for k, v in stats.items() if k != "char_freq"}}
    for path in vocab_paths:
        vocab = DemoVocab(json.loads(Path(path).read_text(encoding="utf-8")))
        total_chars = 0
        raw_chars = 0
        total_ids = 0
        unk_ids = 0
        for text in texts:
            ids = vocab.encode(text)
            total_ids += len(ids)
            raw_chars += len(text)
            total_chars += sum(1 for ch in text if not ch.isspace())
            unk_ids += sum(1 for i in ids if i == vocab.unk_id)
        out[Path(path).name] = {
            "size": vocab.size,
            "vocab_size": vocab.vocab_size,
            "raw_chars": raw_chars,
            "visible_chars": total_chars,
            "tokens": total_ids,
            "chars_per_token": round(total_chars / max(total_ids, 1), 4),
            "tokens_per_100_chars": round(100.0 * total_ids / max(total_chars, 1), 4),
            "unk_tokens": unk_ids,
            "unk_token_rate": round(unk_ids / max(total_ids, 1), 6),
            "unk_char_rate": round(unk_ids / max(total_chars, 1), 6),
        }
    return out


def train_smoke_64k(vocab_path: Path, config_path: Path, steps: int = 20,
                    out_dir: Optional[Path] = None, data_dir: Optional[Path] = None,
                    light_root: Optional[Path] = None, seq_len: int = 512,
                    batch_size: int = 8, prompts: int = 20,
                    baseline_config: Optional[Path] = None,
                    baseline_out: Optional[Path] = None,
                    verbose: bool = True) -> Dict[str, object]:
    """用 64K 词表跑 20 步真训练 + 受控生成 + ``light ai check``，可选跑 4096 对照。

    **为什么要注入**：``train_demo.DemoTrainer`` 目前把词表路径硬绑到
    ``data/unified/vocab_4096.json``，并在内部用 ``vocab.vocab_size`` 覆盖 config 的
    ``vocab_size``。该文件不在 T-0373 的 paths 内，按契约**不改**；这里在进程内注入
    ``load_demo_vocab``，复用同一个 DemoTrainer，让 64K 训练真的落在 V=65536 上。
    同时先做**尺寸对账**：config 声明的 V 与词表实际 embedding 尺寸不一致就直接报错
    （卡面 §2.2 要求「不许把 64K 词表塞给 V=4096 的模型还声称跑通」）。
    """
    import torch  # noqa: F401 - 缺 torch 时这里就失败，给出明确错误

    from . import train_demo as td
    from .infer_demo import (ConstrainedDemoGenerator, extract_code,
                             summarize_structure)

    root = Path(__file__).resolve().parents[3]
    data_dir = Path(data_dir) if data_dir else root / "data"
    out_dir = Path(out_dir) if out_dir else root / "checkpoints" / "t0373-64k-20"
    light_root = Path(light_root) if light_root else root.parent / "light-merge"

    vocab_json = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    vocab = td.DemoVocab(vocab_json)
    raw_cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    declared_v = int(raw_cfg.get("vocab_size") or 0)
    if declared_v != vocab.vocab_size:
        raise ValueError(
            f"embedding 尺寸不匹配：config {config_path.name} 声明 V={declared_v}，"
            f"词表实际 {vocab.vocab_size}；必须按词表重建 config")

    model_cfg = td._load_model_config(Path(config_path))
    prompts_list = [t for t in _task_texts(data_dir)]
    prompts_list = prompts_list[: prompts] if prompts_list else ["写一个加法程序"]
    eval_prompts = prompts_list[:prompts]

    def _eval_model(model: "torch.nn.Module", vocab_obj: td.DemoVocab) -> Dict[str, object]:
        gen = ConstrainedDemoGenerator(model, vocab_obj, max_new_tokens=48,
                                       constrain_structure=False, force_structure=False)
        texts = [gen.generate(task, max_new_tokens=48)["text"] for task in eval_prompts]
        struct = summarize_structure(texts)
        checked = passed = 0
        segments: List[Dict[str, object]] = []
        for text in texts:
            code = extract_code(text)
            if not code:
                segments.append({"code": "", "ok": None, "rc": None})
                continue
            checked += 1
            res = td._light_ai_check(light_root, code, timeout=30)
            ok = bool(res.get("ok"))
            passed += 1 if ok else 0
            segments.append({"code": code[:400], "ok": ok,
                             "rc": res.get("rc"), "error": res.get("error")})
        return {
            "structure": struct,
            "code_check": {"checked": checked, "passed": passed,
                           "empty": len(texts) - checked,
                           "rate": round(passed / checked, 4) if checked else 0.0},
            "samples": [{"task": task, "text": text[:1200]}
                        for task, text in zip(eval_prompts, texts)],
            "segments": segments,
        }

    results: Dict[str, object] = {"config": str(config_path), "vocab": str(vocab_path),
                                  "steps": steps, "prompts": len(eval_prompts)}

    # ---- 64K -----------------------------------------------------------
    cfg = td.DemoTrainConfig(config=str(config_path), steps=steps, seq_len=seq_len,
                             batch_size=batch_size, out_dir=str(out_dir),
                             data_dir=str(data_dir), light_root=str(light_root),
                             eval_every=steps, log_every=1, model=model_cfg)
    orig_loader = td.load_demo_vocab
    td.load_demo_vocab = lambda path=None: vocab  # type: ignore[assignment]
    try:
        trainer = td.DemoTrainer(cfg)
        summary = trainer.train()
        ckpt = trainer.save()
        results["64k"] = {
            "summary": summary,
            "checkpoint": str(ckpt),
            "eval": _eval_model(trainer.model, trainer.vocab),
        }
    finally:
        td.load_demo_vocab = orig_loader
    if verbose:
        print(f"[smoke64k] 64K：{steps} 步完成，checkpoint {ckpt}")

    # ---- 4096 对照（可选） ---------------------------------------------
    if baseline_config:
        b_cfg_raw = json.loads(Path(baseline_config).read_text(encoding="utf-8"))
        b_model_cfg = td._load_model_config(Path(baseline_config))
        b_out = Path(baseline_out) if baseline_out else root / "checkpoints" / "t0373-4096-20"
        b_cfg = td.DemoTrainConfig(config=str(baseline_config), steps=steps,
                                   seq_len=seq_len, batch_size=batch_size,
                                   out_dir=str(b_out), data_dir=str(data_dir),
                                   light_root=str(light_root), eval_every=steps,
                                   log_every=1, model=b_model_cfg)
        b_trainer = td.DemoTrainer(b_cfg)
        b_declared = int(b_cfg_raw.get("vocab_size") or 0)
        if b_declared != b_trainer.vocab.vocab_size:
            raise ValueError(
                f"baseline config 声明 V={b_declared} 与实际 {b_trainer.vocab.vocab_size} 不一致")
        b_summary = b_trainer.train()
        b_ckpt = b_trainer.save()
        results["4096"] = {
            "summary": b_summary, "checkpoint": str(b_ckpt),
            "eval": _eval_model(b_trainer.model, b_trainer.vocab),
        }
        if verbose:
            print(f"[smoke64k] 4096 对照：{steps} 步完成，checkpoint {b_ckpt}")
    return results


# ---- CLI ----------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lightprism.demo.vocab_build",
        description="Demo 4096 实词表构建与校验；T-0373 64K 实词表构建/校验/冒烟")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="从 light 仓语料构建 4096 词表")
    mode.add_argument("--check", metavar="VOCAB_JSON", help="校验给定 4096 词表，全绿返回 0")
    mode.add_argument("--build-64k", action="store_true",
                      help="按 DEFAULT_BUDGET 构建 64K 实词表 + manifest（T-0373）")
    mode.add_argument("--check-64k", metavar="VOCAB_JSON",
                      help="校验 64K 词表 + manifest，全绿返回 0")
    mode.add_argument("--coverage", metavar="VOCAB_JSON",
                      help="覆盖率对照（逗号分隔多个词表，如 4096.json,64k.json）")
    mode.add_argument("--train-smoke", action="store_true",
                      help="64K 词表 20 步真训练 + 生成 + light ai check（可选 4096 对照）")
    parser.add_argument("--light-root", default=os.environ.get("LIGHT_ROOT", ""),
                        help="light 仓根目录（构建/冒烟用；或设 LIGHT_ROOT）")
    parser.add_argument("--output", default=None,
                        help="词表输出路径（默认 4096→data/vocab_4096.json，64K→data/vocab_64k.json）")
    parser.add_argument("--manifest", default="data/vocab_64k.manifest.json",
                        help="64K manifest 路径")
    parser.add_argument("--data-dir", default="data", help="统一数据目录（默认 data/unified）")
    parser.add_argument("--corpus-dir", default="data/corpus", help="语料产物目录")
    parser.add_argument("--smoke-config", default="configs/demo-3m-64k.json")
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--smoke-out", default="checkpoints/t0373-64k-20")
    parser.add_argument("--baseline-config", default="",
                        help="4096 对照 config（非空则同时跑同 seeds/steps 的 4096）")
    parser.add_argument("--baseline-out", default="checkpoints/t0373-4096-20")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompts", type=int, default=20)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - 老解释器/被重定向
        pass

    def _resolve_light_root(required: bool) -> Optional[Path]:
        raw = args.light_root or str(Path(__file__).resolve().parents[3].parent / "light-merge")
        root = Path(raw).expanduser()
        if not root.exists():
            if required:
                raise SystemExit(f"[build] light 仓不存在: {root}")
            return None
        return root

    if args.build:
        if not args.light_root:
            parser.error("--build 需要 --light-root（或环境变量 LIGHT_ROOT）")
        root = Path(args.light_root).expanduser()
        if not root.exists():
            print(f"[build] light 仓不存在: {root}", file=sys.stderr)
            return 2
        try:
            build_vocab(root, Path(args.output or "data/vocab_4096.json"),
                        Path(args.corpus_dir))
        except Exception as exc:  # noqa: BLE001 - CLI 边界给一条人能看懂的错误
            print(f"[build] 失败: {exc!r}", file=sys.stderr)
            return 2
        return 0

    if args.build_64k:
        light_root = _resolve_light_root(required=False)
        output = Path(args.output or "data/vocab_64k.json")
        try:
            build_vocab_64k(light_root, output, Path(args.manifest),
                            data_dir=Path(args.data_dir))
        except Exception as exc:  # noqa: BLE001 - CLI 边界给一条人能看懂的错误
            print(f"[build64k] 失败: {exc!r}", file=sys.stderr)
            return 2
        return 0

    if args.check_64k:
        good, lines = check_vocab_64k(Path(args.check_64k), Path(args.manifest),
                                      data_dir=Path(args.data_dir))
        print("\n".join(lines))
        return 0 if good else 1

    if args.coverage:
        paths = [Path(x.strip()) for x in args.coverage.split(",") if x.strip()]
        print(json.dumps(coverage_compare(paths, Path(args.data_dir)),
                         ensure_ascii=False, indent=2))
        return 0

    if args.train_smoke:
        light_root = _resolve_light_root(required=False)
        results = train_smoke_64k(
            Path(args.output or "data/vocab_64k.json"),
            Path(args.smoke_config),
            steps=args.smoke_steps,
            out_dir=Path(args.smoke_out),
            data_dir=Path(args.data_dir),
            light_root=light_root,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            prompts=args.prompts,
            baseline_config=Path(args.baseline_config) if args.baseline_config else None,
            baseline_out=Path(args.baseline_out) if args.baseline_config else None,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    good, lines = check_vocab(Path(args.check), Path(args.corpus_dir))
    print("\n".join(lines))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
