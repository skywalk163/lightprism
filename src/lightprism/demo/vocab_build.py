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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..tokens import LIGHT_L0_CORE, VALIDATION_CODES, TokenSpace
from ..tokenizer import STRUCTURE_MARKERS, ControlTokenLayout
from ..vocab import check_l0_atomicity

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


# ---- CLI ----------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lightprism.demo.vocab_build",
        description="Demo 4096 实词表构建与校验")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="从 light 仓语料构建词表")
    mode.add_argument("--check", metavar="VOCAB_JSON", help="校验给定词表，全绿返回 0")
    parser.add_argument("--light-root", default=os.environ.get("LIGHT_ROOT", ""),
                        help="light 仓根目录（--build 必填，或设 LIGHT_ROOT）")
    parser.add_argument("--output", default="data/vocab_4096.json", help="词表输出路径")
    parser.add_argument("--corpus-dir", default="data/corpus", help="语料产物目录")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - 老解释器/被重定向
        pass

    if args.build:
        if not args.light_root:
            parser.error("--build 需要 --light-root（或环境变量 LIGHT_ROOT）")
        root = Path(args.light_root).expanduser()
        if not root.exists():
            print(f"[build] light 仓不存在: {root}", file=sys.stderr)
            return 2
        try:
            build_vocab(root, Path(args.output), Path(args.corpus_dir))
        except Exception as exc:  # noqa: BLE001 - CLI 边界给一条人能看懂的错误
            print(f"[build] 失败: {exc!r}", file=sys.stderr)
            return 2
        return 0

    good, lines = check_vocab(Path(args.check), Path(args.corpus_dir))
    print("\n".join(lines))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
