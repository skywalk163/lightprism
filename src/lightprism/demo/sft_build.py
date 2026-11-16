# -*- coding: utf-8 -*-
"""T-0333：数据配比与 JSON 输出任务集（``prompt → JSON`` / ``prompt → 代码``）。

## 这张卡解决什么

Demo 要能「输入 prompt，输出代码 **或 JSON**」，而语料里只有光明代码，没有一条
「prompt → JSON」的样本。这里用**小而精、可复核**的方式造第二批 SFT 数据：

* 6 类 JSON 任务，每类 50 条，共 300 条；
* 63 条代码任务，取自 T-0330 语料的上游 ``light/**/*.light``，且每条都真跑过
  ``light ai check --run``；
* 合计 363 条，不超过 500 条。

## 数据从哪里来（不编造）

所有素材都在 :data:`data/sft/fixtures.json` 里，逐项写明了出处：

============  ==========================================================
类别          出处
============  ==========================================================
file_list     lightprism 仓真实文件的 ``os.stat`` 字节数
diagnostic    prompt 里显式给出的指标值与阈值（答案是唯一可推的计算）
subtasks      tasks/*.md 与 Demo开发计划.md 里的真实任务标题与步骤
decision      tokens.py 的 ``EDIT`` 合法选项（FR/RW/PATCH）
validation    tokens.py 的 ``DIAGNOSTIC_TO_VALIDATION`` 真实映射（19 个码）
config        configs/300m.json、configs/1b.json、Demo开发计划.md 的真实数值
code          light 仓短小完整的 ``.light`` 程序，描述取自源码自身的注释/导出
============  ==========================================================

**没有调用任何模型批量生成样本**。生成器只是把上述真实素材套进固定模板，
答案由输入确定，可以在 ``--validate`` 里逐条重算/解析校验。

## source 字段的口径（重要）

卡片正文要求「先给 ``SOURCES`` 加 ``demo_json``，并同步 ``docs/样本导出.md``」。
但本卡允许修改的 paths 只有 ``data/sft/``、本文件、``tests/test_sft_data.py``
与 ``reports/数据配比报告.md`` —— **``src/lightprism/data.py`` 与
``docs/样本导出.md`` 不在其中**。执行契约优先，因此不改这两个文件：
本批样本的 ``source`` 使用 ``SOURCES`` 里已登记的 ``human_edit``，
并在 ``meta.source_kind`` 标记 ``demo_json``，将来负责 ``data.py`` 的卡把
``demo_json`` 登记进去后可平滑迁移。这样 ``load_jsonl`` 在**任何进程**里
直接读 ``json_tasks.jsonl`` 都不会因 source 未登记而丢行。

## 序列格式

训练序列在标准 ``<TASK>/<CODE>/<V:PASS>`` 前加 schema 前缀，让模型学会
在 JSON 模式下先输出 ``{``：

```
<FORMAT>json</FORMAT>
<TASK>列出当前目录的 py 文件及其大小</TASK>
<CODE>{"files":[{"name":"a.py","size":123}]}</CODE>
<V:PASS>
```

``<FORMAT>`` 是 demo 训练侧新增的前缀标记，不属于 ``parse_sequence`` 的
v0.1 四类片段（见 docs/序列规范.md）；因此本模块用 :func:`render_training_sequence`
单独渲染训练序列，同时仍保证每行 JSONL 是合法的 ``Sample``，能被
``load_jsonl`` 与 ``Sample.to_sequence()`` 消费。

## 用法

```
PYTHONPATH=src python -m lightprism.demo.sft_build --build
PYTHONPATH=src python -m lightprism.demo.sft_build --validate data/sft/json_tasks.jsonl
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from lightprism.data import SOURCES, Sample, Step, load_jsonl, write_jsonl
from lightprism.tokens import DIAGNOSTIC_TO_VALIDATION
from lightprism.vocab import SEGMENT_MARKERS

__all__ = [
    "CATEGORIES",
    "MAX_SAMPLES",
    "PER_CATEGORY",
    "SEGMENT_MODES",
    "SFT_SOURCE",
    "apply_segment_markers",
    "build",
    "extract_light_segment",
    "format_ok",
    "render_inference_translation",
    "render_training_sequence",
    "render_training_translation",
    "translation_mixed_segment_error",
    "validate",
]

#: 仓库根：src/lightprism/demo/sft_build.py → parents[3]
ROOT = Path(__file__).resolve().parents[3]
SFT_DIR = ROOT / "data" / "sft"
UNIFIED_DIR = ROOT / "data" / "unified"
FIXTURES_PATH = SFT_DIR / "fixtures.json"
JSON_TASKS_PATH = SFT_DIR / "json_tasks.jsonl"
CODE_TASKS_PATH = SFT_DIR / "code_tasks.jsonl"
SEQUENCES_PATH = SFT_DIR / "train_sequences.jsonl"
MANIFEST_PATH = SFT_DIR / "manifest.json"

CATEGORIES: Tuple[str, ...] = (
    "file_list",
    "diagnostic",
    "task_decomposition",
    "decision",
    "validation",
    "config",
)
CATEGORY_LABELS: Dict[str, str] = {
    "file_list": "文件清单",
    "diagnostic": "诊断结果",
    "task_decomposition": "任务分解",
    "decision": "决策选择",
    "validation": "验证结果",
    "config": "配置生成",
}
PER_CATEGORY = 50
MIN_SAMPLES = 200
MAX_SAMPLES = 500

#: 见模块 docstring「source 字段的口径」。
DEMO_JSON_SOURCE = "demo_json"
SFT_SOURCE = "human_edit"
assert SFT_SOURCE in SOURCES, f"{SFT_SOURCE} 必须已在 lightprism.data.SOURCES 中登记"

CODE_SEGMENT_RE = re.compile(r"<CODE>(.*?)</CODE>", re.DOTALL)
FORMAT_PREFIX_RE = re.compile(r"^<FORMAT>(json|code)</FORMAT>\n")


# --------------------------------------------------------------------------
# 样本构造
# --------------------------------------------------------------------------
def _sample(
    task: str,
    answer: Mapping[str, Any],
    category: str,
    *,
    origin: str,
    template: str,
) -> Sample:
    """一条 ``prompt → JSON`` 样本：``<CODE>`` 段就是紧凑 JSON 文本。"""
    code = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    return Sample(
        task=task,
        steps=[Step(decisions={}, code=code, validation="PASS")],
        source=SFT_SOURCE,
        meta={
            "category": category,
            "format": "json",
            "source_kind": DEMO_JSON_SOURCE,
            "origin": origin,
            "template": template,
        },
    )


def _file_list_samples(fx: Mapping[str, Any]) -> List[Sample]:
    """真实文件 + 真实字节数；prompt 里显式列出输入，答案由过滤/排序规则唯一确定。"""
    files: Sequence[Mapping[str, Any]] = fx["files"]
    out: List[Sample] = []
    for i in range(PER_CATEGORY):
        start = (i * 7) % len(files)
        n = 3 + i % 4
        window = [files[(start + j) % len(files)] for j in range(n)]
        listing = "、".join(f"{f['name']} {f['size']} 字节" for f in window)
        base = f"已知目录内容为：{listing}。"
        variant = i % 5
        if variant == 0:
            picked = [f for f in window if f["ext"] == ".py"]
            task = base + "列出其中的 py 文件及其大小，输出 JSON。"
            tpl = "filter_ext_py"
        elif variant == 1:
            picked = sorted(window, key=lambda f: f["size"])
            task = base + "按字节数从小到大列出文件及大小，输出 JSON。"
            tpl = "sort_by_size"
        elif variant == 2:
            picked = [f for f in window if f["size"] > 1000]
            task = base + "列出其中超过 1000 字节的文件及大小，输出 JSON。"
            tpl = "filter_size_gt_1000"
        elif variant == 3:
            picked = [f for f in window if f["ext"] == ".md"]
            task = base + "列出其中的 md 文档及其大小，输出 JSON。"
            tpl = "filter_ext_md"
        else:
            picked = list(window)
            task = base + "统计这些文件的个数和总字节数，输出 JSON。"
            tpl = "count_and_total"
        if tpl == "count_and_total":
            answer = {"count": len(picked), "total_bytes": sum(f["size"] for f in picked)}
        else:
            answer = {
                "files": [{"name": f["name"], "size": f["size"]} for f in picked],
                "count": len(picked),
            }
        out.append(_sample(task, answer, "file_list", origin="lightprism/**", template=tpl))
    return out


def _diagnostic_samples(fx: Mapping[str, Any]) -> List[Sample]:
    """指标值在 prompt 里显式给出；alert 是唯一可推的布尔结果。"""
    metrics: Sequence[Mapping[str, Any]] = fx["metrics"]
    out: List[Sample] = []
    for i in range(PER_CATEGORY):
        m = metrics[i % len(metrics)]
        values = m["values"]
        value = values[i % len(values)]
        threshold = m["threshold"]
        alert = value > threshold
        task = (
            f"{m['label']}当前为 {value}{m['unit']}，告警阈值为 {threshold}{m['unit']}。"
            f"报告{m['label']}并判断是否告警，输出 JSON。"
        )
        answer = {
            m["key"]: {
                m["field"]: value,
                "threshold": threshold,
                "alert": alert,
            }
        }
        out.append(_sample(task, answer, "diagnostic",
                           origin="data/sft/fixtures.json#metrics", template="threshold_alert"))
    return out


def _task_decomposition_samples(fx: Mapping[str, Any]) -> List[Sample]:
    """任务标题与步骤逐条抄自 tasks/*.md 与 Demo开发计划.md。"""
    tasks: Sequence[Mapping[str, Any]] = fx["tasks"]
    out: List[Sample] = []
    for i in range(PER_CATEGORY):
        t = tasks[i % len(tasks)]
        k = 3 + i % 3
        steps = list(t["steps"])[:k]
        task = f"把「{t['title']}」拆成 {len(steps)} 个子任务，输出 JSON。"
        answer = {
            "task": t["title"],
            "subtasks": [{"id": j + 1, "desc": s} for j, s in enumerate(steps)],
        }
        out.append(_sample(task, answer, "task_decomposition",
                           origin="tasks/*.md", template="decompose"))
    return out


def _decision_samples(fx: Mapping[str, Any]) -> List[Sample]:
    """FR / RW / PATCH 是 TokenSpace 里真实登记的三个 EDIT 选项。"""
    scenarios: Sequence[Mapping[str, str]] = fx["decisions"]["scenarios"]
    out: List[Sample] = []
    for i in range(PER_CATEGORY):
        sc = scenarios[i % len(scenarios)]
        task = (
            f"{sc['scenario']}。这段代码该整体重写还是局部替换？"
            "从 FR / RW / PATCH 中选一个并给出理由，输出 JSON。"
        )
        answer = {"decision": sc["option"], "reason": sc["reason"]}
        out.append(_sample(task, answer, "decision",
                           origin="src/lightprism/tokens.py#DEFAULT_DECISION_POINTS",
                           template="edit_choice"))
    return out


def _validation_samples(fx: Mapping[str, Any]) -> List[Sample]:
    """诊断码与验证码映射来自 tokens.py 的 DIAGNOSTIC_TO_VALIDATION。"""
    rows: Sequence[Mapping[str, str]] = fx["validations"]
    out: List[Sample] = []
    for i in range(PER_CATEGORY):
        v = rows[i % len(rows)]
        code = v["code"]
        token = v["validation"]
        line = 0 if code == "PASS" else 2 + i % 9
        ok = not token.startswith("ERR=")
        if code == "PASS":
            task = "检查这段代码：light ai check 没有报出任何诊断。给出验证结果，输出 JSON。"
        else:
            task = (
                f"检查这段代码：light ai check 返回 {code}（映射到 {token}），"
                f"出错行 {line}。给出验证结果，输出 JSON。"
            )
        answer = {"ok": ok, "code": code, "line": line, "validation": token}
        out.append(_sample(task, answer, "validation",
                           origin="src/lightprism/tokens.py#DIAGNOSTIC_TO_VALIDATION",
                           template="validation_result"))
    return out


def _config_samples(fx: Mapping[str, Any]) -> List[Sample]:
    """字段值来自 configs/*.json 与 Demo开发计划.md 的 4096 词表 demo 配置。"""
    presets: Sequence[Mapping[str, Any]] = fx["configs"]["presets"]
    out: List[Sample] = []
    for i in range(PER_CATEGORY):
        preset = presets[i % len(presets)]
        variant = i % 3
        if variant == 0:
            keys = ["mode", "threads", "vocab_size"]
        elif variant == 1:
            keys = ["mode", "threads", "vocab_size", "dim", "n_layers", "max_seq"]
        else:
            keys = ["name", "mode", "threads"]
        gives = "，".join(f"{k} = {preset[k]}" for k in keys)
        task = f"生成一个 JSON 配置，只包含这些字段：{gives}。只输出 JSON 对象。"
        answer = {k: preset[k] for k in keys}
        out.append(_sample(task, answer, "config",
                           origin="configs/300m.json;configs/1b.json;docs/Demo开发计划.md",
                           template="config_generate"))
    return out


GENERATORS = (
    _file_list_samples,
    _diagnostic_samples,
    _task_decomposition_samples,
    _decision_samples,
    _validation_samples,
    _config_samples,
)


def _code_samples(fx: Mapping[str, Any]) -> List[Sample]:
    """T-0330 语料上游的短小完整 .light 程序；描述取自源码自身。"""
    out: List[Sample] = []
    for item in fx["code_samples"]:
        out.append(
            Sample(
                task=item["task"],
                steps=[Step(decisions={}, code=item["code"], validation="PASS")],
                source=SFT_SOURCE,
                meta={
                    "category": "code",
                    "format": "code",
                    "source_kind": "t0330_corpus_upstream",
                    "origin": item["origin"],
                    "light_ai_check": item["check"],
                },
            )
        )
    return out


# --------------------------------------------------------------------------
# 训练序列（schema 前缀）
# --------------------------------------------------------------------------
def render_training_sequence(sample: Sample, fmt: str | None = None) -> str:
    """渲染 demo 训练序列，带 ``<FORMAT>`` schema 前缀。

    JSON 类用单行 ``<CODE>{...}</CODE>``，让模型在 ``<CODE>`` 后第一个字符
    就输出 ``{``；代码类保留多行 ``<CODE>`` 段。
    """
    fmt = fmt or str(sample.meta.get("format") or "json")
    if fmt not in ("json", "code"):
        raise ValueError(f"未知 format {fmt!r}")
    code = "\n".join(s.code for s in sample.steps)
    if fmt == "json":
        block = f"<CODE>{code}</CODE>"
    else:
        block = "<CODE>\n" + code + "\n</CODE>"
    return (
        f"<FORMAT>{fmt}</FORMAT>\n"
        f"<TASK>{sample.task}</TASK>\n"
        f"{block}\n"
        f"<V:PASS>\n"
    )


# --------------------------------------------------------------------------
# 翻译方向：共享段标记渲染器（T-0371）
# --------------------------------------------------------------------------
#: 两种序列格式。``baseline`` 是旧的双 ``<CODE>`` 块（A/B 对照用）；
#: ``markers`` 是 T-0371 的 ``<PY>`` / ``<LIGHT>`` 独立段标记。
SEGMENT_MODES: Tuple[str, ...] = ("baseline", "markers")
DEFAULT_SEGMENT_MODE = "markers"

_PYTHON_LINE_RE = re.compile(
    r"(?m)^\s*(?:def |import |from |class |print\(|return |for |while |"
    r"elif |else:|try:|except |with |lambda )"
)


def _render_translation_core(
    task: str,
    py: str,
    light: str,
    *,
    mode: str,
    include_target: bool,
) -> str:
    """训练构造与推理 prompt 的**唯一**渲染核心（T-0371 反向验证的锚点）。

    ``include_target=False`` 时只渲染到目标段起点（推理 prompt）；``True`` 时
    补齐光明段与 ``<V:PASS>``（训练序列）。两边共用本函数，所以「改了训练侧
    格式却忘了改推理 prompt」在 ``test_training_and_inference_share_renderer``
    会直接变红。
    """
    if mode not in SEGMENT_MODES:
        raise ValueError(f"未知翻译序列格式 {mode!r}；合法值: {SEGMENT_MODES}")
    task = task.strip()
    py = py.strip()
    head = f"<FORMAT>translate</FORMAT>\n<TASK>{task}</TASK>\n"
    if mode == "baseline":
        head += f"<CODE>\n{py}\n</CODE>\n"
        if not include_target:
            return head
        return head + f"<CODE>\n{light.strip()}\n</CODE>\n<V:PASS>\n"
    # mode == "markers"：<PY> 段起点 → <LIGHT> 段起点 → <V:PASS> 结束。
    head += f"<PY>\n{py}\n"
    if not include_target:
        return head
    return head + f"<LIGHT>\n{light.strip()}\n<V:PASS>\n"


def render_training_translation(
    task: str, py: str, light: str, *, mode: str = DEFAULT_SEGMENT_MODE
) -> str:
    """训练侧翻译序列（含目标光明段与验证 token）。"""
    return _render_translation_core(task, py, light, mode=mode, include_target=True)


def render_inference_translation(
    task: str, py: str, *, mode: str = DEFAULT_SEGMENT_MODE
) -> str:
    """推理侧翻译 prompt（只含任务与 Python 源段，模型续写光明段）。"""
    return _render_translation_core(task, py, "", mode=mode, include_target=False)


def render_translation_record(
    rec: Mapping[str, Any], *, mode: str = DEFAULT_SEGMENT_MODE
) -> str:
    """把 ``data/unified/translation_tasks.jsonl`` 的一行渲染成翻译序列。"""
    kind = str(rec.get("kind") or "pair")
    task = str(rec.get("instruction") or rec.get("note")
               or rec.get("category") or "").strip()
    if kind == "pair":
        return render_training_translation(
            task, str(rec.get("input") or ""), str(rec.get("output") or ""), mode=mode)
    light = str(rec.get("light") or rec.get("input") or rec.get("output") or "").strip()
    if mode == "baseline":
        return (
            "<FORMAT>light</FORMAT>\n"
            f"<TASK>{task}</TASK>\n"
            f"<CODE>\n{light}\n</CODE>\n<V:PASS>\n"
        )
    return (
        "<FORMAT>light</FORMAT>\n"
        f"<TASK>{task}</TASK>\n"
        f"<LIGHT>\n{light}\n<V:PASS>\n"
    )


def extract_light_segment(text: str, *, mode: str = DEFAULT_SEGMENT_MODE) -> str:
    """从生成结果里抽出光明代码段；抽不到返回空串（**不把全文当代码**）。"""
    if mode == "markers":
        m = re.search(r"<LIGHT>(.*?)(?:</LIGHT>|(?=<V:)|$)", text or "", re.DOTALL)
        return m.group(1).strip() if m else ""
    blocks = CODE_SEGMENT_RE.findall(text or "")
    return blocks[1].strip() if len(blocks) >= 2 else ""


def format_ok(text: str, *, mode: str = DEFAULT_SEGMENT_MODE) -> bool:
    """段格式是否可无歧义分段（空输出、缺段、乱序都判不合法）。"""
    t = text or ""
    if mode == "markers":
        if t.count("<CODE>") or "<PY>" not in t or "<LIGHT>" not in t:
            return False
        if t.index("<PY>") > t.index("<LIGHT>"):
            return False
        # 只允许各一个起点标记；重复出现说明段边界已不可判。
        if t.count("<PY>") != 1 or t.count("<LIGHT>") != 1:
            return False
        tail = t[t.index("<LIGHT>"):]
        return bool(re.search(r"<V:[A-Z_=]+>", tail)) and bool(extract_light_segment(t))
    if not format_ok_baseline(t):
        return False
    return bool(extract_light_segment(t, mode="baseline"))


def format_ok_baseline(text: str) -> bool:
    t = text or ""
    if t.count("<CODE>") != 2 or t.count("</CODE>") != 2:
        return False
    first = t.index("<CODE>")
    second = t.index("<CODE>", first + 1)
    if t.index("</CODE>") > second:
        return False
    tail = t[t.index("</CODE>", t.index("</CODE>") + 1):]
    return bool(re.search(r"<V:[A-Z_=]+>", tail))


def translation_mixed_segment_error(text: str, *, mode: str = DEFAULT_SEGMENT_MODE) -> bool:
    """混段错误（一次错误统计，口径写死）：

    * ``markers``：段边界不可判（缺/重/乱序）**或**光明段里出现 Python 源码行
      （``def`` / ``import`` / ``print(`` 等），即 Python 段漏进了光明段；
    * ``baseline``：两个 ``<CODE>`` 块不可判，**或**第二个块是 Python 源码。
    """
    t = text or ""
    if mode == "markers":
        if not format_ok(t, mode="markers"):
            return True
        light = extract_light_segment(t, mode="markers")
        return bool(_PYTHON_LINE_RE.search(light))
    if not format_ok(t, mode="baseline"):
        return True
    return bool(_PYTHON_LINE_RE.search(extract_light_segment(t, mode="baseline")))


def apply_segment_markers(vocab_path: str | Path) -> Dict[str, Any]:
    """把 ``<PY>`` / ``<LIGHT>`` 追加进 demo 4096 词表的控制区（幂等）。

    追加位置在验证 token 之后、UNK/PAD 之前，所以**旧 token 的 id 全部不变**；
    旧 checkpoint 的嵌入层仍是 4096 行，可直接加载（方案 b）。产物写回前做
    结构校验，失败不落盘。
    """
    path = Path(vocab_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens: List[str] = list(data["tokens"])
    kinds: List[str] = list(data["kinds"])
    layout = data["control_layout"]
    start, end = int(layout["start"]), int(layout["end"])
    if end != len(tokens):
        raise ValueError(f"控制区 end={end} 与词表长度 {len(tokens)} 不一致，拒绝改写")
    existing = [m for m in SEGMENT_MARKERS if m in tokens]
    if existing:
        if existing != list(SEGMENT_MARKERS):
            raise ValueError(f"段标记只存在一部分: {existing}")
        for i, m in enumerate(SEGMENT_MARKERS):
            if int(layout["structure"].get(m, -1)) != end - len(SEGMENT_MARKERS) + i:
                raise ValueError(f"段标记 {m} 的 id 不在控制区尾部，拒绝静默接受")
        return {"path": str(path), "changed": False, "ids": {
            m: int(layout["structure"][m]) for m in SEGMENT_MARKERS}}

    for offset, marker in enumerate(SEGMENT_MARKERS):
        tid = end + offset
        layout["structure"][marker] = tid
        tokens.append(marker)
        kinds.append("control")
        data["layers"]["control"].append(marker)
    data["counts"]["control"] += len(SEGMENT_MARKERS)
    data["tokens"] = tokens
    data["kinds"] = kinds
    data["size"] = len(tokens)
    data["control_layout"]["end"] = end + len(SEGMENT_MARKERS)
    data["control_layout"]["size"] = int(data["control_layout"]["end"]) - start
    limit = int(data.get("limit") or 4096)
    data["headroom"] = limit - data["size"]
    data["headroom_ratio"] = round(data["headroom"] / limit, 6)
    data["segment_markers"] = {
        "tokens": list(SEGMENT_MARKERS),
        "ids": {m: end + i for i, m in enumerate(SEGMENT_MARKERS)},
        "closure": "starter-only（段尾由下一个 <LIGHT>/<V:…> 或序列末尾确定）",
        "budget_slot": "reserved（64K 预算里 carve 前 2 个 id，总额不变）",
        "backward_compat": "旧 token id 不变；旧 checkpoint 嵌入层 4096 行可直接加载",
    }
    assert data["size"] <= limit
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return {"path": str(path), "changed": True,
            "ids": {m: end + i for i, m in enumerate(SEGMENT_MARKERS)},
            "size": data["size"]}


def _sequence_rows(samples: Iterable[Sample]) -> List[Dict[str, str]]:
    rows = []
    for s in samples:
        fmt = str(s.meta.get("format") or "json")
        rows.append({
            "format": fmt,
            "category": str(s.meta.get("category") or ""),
            "task": s.task,
            "sequence": render_training_sequence(s, fmt),
        })
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(path: Path) -> str:
    """行尾归一化后的 sha256（CRLF → LF），与 T-0370 的跨平台口径一致。"""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build() -> Dict[str, Any]:
    """从 :data:`data/sft/fixtures.json` 重建全部产物（确定性，不调用模型）。"""
    if not FIXTURES_PATH.exists():
        raise SystemExit(f"缺少素材文件: {FIXTURES_PATH}")
    fx = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

    json_samples: List[Sample] = []
    for gen in GENERATORS:
        json_samples.extend(gen(fx))
    code_samples = _code_samples(fx)
    total = len(json_samples) + len(code_samples)
    if total > MAX_SAMPLES:
        raise SystemExit(f"样本总数 {total} 超过 {MAX_SAMPLES}，拒绝产出")

    n_json = write_jsonl(json_samples, JSON_TASKS_PATH)
    n_code = write_jsonl(code_samples, CODE_TASKS_PATH)
    n_seq = _write_jsonl(SEQUENCES_PATH, _sequence_rows(json_samples + code_samples))

    by_cat = {c: 0 for c in CATEGORIES}
    for s in json_samples:
        by_cat[str(s.meta["category"])] += 1

    parsed = 0
    for s in json_samples:
        for m in CODE_SEGMENT_RE.finditer(render_training_sequence(s)):
            json.loads(m.group(1).strip())
            parsed += 1

    checked = sum(1 for s in code_samples if s.meta["light_ai_check"].get("ok"))
    manifest = {
        "schema_version": 1,
        "source_field": SFT_SOURCE,
        "source_kind": DEMO_JSON_SOURCE,
        "provenance": fx["provenance"],
        "counts": {
            "json_tasks": n_json,
            "code_tasks": n_code,
            "sequences": n_seq,
            "total": total,
            "by_category": by_cat,
        },
        "json_loads": {"code_segments": parsed, "passed": parsed,
                       "pass_rate": 1.0 if parsed else 0.0},
        "light_ai_check": {
            "command": fx["code_samples"][0]["check"]["command"] if fx["code_samples"] else "",
            "checked": checked,
            "selected": len(code_samples),
            "candidates": fx["code_check_summary"]["candidates"],
            "passed": fx["code_check_summary"]["passed"],
            "failed": fx["code_check_summary"]["failed"],
        },
        "files": {
            "json_tasks.jsonl": _sha256(JSON_TASKS_PATH),
            "code_tasks.jsonl": _sha256(CODE_TASKS_PATH),
            "train_sequences.jsonl": _sha256(SEQUENCES_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
def _read_nonempty_lines(path: Path) -> List[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(path: str | Path, *, verbose: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """双重校验：``load_jsonl`` 的 schema 校验 + ``<CODE>`` 段的 ``json.loads``。

    返回 ``(是否通过, 统计字典)``；不抛异常，便于 CLI 与测试共用。
    """
    p = Path(path)
    problems: List[str] = []
    stats: Dict[str, Any] = {"path": str(p)}

    if not p.exists():
        return False, {"problems": [f"文件不存在: {p}"]}

    lines = _read_nonempty_lines(p)
    samples, load_problems = load_jsonl(p)
    stats["rows"] = len(lines)
    stats["loaded"] = len(samples)
    stats["load_problems"] = list(load_problems)
    problems.extend(load_problems)
    if len(samples) != len(lines):
        problems.append(f"load_jsonl 只读回 {len(samples)}/{len(lines)} 行，有行被丢弃")

    if not (MIN_SAMPLES <= len(lines) <= MAX_SAMPLES):
        problems.append(f"条数 {len(lines)} 不在 {MIN_SAMPLES}—{MAX_SAMPLES} 之间")
    if len(lines) > MAX_SAMPLES:
        problems.append(f"条数 {len(lines)} 超过硬上限 {MAX_SAMPLES}")

    by_cat: Dict[str, int] = {c: 0 for c in CATEGORIES}
    for s in samples:
        cat = str(s.meta.get("category") or "")
        if cat in by_cat:
            by_cat[cat] += 1
        else:
            problems.append(f"未知 category {cat!r}: {s.task[:30]}")
    stats["by_category"] = by_cat
    for cat, n in by_cat.items():
        if not (30 <= n <= 80):
            problems.append(f"类别 {cat} 条数 {n} 不在 30—80 之间")

    # 核心：渲染带 schema 前缀的训练序列，逐段 json.loads
    segments = 0
    parsed = 0
    prefixes = 0
    for idx, s in enumerate(samples, 1):
        fmt = str(s.meta.get("format") or "json")
        if fmt != "json":
            problems.append(f"第 {idx} 行不是 JSON 类样本 (format={fmt!r})")
            continue
        seq = render_training_sequence(s, fmt)
        if FORMAT_PREFIX_RE.match(seq) and "<V:PASS>" in seq:
            prefixes += 1
        found = CODE_SEGMENT_RE.findall(seq)
        if not found:
            problems.append(f"第 {idx} 行没有 <CODE> 段")
        for raw in found:
            segments += 1
            try:
                json.loads(raw.strip())
                parsed += 1
            except json.JSONDecodeError as exc:
                problems.append(f"第 {idx} 行 <CODE> 不是合法 JSON: {exc}")
    stats["code_segments"] = segments
    stats["json_loads_passed"] = parsed
    stats["json_loads_rate"] = round(parsed / segments, 4) if segments else 0.0
    stats["format_prefix_ok"] = prefixes
    if prefixes != len(samples):
        problems.append(f"只有 {prefixes}/{len(samples)} 条带正确的 <FORMAT>/<V:PASS> 前缀")

    # 交叉校验预渲染的训练序列文件
    if SEQUENCES_PATH.exists():
        rows = [json.loads(x) for x in _read_nonempty_lines(SEQUENCES_PATH)]
        seq_json = [r for r in rows if r.get("format") == "json"]
        seq_ok = 0
        for r in seq_json:
            if not FORMAT_PREFIX_RE.match(r["sequence"]):
                problems.append("train_sequences.jsonl 有条记录缺 <FORMAT> 前缀")
            for raw in CODE_SEGMENT_RE.findall(r["sequence"]):
                try:
                    json.loads(raw.strip())
                    seq_ok += 1
                except json.JSONDecodeError:
                    problems.append("train_sequences.jsonl 有 <CODE> 段不是合法 JSON")
        stats["sequence_rows"] = len(rows)
        stats["sequence_json_segments_ok"] = seq_ok
        if seq_json and seq_ok < len(seq_json):
            problems.append("train_sequences.jsonl 的 JSON 段未全部通过")

    # 代码类样本：schema 兼容 + 记录过的 light ai check 结果
    if CODE_TASKS_PATH.exists():
        code_samples, code_problems = load_jsonl(CODE_TASKS_PATH)
        problems.extend(code_problems)
        n_code = len(_read_nonempty_lines(CODE_TASKS_PATH))
        stats["code_tasks"] = n_code
        if not (50 <= n_code <= 100):
            problems.append(f"代码类样本 {n_code} 条，不在 50—100 之间")
        checked = 0
        for s in code_samples:
            if s.meta.get("light_ai_check", {}).get("ok"):
                checked += 1
            if not s.meta.get("origin"):
                problems.append("代码类样本缺 meta.origin")
        stats["code_tasks_checked"] = checked
        if checked != len(code_samples):
            problems.append(f"只有 {checked}/{len(code_samples)} 条代码样本有通过的 light ai check")
        stats["total_samples"] = len(lines) + n_code
        if stats["total_samples"] > MAX_SAMPLES:
            problems.append(f"JSON + 代码合计 {stats['total_samples']} 超过 {MAX_SAMPLES}")
    else:
        stats["code_tasks"] = 0
        stats["code_tasks_checked"] = 0
        stats["total_samples"] = len(lines)

    stats["problems"] = problems
    ok = not problems
    if verbose:
        print(f"[validate] {p}")
        print(f"  条数                : {stats['rows']} (要求 {MIN_SAMPLES}—{MAX_SAMPLES})")
        print(f"  load_jsonl 读回     : {stats['loaded']}，问题 {len(load_problems)} 条")
        print(f"  各类条数            : {by_cat}")
        print(f"  <CODE> 段 json.loads: {parsed}/{segments}，通过率 {stats['json_loads_rate']:.2%}")
        print(f"  <FORMAT> 前缀       : {prefixes}/{len(samples)}")
        print(f"  代码类样本          : {stats['code_tasks']} 条，"
              f"light ai check 通过 {stats['code_tasks_checked']} 条")
        print(f"  合计                : {stats['total_samples']} 条（上限 {MAX_SAMPLES}）")
        if problems:
            print(f"  [FAIL] {len(problems)} 个问题：")
            for msg in problems[:12]:
                print(f"    - {msg}")
        else:
            print("  [PASS] 全部校验通过")
    return ok, stats


def rebuild_unified_sequences(data_dir: str | Path | None = None) -> Dict[str, Any]:
    """重建 ``data/unified/`` 的段标记产物（T-0371，确定性 + 幂等）。

    做三件事：
    1. 把 ``<PY>`` / ``<LIGHT>`` 追加进 ``vocab_4096.json``（旧 id 不变）；
    2. 给 ``translation_tasks.jsonl`` 每行补 ``sequence_marked``（训练格式）；
    3. 更新 ``manifest.json`` 的 vocab size / control span / segment ids / sha256。
    """
    import datetime as _dt

    out = Path(data_dir) if data_dir else UNIFIED_DIR
    vocab_path = out / "vocab_4096.json"
    trans_path = out / "translation_tasks.jsonl"
    manifest_path = out / "manifest.json"
    if not vocab_path.is_file() or not trans_path.is_file():
        raise SystemExit(f"缺少统一数据产物: {vocab_path} / {trans_path}")

    seg = apply_segment_markers(vocab_path)
    rows: List[Dict[str, Any]] = []
    n_marked = 0
    for line in trans_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec["sequence_marked"] = render_translation_record(rec, mode="markers")
        n_marked += 1
        rows.append(rec)
    tmp = trans_path.with_suffix(trans_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(trans_path)

    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    layout = vocab["control_layout"]
    manifest.setdefault("sha256", {})["vocab_4096.json"] = _sha256_text(vocab_path)
    manifest["sha256"]["translation_tasks.jsonl"] = _sha256_text(trans_path)
    manifest["vocab"] = {
        "file": "vocab_4096.json",
        "size": int(vocab["size"]),
        "control_span": [int(layout["start"]), int(layout["end"])],
        "segment_markers": dict(vocab.get("segment_markers", {}).get("ids", {})),
        "note": "demo/train_openi 共用本副本；T-0371 追加 <PY>/<LIGHT>，旧 token id 不变",
    }
    manifest["translation_build"] = dict(manifest.get("translation_build") or {})
    manifest["translation_build"]["sequence_formats"] = {
        "baseline": "双 <CODE> 块（旧格式，A/B 对照）",
        "markers": "``<PY>`` → ``<LIGHT>`` → ``<V:…>``（T-0371 训练格式）",
    }
    manifest["translation_build"]["sequence_marked_rows"] = n_marked
    manifest["task"] = "T-0371"
    manifest["generated_at"] = _dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return {"vocab": seg, "translation_rows": n_marked,
            "manifest": str(manifest_path),
            "sha256": manifest["sha256"]}


# --------------------------------------------------------------------------
# 小规模 A/B 对照（T-0371）：同词表 / 同 seed / 同步数，只换段格式
# --------------------------------------------------------------------------
def _load_translation_rows(data_dir: str | Path | None = None) -> List[Dict[str, Any]]:
    path = (Path(data_dir) if data_dir else UNIFIED_DIR) / "translation_tasks.jsonl"
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _split_heldout(rows: Sequence[Mapping[str, Any]], n_eval: int,
                   seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按 ``sha256`` 稳定排序后取 ``n_eval`` 条做 held-out；不按模板切，只保证不重叠。"""
    ordered = sorted((dict(r) for r in rows),
                     key=lambda r: str(r.get("sha256") or hashlib.sha256(
                         (str(r.get("input")) + str(r.get("output"))).encode("utf-8")
                     ).hexdigest()))
    n_eval = max(0, min(n_eval, len(ordered) - 1))
    # 均匀抽，避免只拿到同一类别（文件前部通常按类别聚集）。
    if n_eval:
        step = max(1, len(ordered) // n_eval)
        eval_rows = ordered[::step][:n_eval]
    else:
        eval_rows = []
    eval_ids = {id(r) for r in eval_rows}
    train_rows = [r for r in ordered if id(r) not in eval_ids]
    return eval_rows, train_rows


def _generate_unconstrained(model, vocab, prompt: str, max_new_tokens: int) -> str:
    """裸 argmax 生成（**不接状态机**，对应卡片的能力列口径）。"""
    import torch

    model.eval()
    ids: List[int] = list(vocab.encode(prompt))
    max_seq = int(model.cfg.max_seq)
    start = len(ids)
    stop_ids = set(vocab.validations.values())
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = torch.tensor([ids[-max_seq:]], dtype=torch.long)
            logits, _ = model(window)
            nxt = int(torch.argmax(logits[0, -1]))
            ids.append(nxt)
            if nxt in stop_ids:
                break
    return vocab.decode(ids[start:])


def _translation_ce(model, vocab, rows: Sequence[Mapping[str, Any]], *,
                    mode: str, seq_len: int = 512) -> float | None:
    """held-out 翻译序列的 teacher-forced 平均交叉熵（按 token，不按样本）。"""
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for rec in rows:
            text = render_translation_record(rec, mode=mode)
            ids = vocab.encode(text)[:seq_len]
            if len(ids) < 2:
                continue
            inp = torch.tensor([ids[:-1]], dtype=torch.long)
            tgt = torch.tensor([ids[1:]], dtype=torch.long)
            weights = torch.ones_like(tgt, dtype=torch.float32)
            _, loss = model(inp, tgt, loss_weights=weights)
            total += float(loss) * (len(ids) - 1)
            count += len(ids) - 1
    return round(total / count, 5) if count else None


def _evaluate_translation(ckpt: Path, rows: Sequence[Mapping[str, Any]], *,
                          mode: str, max_new_tokens: int = 64) -> Dict[str, Any]:
    """生成 → 抽光明段 → ``light ai check`` 真跑；空输出/抽不到一律判不合法。"""
    import lightprism.demo.train_demo as td
    from lightprism.demo.infer_demo import load_demo_generator

    gen = load_demo_generator(ckpt, max_new_tokens=max_new_tokens,
                              constrain_structure=False, mask_empty_slots=True)
    light_root = td._resolve_light_root()
    items: List[Dict[str, Any]] = []
    for rec in rows:
        task = str(rec.get("instruction") or rec.get("note") or rec.get("category") or "")
        prompt = render_inference_translation(task, str(rec.get("input") or ""), mode=mode)
        text = _generate_unconstrained(gen.model, gen.vocab, prompt, max_new_tokens)
        light = extract_light_segment(text, mode=mode)
        fmt_ok = format_ok(text, mode=mode) and bool(light)
        if light:
            check = td._light_ai_check(light_root, light)
        else:
            check = {"ok": False, "mode": "syntax", "diagnostics": [],
                     "duration_ms": None, "rc": None, "empty": True}
        items.append({
            "sha256": rec.get("sha256"),
            "format_ok": bool(fmt_ok),
            "empty": not bool(light.strip()),
            "light_check_ok": bool(check.get("ok")),
            "mixed": translation_mixed_segment_error(text, mode=mode),
            "generated": text[:600],
            "light_excerpt": light[:300],
        })
    n = len(items)
    def _rate(key: str) -> float:
        return round(sum(1 for x in items if x[key]) / n, 4) if n else 0.0
    return {
        "mode": mode,
        "samples": n,
        "format_rate": _rate("format_ok"),
        "empty_rate": _rate("empty"),
        "light_check_pass_rate": _rate("light_check_ok"),
        # 结构合法率（产品口径）：格式可判 + 光明段非空 + light ai check 放行；
        # 全分母，空输出判不合法，不剔除截断。
        "structure_legal_rate": round(sum(
            1 for x in items if x["format_ok"] and not x["empty"] and x["light_check_ok"]
        ) / n, 4) if n else 0.0,
        "mixed_segment_error_rate": _rate("mixed"),
        "items": items,
    }


def run_ablation(*, steps: int = 120, out_dir: str | Path = "/tmp/t0371-ablation",
                 eval_samples: int = 16, seed: int = 20261112,
                 max_new_tokens: int = 64,
                 progress: Callable[[str], None] | None = None) -> Dict[str, Any]:
    """同 seed / 同步数跑 baseline 与 markers 两组，产出真实对照数字。

    只 monkeypatch **实验进程内**的 ``train_demo.render_translation_samples``：
    生产训练文件的渲染路径不在本卡 paths 内，不能改；实验通过共享渲染器
    :func:`render_translation_record` 把两种格式喂给同一个 ``DemoTrainer``。
    """
    import lightprism.demo.train_demo as td

    say = progress or (lambda msg: print(msg, flush=True))
    rows = [r for r in _load_translation_rows()
            if str(r.get("kind") or "pair") == "pair"
            and str(r.get("input") or "").strip()
            and str(r.get("output") or "").strip()]
    if len(rows) < 20:
        raise SystemExit(f"对照语料不足（{len(rows)} 条），无法做 A/B")
    eval_rows, train_rows = _split_heldout(rows, eval_samples, seed)
    say(f"[ablation] 语料 {len(rows)} 条：train {len(train_rows)} / held-out {len(eval_rows)}；"
        f"steps={steps} seed={seed}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    original = td.render_translation_samples
    model_cfg = td._load_model_config(ROOT / "configs" / "demo-3m.json")
    results: Dict[str, Any] = {"config": {
        "steps": steps, "seed": seed, "eval_samples": len(eval_rows),
        "max_new_tokens": max_new_tokens, "model": model_cfg,
        "note": "除段格式外同词表/同 seed/同步数/同超参",
    }}
    try:
        for mode in SEGMENT_MODES:
            def _fake(data_dir=None, _mode=mode):
                return [(str(r.get("kind") or "pair"),
                         render_translation_record(r, mode=_mode)) for r in train_rows]
            td.render_translation_samples = _fake
            cfg = td.DemoTrainConfig(
                steps=steps, seq_len=512, batch_size=8, data_dir=str(UNIFIED_DIR),
                out_dir=str(out / mode), seed=seed,
                log_every=max(1, steps // 10), eval_every=max(10, steps // 5),
                model=dict(model_cfg),
            )
            say(f"[ablation] {mode}: 训练 {steps} 步 …")
            trainer = td.DemoTrainer(cfg)
            summary = trainer.train()
            ckpt = trainer.save()
            ce = _translation_ce(trainer.model, trainer.vocab, eval_rows, mode=mode)
            gen = _evaluate_translation(ckpt, eval_rows, mode=mode,
                                        max_new_tokens=max_new_tokens)
            results[mode] = {
                "val_history": summary.get("val_history"),
                "best_val_loss": summary.get("best_val_loss"),
                "final_train_loss": summary.get("final_loss"),
                "translation_ce": ce,
                "generation": gen,
                "checkpoint": str(ckpt),
                "elapsed_s": summary.get("elapsed_s"),
                "dataset": {k: summary.get("dataset", {}).get(k) for k in
                            ("sample_counts", "sample_repeats", "control_tokens",
                             "control_ratio", "train_blocks", "val_blocks")},
            }
            say(f"[ablation] {mode}: val={summary.get('best_val_loss')} "
                f"CE={ce} 结构合法率={gen['structure_legal_rate']} "
                f"混段错误率={gen['mixed_segment_error_rate']}")
    finally:
        td.render_translation_samples = original
    (out / "ablation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results["output"] = str(out / "ablation.json")
    return results


def _stats() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LightPrism SFT 数据构建 / 校验")
    ap.add_argument("--build", action="store_true", help="从 fixtures.json 重建全部数据")
    ap.add_argument("--validate", metavar="JSONL", help="校验 JSON 任务集（含 <CODE> 段 json.loads）")
    ap.add_argument("--stats", action="store_true", help="打印 manifest 统计")
    ap.add_argument("--apply-segment-markers", metavar="VOCAB_JSON",
                    help="把 <PY>/<LIGHT> 追加进 demo 词表（幂等）")
    ap.add_argument("--rebuild-translation-sequences", action="store_true",
                    help="重建 unified 词表段标记 + 翻译序列 + manifest")
    ap.add_argument("--ablation", action="store_true", help="跑 baseline/markers 小规模 A/B")
    ap.add_argument("--steps", type=int, default=120, help="A/B 每组训练步数")
    ap.add_argument("--eval-samples", type=int, default=16, help="A/B held-out 条数")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="A/B 生成预算")
    ap.add_argument("--seed", type=int, default=20261112, help="A/B 固定 seed")
    ap.add_argument("--out-dir", default="/tmp/t0371-ablation", help="A/B 产物目录")
    args = ap.parse_args(argv)

    if args.apply_segment_markers:
        print(json.dumps(apply_segment_markers(args.apply_segment_markers),
                         ensure_ascii=False, indent=2))
        return 0
    if args.rebuild_translation_sequences:
        print(json.dumps(rebuild_unified_sequences(), ensure_ascii=False, indent=2))
        return 0
    if args.ablation:
        res = run_ablation(steps=args.steps, out_dir=args.out_dir,
                           eval_samples=args.eval_samples, seed=args.seed,
                           max_new_tokens=args.max_new_tokens)
        slim = {k: v for k, v in res.items() if k != "config"}
        print(json.dumps(slim, ensure_ascii=False, indent=2))
        return 0
    if args.build:
        m = build()
        print(json.dumps(m["counts"], ensure_ascii=False, indent=2))
        return 0
    if args.validate:
        ok, _ = validate(args.validate)
        return 0 if ok else 1
    if args.stats:
        print(json.dumps(_stats(), ensure_ascii=False, indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
