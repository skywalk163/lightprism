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
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from lightprism.data import SOURCES, Sample, Step, load_jsonl, write_jsonl
from lightprism.tokens import DIAGNOSTIC_TO_VALIDATION

__all__ = [
    "CATEGORIES",
    "MAX_SAMPLES",
    "PER_CATEGORY",
    "SFT_SOURCE",
    "build",
    "render_training_sequence",
    "validate",
]

#: 仓库根：src/lightprism/demo/sft_build.py → parents[3]
ROOT = Path(__file__).resolve().parents[3]
SFT_DIR = ROOT / "data" / "sft"
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


def _stats() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LightPrism SFT 数据构建 / 校验")
    ap.add_argument("--build", action="store_true", help="从 fixtures.json 重建全部数据")
    ap.add_argument("--validate", metavar="JSONL", help="校验 JSON 任务集（含 <CODE> 段 json.loads）")
    ap.add_argument("--stats", action="store_true", help="打印 manifest 统计")
    args = ap.parse_args(argv)

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
