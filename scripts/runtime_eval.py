# -*- coding: utf-8 -*-
"""阶段 0 验收入口：跑窄任务集 → 产出决策日志 → 输出闭环可用率与失败明细。

    python3 scripts/runtime_eval.py --tasks data/runtime/tasks.jsonl

判据（任务卡 §3）：``light ai check`` 通过 **且** 真跑输出与预期一致；
全分母，空输出 / 截断 / checker 失败一律计入失败，不剔除。

其它用法
--------
* ``--make-tasks``：用模板的独立 Python 语义重建 ``data/runtime/tasks.jsonl``
  （预期结果不参与运行时代码生成，只用于最后比对）。
* ``--report-out reports/零模型系统报告.md``：按本次真实运行结果生成报告，
  失败样例逐条全贴，不挑成功样例。
* ``--json-out``：机器可读摘要。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lightprism.runtime.checker import LightRunner  # noqa: E402
from lightprism.runtime.decisions import (  # noqa: E402
    OBSERVABLE_VALIDATION_CODES, DecisionWriter, validate_record,
)
from lightprism.runtime.engine import (  # noqa: E402
    RETRIEVAL_METHOD, TaskOutcome, load_tasks,
)
from lightprism.runtime.engine import RuntimeEngine  # noqa: E402
from lightprism.runtime.retrieval import BM25Index  # noqa: E402
from lightprism.runtime.templates import (  # noqa: E402
    CLUSTER_LABELS, TEMPLATES, expected_of, get_template,
)

DEFAULT_TASKS = REPO_ROOT / "data" / "runtime" / "tasks.jsonl"
DEFAULT_DECISIONS = REPO_ROOT / "data" / "runtime" / "decisions.jsonl"

# --------------------------------------------------------------------------
# 任务集：文本只给「用户会怎么说」，预期输出由模板的独立 Python 语义算出。
# 系统运行时**看不到** template_id / slots / 这段表，只能从 text 里检索与抽槽。
# --------------------------------------------------------------------------
TaskDef = Dict[str, Any]

TASK_DEFS: List[TaskDef] = [
    # ---- 算术表达式 ----
    dict(id="A01", cluster="arith", tpl="arith_binop", text="计算 3 加 4 的结果，输出数字。",
         slots={"A": 3, "B": 4, "OP": "加"}),
    dict(id="A02", cluster="arith", tpl="arith_binop", text="计算 10 减 3 的差，输出数字。",
         slots={"A": 10, "B": 3, "OP": "减"}),
    dict(id="A03", cluster="arith", tpl="arith_binop", text="计算 6 乘 7 的积，输出数字。",
         slots={"A": 6, "B": 7, "OP": "乘"}),
    dict(id="A04", cluster="arith", tpl="arith_binop", text="计算 20 除 4 的商，输出数字。",
         slots={"A": 20, "B": 4, "OP": "除"}),
    # ---- 单函数 / 段落 ----
    dict(id="F01", cluster="func", tpl="paragraph_inc", text="写一个段落函数把输入数加上 2，并对 5 求值，输出结果。",
         slots={"K": 2, "N": 5}),
    dict(id="F02", cluster="func", tpl="factorial", text="写递归段落求 5 的阶乘，输出结果。",
         slots={"N": 5}),
    dict(id="F03", cluster="func", tpl="loop_sum", text="累加从 1 到 5 的连续整数，输出总和。",
         slots={"N": 5}),
    dict(id="F04", cluster="func", tpl="paragraph_inc", text="写一个段落函数把输入数加上 10，并对 7 求值，输出结果。",
         slots={"K": 10, "N": 7}),
    # ---- 列表处理 ----
    dict(id="L01", cluster="list", tpl="list_sum", text="列表 [1, 2, 3, 4] 求和，输出所有元素的总和。",
         slots={"LIST": "[1, 2, 3, 4]"}),
    dict(id="L02", cluster="list", tpl="list_len", text="统计列表 [10, 20, 30] 的长度，输出元素个数。",
         slots={"LIST": "[10, 20, 30]"}),
    dict(id="L03", cluster="list", tpl="list_index", text="取列表 [10, 20, 30] 的第 1 个元素，输出该元素。",
         slots={"LIST": "[10, 20, 30]", "I": 1}),
    dict(id="L04", cluster="list", tpl="list_max", text="求列表 [4, 8, 2] 的最大值，输出结果。",
         slots={"LIST": "[4, 8, 2]"}),
    dict(id="L05", cluster="list", tpl="list_sum", text="列表 [5, 10, 15] 求和，输出所有元素的总和。",
         slots={"LIST": "[5, 10, 15]"}),
    # ---- 文本 / 文件处理 ----
    dict(id="T01", cluster="text", tpl="text_split_count", text="统计文本「a,b,c」按逗号分割后的段数，输出结果。",
         slots={"TEXT": "a,b,c", "SEP": ","}),
    dict(id="T02", cluster="text", tpl="text_count", text="统计文本「banana」中字符「a」的出现次数，输出结果。",
         slots={"TEXT": "banana", "CH": "a"}),
    dict(id="T03", cluster="text", tpl="text_affix", text="判断文件名「hello.log」是否以后缀「.log」结尾，输出判断结果。",
         slots={"TEXT": "hello.log", "PART": ".log", "FUNC": "结尾"}),
    dict(id="T04", cluster="text", tpl="text_affix", text="判断日志行「ERROR: disk full」是否以前缀「ERROR」开头，输出判断结果。",
         slots={"TEXT": "ERROR: disk full", "PART": "ERROR", "FUNC": "开头"}),
    dict(id="T05", cluster="text", tpl="text_replace", text="把文本「a b c」中的空格去掉，输出清理后的文本。",
         slots={"TEXT": "a b c", "OLD": " ", "NEW": ""}),
    # ---- 系统诊断（输入内嵌在任务文本里，输出确定，不依赖真实机器状态） ----
    dict(id="S01", cluster="sysdiag", tpl="sysdiag_threshold",
         text="系统诊断：给定磁盘用量 [70, 85, 40, 92]，统计超过阈值 80 的个数。",
         slots={"LIST": "[70, 85, 40, 92]", "TH": 80}),
    dict(id="S02", cluster="sysdiag", tpl="sysdiag_state",
         text="系统诊断：进程状态文本「run,stop,run」中状态「run」出现的次数。",
         slots={"TEXT": "run,stop,run", "STATE": "run"}),
    dict(id="S03", cluster="sysdiag", tpl="sysdiag_avg",
         text="系统诊断：温度读数 [20, 25, 30] 的平均值是多少，输出结果。",
         slots={"LIST": "[20, 25, 30]"}),
    dict(id="S04", cluster="sysdiag", tpl="sysdiag_threshold",
         text="系统诊断：磁盘用量 [10, 90, 95, 20] 中超过阈值 50 的个数。",
         slots={"LIST": "[10, 90, 95, 20]", "TH": 50}),
    # ---- 小 CLI / 格式化输出 ----
    dict(id="C01", cluster="cli", tpl="cli_format",
         text="命令行工具输出格式化结果行：结果=7，数值取 7。", slots={"N": 7}),
    dict(id="C02", cluster="cli", tpl="cli_report",
         text="小 CLI 输出两行汇总报告，数据列表 [2, 4, 6]，输出总数与总和。",
         slots={"LIST": "[2, 4, 6]"}),
    # ---- 边界 / 真实失败 ----
    dict(id="E01", cluster="edge", tpl="list_index",
         text="取列表 [10, 20, 30] 的第 5 个元素；若索引越界则输出最后一个元素 30。",
         slots={"LIST": "[10, 20, 30]", "I": 5}, expected="30\n",
         note="首次展开按字面索引会越界；预期系统依据 L3001 诊断做 find-replace 修复后输出最后一个元素"),
    dict(id="E02", cluster="edge", tpl=None,
         text="把这段音频转写成乐谱并渲染五线谱，输出乐谱摘要。",
         expected="乐谱摘要: C大调 4/4\n",
         note="模板库没有音频/乐谱能力，检索应未命中；这是真实失败样例，不剔除分母"),
]


def make_task_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in TASK_DEFS:
        expected = d.get("expected")
        if expected is None:
            expected = expected_of(get_template(d["tpl"]), d["slots"])
        rows.append({
            "task_id": d["id"], "cluster": d["cluster"], "text": d["text"],
            "expected": expected, "note": d.get("note", ""),
        })
    return rows


def write_tasks(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = make_task_rows()
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    index = BM25Index()
    print(f"已生成 {len(rows)} 条任务 -> {path}")
    for r, d in zip(rows, TASK_DEFS):
        top = index.search(r["text"], top_k=1)[0]
        want = d["tpl"]
        flag = "OK " if (want is not None and top.template_id == want) else (
            "NO_MATCH " if want is None and not top.trigger_hits else "MISMATCH")
        print(f"  {flag:9s} {r['task_id']} 期望={want} 实得={top.template_id} "
              f"score={top.score:.2f} triggers={list(top.trigger_hits)}")


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def _attempt_rows(outcome: TaskOutcome) -> List[Dict[str, Any]]:
    """把一次任务的全部 attempt 摊平，失败 / 修复样例共用，不挑样例。"""
    return [
        {
            "attempt": a.attempt,
            "template_id": a.template_id,
            "edit_mode": a.edit_mode,
            "validation_code": a.validation_code,
            "checker_error_codes": a.checker_error_codes,
            "decision_tokens": a.record.decision_tokens,
            "edit_action": a.record.retrieval.get("edit_action", ""),
            "stdout": a.stdout,
            "failure_reason": a.failure_reason,
            "source": a.source,
        }
        for a in outcome.attempts
    ]


def build_summary(
    outcomes: Sequence[TaskOutcome],
    records: Sequence[Mapping[str, Any]],
    runner: LightRunner,
    elapsed: float,
) -> Dict[str, Any]:
    total = len(outcomes)
    ok = sum(1 for o in outcomes if o.final_ok)
    clusters: Dict[str, Dict[str, Any]] = {}
    for o in outcomes:
        c = clusters.setdefault(o.cluster, {"total": 0, "ok": 0, "failures": []})
        c["total"] += 1
        if o.final_ok:
            c["ok"] += 1
        else:
            c["failures"].append(o.task_id)
    for c in clusters.values():
        c["rate"] = round(c["ok"] / c["total"], 4) if c["total"] else 0.0

    failures: List[Dict[str, Any]] = []
    for o in outcomes:
        if o.final_ok:
            continue
        failures.append({
            "task_id": o.task_id,
            "cluster": o.cluster,
            "text": o.text,
            "expected": o.expected,
            "failure_reason": o.failure_reason,
            "retries": o.retries,
            "retrieval_matched": o.retrieval.get("matched"),
            "top_templates": [h["template_id"] for h in o.retrieval.get("top", [])[:3]],
            "attempts": _attempt_rows(o),
        })

    # 成功但经历过真实重试的修复闭环：证明 <D:FIX>/<D:RETRY> 来自实际动作
    repairs: List[Dict[str, Any]] = [
        {
            "task_id": o.task_id, "cluster": o.cluster, "text": o.text,
            "expected": o.expected, "retries": o.retries,
            "attempts": _attempt_rows(o),
        }
        for o in outcomes if o.final_ok and o.retries > 0
    ]

    decision_points: Counter = Counter()
    decision_options: Counter = Counter()
    validation_codes: Counter = Counter()
    templates_used: Counter = Counter()
    retry_counts: Counter = Counter()
    bad_records: List[str] = []
    for rec in records:
        issues = validate_record(rec)
        if issues:
            bad_records.append(f"{rec.get('task_id')}#{rec.get('attempt')}: {'; '.join(issues)}")
        for point, option in (rec.get("decisions") or {}).items():
            decision_points[point] += 1
            decision_options[f"{point}={option}"] += 1
        validation_codes[str(rec.get("validation_code"))] += 1
        templates_used[str(rec.get("template_id"))] += 1
        retry_counts[int(rec.get("retry") or 0)] += 1

    from lightprism.tokens import DEFAULT_DECISION_POINTS, TokenSpace
    all_points = [p.name for p in DEFAULT_DECISION_POINTS]
    all_options = [f"{p.name}={o}" for p in DEFAULT_DECISION_POINTS for o in p.options]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "env": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "light_root": str(runner.light_root) if runner.light_root else "",
            "light_available": runner.available(),
            "retrieval": RETRIEVAL_METHOD,
            "checker_origin": "light ai check --json --no-corpus --run",
            "runner": "light run",
            "elapsed_sec": round(elapsed, 1),
        },
        "total_tasks": total,
        "ok_tasks": ok,
        "closed_loop_rate": round(ok / total, 4) if total else 0.0,
        "failed_tasks": total - ok,
        "by_cluster": clusters,
        "failures": failures,
        "repairs": repairs,
        "decision_log_records": len(records),
        "decision_token_total": sum(decision_points.values()),
        "decision_point_counts": dict(decision_points),
        "decision_option_counts": dict(decision_options),
        "unobserved_decision_points": [p for p in all_points if p not in decision_points],
        "unobserved_decision_options": [o for o in all_options if o not in decision_options],
        "validation_code_counts": dict(validation_codes),
        "unobserved_validation_codes": [c for c in OBSERVABLE_VALIDATION_CODES
                                        if c not in validation_codes],
        "template_counts": dict(templates_used),
        "retry_counts": {str(k): v for k, v in sorted(retry_counts.items())},
        "invalid_records": bad_records,
        "task_count_by_cluster": {k: v["total"] for k, v in clusters.items()},
    }


def print_summary(s: Mapping[str, Any]) -> None:
    print()
    print("=" * 72)
    print("阶段 0 零模型确定性系统 · 闭环评估")
    print("=" * 72)
    print(f"环境：{s['env']['platform']} / Python {s['env']['python']}")
    print(f"light 仓：{s['env']['light_root'] or '不可用'}")
    print(f"检索：{s['env']['retrieval']}")
    print(f"验证：<V> 一律取自 {s['env']['checker_origin']} 的真实返回；真跑输出取自 {s['env']['runner']}")
    print(f"耗时：{s['env']['elapsed_sec']}s")
    print()
    print(f"闭环可用率：{s['ok_tasks']}/{s['total_tasks']} = "
          f"{s['closed_loop_rate'] * 100:.2f}%（全分母，失败 {s['failed_tasks']} 条）")
    print()
    print("按模板簇：")
    for cluster, c in s["by_cluster"].items():
        label = CLUSTER_LABELS.get(cluster, cluster)
        print(f"  {cluster:<8s} {label:<16s} {c['ok']:>2d}/{c['total']:<2d} "
              f"= {c['rate'] * 100:6.2f}%  失败={','.join(c['failures']) or '-'}")
    print()
    repairs = s.get("repairs") or []
    if repairs:
        print("修复闭环样例（成功但经历真实重试）："
              + ", ".join("%s(%d次)" % (r["task_id"], r["retries"]) for r in repairs))
    print(f"决策日志：{s['decision_log_records']} 条，决策 token {s['decision_token_total']} 个")
    print("  决策点分布：" + ", ".join(f"{k}={v}" for k, v in sorted(s["decision_point_counts"].items())))
    print("  未观测决策点：" + (", ".join(s["unobserved_decision_points"]) or "无"))
    print("  验证码分布：" + ", ".join(f"{k}={v}" for k, v in sorted(s["validation_code_counts"].items())))
    print("  未观测验证码：" + (", ".join(s["unobserved_validation_codes"]) or "无"))
    print("  重试次数分布：" + ", ".join(f"retry={k}:{v}" for k, v in s["retry_counts"].items()))
    if s["invalid_records"]:
        print(f"  [警告] {len(s['invalid_records'])} 条日志校验不通过：{s['invalid_records'][:3]}")
    else:
        print("  日志校验：全部通过（字段齐 / 验证码可观测 / 决策合法 / checker 可回溯）")
    print()
    print(f"失败明细（全贴，共 {len(s['failures'])} 条，不挑样例）：")
    if not s["failures"]:
        print("  （本次无失败任务）")
    for f in s["failures"]:
        print(f"  - {f['task_id']} [{f['cluster']}] {f['text']}")
        print(f"      预期 stdout: {f['expected']!r}")
        print(f"      最终失败: {f['failure_reason']}")
        print(f"      检索候选: {f['top_templates']}  重试 {f['retries']} 次")
        for a in f["attempts"]:
            print(f"      attempt {a['attempt']}: 模板={a['template_id']} 编辑={a['edit_mode']} "
                  f"验证={a['validation_code']} 诊断码={a['checker_error_codes']}")
            print(f"          stdout={a['stdout']!r}  原因={a['failure_reason']}")
    print("=" * 72)


def markdown_report(s: Mapping[str, Any]) -> str:
    """按本次真实运行结果生成报告：失败样例逐条全贴，不挑成功样例。"""
    e = s["env"]
    lines: List[str] = []
    add = lines.append
    add("# 零模型系统报告（阶段 0）")
    add("")
    add(f"> 生成时间：{s['generated_at']}　|　任务卡：T-0346　|　代码仓：lightprism")
    add("> 本报告由 `python3 scripts/runtime_eval.py --tasks data/runtime/tasks.jsonl "
        "--report-out reports/零模型系统报告.md` 依据**本次真实运行**自动生成，"
        "失败样例逐条全贴，不挑成功样例。")
    add("")
    add("## 一、系统与环境")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 平台 | {e['platform']} |")
    add(f"| Python | {e['python']} |")
    add(f"| light 仓 | `{e['light_root'] or '不可用'}`（可用：{e['light_available']}） |")
    add(f"| 检索方式 | {e['retrieval']}（无词向量、无 LLM） |")
    add(f"| 验证信号 `<V>` | `{e['checker_origin']}` 的真实 JSON 返回 |")
    add(f"| 真跑输出 | `{e['runner']}`（check 的 JSON **不含**程序 stdout，必须另跑） |")
    add(f"| 本次耗时 | {e['elapsed_sec']}s |")
    add("")
    add("流程：任务文本 → 意图归一 / BM25+规则检索 → 选 `light ai snippets` 占位符模板 → "
        "正则填槽 → 确定性展开 `.light` → `light ai check --json --no-corpus --run` → "
        "`light run` 真跑比对预期 → 不过则换候选 / 带诊断反馈 find-replace 重试 1–2 次。")
    add("")
    add("**判据**：`check` 通过 **且** 真跑输出与预期一致。不拿 checker exit 0 当分子"
        "（`docs/Demo验收报告.md` §3.4 实测空文件也判 `ok:true`），不做 JSON 精确匹配，"
        "空输出 / 截断 / checker 自身失败一律计入失败分母。")
    add("")
    add("## 二、闭环可用率（全分母）")
    add("")
    add(f"**{s['ok_tasks']} / {s['total_tasks']} = {s['closed_loop_rate'] * 100:.2f}%**，"
        f"失败 {s['failed_tasks']} 条全部计入分母。")
    add("")
    add("| 模板簇 | 说明 | 条数 | 可用 | 可用率 | 失败任务 |")
    add("|---|---|---:|---:|---:|---|")
    for cluster, c in s["by_cluster"].items():
        add(f"| `{cluster}` | {CLUSTER_LABELS.get(cluster, cluster)} | {c['total']} | "
            f"{c['ok']} | {c['rate'] * 100:.2f}% | {', '.join(c['failures']) or '-'} |")
    add("")
    add("## 三、失败样例（逐条全贴）")
    add("")
    if not s["failures"]:
        add("本次运行没有失败任务。")
    for f in s["failures"]:
        add(f"### {f['task_id']} · `{f['cluster']}`")
        add("")
        add(f"- 任务文本：{f['text']}")
        add(f"- 预期执行结果（stdout）：`{f['expected']!r}`")
        add(f"- 检索命中：{f['retrieval_matched']}；候选模板：`{f['top_templates']}`")
        add(f"- 重试次数：{f['retries']}")
        add(f"- 最终失败原因：{f['failure_reason']}")
        add("")
        add("| attempt | 模板 | 编辑策略 | 验证码 | 真实诊断码 | 实得 stdout | 失败原因 |")
        add("|---:|---|---|---|---|---|---|")
        for a in f["attempts"]:
            add(f"| {a['attempt']} | `{a['template_id']}` | `{a['edit_mode']}` | "
                f"`{a['validation_code']}` | `{a['checker_error_codes']}` | "
                f"`{a['stdout']!r}` | {a['failure_reason']} |")
        add("")
        add("<details><summary>每次 attempt 的 .light 源码</summary>")
        add("")
        for a in f["attempts"]:
            add(f"attempt {a['attempt']} / `{a['template_id']}`：")
            add("")
            add("```light")
            add(a["source"].rstrip())
            add("```")
            add("")
        add("</details>")
        add("")
    add("## 三·补 修复闭环样例（成功但经历真实重试）")
    add("")
    add("这一节贴的是**最终成功**但中途真的失败过、并依据 checker 诊断做了实际编辑动作的任务。"
        "它证明日志里的 `<D:RETRY=YES>` / `<D:FIX=REFACTOR>` / `<D:EDIT=FR>` 不是标注出来的，"
        "而是那一轮实际发生了「换候选 / find-replace」之后才落的盘。")
    add("")
    if not s["repairs"]:
        add("本次运行没有「先失败后修复成功」的任务。")
    for r in s["repairs"]:
        add(f"### {r['task_id']} · `{r['cluster']}`（重试 {r['retries']} 次）")
        add("")
        add(f"- 任务文本：{r['text']}")
        add(f"- 预期执行结果：`{r['expected']!r}` → 最终达成")
        add("")
        add("| attempt | 模板 | 编辑 | 验证码 | 真实诊断码 | stdout | 实际动作 / 决策 token |")
        add("|---:|---|---|---|---|---|---|")
        for a in r["attempts"]:
            toks = " ".join(f"`{t}`" for t in a["decision_tokens"])
            add(f"| {a['attempt']} | `{a['template_id']}` | `{a['edit_mode']}` | "
                f"`{a['validation_code']}` | `{a['checker_error_codes']}` | `{a['stdout']!r}` | "
                f"{a['edit_action']}<br>{toks} |")
        add("")
        add("<details><summary>每次 attempt 的 .light 源码</summary>")
        add("")
        for a in r["attempts"]:
            add(f"attempt {a['attempt']}：")
            add("")
            add("```light")
            add(a["source"].rstrip())
            add("```")
            add("")
        add("</details>")
        add("")
    add("## 四、决策 / 验证日志分布")
    add("")
    add(f"- 日志条数：**{s['decision_log_records']}**（每次 attempt 一条，含重试）")
    add(f"- 决策 token 总数：**{s['decision_token_total']}**")
    add(f"- 日志校验：{'全部通过' if not s['invalid_records'] else str(s['invalid_records'][:5])}")
    add("")
    add("### 4.1 决策点分布（`<D:...>`）")
    add("")
    add("| 决策点 | 条数 |")
    add("|---|---:|")
    for k, v in sorted(s["decision_point_counts"].items()):
        add(f"| `{k}` | {v} |")
    add("")
    add("选项分布：")
    add("")
    add("| 选项 | 条数 |")
    add("|---|---:|")
    for k, v in sorted(s["decision_option_counts"].items()):
        add(f"| `<D:{k}>` | {v} |")
    add("")
    add(f"**未观测到的决策点**：{', '.join('`' + x + '`' for x in s['unobserved_decision_points']) or '无'}。")
    add(f"**未观测到的选项**（共 {len(s['unobserved_decision_options'])} 个）："
        + (", ".join("`" + x + "`" for x in s["unobserved_decision_options"]) or "无") + "。")
    add("原因如实说明：本系统只对本地临时文件做自动执行（无人工确认节点，故 `SAFE=L1/L2` 不出现）；"
        "任务全部单模板直接展开（无子任务分解，故 `PLAN=DECOMPOSE` 不出现）；"
        "修复动作只有「换候选重写」与「find-replace 改边界」，方向都记为 `REFACTOR`，"
        "故 `FIX=ADD_TYPE/RENAME`、`EDIT=PATCH`、`CTX=DROP` 未观测。"
        "这些不是缺失数据，是零模型系统当前真实能力边界，**不补造**。")
    add("")
    add("### 4.2 验证码分布（`<V:...>`）")
    add("")
    add("只保留可观测的 3 类，全部来自真实 checker payload：")
    add("")
    add("| 验证码 | 条数 | 来源 |")
    add("|---|---:|---|")
    origin = {
        "PASS": "check 无 error 级诊断",
        "ERR=RUNTIME": "诊断码 L3001（列表索引越界）",
        "ERR=CHECKER_FAIL": "L0000 / 未登记码 / checker 未跑成（检查器未细分，保留原始码）",
    }
    for code in OBSERVABLE_VALIDATION_CODES:
        add(f"| `{code}` | {s['validation_code_counts'].get(code, 0)} | {origin[code]} |")
    add("")
    add(f"**未观测到的验证码**：{', '.join('`' + x + '`' for x in s['unobserved_validation_codes']) or '无'}。")
    add("`reports/ERR样本覆盖.md` 实测 8 类里 7 类不可达（SYNTAX / TYPE_MISMATCH / "
        "MISSING_TYPE / UNDEFINED_VAR / IMPORT_FAIL / NON_CANONICAL 等全回落 `L0000`）。"
        "本报告**不拿 `L0000` 冒充具体错误类型**：原始诊断码完整保留在每条日志的 "
        "`checker_diagnostics` / `checker_error_codes` 里，只把验证 token 归为 "
        "`ERR=CHECKER_FAIL`，留待 light 诊断粒度细化后再细分。")
    add("")
    add("### 4.3 模板与重试")
    add("")
    add("| 模板 | 次数 |")
    add("|---|---:|")
    for k, v in sorted(s["template_counts"].items()):
        add(f"| `{k}` | {v} |")
    add("")
    add("重试分布：" + ", ".join(f"`retry={k}` {v} 条" for k, v in s["retry_counts"].items()) + "。")
    add("")
    add("## 五、监督信号：为什么这批日志能替代空标签 SFT")
    add("")
    add("任务卡背景：本机 `--self-check` 在 363 条空标签 SFT 上得到决策密度 "
        "**0.083–0.096%**（方案要求 1–2%）、决策点 test 覆盖 **1/7**、`err_tasks` 仅 **8 条**。"
        "缺的是监督信号，不是 GPU 或超参。")
    add("")
    add(f"本卡产出 `data/runtime/decisions.jsonl` 共 **{s['decision_log_records']}** 条，"
        f"覆盖 **{len(s['decision_point_counts'])}/7** 个决策点、"
        f"**{len(s['decision_option_counts'])}** 个决策选项、"
        f"**{len(s['validation_code_counts'])}** 类真实验证码。每条记录都带：")
    add("")
    add("- `source` + `source_sha256`：真实展开的 `.light` 代码体（`<CODE>` 内容）；")
    add("- `decisions` / `decision_tokens`：由**实际动作**映射的 `<D:...>`；")
    add("- `validation_code` / `validation_token_text`：由真实 checker 返回映射的 `<V:...>`；")
    add("- `checker_cmd` / `checker_cwd` / `checker_rc` / `checker_payload` / "
        "`checker_diagnostics` / `checker_stdout`：**可回溯到那一次真实 checker 调用**；")
    add("- `run_cmd` / `run_rc` / `stdout` / `expected_stdout` / `failure_reason`：真跑比对证据；")
    add("- `retry` / `edit_action` / `retrieval`：重试与编辑序列，可直接导出修复对。")
    add("")
    add("即：`<D>` 从实际编辑动作、`<V>` 从 `light ai check` 日志、`<CODE>` 从实际展开源码，"
        "三者按 attempt 对齐，**无需人工标注**（方案一 §4.3 原本就这么规定）。"
        "这批日志是**监督信号源**，还不是最终训练序列；密度与配比需在后续导出阶段按 "
        "§8 的 1–2% 上限核算，本报告不宣称已达成训练指标。")
    add("")
    add("## 六、纪律与边界")
    add("")
    add("- 不训练任何模型、不用别的模型批量生成样本（T-0333 硬约束）；纯标准库 + light CLI。")
    add("- 不编造决策日志；`<V>` 不许模型自判，不许拿 `L0000` 冒充具体错误类型。")
    add("- 不拿 checker exit 0 当分子；不剔除失败分母；空输出 / 截断按失败计入。")
    add("- 预期执行结果由**独立于 .light 代码的 Python 语义**离线算出，运行时不参与代码生成，"
        "避免自己给自己判卷。")
    add("- 系统诊断类任务的输入内嵌在任务文本里（不读真实机器状态），保证输出可复现；"
        "这是阶段 0 的窄任务边界，不代表已具备真实系统工具调用能力。")
    add("- 检索是零模型 BM25 + 关键词规则，报告如实标注；未命中的任务保留为真实失败样例。")
    add("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段 0 零模型系统闭环评估")
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS), help="窄任务集 JSONL")
    ap.add_argument("--decisions", default=str(DEFAULT_DECISIONS), help="决策日志输出 JSONL")
    ap.add_argument("--light-root", default="", help="light 仓根（默认自动探测 / LIGHT_ROOT）")
    ap.add_argument("--workers", type=int, default=4, help="并发任务数")
    ap.add_argument("--max-retries", type=int, default=2, help="单任务最大重试次数")
    ap.add_argument("--timeout", type=float, default=240.0, help="单次 light 子进程超时（秒）")
    ap.add_argument("--append", action="store_true", help="追加而非重写决策日志")
    ap.add_argument("--json-out", default="", help="机器可读摘要输出路径")
    ap.add_argument("--report-out", default="", help="Markdown 报告输出路径")
    ap.add_argument("--make-tasks", action="store_true", help="重建任务集后退出")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    args = ap.parse_args(argv)

    if args.make_tasks:
        write_tasks(Path(args.tasks))
        return 0

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    runner = LightRunner(args.light_root or None, timeout=args.timeout)
    if not runner.available():
        print("[警告] 找不到 light 仓（cli/light.py）。本次所有 checker 调用将如实记为 "
              "ERR=CHECKER_FAIL，不会伪造通过。可设置 LIGHT_ROOT 或 --light-root。",
              file=sys.stderr)

    decisions_path = Path(args.decisions)
    writer = DecisionWriter(decisions_path, append=args.append)
    engine = RuntimeEngine(runner, writer, max_retries=args.max_retries)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        outcomes = list(pool.map(engine.run_task, tasks))
    elapsed = time.monotonic() - started

    summary = build_summary(outcomes, writer.records(), runner, elapsed)
    print_summary(summary)

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"机器可读摘要 -> {p}")
    if args.report_out:
        p = Path(args.report_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(summary), encoding="utf-8")
        print(f"报告 -> {p}")
    print(f"决策日志 -> {decisions_path}（{writer.count} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
