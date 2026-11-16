# -*- coding: utf-8 -*-
"""ERR 类样本补强：只给**实测可达**的诊断码造正样本。

2026-11-16 定因复测（28 个片段、逐个真跑 ``light ai check --json --no-corpus --run``）
的结论：这台 light（gitcode/main ``89948c47a``）只有 **L3001（列表索引越界）** 与
**L2002（关键字作变量名）** 被细分成功，其余语法/类型/未定义/导入/除零等**全部回落
L0000**（详见 ``DIAGNOSTIC_REACHABILITY`` 的逐码定因）。所以本模块只产
L3001（``ERR=RUNTIME``）与 L0000（``ERR=CHECKER_FAIL``）两类样本，不伪造
L100x/L200x/L400x——``L2002`` 虽然在真机可达，但它映射到的 ``ERR=MISSING_TYPE``
不在可观测验证码集合，按口径只能记 ``ERR=CHECKER_FAIL``。

每一条样本都是先真跑、读到该诊断码之后才写进数据的（``build_entries`` 会在
任何一条对不上时剔除），不是手写猜出来的。

用法::

    PYTHONPATH=src python -m lightprism.runtime.err_samples --check   # 只探针
    PYTHONPATH=src python -m lightprism.runtime.err_samples --write   # 真跑+落盘
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .checker import LightRunner
from .decisions import OBSERVABLE_VALIDATION_CODES, map_check_payload

__all__ = [
    "CATALOG",
    "CODE_META",
    "DIAGNOSTIC_REACHABILITY",
    "SAMPLE_MIN_PER_CODE",
    "TEST_MIN_PER_CODE",
    "load_catalog_result",
    "build_entries",
    "write_unified",
    "refresh_manifest",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
UNIFIED_ERR = REPO_ROOT / "data" / "unified" / "err_tasks.jsonl"
UNIFIED_DIR = REPO_ROOT / "data" / "unified"
MANIFEST = UNIFIED_DIR / "manifest.json"
RUNTIME_TASKS = REPO_ROOT / "data" / "runtime" / "tasks.jsonl"
RUNTIME_DECISIONS = REPO_ROOT / "data" / "runtime" / "decisions.jsonl"

#: 每个可达码至少要有的样本条数（任务卡「ERR 类补 test 正样本」）
SAMPLE_MIN_PER_CODE = 10
#: 每个可达码在 test 侧至少要有的条数
TEST_MIN_PER_CODE = 3

#: 诊断码 → （验证码, 是否运行时阶段）。验证码只取 OBSERVABLE_VALIDATION_CODES 里的。
CODE_META: Dict[str, Dict[str, str]] = {
    "L3001": {
        "validation": "ERR=RUNTIME",
        "title": "列表索引越界",
        "stage": "运行",
        "edit": "FR",
    },
    "L0000": {
        "validation": "ERR=CHECKER_FAIL",
        "title": "未细分诊断（回落 L0000）",
        "stage": "",
        "edit": "RW",
    },
}

#: 2026-11-16 定因复测矩阵：``tokens.py`` 里**已登记**的每个诊断码 → 一条输入片段
#: + 这台 light 上**实测**拿到的码 + 定因备注。
#:
#: ``measured`` 是跑出来的结果，不是期望值：能拿到细分码说明「可达」，拿到
#: ``L0000`` 就是兜底。``snippet=None`` 表示本轮还没构造出能触发它的片段，
#: 必须在 ``reason`` 里写清——不允许留空，也不允许把 L0000 写成细分码。
#:
#: 定因结论（``light`` 侧，见 ``cli/light.py`` 的 ``_ai_explain`` → ``registry.explain``）：
#: 规则**都已经接线**——每条诊断都会过一次 `registry.explain(message)`，没有兜底
#: 优先级抢码，匹配不上才由 ``cli/light.py`` 发 ``L0000``。真正的原因是
#: **码表里的正则 pattern 与 light 实际产出的报错文本对不上**（多为英文 parser 文案、
#: 或措辞与 pattern 的「该函数来自『…』模块」「没有 math 模块」等不一致）。
#: 因此修复点在 light 仓的 ``src/diagnostics/registry.py``（pattern 或上报文本），
#: 不在本仓；本仓只负责记录矩阵 + 回归用例，等 light 改了再由用例提醒更新。
DIAGNOSTIC_REACHABILITY: Dict[str, Dict[str, Any]] = {
    "L3001": {
        "snippet": "设 甲 为 [1, 2]\n打印(甲[9])",
        "measured": "L3001",
        "reason": "可达：运行期报「列表索引越界」，码表 pattern 对得上",
    },
    "L2002": {
        "snippet": "设 空 为 []\n打印(空[0])",
        "measured": "L2002",
        "reason": "可达：关键字「空」作变量名 → 语法检查阶段报 L2002（旧报告把它列为不可达，实测纠正）",
    },
    "L0000": {
        "snippet": "设 甲 为 (\n",
        "measured": "L0000",
        "reason": "兜底码本身可达：括号不闭合，语法检查阶段没命中任何细分 pattern",
    },
    "L1001": {
        "snippet": "def 加一(甲)：\n  返回 甲 加 1。\n",
        "measured": "L0000",
        "reason": "实际报错文本是「无法识别的语法元素：'：'」，L1001 的 pattern 只认 def/function 关键字",
    },
    "L2001": {
        "snippet": "段落 加一 接收 甲：\n  返回 甲。\n打印(加一(1, 2))。\n",
        "measured": "L0000",
        "reason": "实际只报「参数缺少类型标注」的警告，L2001 的 pattern 认的是「参数数量错误」文案",
    },
    "L2003": {
        "snippet": "段落 加一 接收 甲 为 1：\n  返回 甲。\n打印(加一())。\n",
        "measured": "L0000",
        "reason": "实际报「期望 冒号「:」，但得到 数字」，L2003 的 pattern 认的是默认参数写法",
    },
    "L4001": {
        "snippet": "设 甲 为 1。\n打印(乙)。\n",
        "measured": "L0000",
        "reason": "实际报「名称错误 name '乙' is not defined」（英文），L4001 的 pattern 认「缺导入/该函数来自『…』模块」",
    },
    "L4002": {
        "snippet": "导入《不存在的模块》。\n",
        "measured": "L0000",
        "reason": "实际报「模块未找到 No module named …」，L4002 的 pattern 认「从 math 导入 / print( / 没有 math 模块」",
    },
}
for _code in ("L1002", "L1003", "L1004", "L1005", "L1006", "L1007",
              "L3002", "L3003", "L3004", "L5001", "L5002", "L5003"):
    DIAGNOSTIC_REACHABILITY[_code] = {
        "snippet": None,
        "measured": None,
        "reason": "本轮未构造出能触发它的片段（只覆盖到上表 8 个码），留待 light 细化诊断后再补",
    }

#: 候选片段。前 8 条 L3001 与 data/sft/err_tasks.jsonl 现有样本同源（保持续接）。
CATALOG: Dict[str, List[str]] = {
    "L3001": [
        "设 甲 为 [1, 2]\n打印(甲[9])",
        "设 乙 为 [10, 20, 30]\n打印(乙[5])",
        "设 丙 为 [5, 6, 7, 8]\n打印(丙[10])",
        "设 数 为 [1, 2, 3]\n设 i 为 5\n打印(数[i])",
        "设 空 为 []\n打印(空[0])",
        "设 表 为 [[1, 2], [3, 4]]\n打印(表[1][5])",
        "打印(\"测试\")\n设 甲 为 [1, 2, 3]\n打印(甲[99])",
        "设 无关 为 123\n设 丁 为 [100, 200]\n打印(丁[3])\n打印(无关)",
        "设 戊 为 [7, 8]\n打印(戊[3])",
        "设 己 为 [1, 2, 3, 4, 5]\n设 位 为 7\n打印(己[位])",
        "设 庚 为 [「甲」, 「乙」]\n打印(庚[2])",
        "设 表 为 [[1, 2], [3, 4]]\n设 i 为 2\n打印(表[i][0])",
        "设 数 为 [9]\n打印(数[1])",
        "设 辛 为 [1, 2, 3]\n设 末 为 3\n打印(辛[末])",
    ],
    "L0000": [
        "设 甲 为 (\n",
        "打印(甲)。\n",
        "设 甲 为 1 加 「文」。\n打印(甲)。\n",
        "段落 加一(数)：\n  返回 数 加 1。\n打印(加一(2))。\n",
        "从《不存在的模块》导入《甲》。\n",
        "打印(不存在的函数(1))。\n",
        "设 甲 为 [1, 2]。\n打印(甲[「甲」])。\n",
        "打印(1 除 0)。\n",
        "设 甲 为 「文」 减 1。\n打印(甲)。\n",
        "设 总数 为 3。\n打印(总和)。\n",
        "段落 加一(数 为 甲)：\n  返回 甲 加 1。\n打印(加一(2))。\n",
        "设 甲 为 1。\n打印(甲 加 未定义数)。\n",
        "如果 甲 大于 0：\n  打印(甲)。\n",
        "设 甲 为 真。\n打印(甲 加 1)。\n",
    ],
}


class SampleError(RuntimeError):
    """某条候选片段的实测诊断码与预期不符。"""


def _sha256(path: Path) -> str:
    """文件内容的 sha256，先把 CRLF 归一成 LF 再算。

    直接 hash ``read_bytes()`` 会让同一条数据在 Windows（工作树 CRLF）与
    Linux/macOS（工作树 LF）下算出两个值：manifest 在 Windows 上生成、提交时
    被 git 归一成 LF，落在仓里就对不上，``test_unified_manifest_tracks_err_tasks``
    必红（T-0370 现场）。归一化之后 sha 描述的是「内容」而不是「某个检出的行尾」。
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _probe(runner: LightRunner, snippet: str) -> Dict[str, Any]:
    """把片段写成临时文件真跑一次 check，返回实测诊断信息。"""
    with tempfile.TemporaryDirectory(prefix="err-sample-") as tmp:
        f = Path(tmp) / "sample.light"
        f.write_text(snippet if snippet.endswith("\n") else snippet + "\n", encoding="utf-8")
        res = runner.check(str(f))
    outcome = map_check_payload(
        res.payload if res.payload else None, res.error,
        ok_flag=(res.payload or {}).get("ok"),
    )
    diags = res.payload.get("diagnostics") if res.payload else []
    first = diags[0] if diags else {}
    return {
        "rc": res.rc,
        "validation": outcome.code,
        "codes": list(outcome.error_codes),
        "stage": str(first.get("stage") or ""),
        "line": first.get("line"),
        "message": str(first.get("message") or outcome.checker_error or ""),
        "elapsed_ms": res.elapsed_ms,
    }


def load_catalog_result(
    runner: LightRunner,
    *,
    catalog: Optional[Mapping[str, Sequence[str]]] = None,
    out_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """逐个片段真跑，把实测结果摊平成可打印/可断言的列表（不改数据）。

    给了 ``out_path`` 就**逐条实时**追加到 JSONL，进程中途被杀也能保住已跑完的部分
    （``--write --from-json`` 直接复用这份实测结果，不必再跑第二遍）。
    """
    rows: List[Dict[str, Any]] = []
    catalog = catalog or CATALOG
    fh = out_path.open("w", encoding="utf-8") if out_path else None
    try:
        for code, snippets in catalog.items():
            for variant, snippet in enumerate(snippets):
                got = _probe(runner, snippet)
                row = {
                    "diagnostic_code": code,
                    "variant": variant,
                    "snippet": snippet,
                    "got_code": got["codes"][0] if got["codes"] else "",
                    "validation": got["validation"],
                    "stage": got["stage"],
                    "line": got["line"],
                    "message": got["message"],
                    "rc": got["rc"],
                }
                rows.append(row)
                if fh is not None:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
    finally:
        if fh is not None:
            fh.close()
    return rows


def entries_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_variants: int = 8,
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """把**实测**结果（真跑得到的诊断码）组装成 err_tasks 记录。

    只有 ``got_code == 目标码`` 且 ``validation`` 对得上的行才会进数据——候选片段里
    那些实测跑成别的码 / 直接 PASS 的，一律剔除并列进 ``rejected`` 返回（写进报告）。
    ``strict=True`` 时任何一条对不上就直接抛错（默认关，因为候选表本来就允许试探）。
    最后仍要过 ``_assert_gate``：每个码的条数与 test 侧条数不够就不写盘。
    """
    entries: List[Dict[str, Any]] = []
    rejected: List[str] = []
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    for row in rows:
        code = str(row["diagnostic_code"])
        if code not in CODE_META:
            rejected.append(f"{code}#{row.get('variant')}: 未登记的目标码")
            continue
        meta = CODE_META[code]
        want = meta["validation"]
        hit = str(row.get("got_code") or "") == code and str(row.get("validation") or "") == want
        if not hit:
            rejected.append(
                f"{code}#{row.get('variant')}: 实测 {row.get('got_code') or '(无码)'} / "
                f"{row.get('validation')!r}（期望 {code} / {want}）"
            )
            continue
        variant = int(row.get("variant") or 0)
        split = "train" if variant < train_variants else "test"
        entries.append({
            "task": f"修复光明代码中的错误（{code}）：{meta['title']}",
            "steps": [{
                "decisions": {"EDIT": meta["edit"]},
                "code": row["snippet"],
                "validation": want,
                "validation_detail": str(row.get("message") or meta["title"]),
            }],
            "meta": {
                "category": "err",
                "format": "code",
                "diagnostic_code": code,
                "stage": str(row.get("stage") or meta["stage"]),
                "source": "light_ai_check",
                "generated_at": generated_at,
                "variant": variant,
                "split": split,
                "probe_rc": row.get("rc"),
                "probe_line": row.get("line"),
            },
        })

    if rejected and strict:
        raise SampleError(
            "以下候选片段的实测诊断码与预期不符（不写盘，先修候选表）：\n  "
            + "\n  ".join(rejected)
        )
    _assert_gate(entries, [str(r["diagnostic_code"]) for r in rows])
    return entries, rejected


def build_entries(
    runner: LightRunner,
    *,
    catalog: Optional[Mapping[str, Sequence[str]]] = None,
    train_variants: int = 8,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """真跑候选片段 → 只保留实测命中目标诊断码的 → 组装成 err_tasks 记录。"""
    rows = load_catalog_result(runner, catalog=catalog)
    return entries_from_rows(rows, train_variants=train_variants)


def _assert_gate(
    entries: Sequence[Mapping[str, Any]],
    codes: Sequence[str] = (),
) -> None:
    by_code: Dict[str, List[Mapping[str, Any]]] = {}
    for e in entries:
        by_code.setdefault(e["meta"]["diagnostic_code"], []).append(e)
    problems = []
    for code in (codes or tuple(CATALOG)):
        rows = by_code.get(code, [])
        test_rows = [r for r in rows if r["meta"]["split"] == "test"]
        if len(rows) < SAMPLE_MIN_PER_CODE:
            problems.append(f"{code} 只有 {len(rows)} 条，少于 {SAMPLE_MIN_PER_CODE}")
        if len(test_rows) < TEST_MIN_PER_CODE:
            problems.append(f"{code} 的 test 侧只有 {len(test_rows)} 条，少于 {TEST_MIN_PER_CODE}")
        for r in rows:
            if r["steps"][0]["validation"] == "PASS":
                problems.append(f"{code}#{r['meta']['variant']} 是 PASS，不能当 ERR 正样本")
            if r["steps"][0]["validation"] not in OBSERVABLE_VALIDATION_CODES:
                problems.append(f"{code}#{r['meta']['variant']} 验证码不在可观测集合内")
    if problems:
        raise SampleError("样本集未达门禁：\n  " + "\n  ".join(problems))


def write_unified(entries: Sequence[Mapping[str, Any]], path: Path = UNIFIED_ERR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )


def refresh_manifest(
    entries: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path = MANIFEST,
    runtime_decisions: Path = RUNTIME_DECISIONS,
) -> Dict[str, Any]:
    """更新 manifest：err_tasks 的 sha/条数/切分 + runtime 决策日志快照。

    ``data/unified/decisions.jsonl`` 不在本卡允许路径内，**不刷新**；把它的现状
    （条数 + sha）如实记进 manifest，并注明需要 ``--build-unified`` 才算同步。
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = data.setdefault("counts", {})
    shas = data.setdefault("sha256", {})

    err_split = {"train": 0, "test": 0}
    for e in entries:
        err_split[e["meta"]["split"]] = err_split.get(e["meta"]["split"], 0) + 1
    counts["err_tasks"] = len(entries)
    counts["err_tasks_split"] = err_split
    shas["err_tasks.jsonl"] = _sha256(UNIFIED_ERR)

    runtime_rows = _count_lines(runtime_decisions)
    unified_decisions = UNIFIED_DIR / "decisions.jsonl"
    data["runtime"] = {
        "tasks": {"file": "data/runtime/tasks.jsonl",
                  "lines": _count_lines(RUNTIME_TASKS),
                  "sha256": _sha256(RUNTIME_TASKS)},
        "decisions": {"file": "data/runtime/decisions.jsonl",
                      "lines": runtime_rows,
                      "sha256": _sha256(runtime_decisions)},
        "note": "runtime 决策日志是本卡的真实产物；unified 副本不在本卡允许路径内",
    }
    if unified_decisions.is_file():
        data["decisions_snapshot"] = {
            "file": "data/unified/decisions.jsonl",
            "lines": _count_lines(unified_decisions),
            "sha256": _sha256(unified_decisions),
            "note": "本卡未刷新该副本（不在允许路径内）；需 --build-unified 才能与 runtime 对齐",
        }
    data["task"] = "T-0370"
    data["generated_at"] = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _report_build(entries: Sequence[Mapping[str, Any]], rejected: Sequence[str]) -> None:
    write_unified(entries)
    data = refresh_manifest(entries)
    split = data["counts"]["err_tasks_split"]
    print(f"已写入 {len(entries)} 条 ERR 样本 -> {UNIFIED_ERR}")
    print(f"切分：train {split['train']} / test {split['test']}")
    by_code: Dict[str, int] = {}
    for e in entries:
        by_code[e["meta"]["diagnostic_code"]] = by_code.get(e["meta"]["diagnostic_code"], 0) + 1
    print("每码条数：" + ", ".join(f"{k}={v}" for k, v in sorted(by_code.items())))
    if rejected:
        print(f"剔除的候选片段 {len(rejected)} 条（实测跑到别的码或直接 PASS，不写进数据）：")
        for r in rejected:
            print(f"  - {r}")
    print(f"manifest 已刷新：{MANIFEST}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ERR 样本补强（只造实测可达的诊断码）")
    ap.add_argument("--check", action="store_true", help="只真跑探针，打印实测诊断码")
    ap.add_argument("--write", action="store_true", help="真跑 + 写 err_tasks + 刷 manifest")
    ap.add_argument("--json", action="store_true", help="探针结果以 JSON 打印")
    ap.add_argument("--json-out", default="", help="把探针结果逐条实时写成 JSONL")
    ap.add_argument("--from-json", default="", help="复用已有的探针 JSONL，不再重跑 light")
    args = ap.parse_args(argv)

    if args.from_json:
        rows = [json.loads(x) for x in
                Path(args.from_json).read_text(encoding="utf-8").splitlines() if x.strip()]
        try:
            entries, rejected = entries_from_rows(rows)
        except SampleError as exc:
            print(f"[失败] {exc}", file=sys.stderr)
            return 1
        _report_build(entries, rejected)
        return 0

    runner = LightRunner()
    print(f"light_root: {runner.light_root}", file=sys.stderr)
    if not runner.available():
        print("[错误] 找不到 light 仓（设置 LIGHT_ROOT）", file=sys.stderr)
        return 2

    if args.check or not args.write:
        rows = load_catalog_result(runner, out_path=Path(args.json_out) if args.json_out else None)
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for r in rows:
                flag = "OK " if r["got_code"] == r["diagnostic_code"] else "MISS"
                print(f"{flag:5s} {r['diagnostic_code']:6s}#{r['variant']:<2d} "
                      f"实得={r['got_code'] or '(无码)':6s} V={r['validation']:16s} "
                      f"stage={r['stage'] or '-':4s} line={r['line']} "
                      f"msg={r['message'][:50]}", flush=True)
            ok = sum(1 for r in rows if r["got_code"] == r["diagnostic_code"])
            print(f"\n可达 {ok}/{len(rows)}")
        if not args.write:
            return 0

    try:
        entries, rejected = build_entries(runner)
    except SampleError as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    _report_build(entries, rejected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
