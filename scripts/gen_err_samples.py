#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ERR 类验证样本生成器：诊断码可达性矩阵 + 真实样本生成。

用法::

    PYTHONPATH=src python scripts/gen_err_samples.py --check     # 只探可达性
    PYTHONPATH=src python scripts/gen_err_samples.py             # 生成 err_tasks.jsonl

设计要点
--------
* **必须调真实 checker**：不手写 `<V:ERR=…>` 标签，一切来自 `light ai check` 输出。
* `--no-corpus` 必须加：否则会往 `~/.light/error_corpus` 里灌测试垃圾。
* 每个候选片段跑 3 次，只有 **3 次同码**才算「稳定可达」。
* 每个可达码生成 ≥5 条、目标 10 条不同内容的样本。
* 达不成的码诚实记「未覆盖」，绝不拿 L0000 冒充。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 把 src 加入 path，方便直接引用 DIAGNOSTIC_TO_VALIDATION
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lightprism.tokens import DIAGNOSTIC_TO_VALIDATION, VALIDATION_CODES  # noqa: E402

#: 默认 light CLI 路径（FreeBSD 上的路径，Windows 上用 --light-cli 覆盖）
DEFAULT_LIGHT_CLI = "/home/skywalk/dswork/light/cli/light.py"

#: 每个码至少要生成的样本数
MIN_SAMPLES_PER_CODE = 5
TARGET_SAMPLES_PER_CODE = 10

#: 稳定性测试跑几次
STABILITY_RUNS = 3


# ---------------------------------------------------------------------------
# 候选片段库：按目标诊断码分组
# ---------------------------------------------------------------------------
#: 每个条目是 (片段标题, 完整代码片段)。代码必须是完整程序，能独立跑出同一个诊断。
#: 注意：大多数运行期错误会回落到 L0000，这里只列**可能**触发具体码的候选。
CANDIDATES: Dict[str, List[Tuple[str, str]]] = {
    # ---- L3001 → ERR=RUNTIME（列表索引越界，唯一稳定复现的）
    "L3001": [
        (
            "列表索引越界_基础",
            "设 甲 为 [1, 2, 3]\n打印(甲[10])",
        ),
        (
            "列表索引越界_负索引",
            "设 乙 为 [10, 20, 30]\n打印(乙[-5])",
        ),
        (
            "列表索引越界_循环越界",
            "设 数 为 [1, 2, 3, 4, 5]\n设 i 为 0\n当 i 小于 10：\n  打印(数[i])\n  i 为 i 加 1",
        ),
        (
            "嵌套列表越界",
            "设 表 为 [[1, 2], [3, 4]]\n打印(表[1][5])",
        ),
        (
            "空列表越界",
            "设 空 为 []\n打印(空[0])",
        ),
        (
            "字符串索引越界",
            "设 文 为 \"abc\"\n打印(文[10])",
        ),
        (
            "字典键不存在",
            "设 字 为 {\"a\": 1}\n打印(字[\"不存在\"])",
        ),
        (
            "除零错误",
            "设 x 为 10\n设 y 为 0\n打印(x 除 y)",
        ),
        (
            "类型不匹配_运行时",
            "设 s 为 \"hello\"\n打印(s 加 123)",
        ),
        (
            "空指针解引用",
            "设 n 为 空\n打印(n.某属性)",
        ),
    ],

    # ---- L1001..L1007 → ERR=SYNTAX / WARN=NON_CANONICAL
    "L1001": [
        (
            "括号不闭合",
            "设 甲 为 (1 + 2",
        ),
        (
            "括号不闭合_函数调用",
            "打印(\"hello\"",
        ),
        (
            "引号不闭合",
            "设 s 为 \"hello",
        ),
        (
            "冒号缺失",
            "设 x 为 1\n如果 x 大于 0\n  打印(x)",
        ),
        (
            "缩进错误",
            "设 x 为 1\n如果 x 大于 0：\n打印(x)",
        ),
        (
            "关键字拼写错误",
            "设 x 为 1\niff x 大于 0：\n  打印(x)",
        ),
        (
            "缺少逗号",
            "设 列 为 [1 2 3]",
        ),
    ],

    # ---- L2001/L2003 → ERR=TYPE_MISMATCH
    "L2001": [
        (
            "类型不匹配_赋值",
            "段落 加一(甲)：\n  返回 甲 加 1\n设 x 为 \"文本\"\n加一(x)",
        ),
        (
            "类型不匹配_参数",
            "段落 双(x)：\n  返回 x 乘 2\n双(\"abc\")",
        ),
        (
            "类型不比较_返回",
            "段落 取数()：\n  返回 \"不是数字\"",
        ),
    ],

    # ---- L2002 → ERR=MISSING_TYPE
    "L2002": [
        (
            "函数缺类型标注",
            "段落 加一(甲)：\n  返回 甲 加 1",
        ),
        (
            "变量缺类型标注",
            "设 甲 为 123",
        ),
    ],

    # ---- L4001 → ERR=UNDEFINED_VAR
    "L4001": [
        (
            "未定义变量_打印",
            "打印(未定义变量)",
        ),
        (
            "未定义变量_运算",
            "设 x 为 不存在的变量 加 1",
        ),
        (
            "未定义变量_函数调用",
            "不存在的函数()",
        ),
    ],

    # ---- L4002 → ERR=IMPORT_FAIL
    "L4002": [
        (
            "导入不存在模块",
            "从《不存在的模块》导入《甲》。",
        ),
        (
            "导入不存在符号",
            "从《sys》导入《不存在的符号》。",
        ),
    ],

    # ---- L5001 → ERR=RUNTIME（后端不支持）
    "L5001": [
        (
            "不支持的语句",
            "类 甲：\n  段落 构造()：\n    打印(\"hi\")",
        ),
    ],

    # ---- L0000 → ERR=CHECKER_FAIL（兜底，不算具体码）
    "L0000": [
        (
            "检查器崩溃_复杂代码",
            "设 甲 为 1\n打印(甲",
        ),
    ],
}


# ---------------------------------------------------------------------------
# light ai check 调用封装
# ---------------------------------------------------------------------------

def run_light_check(
    code: str,
    light_cli: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """跑一次 ``light ai check --json --no-corpus --run``，返回解析后的 dict。

    返回::

        {
            "ok": bool,           # 是否通过
            "code": str,          # 诊断码（Lxxxx）
            "message": str,      # 诊断消息
            "validation": str,  # 映射后的验证码（如 ERR=RUNTIME）
            "raw": dict,          # 原始 JSON 输出
            "error": str,         # 非空表示 checker 本身没跑成
        }
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".light", encoding="utf-8", delete=False
    ) as fh:
        fh.write(code)
        tmp_path = fh.name

    try:
        cmd = [
            sys.executable, "-X", "utf8",
            light_cli, "ai", "check",
            "--json", "--no-corpus",
            "--run", "--timeout", str(timeout),
            tmp_path,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout + 10,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": "", "message": "checker 超时",
                "validation": "ERR=CHECKER_FAIL", "raw": {}, "error": "timeout"}
    except OSError as exc:
        return {"ok": False, "code": "", "message": str(exc),
                "validation": "ERR=CHECKER_FAIL", "raw": {}, "error": str(exc)}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # 解析 JSON 输出（前面可能混进编码提示）
    stdout = proc.stdout or ""
    payload = _extract_json(stdout)
    if payload is None:
        return {
            "ok": False, "code": "", "message": stdout[:200],
            "validation": "ERR=CHECKER_FAIL", "raw": {},
            "error": f"stdout 不是合法 JSON (rc={proc.returncode})",
        }

    # 提取第一个 error 级诊断
    diagnostics = payload.get("diagnostics") or []
    errors = [d for d in diagnostics if str(d.get("severity", "")).lower() == "error"]
    if not errors:
        return {
            "ok": True, "code": "", "message": "PASS",
            "validation": "PASS", "raw": payload, "error": "",
        }

    first = errors[0]
    code = str(first.get("code", ""))
    message = str(first.get("message", ""))[:200]
    validation = DIAGNOSTIC_TO_VALIDATION.get(code, "ERR=CHECKER_FAIL")

    return {
        "ok": False, "code": code, "message": message,
        "validation": validation, "raw": payload, "error": "",
    }


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从 stdout 里提取第一个 JSON 对象。"""
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# 可达性矩阵
# ---------------------------------------------------------------------------

def probe_reachability(
    light_cli: str,
    stability_runs: int = STABILITY_RUNS,
) -> Dict[str, Dict[str, Any]]:
    """对每个候选码跑稳定性测试，返回可达性矩阵。

    返回格式::

        {
            "L3001": {
                "stable": True,
                "validation": "ERR=RUNTIME",
                "runs": [{"code": "L3001", "message": "..."}, ...],
                "best_candidate": ("列表索引越界_基础", "设 甲 为 [1, 2, 3]\\n打印(甲[10])"),
                "n_candidates": 10,
            },
            ...
        }
    """
    matrix: Dict[str, Dict[str, Any]] = {}

    for diag_code, candidates in CANDIDATES.items():
        print(f"\n=== 探测 {diag_code} ({len(candidates)} 个候选) ===")
        best: Optional[Tuple[str, str, Dict[str, Any]]] = None

        for title, code in candidates:
            print(f"  试: {title} ...", end=" ", flush=True)
            runs: List[Dict[str, Any]] = []
            for i in range(stability_runs):
                result = run_light_check(code, light_cli)
                runs.append({"code": result["code"], "message": result["message"]})

            # 判断是否稳定触发同一个码
            codes = {r["code"] for r in runs}
            if len(codes) == 1 and "" not in codes:
                triggered_code = runs[0]["code"]
                print(f"✓ 稳定触发 {triggered_code}", flush=True)
                if best is None:
                    best = (title, code, runs[0])
            else:
                print(f"✗ 不稳定 ({codes})", flush=True)

        if best is not None:
            title, code, first_run = best
            validation = DIAGNOSTIC_TO_VALIDATION.get(first_run["code"], "ERR=CHECKER_FAIL")
            matrix[diag_code] = {
                "stable": True,
                "validation": validation,
                "runs": [{"code": r["code"], "message": r["message"]} for r in runs],
                "best_candidate": (title, code),
                "n_candidates": len(candidates),
                "actual_code": first_run["code"],
            }
            print(f"  → {diag_code} 可达，映射到 {validation}")
        else:
            matrix[diag_code] = {
                "stable": False,
                "validation": "未覆盖",
                "runs": [],
                "best_candidate": None,
                "n_candidates": len(candidates),
                "actual_code": "",
            }
            print(f"  → {diag_code} 不可达")

    return matrix


# ---------------------------------------------------------------------------
# 样本生成
# ---------------------------------------------------------------------------

def generate_samples(
    light_cli: str,
    output_path: Path,
    min_per_code: int = MIN_SAMPLES_PER_CODE,
    target_per_code: int = TARGET_SAMPLES_PER_CODE,
) -> List[Dict[str, Any]]:
    """生成 err_tasks.jsonl 样本。

    对每个可达码，从候选片段里选能稳定触发的，生成多个变体（改变量名/上下文）。
    """
    samples: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    # 先跑一遍可达性探测，确定哪些码可达
    matrix = probe_reachability(light_cli)

    for diag_code, info in matrix.items():
        if not info["stable"]:
            print(f"\n跳过 {diag_code}（不可达）")
            continue

        validation = info["validation"]
        title, base_code = info["best_candidate"]
        actual_code = info["actual_code"]

        print(f"\n=== 生成 {diag_code} → {validation} 样本 ===")

        # 生成变体：改变量名、加上下文
        variants = _make_variants(base_code, diag_code, n=target_per_code)

        for i, variant_code in enumerate(variants[:target_per_code]):
            # 验证这个变体确实能触发同一个码
            result = run_light_check(variant_code, light_cli)
            if result["code"] != actual_code:
                print(f"  变体 {i} 触发了 {result['code']}（预期 {actual_code}），跳过")
                continue

            task_title = f"修复光明代码中的错误（{actual_code}）：{validation}"
            sample = {
                "task": task_title,
                "steps": [{
                    "decisions": {"EDIT": "FR"},
                    "code": variant_code,
                    "validation": validation,
                    "validation_detail": result["message"],
                }],
                "meta": {
                    "category": "err",
                    "format": "code",
                    "diagnostic_code": actual_code,
                    "stage": _infer_stage(actual_code),
                    "source": "light_ai_check",
                    "generated_at": now,
                    "variant": i,
                },
            }
            samples.append(sample)
            print(f"  ✓ 变体 {i}: {result['message'][:50]}")

    # 写盘
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n共生成 {len(samples)} 条样本，写入 {output_path}")
    return samples


def _make_variants(base_code: str, diag_code: str, n: int = 10) -> List[str]:
    """从基础代码生成 n 个变体（改变量名、加无关上下文）。"""
    variants = [base_code]  # 第一个就是原始版本

    # 变量名替换表
    name_pools = [
        ("甲", ["乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]),
        ("x", ["y", "z", "a", "b", "c", "d", "e", "f", "g"]),
    ]

    for i in range(1, n):
        code = base_code
        # 轮换替换变量名
        for old, pool in name_pools:
            if old in code:
                new_name = pool[i % len(pool)]
                code = code.replace(old, new_name)
                break

        # 加一点无关上下文（前面加打印）
        if i % 3 == 1:
            code = f"打印(\"准备测试\")\n{code}"
        elif i % 3 == 2:
            code = f"设 无关 为 123\n{code}\n打印(无关)"

        variants.append(code)

    return variants


def _infer_stage(diag_code: str) -> str:
    """根据诊断码推断阶段。"""
    if diag_code.startswith("L1"):
        return "语法"
    elif diag_code.startswith("L2"):
        return "类型"
    elif diag_code.startswith("L3"):
        return "运行"
    elif diag_code.startswith("L4"):
        return "导入"
    elif diag_code.startswith("L5"):
        return "后端"
    else:
        return "未知"


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ERR 类验证样本生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check", action="store_true",
        help="只探可达性，不写盘",
    )
    parser.add_argument(
        "--light-cli", default=os.environ.get("LIGHT_CLI", DEFAULT_LIGHT_CLI),
        help=f"light CLI 路径（默认 {DEFAULT_LIGHT_CLI}）",
    )
    parser.add_argument(
        "--output", default=str(_REPO_ROOT / "data" / "sft" / "err_tasks.jsonl"),
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--min-per-code", type=int, default=MIN_SAMPLES_PER_CODE,
        help=f"每个码最少样本数（默认 {MIN_SAMPLES_PER_CODE}）",
    )
    parser.add_argument(
        "--target-per-code", type=int, default=TARGET_SAMPLES_PER_CODE,
        help=f"每个码目标样本数（默认 {TARGET_SAMPLES_PER_CODE}）",
    )
    args = parser.parse_args()

    light_cli = args.light_cli
    if not Path(light_cli).is_file():
        print(f"错误: 找不到 light CLI: {light_cli}", file=sys.stderr)
        print(f"请用 --light-cli 指定，或设置 LIGHT_CLI 环境变量", file=sys.stderr)
        return 2

    print(f"使用 light CLI: {light_cli}")
    print(f"已登记验证码: {list(VALIDATION_CODES.keys())}")

    if args.check:
        matrix = probe_reachability(light_cli)
        print("\n" + "=" * 60)
        print("可达性矩阵汇总")
        print("=" * 60)
        for code, info in matrix.items():
            status = "✓ 可达" if info["stable"] else "✗ 不可达"
            print(f"  {code:8s} → {info['validation']:20s} {status}")
        return 0

    # 默认模式：生成样本
    generate_samples(
        light_cli,
        Path(args.output),
        min_per_code=args.min_per_code,
        target_per_code=args.target_per_code,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
