# -*- coding: utf-8 -*-
"""``lightprism`` 命令行：预算核算、控制 token 清单、``light ai check`` 结果映射。

只依赖标准库，不调用模型、不联网。四个子命令都是**只读**的，
输出机器可读 JSON（``--json``）或人读文本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .sequence import decision_density, parse_sequence
from .tokens import CONTROL_TOKEN_MAX_RATIO, VALIDATION_CODES, TokenSpace, map_diagnostics
from .vocab import DEFAULT_BUDGET, VocabBudget, check_l0_atomicity, load_light_l0_core


def _emit(payload: Dict[str, Any], as_json: bool, lines: Sequence[str]) -> int:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for line in lines:
            print(line)
    return 0 if payload.get("ok", True) else 1


# --------------------------------------------------------------------------
# 子命令实现
# --------------------------------------------------------------------------
def cmd_tokens(args: argparse.Namespace) -> int:
    """列出全部决策 / 验证 token（词表保留区的权威清单）。"""
    space = TokenSpace()
    decisions = [t.text for t in space.decision_tokens()]
    validations = [f"<V:{code}>" for code in VALIDATION_CODES]
    payload = {
        "ok": True,
        "schema_version": space.schema_version,
        "decision_points": [
            {"name": p.name, "options": list(p.options), "description": p.description}
            for p in space.decision_points
        ],
        "decision_tokens": decisions,
        "validation_tokens": validations,
        "counts": {"decision": len(decisions), "validation": len(validations)},
    }
    lines = [
        f"决策 token（{len(decisions)} 个，{len(space.decision_points)} 个决策点）:",
        *[f"  {t}" for t in decisions],
        f"验证 token（{len(validations)} 个）:",
        *[f"  {t}" for t in validations],
    ]
    return _emit(payload, args.json, lines)


def cmd_budget(args: argparse.Namespace) -> int:
    """核算 64K 词表预算；``--light-root`` 时交叉校验 L0 字表来源。"""
    space = TokenSpace()
    alloc: Dict[str, int] = {}
    for key, value in args.set_ or []:
        alloc[key] = value
    budget = VocabBudget(alloc=alloc)
    issues = budget.problems(space)
    core, source = load_light_l0_core(args.light_root)

    payload = {
        "ok": not issues,
        "total": budget.total,
        "allocated": budget.allocated(),
        "total_allocated": budget.total_allocated(),
        "remaining": budget.remaining(),
        "l0_core_size": len(core),
        "l0_core_source": source,
        "decision_density_cap": CONTROL_TOKEN_MAX_RATIO,
        "issues": issues,
    }
    lines = [
        f"词表预算（总量 {budget.total}）:",
        *[
            f"  {slot.name:<16} {budget.value_of(slot.name):>6}  "
            f"[{slot.low}{'=' if slot.exact else '—'}{'' if slot.exact else slot.high}]  {slot.note}"
            for slot in DEFAULT_BUDGET
        ],
        f"  {'合计':<16} {budget.total_allocated():>6}  余量 {budget.remaining():+d}",
        f"L0 核心字 {len(core)} 个；来源：{source}",
        ("预算检查：通过" if not issues else "预算检查：不通过"),
        *[f"  ✗ {issue}" for issue in issues],
    ]
    return _emit(payload, args.json, lines)


def cmd_map_check(args: argparse.Namespace) -> int:
    """把 ``light ai check --json`` 的输出映射为**一个**验证 token。

    这是方案文档 §5「验证 token 必须与真实返回一一对应」的落地实现：
    输入是外部命令产物，输出是序列里可用的 token，中间不含任何模型判断。
    """
    raw = Path(args.input).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        payload = {"ok": False, "error": f"不是合法 JSON: {exc}"}
        return _emit(payload, args.json, [f"✗ 不是合法 JSON: {exc}"])

    diagnostics: List[Dict[str, Any]] = list(data.get("diagnostics") or [])
    token = map_diagnostics(diagnostics)
    payload = {
        "ok": True,
        "file": data.get("file"),
        "checker_ok": data.get("ok"),
        "validation_token": token.text,
        "code": token.code,
        "detail": token.detail,
        "diagnostic_count": len(diagnostics),
        "diagnostic_codes": [str(d.get("code", "")) for d in diagnostics],
    }
    return _emit(
        payload,
        args.json,
        [f"{token.text}" + (f"  ({token.detail})" if token.detail else "")],
    )


def cmd_parse(args: argparse.Namespace) -> int:
    """解析一个 LightPrism 序列文件并校验结构合法性。"""
    text = Path(args.input).read_text(encoding="utf-8")
    try:
        parsed = parse_sequence(text)
    except Exception as exc:  # noqa: BLE001 - ParseError 或语法之外的解码问题都报出来
        payload = {"ok": False, "error": str(exc)}
        return _emit(payload, args.json, [f"✗ 解析失败: {exc}"])

    problems = parsed.validate()
    density = decision_density(parsed.items)
    payload = {
        "ok": not problems,
        "counts": parsed.counts(),
        "decision_density": round(density, 4),
        "decision_density_ok": density <= CONTROL_TOKEN_MAX_RATIO,
        "problems": problems,
    }
    lines = [
        f"序列：决策 {payload['counts']['decision']} / 代码 {payload['counts']['code']} / "
        f"验证 {payload['counts']['validation']}",
        f"决策密度 {density:.2%}（按序列段计数，仅作参考；训练侧的 2% 硬限按子词统计，"
        f"两者不可直接比较）",
    ]
    if not payload["decision_density_ok"]:
        lines.append(
            f"  ! 段计数密度超过 {CONTROL_TOKEN_MAX_RATIO:.0%}：短样例天然偏高，"
            f"须用子词口径复核后再判违规"
        )
    lines.append("结构校验：通过" if not problems else "结构校验：不通过")
    lines.extend(f"  ✗ {p}" for p in problems)
    return _emit(payload, args.json, lines)


def cmd_l0(args: argparse.Namespace) -> int:
    """检查一份 token 列表里 L0 核心字是否保持一字一 token。"""
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    tokens = data.get("tokens") if isinstance(data, dict) else data
    if not isinstance(tokens, list):
        payload = {"ok": False, "error": "输入应为 token 列表，或含 tokens 字段的对象"}
        return _emit(payload, args.json, [f"✗ {payload['error']}"])
    core, source = load_light_l0_core(args.light_root)
    issues = check_l0_atomicity([str(t) for t in tokens], core)
    payload = {
        "ok": not issues,
        "token_count": len(tokens),
        "l0_core_size": len(core),
        "l0_core_source": source,
        "issues": issues,
    }
    return _emit(
        payload,
        args.json,
        [f"检查 {len(tokens)} 个 token；L0 字表 {len(core)} 字（{source}）",
         "L0 原子性：通过" if not issues else "L0 原子性：不通过",
         *[f"  ✗ {i}" for i in issues]],
    )


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lightprism",
        description="LightPrism 光棱：决策 / 代码 / 验证 的统一序列工具",
    )
    parser.add_argument("--version", action="version", version=f"lightprism {__version__}")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tokens", help="列出决策 / 验证 token 清单")
    p.set_defaults(func=cmd_tokens)

    p = sub.add_parser("budget", help="核算 64K 词表预算")
    p.add_argument("--set", dest="set_", nargs=2, type=str, action="append",
                   metavar=("SLOT", "N"), help="覆盖某槽位条数，可重复")
    p.add_argument("--light-root", help="light 仓根目录（交叉校验 L0 字表）")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("map-check", help="light ai check --json → 验证 token")
    p.add_argument("input", help="light ai check --json 的输出文件")
    p.set_defaults(func=cmd_map_check)

    p = sub.add_parser("parse", help="解析并校验一个序列文件")
    p.add_argument("input", help="序列文件（.prism.txt）")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("l0", help="检查 token 列表的 L0 原子性")
    p.add_argument("input", help="token 列表 JSON")
    p.add_argument("--light-root", help="light 仓根目录")
    p.set_defaults(func=cmd_l0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
