# -*- coding: utf-8 -*-
"""``lightprism`` 命令行：预算核算、控制 token 清单、``light ai check`` 结果映射，
以及 ``chat`` 单轮推理入口。

除 ``chat`` 外只依赖标准库、不调用模型、不联网；``chat`` 才需要 torch 与
demo checkpoint（缺任一都给出可读错误并**非零退出**）。
输出统一支持人读文本与机器可读 JSON（``--json``，或等价的 ``--format json``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .demo.chat import DEFAULT_CKPT as _DEFAULT_CKPT
from .demo.chat import DEFAULT_MAX_TOKENS as _DEFAULT_MAX_TOKENS
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


def cmd_config(args: argparse.Namespace) -> int:
    """校验模型配置并报参数量（不需要 torch）。"""
    from .model_cfg import CONFIG_300M, PrismConfig, count_parameters, from_json_file

    if args.input:
        cfg, unknown = from_json_file(args.input)
    else:
        cfg, unknown = CONFIG_300M, set()
    total = count_parameters(cfg)
    dense = count_parameters(cfg, non_embedding=True)
    payload = {
        "ok": True,
        "ignored_fields": sorted(unknown),
        "dim": cfg.dim, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
        "vocab_size": cfg.vocab_size, "max_seq": cfg.max_seq,
        "params": total,
        "params_non_embedding": dense,
        "params_human": f"{total / 1e9:.3f}B" if total >= 1e9 else f"{total / 1e6:.0f}M",
    }
    return _emit(payload, args.json, [
        f"模型：d={cfg.dim} L={cfg.n_layers} H={cfg.n_heads} V={cfg.vocab_size}",
        f"参数量：{payload['params_human']}（{total:,}），非嵌入 {dense:,}",
        *([f"忽略的说明字段：{sorted(unknown)}"] if unknown else []),
    ])


def cmd_export(args: argparse.Namespace) -> int:
    """从 ``light ai check`` 归档导出训练样本。

    ``--light-root`` 决定**有没有 PASS 来源**：给了就现跑 check（带 ``--run``），
    不给就只能导「当前仍报错」的单步样本。两者产出物的用途完全不同，输出里必须写明。
    """
    from .exporters import export_samples

    result = export_samples(
        corpus_dir=args.corpus_dir,
        source_roots=args.source_root or [],
        light_root=args.light_root,
        min_hits=args.min_hits,
        limit=args.limit,
    )
    stats = result.stats.as_dict()

    if args.out:
        from .data import write_jsonl

        written = write_jsonl(result.samples, args.out)
    else:
        written = 0

    payload = {
        "ok": True,
        "stats": stats,
        "written": written,
        "live_check_enabled": bool(args.light_root),
        # 本导出器**不产出修复样本**（hash 回找结构上做不到），这里显式写明，
        # 防止下游把失败样本当修复样本用。
        "produces_repair_samples": False,
        "problems": result.problems[:20],
        "problem_count": len(result.problems),
    }
    lines = [
        f"归档记录 {stats['total_records']} 条 → 导出 {stats['exported_total']} 条样本"
        f"（覆盖率 {stats['coverage']:.1%}）",
        f"  失败样本（当前仍报错）: {stats['exported_failed']}",
        f"  陈旧样本（文件现已通过）: {stats['exported_stale']}",
        f"  跳过：无源文件 {stats['skipped_no_source']}"
        f" / 用法错误 {stats['skipped_usage_error']}"
        f" / 未知码 {stats['skipped_unknown_code']}"
        f" / 现跑未成功 {stats['skipped_checker_failed']}",
        ("未现跑 check（未给 --light-root）：样本未做复现确认"
         if not args.light_root
         else "已现跑 light ai check（含 --run）确认「当前是否仍报错」"),
        "注意：本导出器只产出失败样本，修复样本需 lightharness 编辑日志（见 docs/样本导出.md）",
        *([f"输出：{args.out}（{written} 行）"] if args.out else []),
        *([f"问题：{payload['problem_count']} 条（前 3 条见 --json）"]
          if payload["problem_count"] else []),
    ]
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


def cmd_chat(args: argparse.Namespace) -> int:
    """加载 demo checkpoint，跑一次 ``prompt → 文本`` 推理。

    退出码口径：

    * ``0``：推理完整跑完；``ok`` 表示文本有没有可见字符（空输出为 False，
      **不冒充成功**）；``--format json`` 的输出必定是合法 JSON。
    * ``2``：参数错误 / checkpoint 缺失 / 缺 torch / 推理异常——错误信息在
      text 模式写 stderr，在 json 模式写 ``{"ok": false, "error": ...}``。
    """
    from .demo.chat import ChatError, run_chat

    as_json = bool(getattr(args, "json", False)) or getattr(args, "format", "text") == "json"
    try:
        payload = run_chat(args.prompt, ckpt=args.ckpt, max_tokens=args.max_tokens)
    except ChatError as exc:
        if as_json:
            print(json.dumps(
                {"ok": False, "error": str(exc), "prompt": args.prompt},
                ensure_ascii=False, indent=2,
            ))
        else:
            print(f"✗ {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(
        f"checkpoint: {payload['ckpt']} | tokens: {payload['tokens']} | "
        f"{payload['tok_s']:.2f} tok/s"
    )
    print(payload["text"])
    if payload["empty_output"]:
        print(
            f"⚠ 模型未产出可见字符（{payload['tokens']} 个 token 全为空白）；"
            "本次不计为有效输出",
            file=sys.stderr,
        )
    return 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def _json_flag(*, suppress_default: bool) -> argparse.ArgumentParser:
    """`--json` 开关，主命令与子命令各挂一份。

    `lightprism budget --json` 与 `lightprism --json budget` 必须**等价**。
    argparse 的坑在于：子解析器会用自己的默认值覆盖主解析器已赋的值，所以
    子命令那份必须用 ``default=SUPPRESS``（不给属性赋值），否则主命令上的
    `--json` 会被子命令默认的 False 悄悄抹掉 —— 这个 bug 在 v0.1.0 首次
    fresh-clone 自检时被实测抓到。
    """
    flag = argparse.ArgumentParser(add_help=False)
    flag.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS if suppress_default else False,
        help="输出机器可读 JSON",
    )
    return flag


def _format_flag(*, suppress_default: bool) -> argparse.ArgumentParser:
    """``--format {text,json}``，与 ``--json`` 等价但贴合计计划书判据写法。

    计划书 §2 的 V3 判据写的是 ``--format json``，而 CLI 早期只有 ``--json``。
    两个开关都保留：``--json`` 不破坏既有脚本，``--format json`` 让判据原样可跑。
    与 ``--json`` 一样，子命令那份必须 ``SUPPRESS``，否则会覆盖主命令上的值。
    """
    flag = argparse.ArgumentParser(add_help=False)
    flag.add_argument(
        "--format",
        choices=("text", "json"),
        default=argparse.SUPPRESS if suppress_default else "text",
        help="输出格式：text（默认）或 json（等价于 --json）",
    )
    return flag


def build_parser() -> argparse.ArgumentParser:
    top = _json_flag(suppress_default=False)
    sub_flag = _json_flag(suppress_default=True)
    top_fmt = _format_flag(suppress_default=False)
    sub_fmt = _format_flag(suppress_default=True)
    parser = argparse.ArgumentParser(
        prog="lightprism",
        description="LightPrism 光棱：决策 / 代码 / 验证 的统一序列工具",
        parents=[top, top_fmt],
    )
    parser.add_argument("--version", action="version", version=f"lightprism {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tokens", help="列出决策 / 验证 token 清单", parents=[sub_flag, sub_fmt])
    p.set_defaults(func=cmd_tokens)

    p = sub.add_parser("budget", help="核算 64K 词表预算", parents=[sub_flag, sub_fmt])
    p.add_argument("--set", dest="set_", nargs=2, type=str, action="append",
                   metavar=("SLOT", "N"), help="覆盖某槽位条数，可重复")
    p.add_argument("--light-root", help="light 仓根目录（交叉校验 L0 字表）")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("map-check", help="light ai check --json → 验证 token", parents=[sub_flag, sub_fmt])
    p.add_argument("input", help="light ai check --json 的输出文件")
    p.set_defaults(func=cmd_map_check)

    p = sub.add_parser("parse", help="解析并校验一个序列文件", parents=[sub_flag, sub_fmt])
    p.add_argument("input", help="序列文件（.prism.txt）")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("l0", help="检查 token 列表的 L0 原子性", parents=[sub_flag, sub_fmt])
    p.add_argument("input", help="token 列表 JSON")
    p.add_argument("--light-root", help="light 仓根目录")
    p.set_defaults(func=cmd_l0)

    p = sub.add_parser("export", help="从 light ai check 归档导出训练样本",
                       parents=[sub_flag, sub_fmt])
    p.add_argument("--corpus-dir", help="归档目录（默认 $LIGHT_ERROR_CORPUS_DIR 或 ~/.light/error_corpus）")
    p.add_argument("--source-root", action="append",
                   help="回找源文件的目录（可重复；不给则导不出任何样本）")
    p.add_argument("--light-root", help="light 仓根目录；给了才现跑 check 拿 PASS")
    p.add_argument("--min-hits", type=int, default=1, help="最少出现次数（默认 1）")
    p.add_argument("--limit", type=int, default=0, help="最多导出条数（0=不限）")
    p.add_argument("--out", help="输出 JSONL 路径")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("config", help="校验模型配置并报参数量（不需要 torch）",
                       parents=[sub_flag, sub_fmt])
    p.add_argument("input", nargs="?", help="模型配置 JSON（省略则用默认 300M）")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("chat", help="加载 demo checkpoint 推理一次（需要 torch）",
                       parents=[sub_flag, sub_fmt])
    p.add_argument("--prompt", required=True, help="任务提示词")
    p.add_argument("--ckpt", help=f"checkpoint 路径（默认 {_DEFAULT_CKPT}，"
                                  f"或 $LIGHTPRISM_CKPT）")
    p.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS,
                   help=f"最多生成的 token 数（默认 {_DEFAULT_MAX_TOKENS}）")
    p.set_defaults(func=cmd_chat)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # `--format json` 与 `--json` 双向等价：老命令读 args.json，chat 读 args.format。
    if getattr(args, "json", False):
        args.format = "json"
    elif getattr(args, "format", "text") == "json":
        args.json = True
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
