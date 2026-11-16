# -*- coding: utf-8 -*-
"""覆盖核算入口：按 train/test 切分统计决策点 × 选项条数，并给出硬判据。

用法::

    PYTHONPATH=src python -m lightprism.runtime.coverage --json
    PYTHONPATH=src python -m lightprism.runtime.coverage            # 人读摘要

判据（任务卡 T-0370 §2.3 / §四-2）
---------------------------------
* ``test.decision_samples >= 30``：判据只认 **test 侧**，train 再多不算数；
* ``test.decision_points_covered == 7``：7 个决策点在 test 侧全部出现；
* ``template_overlap_train_test == 0``：同一模板不得跨侧（group-split 口径）；
* 重复率 ``duplicate_rate <= 0.20``：同一 ``source_sha256`` 的重复记录不算新样本。

切分归属由 :mod:`lightprism.runtime.splits` 的登记表决定（按模板分组），
本模块只做核算，不改任何记录、不改切分。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .decisions import OBSERVABLE_VALIDATION_CODES, validate_record
from .splits import TRAIN, TEST, SplitError, family_of, split_of
from ..tokens import DEFAULT_DECISION_POINTS

__all__ = [
    "DIAGNOSTICS_PATH",
    "MIN_TEST_SAMPLES",
    "MAX_DUPLICATE_RATE",
    "load_records",
    "audit",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DIAGNOSTICS_PATH = REPO_ROOT / "data" / "runtime" / "decisions.jsonl"

#: test 侧决策样本下限（任务卡门禁）
MIN_TEST_SAMPLES = 30
#: 重复率上限（任务卡 §五-2：>20% 判 incomplete）
MAX_DUPLICATE_RATE = 0.20


@dataclass
class SideStats:
    split: str
    decision_samples: int = 0
    decision_points_covered: int = 0
    templates: List[str] = field(default_factory=list)
    families: List[str] = field(default_factory=list)
    by_point: Dict[str, int] = field(default_factory=dict)
    by_option: Dict[str, int] = field(default_factory=dict)
    by_point_option: Dict[str, Dict[str, int]] = field(default_factory=dict)
    validation_codes: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_records(path: str | Path = DIAGNOSTICS_PATH) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{lineno}: 不是合法 JSON: {exc}") from exc
    return rows


def _side() -> Dict[str, SideStats]:
    return {TRAIN: SideStats(TRAIN), TEST: SideStats(TEST)}


def audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """核算覆盖。切分归属**只**由模板决定，不看记录里别的元数据。"""
    sides = _side()
    invalid: List[str] = []
    unknown_templates: List[str] = []
    family_conflicts: List[str] = []
    shas: Dict[str, str] = {}
    dup = 0

    for rec in records:
        tid = str(rec.get("template_id") or "")
        issues = validate_record(rec)
        if issues:
            invalid.append(f"{rec.get('task_id')}#{rec.get('attempt')}: {'; '.join(issues)}")
        try:
            fam = family_of(tid)
        except SplitError as exc:
            unknown_templates.append(f"{tid}: {exc}")
            continue
        side = sides[fam.split]
        declared = (rec.get("retrieval") or {}).get("family_split")
        if declared and declared != fam.split:
            family_conflicts.append(
                f"{rec.get('task_id')}: 记录声明 {declared}，登记表 {fam.split}"
            )
        side.decision_samples += 1
        if tid not in side.templates:
            side.templates.append(tid)
        if fam.name not in side.families:
            side.families.append(fam.name)
        for point, option in (rec.get("decisions") or {}).items():
            side.by_point[point] = side.by_point.get(point, 0) + 1
            key = f"{point}={option}"
            side.by_option[key] = side.by_option.get(key, 0) + 1
            side.by_point_option.setdefault(point, {})
            side.by_point_option[point][option] = side.by_point_option[point].get(option, 0) + 1
        code = str(rec.get("validation_code") or "")
        side.validation_codes[code] = side.validation_codes.get(code, 0) + 1
        sha = str(rec.get("source_sha256") or "")
        if sha:
            if sha in shas:
                dup += 1
            else:
                shas[sha] = str(rec.get("task_id") or "")

    for side in sides.values():
        side.by_point = dict(sorted(side.by_point.items()))
        side.by_option = dict(sorted(side.by_option.items()))
        side.by_point_option = {k: dict(sorted(v.items()))
                                for k, v in sorted(side.by_point_option.items())}
        side.decision_points_covered = len(
            [p for p in side.by_point if side.by_point[p] > 0])
        side.templates = sorted(side.templates)
        side.families = sorted(side.families)

    all_points = [p.name for p in DEFAULT_DECISION_POINTS]
    overlap = sorted(set(sides[TRAIN].templates) & set(sides[TEST].templates))
    total = len(records)
    duplicate_rate = round(dup / total, 4) if total else 0.0

    checks = {
        "test_decision_samples_ge_30": sides[TEST].decision_samples >= MIN_TEST_SAMPLES,
        "test_decision_points_7_of_7": sides[TEST].decision_points_covered == len(all_points),
        "template_overlap_zero": not overlap,
        "duplicate_rate_le_20pct": duplicate_rate <= MAX_DUPLICATE_RATE,
        "no_invalid_records": not invalid,
        "no_unknown_templates": not unknown_templates,
        "no_family_conflicts": not family_conflicts,
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records": total,
        "unique_sources": len(shas),
        "duplicate_records": dup,
        "duplicate_rate": duplicate_rate,
        "decision_points": all_points,
        "train": sides[TRAIN].to_dict(),
        "test": sides[TEST].to_dict(),
        "template_overlap_train_test": len(overlap),
        "overlap_templates": overlap,
        "unobserved_test_points": [
            p for p in all_points if sides[TEST].by_point.get(p, 0) == 0],
        "unobserved_test_options": [
            f"{p}={o}" for p in all_points
            for o in next((dp.options for dp in DEFAULT_DECISION_POINTS if dp.name == p), ())
            if sides[TEST].by_point_option.get(p, {}).get(o, 0) == 0],
        "validation_codes": list(OBSERVABLE_VALIDATION_CODES),
        "invalid_records": invalid[:5],
        "unknown_templates": unknown_templates[:5],
        "family_conflicts": family_conflicts[:5],
        "checks": checks,
        "ok": all(checks.values()),
    }


def _print_human(report: Mapping[str, Any], path: Path) -> None:
    print(f"日志：{path}")
    print(f"记录：{report['records']} 条（唯一源码 {report['unique_sources']}，"
          f"重复 {report['duplicate_records']} = {report['duplicate_rate'] * 100:.2f}%）")
    for split in (TRAIN, TEST):
        s = report[split]
        print(f"\n[{split}] 决策样本 {s['decision_samples']}，"
              f"决策点 {s['decision_points_covered']}/7，模板 {len(s['templates'])} 个")
        print("  决策点：" + ", ".join(f"{k}={v}" for k, v in s["by_point"].items()))
        print("  选项：" + ", ".join(f"{k}={v}" for k, v in s["by_option"].items()))
        print("  验证码：" + ", ".join(f"{k}={v}" for k, v in s["validation_codes"].items()))
    print(f"\n模板重合（train∩test）：{report['template_overlap_train_test']}"
          f" {report['overlap_templates']}")
    print("判据：")
    for k, v in report["checks"].items():
        print(f"  {'✓' if v else '✗'} {k}")
    print("结论：" + ("PASS" if report["ok"] else "未达门禁"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="决策日志覆盖核算（train/test 分组切分）")
    ap.add_argument("--decisions", default=str(DIAGNOSTICS_PATH), help="决策日志 JSONL")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    ap.add_argument("--json-out", default="", help="把 JSON 摘要写到文件")
    args = ap.parse_args(argv)

    path = Path(args.decisions)
    if not path.is_file():
        print(f"[错误] 找不到决策日志 {path}", file=sys.stderr)
        return 2
    report = audit(load_records(path))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report, path)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
