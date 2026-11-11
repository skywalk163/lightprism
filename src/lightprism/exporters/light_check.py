# -*- coding: utf-8 -*-
"""从 ``light ai check`` 的报错归档里导出训练样本（方案 §4.3 的第一个来源）。

## 数据源的真实形状（2026-11-11 核对 light 仓源码，不是猜的）

``light ai check`` 的归档走 ``tools/ai_copilot/error_corpus.py::record()``：

- 落盘位置：``$LIGHT_ERROR_CORPUS_DIR`` 或 ``~/.light/error_corpus/<YYYY-MM>.jsonl``；
- 每条记录字段：``ts`` / ``last_seen`` / ``code`` / ``message`` / ``hint`` / ``fixes`` /
  ``source_snippet`` / ``source_hash`` / ``lang_level`` / ``backend`` / ``light_version`` / ``hits``；
- **只归档 error 级诊断**（``_ai_archive_diagnostics`` 里 ``errors = [...] severity == 'error'``），
  PASS 与 warning 都不入档；
- **同一 ``(code, source_hash)`` 只落一行**，命中时只更新 ``index.json`` 的 ``hits`` 与
  ``last_seen``，不重写 JSONL。

## 由此推出的三条设计约束（这是本导出器的全部难点）

1. **PASS 结构性拿不到，不是「暂时没实现」**。

   corpus 只归档 error 级诊断，**没有成功记录**；而 ``source_hash`` 是对**出错那一刻**的
   源码算的，代码一改 hash 就变了。所以：

   - 用 hash 回找，只能找到「坏代码原样还在」的情况 → 现跑必然仍失败；
   - 代码修好后，hash 对不上 → 这条记录再也关联不到当前文件。

   两条路都拿不到「同一个文件从坏到好」的配对。结论：**本导出器只产出失败样本**
   （``EDIT`` + ``V:ERR=<码>``），产出修复样本需要另一个数据源。

   修好之后的 PASS 判定，要靠 **lightharness 的编辑日志**（记录「对哪个文件做了什么编辑、
   编辑后 check 结果」），那是方案 §4.3 的第二个来源，见 ``docs/样本导出.md``「下一来源」。

   导出器**仍然支持** ``--light-root`` 现跑 check，用途是：给能匹配上的失败样本补一个
   「此刻的复现结果」，确认它**现在依然失败**（防止导出陈旧样本）。**不**用来声称已修复。

2. **snippet 只有 ±2 行**（``SNIPPET_RADIUS = 2``），不能当代码体用。要靠 ``source_hash``
   回找当前源文件（见 :class:`SourceIndex`），找不到就**跳过并计数**，绝不拿片段凑代码。

3. **``hits`` 只存在于索引里**，JSONL 每行的 ``hits`` 是首次写入时的值（恒为 1）。
   要真实的出现次数必须读 ``index.json``。实测：真实 corpus 里 JSONL 写 1，index 里是 2。

## 导出规则

| 情形 | 结果 |
|---|---|
| 记录是 ``L0000``（文件不存在/读取失败/检查器崩溃） | 跳过 → ``skipped_usage_error``；那不是代码问题 |
| 诊断码没有验证 token 映射 | 跳过 → ``skipped_unknown_code``；**不猜** |
| ``source_hash`` 回找不到当前源文件 | 跳过 → ``skipped_no_source``（代码已被修改或不在扫描目录内） |
| 找到了源文件，现跑仍失败 | 导出失败样本：``EDIT=FR`` + ``<CODE>`` + ``V:ERR=<码>`` |
| 找到了源文件，但现跑通过了 | **也导出失败样本**，但 ``meta.stale=True`` 标记「该文件现已通过，这条报错已陈旧」 |

最后一行是关键：现跑通过**不代表**「模型学会了修复」，只说明「这条历史报错对应的文件
现在能过」。把它标成 ``stale`` 而不是造一条假的修复样本，是这套导出器的核心纪律。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..data import Sample, Step
from ..tokens import DIAGNOSTIC_TO_VALIDATION, ValidationToken

__all__ = [
    "CorpusRecord",
    "LightChecker",
    "SourceIndex",
    "ExportStats",
    "read_corpus",
    "default_corpus_dir",
    "fault_to_fix_decision",
    "export_samples",
    "ExportResult",
]

#: 判断「是不是用法错误」的码：``L0000`` 覆盖文件不存在 / 读取失败 / 检查器自身崩溃。
USAGE_ERROR_CODES = frozenset({"L0000"})

#: 诊断码 → ``FIX`` 决策。只对有明确修复方向映射的码给映射；
#: 其余**不猜**，返回 None 并由调用方决定是否跳过（避免造出语义错误的监督信号）。
DIAGNOSTIC_TO_FIX: Dict[str, str] = {
    "L2002": "ADD_TYPE",      # 布尔/空值写法 → 补类型
    "L2001": "ADD_TYPE",      # 实参数量不匹配 → 补/删类型标注
    "L2003": "ADD_TYPE",      # 默认参数写法 → 补类型
    "L1001": "REFACTOR",      # 用 def/function → 改写为「段」
    "L1002": "REFACTOR",      # 用英文控制流关键字 → 改写为 L0 字
    "L1003": "REFACTOR",      # 逻辑运算符写成 &&/||/!
    "L1004": "REFACTOR",      # C 风格三元表达式
    "L4001": "ADD_TYPE",      # 缺导入 → 补导入
    "L4002": "RENAME",        # 英文模块名 → 改成光明名
    "L3001": "REFACTOR",      # 索引越界 → 改边界逻辑
}


def _sha16(text: str) -> str:
    """与 ``error_corpus._source_hash`` 完全一致的算法（归一化换行后 sha256 前 16 位）。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def default_corpus_dir() -> Path:
    """与 light 仓 ``error_corpus.DEFAULT_ROOT`` 同口径。"""
    raw = os.environ.get("LIGHT_ERROR_CORPUS_DIR")
    if raw:
        return Path(os.path.expanduser(raw))
    return Path.home() / ".light" / "error_corpus"


# --------------------------------------------------------------------------
# corpus 读取
# --------------------------------------------------------------------------
@dataclass
class CorpusRecord:
    """一条归档记录（字段名与 ``error_corpus.record()`` 的落盘格式一一对应）。"""

    code: str
    message: str
    source_hash: str
    source_snippet: str = ""
    hint: str = ""
    fixes: List[str] = field(default_factory=list)
    ts: str = ""
    last_seen: str = ""
    lang_level: Optional[str] = None
    backend: Optional[str] = None
    light_version: Optional[str] = None
    #: 真实出现次数：来自 ``index.json``，不是 JSONL 里的 ``hits``（后者恒为首次值）。
    hits: int = 1
    origin_file: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], origin_file: str = "") -> "CorpusRecord":
        return cls(
            code=str(data.get("code") or "UNKNOWN"),
            message=str(data.get("message") or ""),
            source_hash=str(data.get("source_hash") or ""),
            source_snippet=str(data.get("source_snippet") or ""),
            hint=str(data.get("hint") or ""),
            fixes=[str(x) for x in (data.get("fixes") or [])],
            ts=str(data.get("ts") or ""),
            last_seen=str(data.get("last_seen") or ""),
            lang_level=data.get("lang_level"),
            backend=data.get("backend"),
            light_version=data.get("light_version"),
            hits=int(data.get("hits") or 1),
            origin_file=origin_file,
        )


def _read_index_hits(root: Path) -> Dict[str, int]:
    """读 ``index.json`` 拿真实 ``hits``。key 是 ``"{code}\\x1f{source_hash}"``。"""
    path = root / "index.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: Dict[str, int] = {}
    for key, entry in (data.get("entries") or {}).items():
        if isinstance(entry, dict):
            try:
                out[key] = int(entry.get("hits") or 1)
            except (TypeError, ValueError):
                out[key] = 1
    return out


def read_corpus(root: Optional[Path] = None) -> Tuple[List[CorpusRecord], List[str]]:
    """读归档目录下所有 ``<YYYY-MM>.jsonl``，返回 ``(记录列表, 问题列表)``。

    坏行计问题而不抛 —— 归档是长期累积的，一行坏掉不该让整批导不出来。
    同一 ``(code, source_hash)`` 在多月文件里重复出现时**只保留一条**（取 ``last_seen`` 最新的），
    因为 light 侧本来就是「同键只落一行」，跨月出现只说明期间索引被清过。
    """
    root = Path(root or default_corpus_dir())
    problems: List[str] = []
    records: List[CorpusRecord] = []

    if not root.is_dir():
        return [], [f"归档目录不存在: {root}"]

    hits_map = _read_index_hits(root)
    best: Dict[Tuple[str, str], CorpusRecord] = {}

    for path in sorted(root.glob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{path.name}: 读取失败 {exc}")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{path.name}:{lineno}: JSON 解析失败 {exc}")
                continue
            if not isinstance(data, dict):
                problems.append(f"{path.name}:{lineno}: 不是 JSON 对象")
                continue
            record = CorpusRecord.from_dict(data, origin_file=path.name)
            key = (record.code, record.source_hash)
            record.hits = hits_map.get(f"{record.code}\x1f{record.source_hash}", record.hits)
            prev = best.get(key)
            if prev is None or record.last_seen >= prev.last_seen:
                best[key] = record

    records = sorted(best.values(), key=lambda r: (r.last_seen, r.code), reverse=True)
    return records, problems


# --------------------------------------------------------------------------
# 源文件回找
# --------------------------------------------------------------------------
class SourceIndex:
    """按 ``source_hash`` 回找当前源文件内容。

    corpus 只存 ±2 行片段，不能当代码体。所以要扫一遍候选目录、算出每个文件的
    hash 再比对。**这是唯一可靠的办法** —— 匹配上了才敢说「这条报错对应的代码
    现在长这样」。
    """

    def __init__(self, suffixes: Sequence[str] = (".light",), max_file_bytes: int = 512 * 1024) -> None:
        self.suffixes = tuple(suffixes)
        self.max_file_bytes = max_file_bytes
        self._index: Dict[str, List[Path]] = {}
        self.scanned = 0
        self.skipped_large = 0
        self.unreadable = 0

    def build(self, roots: Iterable[str | Path]) -> "SourceIndex":
        for root in roots:
            path = Path(root).expanduser()
            if not path.exists():
                continue
            candidates = [path] if path.is_file() else list(path.rglob("*"))
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                if self.suffixes and candidate.suffix not in self.suffixes:
                    continue
                try:
                    if candidate.stat().st_size > self.max_file_bytes:
                        self.skipped_large += 1
                        continue
                    text = candidate.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    self.unreadable += 1
                    continue
                self.scanned += 1
                self._index.setdefault(_sha16(text), []).append(candidate)
        return self

    def find(self, source_hash: str) -> Optional[Path]:
        paths = self._index.get(source_hash)
        if not paths:
            return None
        # 多个文件内容完全相同时取路径最短的（更可能是「原始那份」）
        return sorted(paths, key=lambda p: (len(str(p)), str(p)))[0]

    def read(self, source_hash: str) -> Optional[str]:
        path = self.find(source_hash)
        if path is None:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None


# --------------------------------------------------------------------------
# 现跑 light ai check
# --------------------------------------------------------------------------
class LightChecker:
    """导出时现跑 ``light ai check --json``，拿到**当前**的验证结果。

    这是本导出器唯一的 PASS 来源（corpus 不记 PASS）。实现上必须做两件事：

    1. ``--no-corpus``：否则每导出一次就往归档里再写一遍，把数据源污染成自反馈；
    2. 用临时文件当被测对象，避免在用户仓库里留下痕迹。
    """

    def __init__(
        self,
        light_root: str | Path,
        timeout: float = 60.0,
        run: bool = True,
        run_timeout: int = 30,
    ) -> None:
        """``run=True`` 会带 ``--run`` 真跑一遍代码。

        **这一项不是可选的性能取舍，是正确性要求**：``light ai check`` 不带 ``--run``
        时只做语法检查，运行时类错误（L3001 索引越界、L5001 后端不支持）一律返回
        ``ok: true``。实测确认：同一份越界代码，不带 ``--run`` 判 ``PASS``，
        带 ``--run`` 才判 ``L3001``。导出器若默认不跑，会把「根本没修」的样本
        标成「已修复」，直接污染监督信号。
        """
        self.light_root = Path(light_root).expanduser()
        self.timeout = timeout
        self.run = run
        # argparse 侧声明的是 type=int，传 30.0 会被直接拒绝（rc=2，stdout 空）。
        # 第一版就踩了这个：--timeout 必须是整数。
        if int(run_timeout) != run_timeout:
            raise ValueError(f"run_timeout 必须是整数秒，收到 {run_timeout!r}")
        self.run_timeout = int(run_timeout)
        self._available: Optional[bool] = None

    # ---- 可用性 -------------------------------------------------------
    def available(self) -> bool:
        """检查 ``light`` 仓能否用 ``python -m cli.light`` 跑起来。"""
        if self._available is not None:
            return self._available
        cli = self.light_root / "cli" / "light.py"
        self._available = cli.is_file()
        return self._available

    # ---- 主入口 -------------------------------------------------------
    def check_source(self, source: str) -> Dict[str, Any]:
        """对一段源码现跑 check，返回 ``{ok, validation, diagnostics, error}``。

        ``error`` 非空表示**没跑成**（light 仓缺失、超时、子进程崩溃）。
        这种情况下 ``ok`` 恒为 False 且 ``validation`` 为 ``ERR=CHECKER_FAIL``，
        调用方必须把它与「真的跑出错误」区分开 —— 见 :func:`export_samples`。
        """
        if not self.available():
            return self._fail(f"light 仓不可用（找不到 {self.light_root / 'cli' / 'light.py'}）")

        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".light", encoding="utf-8", delete=False
        ) as fh:
            fh.write(source)
            tmp_path = fh.name
        try:
            return self._run(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _run(self, file_path: str) -> Dict[str, Any]:
        cmd = [
            sys.executable, "-X", "utf8", "-m", "cli.light", "ai", "check",
            "--json", "--no-corpus",
        ]
        if self.run:
            cmd.extend(["--run", "--timeout", str(int(self.run_timeout))])
        cmd.append(file_path)
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.light_root), capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return self._fail(f"light ai check 超时（{self.timeout}s）")
        except OSError as exc:
            return self._fail(f"无法启动子进程: {exc}")

        payload = self._parse_json(proc.stdout)
        if payload is None:
            return self._fail(
                f"stdout 不是合法 JSON（rc={proc.returncode}）: {proc.stdout[:200]}"
            )

        from ..tokens import map_diagnostics

        token = map_diagnostics(payload.get("diagnostics") or [])
        return {
            "ok": bool(payload.get("ok")),
            "validation": token.code,
            "diagnostics": list(payload.get("diagnostics") or []),
            "error": "",
            "rc": proc.returncode,
            "mode": "syntax+run" if self.run else "syntax-only",
        }

    def _fail(self, message: str) -> Dict[str, Any]:
        """「没跑成」的统一形状。**与「跑出了错误」必须可区分**：靠 ``error`` 非空。"""
        return {
            "ok": False,
            "validation": "ERR=CHECKER_FAIL",
            "diagnostics": [],
            "error": message,
            "rc": None,
            "mode": "syntax+run" if self.run else "syntax-only",
        }

    @staticmethod
    def _parse_json(stdout: str) -> Optional[Dict[str, Any]]:
        """stdout 里只有一个 JSON 对象，但前面可能混进编码提示，做一次容错提取。"""
        text = (stdout or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


# --------------------------------------------------------------------------
# 决策映射
# --------------------------------------------------------------------------
def fault_to_fix_decision(code: str) -> Optional[str]:
    """诊断码 → ``FIX`` 选项；没有明确映射时返回 None（**不猜**）。"""
    return DIAGNOSTIC_TO_FIX.get(code)


def _validation_for(code: str) -> Optional[str]:
    """诊断码 → 验证 token 码；未登记码返回 None。"""
    return DIAGNOSTIC_TO_VALIDATION.get(code)


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------
@dataclass
class ExportStats:
    """每个分类的计数。**跳过的东西必须报出来**，这是本导出器的硬规矩。"""

    total_records: int = 0
    exported_failed: int = 0       # 正常失败样本（现跑仍失败 / 未现跑）
    exported_stale: int = 0        # 源文件现已通过，属陈旧报错
    skipped_no_source: int = 0     # 找不到当前源文件
    skipped_usage_error: int = 0   # L0000 之类
    skipped_unknown_code: int = 0  # 诊断码没有验证 token 映射
    skipped_checker_failed: int = 0  # 现跑没跑成（与「跑出错误」不同）

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "exported_failed": self.exported_failed,
            "exported_stale": self.exported_stale,
            "skipped_no_source": self.skipped_no_source,
            "skipped_usage_error": self.skipped_usage_error,
            "skipped_unknown_code": self.skipped_unknown_code,
            "skipped_checker_failed": self.skipped_checker_failed,
            "exported_total": self.exported_failed + self.exported_stale,
            # 覆盖率是「能导出多少」的唯一诚实指标
            "coverage": round(
                (self.exported_failed + self.exported_stale) / self.total_records, 4
            ) if self.total_records else 0.0,
        }


@dataclass
class ExportResult:
    samples: List[Sample] = field(default_factory=list)
    stats: ExportStats = field(default_factory=ExportStats)
    problems: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"stats": self.stats.as_dict(), "problems": self.problems[:50],
                "problem_count": len(self.problems)}


def export_samples(
    corpus_dir: Optional[str | Path] = None,
    source_roots: Sequence[str | Path] = (),
    light_root: Optional[str | Path] = None,
    *,
    checker: Optional[Any] = None,
    min_hits: int = 1,
    limit: int = 0,
) -> ExportResult:
    """主流程：读 corpus → 回找源文件 → 现跑 check → 造样本。

    ``checker`` 可注入（测试用假 checker）；不传且 ``light_root`` 有值时用
    :class:`LightChecker`；两者都没有则**只导 ``V:ERR`` 单步样本**，并在 problems 里
    写明「没有 PASS 来源，导出的不是修复样本」。
    """
    result = ExportResult()
    records, problems = read_corpus(corpus_dir)
    result.problems.extend(problems)
    result.stats.total_records = len(records)

    index = SourceIndex().build(source_roots)

    if checker is None and light_root is not None:
        checker = LightChecker(light_root)
    if checker is None:
        result.problems.append(
            "未提供 light_root / checker：本次导出没有 V:PASS 来源，"
            "只会产出「当前仍报错」的单步样本"
        )

    for record in records:
        if limit and len(result.samples) >= limit:
            break
        if record.hits < min_hits:
            continue
        if record.code in USAGE_ERROR_CODES:
            result.stats.skipped_usage_error += 1
            continue

        validation = _validation_for(record.code)
        if validation is None:
            result.stats.skipped_unknown_code += 1
            result.problems.append(
                f"{record.code}: 没有验证 token 映射，跳过（需补 DIAGNOSTIC_TO_VALIDATION）"
            )
            continue

        source = index.read(record.source_hash)
        if source is None:
            result.stats.skipped_no_source += 1
            continue

        fix = fault_to_fix_decision(record.code)
        # DIAGNOSTIC_TO_VALIDATION 直接给 "ERR=MISSING_TYPE" 这类，用它做验证 token
        err_token = validation

        steps: List[Step] = []
        # 唯一一步：当年那份代码 + 当年的错误。
        # 修复样本**不在这里造** —— hash 回找只能命中「码没改」的情况，
        # 那种情况现跑必然仍失败；真修好了 hash 就变了、根本匹配不上。
        broken = Step(
            decisions={"EDIT": "FR"},
            code=source,
            validation=err_token,
            validation_detail=record.message[:200],
        )
        steps.append(broken)

        stale = False
        if checker is not None:
            outcome = checker.check_source(source) if hasattr(checker, "check_source") \
                else checker(source)
            if outcome.get("error"):
                result.stats.skipped_checker_failed += 1
                result.problems.append(
                    f"{record.code}({record.source_hash}): 现跑 check 未成功：{outcome['error']}"
                )
                continue
            # 现跑通过 = 这条历史报错已陈旧，**不是**修复样本
            stale = bool(outcome.get("ok"))

        sample = Sample(
            task=_task_from_record(record),
            steps=steps,
            source="light_ai_check",
            meta={
                "corpus_code": record.code,
                "source_hash": record.source_hash,
                "hits": record.hits,
                "first_seen": record.ts,
                "last_seen": record.last_seen,
                "lang_level": record.lang_level,
                "light_version": record.light_version,
                "fix_derived": fix or "",
                "stale": stale,
                "live_check": "pass_now" if stale else "still_failing",
                "origin_file": record.origin_file,
            },
        )
        issues = sample.validate()
        if issues:
            result.problems.append(f"{record.code}({record.source_hash}): 样本不合法 "
                                   + "；".join(issues))
            continue
        result.samples.append(sample)
        if stale:
            result.stats.exported_stale += 1
        else:
            result.stats.exported_failed += 1

    return result


def _task_from_record(record: CorpusRecord) -> str:
    """从记录里造一句任务描述。

    **不能编造具体任务**：归档里只有错误码与消息，没有「用户当时想做什么」。
    所以任务描述只能是「修掉这个错误」，并带上错误摘要 —— 这是诚实的最小值。
    """
    msg = re.sub(r"\s+", " ", record.message).strip()
    if len(msg) > 80:
        msg = msg[:80] + "…"
    return f"修复光明代码中的错误（{record.code}）：{msg}"
