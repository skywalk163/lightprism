# -*- coding: utf-8 -*-
"""LightPrism（光棱）决策 / 验证 token 规范（纯标准库，可独立测试）。

设计依据：``docs/方案一_统一决策Token词表_v0.1.md`` 第四、五节。

三条硬约束
----------
1. **封闭性**：每个决策点（``DecisionPoint``）的选项必须覆盖全部合法取值，
   不存在"没有 token 可选"的状态。
2. **互斥性**：同一决策点内一次只能选一个选项。
3. **可解析性**：序列化形态 ``<D:EDIT=FR>`` / ``<V:ERR=SYNTAX>`` 必须能被
   lightharness 无歧义地解析为结构化动作；解析失败**返回错误**，不猜。

命名空间隔离
------------
决策 token 用 ``<D:`` 前缀、验证 token 用 ``<V:`` 前缀、代码段用
``<CODE>``/``</CODE>`` 包裹。这三类都属于「控制 token」，在词表中排在
保留区（见 :data:`CONTROL_TOKEN_BUDGET`），不与中文 BPE、代码 token 混排。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CONTROL_TOKEN_BUDGET",
    "CONTROL_TOKEN_MAX_RATIO",
    "VALIDATION_CODES",
    "DEFAULT_DECISION_POINTS",
    "DecisionPoint",
    "DecisionToken",
    "ValidationToken",
    "TokenSpace",
    "ParseError",
    "LIGHT_L0_CORE",
]

#: 64K 词表预算中控制 token（决策 + 验证 + 保留）的上限条数。
#: 方案文档第三节：决策 128—512、验证 64—256、保留 1K—2K。
CONTROL_TOKEN_BUDGET: Dict[str, Tuple[int, int]] = {
    "decision": (128, 512),
    "validation": (64, 256),
    "reserved": (1024, 2048),
}

#: 决策 token 在训练序列中的密度上限（占全部 token 比例）。
#: 超过这个比例说明模型开始"为了决策而决策"，须降采样（方案文档第八节第 5 条）。
CONTROL_TOKEN_MAX_RATIO: float = 0.02

#: ``light ai check`` 语义 → 验证 token 码。与 ``src/diagnostics/registry.py``
#: 的 L1xxx—L5xxx 码段对齐（词法/语法 L1、类型 L2、运行时 L3、模块 L4、后端 L5）。
VALIDATION_CODES: Dict[str, str] = {
    "PASS": "通过",
    "ERR=MISSING_TYPE": "缺少类型标注（L2）",
    "ERR=UNDEFINED_VAR": "未定义变量（L2/L3）",
    "ERR=TYPE_MISMATCH": "类型不匹配（L2）",
    "ERR=IMPORT_FAIL": "模块导入失败（L4）",
    "ERR=RUNTIME": "运行时错误（L3：越界/空值/类型运行时失败）",
    "ERR=SYNTAX": "语法错误（L1）",
    "WARN=NON_CANONICAL": "规范非强制警告",
    "ERR=CHECKER_FAIL": "验证器自身超时/异常",
}

#: 诊断码 → 验证 token 的确定性映射（不是模型自判，是 ``light ai check`` 的
#: 真实返回码映射）。未登记的码一律落到 ``ERR=CHECKER_FAIL`` 并置 ``unknown``，
#: 便于暴露"词表没覆盖到的新错误码"而不是静默归并。
DIAGNOSTIC_TO_VALIDATION: Dict[str, str] = {
    "L0000": "ERR=CHECKER_FAIL",
    "L1001": "ERR=SYNTAX",
    "L1002": "ERR=SYNTAX",
    "L1003": "ERR=SYNTAX",
    "L1004": "ERR=SYNTAX",
    "L1005": "WARN=NON_CANONICAL",
    "L1006": "ERR=SYNTAX",
    "L1007": "WARN=NON_CANONICAL",
    "L2001": "ERR=TYPE_MISMATCH",
    "L2002": "ERR=MISSING_TYPE",
    "L2003": "ERR=TYPE_MISMATCH",
    # L3001=列表索引越界。原先误映射到 ERR=UNDEFINED_VAR（未定义变量），语义错：
    # 越界不是「变量不存在」。实测导出真实 corpus 时才发现，改为专门的 ERR=RUNTIME。
    "L3001": "ERR=RUNTIME",
    "L3002": "WARN=NON_CANONICAL",
    "L3003": "WARN=NON_CANONICAL",
    "L3004": "WARN=NON_CANONICAL",
    "L4001": "ERR=UNDEFINED_VAR",
    "L4002": "ERR=IMPORT_FAIL",
    # L5001=原生后端不支持该语句 —— 那是编译期能力边界，不是验证器自身故障。
    "L5001": "ERR=RUNTIME",
    "L5002": "WARN=NON_CANONICAL",
    "L5003": "WARN=NON_CANONICAL",
}

#: L0 核心字（``docs/language/l0-core.md``，44 字，一字符一 token，永不拆碎）。
#: 注意：方案文档写"30 字"是旧口径；light 仓在 T-0318 前后已把这套冻结表的
#: 权威定义收敛到 44 字，本项目跟随仓库现状（见 README「口径对齐」）。
LIGHT_L0_CORE: Tuple[str, ...] = (
    "设", "为", "若", "则", "否", "当", "遍", "断", "跃", "段", "返",
    "类", "承", "性", "构", "新", "己", "父", "约", "现", "引", "出", "自",
    "且", "或", "非", "真", "假", "空", "印", "写",
    "试", "捕", "掷", "匹", "例", "异", "等",
    "常", "护", "私", "公", "静", "是",
)

#: 单字 token 的解析正则。中文按字、英文按标识符片段，杜绝把 L0 字拆碎。
_L0_PATTERN = re.compile("|".join(map(re.escape, sorted(LIGHT_L0_CORE, key=len, reverse=True))))


class ParseError(ValueError):
    """控制 token 解析失败。**不兜底**：解析不出来就是错误状态。"""


@dataclass(frozen=True)
class DecisionPoint:
    """一个封闭的决策点。

    ``options`` 的**顺序即优先级**：探针实验里做 argmax 时取第一个命中项，
    所以顺序必须稳定（改动顺序 = 改动模型语义，需同步升 ``schema_version``）。
    """

    name: str
    options: Tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("决策点名称不能为空")
        if not self.options:
            raise ValueError(f"决策点 {self.name} 必须至少有一个选项")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"决策点 {self.name} 的选项重复: {self.options}")
        if any(not o for o in self.options):
            raise ValueError(f"决策点 {self.name} 含空选项")

    def validate(self, option: str) -> str:
        if option not in self.options:
            raise ParseError(
                f"决策点 {self.name} 无此选项 {option!r}；合法值: {list(self.options)}"
            )
        return option

    def tokens(self) -> Iterator["DecisionToken"]:
        for order, option in enumerate(self.options):
            yield DecisionToken(self.name, option, order)


@dataclass(frozen=True)
class DecisionToken:
    """一个具体的决策 token，形如 ``<D:EDIT=FR>``。"""

    point: str
    option: str
    order: int = 0

    @property
    def text(self) -> str:
        return f"<D:{self.point}={self.option}>"

    def __str__(self) -> str:  # pragma: no cover - 便利方法
        return self.text


@dataclass(frozen=True)
class ValidationToken:
    """一个验证反馈 token，形如 ``<V:PASS>`` / ``<V:ERR=SYNTAX>``。"""

    code: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.code not in VALIDATION_CODES:
            raise ParseError(
                f"未登记的验证码 {self.code!r}；合法值: {sorted(VALIDATION_CODES)}"
            )

    @property
    def text(self) -> str:
        return f"<V:{self.code}>"

    @property
    def is_error(self) -> bool:
        return self.code.startswith("ERR=")

    @property
    def is_warning(self) -> bool:
        return self.code.startswith("WARN=")

    def __str__(self) -> str:  # pragma: no cover - 便利方法
        return self.text


#: 首批 7 个决策点（方案文档 §4.2）。总数 7 个点共 19 个选项 = 19 个 token，
#: 远低于 128 的下限 —— 预留空间给后续"子程序地图"里长出来的决策点。
DEFAULT_DECISION_POINTS: Tuple[DecisionPoint, ...] = (
    DecisionPoint("EDIT", ("FR", "RW", "PATCH"), "编辑策略：find-replace / 整文件重写 / 补丁"),
    DecisionPoint("SAFE", ("L0", "L1", "L2"), "安全等级：L0/L1 自动执行，L2 需人工确认"),
    DecisionPoint("RETRY", ("YES", "NO"), "验证失败后的下一步是否重试"),
    DecisionPoint("FIX", ("ADD_TYPE", "RENAME", "REFACTOR"), "按错误类型选修复方向"),
    DecisionPoint("NEXT", ("TEST", "EDIT", "ASK"), "继续测试 / 继续编辑 / 请求人工"),
    DecisionPoint("CTX", ("KEEP", "DROP", "FETCH"), "记忆门控：保留 / 丢弃 / 重新检索"),
    DecisionPoint("PLAN", ("DECOMPOSE", "DIRECT"), "任务是否拆分子任务"),
)


@dataclass
class TokenSpace:
    """决策 + 验证 token 的封闭集合，负责序列化 / 反序列化 / 预算核算。"""

    decision_points: Sequence[DecisionPoint] = DEFAULT_DECISION_POINTS
    schema_version: str = "0.1"

    _by_point: Dict[str, DecisionPoint] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_point = {}
        for point in self.decision_points:
            if point.name in self._by_point:
                raise ValueError(f"决策点重名: {point.name}")
            self._by_point[point.name] = point

    # ---- 决策 token -------------------------------------------------
    def point(self, name: str) -> DecisionPoint:
        try:
            return self._by_point[name]
        except KeyError:
            raise ParseError(f"未登记的决策点 {name!r}；合法值: {sorted(self._by_point)}") from None

    def decision(self, point: str, option: str) -> DecisionToken:
        self.point(point).validate(option)
        return DecisionToken(point, option, self._by_point[point].options.index(option))

    def decision_tokens(self) -> List[DecisionToken]:
        return [tok for point in self.decision_points for tok in point.tokens()]

    def validation_tokens(self) -> List[ValidationToken]:
        return [ValidationToken(code) for code in VALIDATION_CODES]

    # ---- 预算核算 ---------------------------------------------------
    def budget_report(self) -> Dict[str, object]:
        """按 64K 词表预算核对控制 token 占用。返回可断言的字典。"""
        n_decision = len(self.decision_tokens())
        n_validation = len(self.validation_tokens())
        lo_d, hi_d = CONTROL_TOKEN_BUDGET["decision"]
        lo_v, hi_v = CONTROL_TOKEN_BUDGET["validation"]
        return {
            "decision": n_decision,
            "validation": n_validation,
            "decision_within_budget": lo_d <= n_decision <= hi_d,
            "validation_within_budget": lo_v <= n_validation <= hi_v,
            # 决策 token 占 64K 词表的比例，硬上限 1%—2%
            "decision_ratio_of_64k": round(n_decision / 65536, 6),
            "decision_ratio_ok": n_decision / 65536 <= CONTROL_TOKEN_MAX_RATIO,
        }

    # ---- 解析 -------------------------------------------------------
    def parse_decision(self, text: str) -> DecisionToken:
        m = re.fullmatch(r"<D:([A-Z_]+)=([A-Z0-9_]+)>", text.strip())
        if not m:
            raise ParseError(f"不是合法的决策 token: {text!r}")
        return self.decision(m.group(1), m.group(2))

    def parse_validation(self, text: str) -> ValidationToken:
        m = re.fullmatch(r"<V:([A-Z]+(?:=[A-Z_]+)?)>", text.strip())
        if not m:
            raise ParseError(f"不是合法的验证 token: {text!r}")
        return ValidationToken(m.group(1))


def map_diagnostics(diagnostics: Iterable[Mapping[str, object]]) -> ValidationToken:
    """把 ``light ai check --json`` 的 diagnostics 列表映射成**一个**验证 token。

    规则（确定性，不含模型判断）：
    - 无 error 级诊断 → ``PASS``；只有 warning → ``WARN=NON_CANONICAL``；
    - 否则取**第一个 error 级诊断**的码做映射，未登记码 → ``ERR=CHECKER_FAIL``。
    """
    diags = list(diagnostics)
    errors = [d for d in diags if str(d.get("severity", "")).lower() == "error"]
    if not errors:
        warnings = [d for d in diags if str(d.get("severity", "")).lower() == "warning"]
        return ValidationToken("WARN=NON_CANONICAL" if warnings else "PASS")
    first = errors[0]
    code = str(first.get("code", ""))
    mapped = DIAGNOSTIC_TO_VALIDATION.get(code)
    if mapped is None:
        return ValidationToken("ERR=CHECKER_FAIL", detail=f"未登记码 {code or '<空>'}")
    return ValidationToken(mapped, detail=str(first.get("message", ""))[:200])
