# -*- coding: utf-8 -*-
"""确定性 .light 模板库：占位符模板 + 独立的预期结果计算。

来源与纪律
----------
* 模板结构取自 ``light ai snippets``（``tools/ai_copilot/snippets.py`` 的
  ``__槽__`` 占位符风格），代码体逐条按 ``light run`` **实跑校准**过语法
  （例：教学片段写 ``段落 X 接收 p：``，本机实测可用的是 ``段落 X(p)：``）。
* 每条模板的 ``expected_fn`` 是**独立于 .light 代码的 Python 语义实现**，
  只用于离线生成 ``data/runtime/tasks.jsonl`` 的「预期执行结果」。
  运行时引擎**不读** ``expected_fn``、也不读任务文件里的 ``expected`` 字段来
  生成代码 —— 代码只由「任务文本 → 检索 → 抽槽 → 填槽」得到。
  这样「真跑输出 vs 预期」才是真比对，而不是自己给自己判卷。
* 缺槽时**保留** ``__槽__`` 占位符，让 ``light ai check`` 如实报错，绝不编数值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

__all__ = [
    "Template",
    "TEMPLATES",
    "get_template",
    "templates_by_cluster",
    "clusters",
    "render",
    "expected_of",
    "CLUSTER_LABELS",
]

CLUSTER_LABELS: Dict[str, str] = {
    "arith": "算术表达式",
    "func": "单函数 / 段落",
    "list": "列表处理",
    "text": "文本与文件处理",
    "sysdiag": "系统诊断",
    "cli": "小 CLI / 格式化输出",
    "edge": "边界 / 检索失败",
}


@dataclass(frozen=True)
class Template:
    id: str
    cluster: str
    description: str
    #: BM25 文档词（中文 bigram 切分后建索引）
    keywords: Tuple[str, ...]
    #: 精确命中加权词；零模型规则检索的一部分
    triggers: Tuple[str, ...]
    #: ``__槽__`` 占位符模板
    snippet: str
    #: 槽名 -> (抽取规则, 规则参数)
    slot_plan: Mapping[str, Tuple[str, object]]
    #: 独立 Python 语义：给定槽值算预期 stdout（**不参与运行时生成**）
    expected_fn: Optional[Callable[[Mapping[str, object]], str]] = None
    origin: str = "light ai snippets 占位符模板 + light run 实测校准"


def _n(slots: Mapping[str, object], key: str) -> int:
    return int(str(slots[key]))


def _num_list(slots: Mapping[str, object], key: str = "LIST") -> list:
    raw = str(slots[key]).strip().strip("[]")
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",")]


def _e_arith(s: Mapping[str, object]) -> str:
    a, b, op = _n(s, "A"), _n(s, "B"), str(s["OP"])
    r = {"加": lambda: a + b, "减": lambda: a - b,
         "乘": lambda: a * b, "除": lambda: a / b}[op]()
    return f"{r}\n"


def _e_paragraph(s: Mapping[str, object]) -> str:
    return f"{_n(s, 'N') + _n(s, 'K')}\n"


def _e_factorial(s: Mapping[str, object]) -> str:
    return f"{math.factorial(_n(s, 'N'))}\n"


def _e_loop_sum(s: Mapping[str, object]) -> str:
    n = _n(s, "N")
    return f"{n * (n + 1) // 2}\n"


def _e_list_sum(s: Mapping[str, object]) -> str:
    return f"{sum(_num_list(s))}\n"


def _e_list_len(s: Mapping[str, object]) -> str:
    return f"{len(_num_list(s))}\n"


def _e_list_index(s: Mapping[str, object]) -> str:
    return f"{_num_list(s)[_n(s, 'I')]}\n"


def _e_list_max(s: Mapping[str, object]) -> str:
    return f"{max(_num_list(s))}\n"


def _e_split_count(s: Mapping[str, object]) -> str:
    return f"{len(str(s['TEXT']).split(str(s['SEP'])))}\n"


def _e_text_count(s: Mapping[str, object]) -> str:
    return f"{str(s['TEXT']).count(str(s['CH']))}\n"


def _e_affix(s: Mapping[str, object]) -> str:
    text, part, func = str(s["TEXT"]), str(s["PART"]), str(s["FUNC"])
    ok = text.startswith(part) if func == "开头" else text.endswith(part)
    return f"{ok}\n"


def _e_replace(s: Mapping[str, object]) -> str:
    return f"{str(s['TEXT']).replace(str(s['OLD']), str(s['NEW']))}\n"


def _e_threshold(s: Mapping[str, object]) -> str:
    th = _n(s, "TH")
    return f"{sum(1 for x in _num_list(s) if x > th)}\n"


def _e_state(s: Mapping[str, object]) -> str:
    return f"{str(s['TEXT']).split(',').count(str(s['STATE']))}\n"


def _e_avg(s: Mapping[str, object]) -> str:
    xs = _num_list(s)
    return f"{sum(xs) / len(xs)}\n"


def _e_cli_format(s: Mapping[str, object]) -> str:
    return f"结果={s['N']}\n"


def _e_cli_report(s: Mapping[str, object]) -> str:
    xs = _num_list(s)
    return f"总数={len(xs)}\n总和={sum(xs)}\n"


TEMPLATES: Tuple[Template, ...] = (
    Template(
        id="arith_binop", cluster="arith", description="二元算术表达式直接输出",
        keywords=("计算 两数 算术 加法 减法 乘法 除法 结果 数字 输出",
                  "计算 表达式 求值 加 减 乘 除"),
        triggers=("计算", "加", "减", "乘", "除"),
        snippet="打印(__A__ __OP__ __B__)。\n",
        slot_plan={"A": ("numbers", 0), "B": ("numbers", 1), "OP": ("op", "加")},
        expected_fn=_e_arith,
    ),
    Template(
        id="paragraph_inc", cluster="func", description="带返回值的段落函数并求值",
        keywords=("段落 函数 定义 返回 求值 输入 加上 单函数 调用",),
        triggers=("段落", "函数"),
        snippet="段落 加一(数)：\n  返回 数 加 __K__。\n打印(加一(__N__))。\n",
        slot_plan={"K": ("numbers", 0), "N": ("numbers", 1)},
        expected_fn=_e_paragraph,
    ),
    Template(
        id="factorial", cluster="func", description="递归段落求阶乘",
        keywords=("阶乘 递归 段落 终止条件 返回值 数学",),
        triggers=("阶乘",),
        snippet=("段落 阶乘(数)：\n  如果 数 小于等于 1：\n    返回 1。\n"
                 "  返回 数 乘 阶乘(数 减 1)。\n打印(阶乘(__N__))。\n"),
        slot_plan={"N": ("numbers", 0)},
        expected_fn=_e_factorial,
    ),
    Template(
        id="loop_sum", cluster="func", description="当循环累加连续整数",
        keywords=("累加 连续整数 从 到 循环 总和 求和 当循环",),
        triggers=("累加", "连续整数"),
        snippet=("设 总 为 0。\n设 次 为 1。\n当 次 小于等于 __N__：\n"
                 "  设 总 为 总 加 次。\n  设 次 为 次 加 1。\n打印(总)。\n"),
        slot_plan={"N": ("range_end", 0)},
        expected_fn=_e_loop_sum,
    ),
    Template(
        id="list_sum", cluster="list", description="列表元素求和",
        keywords=("列表 数组 元素 求和 总和 所有元素 合计",),
        triggers=("列表", "求和"),
        snippet="设 甲 为 __LIST__。\n打印(求和(甲))。\n",
        slot_plan={"LIST": ("lists", 0)},
        expected_fn=_e_list_sum,
    ),
    Template(
        id="list_len", cluster="list", description="列表元素个数",
        keywords=("列表 数组 长度 个数 元素个数 有多少个 计数",),
        triggers=("列表", "长度"),
        snippet="设 甲 为 __LIST__。\n打印(列表长度(甲))。\n",
        slot_plan={"LIST": ("lists", 0)},
        expected_fn=_e_list_len,
    ),
    Template(
        id="list_index", cluster="list", description="按索引取列表元素",
        keywords=("列表 索引 第 个元素 取出 取第 下标 位置",),
        triggers=("第", "个元素", "索引"),
        snippet="设 甲 为 __LIST__。\n打印(甲[__I__])。\n",
        slot_plan={"LIST": ("lists", 0), "I": ("index", 0)},
        expected_fn=_e_list_index,
    ),
    Template(
        id="list_max", cluster="list", description="遍历求列表最大值",
        keywords=("列表 最大值 最大元素 遍历 比较 峰值 最大",),
        triggers=("最大值", "最大元素"),
        snippet=("设 甲 为 __LIST__。\n设 最大 为 甲[0]。\n遍历 项 于 甲：\n"
                 "  如果 项 大于 最大：\n    设 最大 为 项。\n打印(最大)。\n"),
        slot_plan={"LIST": ("lists", 0)},
        expected_fn=_e_list_max,
    ),
    Template(
        id="text_split_count", cluster="text", description="文本按分隔符切分后的段数",
        keywords=("文本 分割 分隔符 逗号 段数 行数 切分 统计 文件",),
        triggers=("分割", "段数"),
        snippet="设 文 为 \"__TEXT__\"。\n打印(长度(分割字符串(文, \"__SEP__\")))。\n",
        slot_plan={"TEXT": ("quoted", 0), "SEP": ("sep", ",")},
        expected_fn=_e_split_count,
    ),
    Template(
        id="text_count", cluster="text", description="统计子串在文本中出现次数",
        keywords=("文本 字符 子串 出现次数 计数 统计 次数",),
        triggers=("出现次数", "统计文本"),
        snippet="设 文 为 \"__TEXT__\"。\n打印(字符串计数(文, \"__CH__\"))。\n",
        slot_plan={"TEXT": ("quoted", 0), "CH": ("quoted", 1)},
        expected_fn=_e_text_count,
    ),
    Template(
        id="text_affix", cluster="text", description="文件名前缀 / 后缀判断",
        keywords=("文件名 后缀 前缀 开头 结尾 判断 扩展名 日志行",),
        triggers=("后缀", "前缀"),
        snippet="设 文 为 \"__TEXT__\"。\n打印(__FUNC__(文, \"__PART__\"))。\n",
        slot_plan={"TEXT": ("quoted", 0), "PART": ("quoted", 1), "FUNC": ("affix", "结尾")},
        expected_fn=_e_affix,
    ),
    Template(
        id="text_replace", cluster="text", description="清理文本中的空格",
        keywords=("文本 替换 去除 空格 清理 去掉 字符串替换",),
        triggers=("去掉", "替换文本"),
        snippet="设 文 为 \"__TEXT__\"。\n打印(替换字符串(文, \"__OLD__\", \"__NEW__\"))。\n",
        slot_plan={"TEXT": ("quoted", 0), "OLD": ("const", " "), "NEW": ("const", "")},
        expected_fn=_e_replace,
    ),
    Template(
        id="sysdiag_threshold", cluster="sysdiag", description="磁盘用量超阈值计数",
        keywords=("系统诊断 磁盘 用量 阈值 超过 告警 监控 巡检",),
        triggers=("磁盘", "阈值"),
        snippet=("设 用量 为 __LIST__。\n设 计数 为 0。\n遍历 项 于 用量：\n"
                 "  如果 项 大于 __TH__：\n    设 计数 为 计数 加 1。\n打印(计数)。\n"),
        slot_plan={"LIST": ("lists", 0), "TH": ("threshold", 0)},
        expected_fn=_e_threshold,
    ),
    Template(
        id="sysdiag_state", cluster="sysdiag", description="进程状态文本中某状态出现次数",
        keywords=("系统诊断 进程 状态 服务 日志 状态计数 出现次数",),
        triggers=("进程", "状态"),
        snippet=("设 文 为 \"__TEXT__\"。\n设 段 为 分割字符串(文, \",\")。\n"
                 "设 计数 为 0。\n遍历 项 于 段：\n"
                 "  如果 项 等于 \"__STATE__\"：\n    设 计数 为 计数 加 1。\n打印(计数)。\n"),
        slot_plan={"TEXT": ("quoted", 0), "STATE": ("state", "run")},
        expected_fn=_e_state,
    ),
    Template(
        id="sysdiag_avg", cluster="sysdiag", description="从数学模块导入平均数算读数均值",
        keywords=("系统诊断 温度 读数 平均值 均值 平均数 数学模块 导入",),
        triggers=("读数", "平均值"),
        snippet="从 数学 导入 平均数。\n设 读数 为 __LIST__。\n打印(平均数(读数))。\n",
        slot_plan={"LIST": ("lists", 0)},
        expected_fn=_e_avg,
        origin="light ai snippets「导入-标准库」+ stdlib 数学模块 + light run 实测校准",
    ),
    Template(
        id="cli_format", cluster="cli", description="带标签的格式化输出行",
        keywords=("命令行 CLI 格式化 输出格式 标签 前缀 结果等号",),
        triggers=("格式化", "结果="),
        snippet="打印(\"结果=\" + 转字符串(__N__))。\n",
        slot_plan={"N": ("numbers", 0)},
        expected_fn=_e_cli_format,
    ),
    Template(
        id="cli_report", cluster="cli", description="两行汇总报告（总数 / 总和）",
        keywords=("命令行 CLI 汇总报告 两行 总数 总和 输出 报表",),
        triggers=("汇总报告", "两行"),
        snippet=("设 数据 为 __LIST__。\n"
                 "打印(\"总数=\" + 转字符串(列表长度(数据)))。\n"
                 "打印(\"总和=\" + 转字符串(求和(数据)))。\n"),
        slot_plan={"LIST": ("lists", 0)},
        expected_fn=_e_cli_report,
    ),
)

_BY_ID: Dict[str, Template] = {t.id: t for t in TEMPLATES}


def get_template(template_id: str) -> Template:
    return _BY_ID[template_id]


def clusters() -> Tuple[str, ...]:
    seen: list = []
    for t in TEMPLATES:
        if t.cluster not in seen:
            seen.append(t.cluster)
    return tuple(seen)


def templates_by_cluster(cluster: str) -> Tuple[Template, ...]:
    return tuple(t for t in TEMPLATES if t.cluster == cluster)


def render(template: Template, slots: Mapping[str, object]) -> str:
    """填 ``__槽__``。槽缺失**保留占位符**（让 checker 如实报错，不编数值）。"""
    out = template.snippet
    for name in template.slot_plan:
        if name in slots:
            out = out.replace(f"__{name}__", str(slots[name]))
    return out


def expected_of(template: Template, slots: Mapping[str, object]) -> str:
    """离线生成任务集用的预期 stdout。引擎运行时不调用本函数。"""
    if template.expected_fn is None:
        raise ValueError(f"模板 {template.id} 没有独立预期语义")
    return template.expected_fn(slots)
