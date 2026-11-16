# -*- coding: utf-8 -*-
"""train / test 切分登记表（按模板分组切分，group-split）。

口径（任务卡 T-0370 §2.3）
--------------------------
* **切分单位是模板**，不是单条记录：同一模板展开出来的全部 attempt 只能落在同一侧。
  这样 ``template_overlap_train_test`` 结构上恒为 0，不做「同一模板一半 train 一半 test」
  这种会泄漏的切法。
* 一个模板只能登记在**一个 family** 里，且 family 自带 split；构造时校验：
  - 同一模板不得出现在两个 family；
  - 同一模板不得被登记成两种 split。
  校验不过直接抛 :class:`SplitError`，**不静默兜底**。
* 历史 29 条日志按它当时用的模板归侧（模板在 test 侧 ⇒ 那几条历史记录也计入 test）。
  这是 group-split 的正常结果，不是把旧数据挪去凑数。

用法::

    from lightprism.runtime.splits import split_of, family_of, FAMILIES
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

__all__ = [
    "Family",
    "FAMILIES",
    "SplitError",
    "split_of",
    "family_of",
    "families_of_split",
    "check_registry",
    "TRAIN",
    "TEST",
]

TRAIN = "train"
TEST = "test"


class SplitError(ValueError):
    """切分登记表自相矛盾（同一模板两种口径）。"""


@dataclass(frozen=True)
class Family:
    """一组共用切分口径的模板。

    ``templates`` 里的模板**全部**归本 family 的 ``split``：
    family 是模板的切分单位，模板是记录的最小归属键。
    """

    name: str
    templates: Tuple[str, ...]
    split: str
    description: str = ""
    #: 该 family 的源码变体是否天然唯一（复合任务按「两段子任务」算一组变体）
    composite: bool = False

    def __post_init__(self) -> None:
        if self.split not in (TRAIN, TEST):
            raise SplitError(f"family {self.name}: split 只能是 {TRAIN}/{TEST}，实得 {self.split!r}")
        if not self.templates:
            raise SplitError(f"family {self.name}: 模板列表不能为空")


#: 登记表：每个模板恰好出现一次。
#: ``NO_TEMPLATE`` 是「检索未命中的真实状态」，历史与新增都只在 train 侧出现。
FAMILIES: Tuple[Family, ...] = (
    # ---------------- train ----------------
    Family("arith_basic", ("arith_binop",), TRAIN, "二元算术表达式"),
    Family("paragraph_func", ("paragraph_inc",), TRAIN, "带返回值段落函数"),
    Family("loop_sum", ("loop_sum",), TRAIN, "当循环累加"),
    Family("list_len", ("list_len",), TRAIN, "列表长度"),
    Family("list_max", ("list_max",), TRAIN, "遍历求最大值"),
    Family("list_sum", ("list_sum",), TRAIN, "列表求和（历史任务，另作检索诱饵）"),
    Family("text_split_count", ("text_split_count",), TRAIN,
           "文本按分隔符切分段数（含 top1 缺槽 ⇒ CTX=DROP 的诱饵任务）"),
    Family("text_count", ("text_count",), TRAIN, "子串出现次数"),
    Family("text_affix", ("text_affix",), TRAIN, "前后缀判断"),
    Family("sysdiag_state", ("sysdiag_state",), TRAIN, "进程状态计数"),
    Family("sysdiag_avg", ("sysdiag_avg",), TRAIN, "读数平均值"),
    Family("cli_format", ("cli_format",), TRAIN, "格式化输出行"),
    Family("no_template", ("NO_TEMPLATE",), TRAIN, "检索未命中（真实失败）"),
    # ---------------- test ----------------
    Family("list_index", ("list_index",), TEST, "按索引取元素（含越界修复）"),
    Family("cli_report", ("cli_report",), TEST, "两行汇总报告"),
    Family("sysdiag_threshold", ("sysdiag_threshold",), TEST, "磁盘用量阈值计数"),
    Family("factorial", ("factorial",), TEST, "递归阶乘"),
    Family("text_replace", ("text_replace",), TEST, "文本替换"),
    # 复合任务：两段子任务各用一条**只属于复合任务**的模板，避免与单步任务共用模板。
    Family("composite_price_label", ("stage_price_total", "stage_label_total"), TEST,
           "复合任务：先算子任务一的价格，再按「合计=N」输出", composite=True),
)


def check_registry(families: Iterable[Family] = FAMILIES) -> None:
    """校验登记表：模板唯一归属、split 无冲突。空表也算错。"""
    owners: Dict[str, Family] = {}
    families = tuple(families)
    if not families:
        raise SplitError("切分登记表为空")
    for fam in families:
        for tid in fam.templates:
            other = owners.get(tid)
            if other is None:
                owners[tid] = fam
                continue
            if other.split != fam.split:
                raise SplitError(
                    f"模板 {tid} 同时登记在 {other.name}({other.split}) 与 "
                    f"{fam.name}({fam.split})，会造成 train/test 模板重合"
                )
            raise SplitError(f"模板 {tid} 重复登记：{other.name} / {fam.name}")
    for name in (f.name for f in families):
        if sum(1 for f in families if f.name == name) > 1:
            raise SplitError(f"family 重名: {name}")


check_registry()

_BY_TEMPLATE: Dict[str, Family] = {}
_BY_FAMILY: Dict[str, Family] = {}
for _fam in FAMILIES:
    _BY_FAMILY[_fam.name] = _fam
    for _tid in _fam.templates:
        _BY_TEMPLATE[_tid] = _fam


def family_of(template_id: str) -> Family:
    try:
        return _BY_TEMPLATE[template_id]
    except KeyError:
        raise SplitError(
            f"模板 {template_id!r} 未登记切分归属；请先在 runtime/splits.py 登记 family"
        ) from None


def split_of(template_id: str) -> str:
    return family_of(template_id).split


def family_of_split(split: str) -> Tuple[Family, ...]:
    if split not in (TRAIN, TEST):
        raise SplitError(f"未知 split {split!r}")
    return tuple(f for f in FAMILIES if f.split == split)


#: 兼容别名（报告里偶尔用复数形式）
families_of_split = family_of_split

#: 模板 → split 的展开表，供报告与测试直接断言
TEMPLATE_SPLIT: Mapping[str, str] = {
    tid: fam.split for fam in FAMILIES for tid in fam.templates
}
