# -*- coding: utf-8 -*-
"""复合任务的两段式拆分（PLAN=DECOMPOSE 的唯一来源）。

为什么单独成模块
----------------
``PLAN=DECOMPOSE`` 不能靠贴标签：只有当系统**真的**把一个任务拆成两段、
先跑完子任务一、把它的**真跑 stdout** 注入子任务二、再跑完子任务二，
这条日志才有 ``<D:PLAN=DECOMPOSE>``。本模块只做两件确定性的事：

1. :func:`split_stages` —— 按显式连接词（「，再」「，然后」「；再」「，接着」）
   把任务文本切成两段；切不开就返回 ``None``，由引擎退回单模板直展（DIRECT）。
2. :data:`STAGE_SPECS` —— 子任务二的槽位由谁注入（目前只有价格合计这一条，
   槽名 ``N``，值取子任务一 stdout 的整数）。

拆分判据只看**文本**与**检索结果**（两段子文本必须各自命中带 trigger 的模板），
不看任务卡里的 cluster 标签，避免「用答案提示自己」。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

__all__ = ["StagePlan", "split_stages", "COMPOSITE_FAMILY", "STAGE_INJECT"]

#: 复合任务在切分登记表里的 family 名（见 :mod:`lightprism.runtime.splits`）
COMPOSITE_FAMILY = "composite_price_label"

#: 子任务二里由「上一段的真跑输出」注入的槽：{模板 id: (槽名, 取值方式)}
STAGE_INJECT: Dict[str, Tuple[str, str]] = {
    "stage_label_total": ("N", "stdout_int"),
}

#: 显式连接词。按出现位置取**第一个**，保证切分确定。
_CONNECTORS = ("，再", "，然后", "；再", "，接着", ",再", ",然后")

_TRAILING = "。！？!?；;，, \t"


@dataclass(frozen=True)
class StagePlan:
    """拆分结果：两段子任务文本 + 连接词。"""

    text_a: str
    text_b: str
    connector: str

    def as_dict(self) -> dict:
        return {"text_a": self.text_a, "text_b": self.text_b, "connector": self.connector}


def split_stages(text: str) -> Optional[StagePlan]:
    """按显式连接词切两段；任一段为空 / 过短就返回 ``None``。"""
    if not text:
        return None
    best = None
    for conn in _CONNECTORS:
        idx = text.find(conn)
        if idx < 0:
            continue
        if best is None or idx < best[0]:
            best = (idx, conn)
    if best is None:
        return None
    idx, conn = best
    head = text[:idx].strip().rstrip(_TRAILING)
    tail = text[idx + len(conn):].strip().lstrip(_TRAILING)
    # 连接词本身可能带「然后再 / 接着再」，去掉残留的「再」
    tail = re.sub(r"^(再|然后|接着)", "", tail).strip()
    if len(head) < 4 or len(tail) < 4:
        return None
    if not head.endswith("。"):
        head += "。"
    if not tail.endswith("。"):
        tail += "。"
    return StagePlan(text_a=head, text_b=tail, connector=conn)
