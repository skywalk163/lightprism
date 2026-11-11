# -*- coding: utf-8 -*-
"""训练样本导出器：把 light / lightharness 的真实日志转成 LightPrism 样本。

每个导出器都必须遵守同一套规矩（见 ``docs/样本导出.md``）：

1. **跳过必须计数**，不允许静默丢弃；
2. **不编造**：日志里没有的字段（任务意图、通过的结论）不许猜；
3. **数据源格式必须写明出处**（源码行号或样例文件），不许「大概是这个形状」。
"""

from __future__ import annotations

from .light_check import (
    CorpusRecord,
    ExportResult,
    ExportStats,
    LightChecker,
    SourceIndex,
    default_corpus_dir,
    export_samples,
    fault_to_fix_decision,
    read_corpus,
)

__all__ = [
    "CorpusRecord",
    "ExportResult",
    "ExportStats",
    "LightChecker",
    "SourceIndex",
    "default_corpus_dir",
    "export_samples",
    "fault_to_fix_decision",
    "read_corpus",
]
