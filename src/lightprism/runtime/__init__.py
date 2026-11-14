# -*- coding: utf-8 -*-
"""LightPrism 阶段 0：零模型确定性系统 + 决策/验证日志采集（T-0346）。

纯标准库 + 真实 ``light`` CLI，不训练模型、不调用任何 LLM。
入口：``python3 scripts/runtime_eval.py --tasks data/runtime/tasks.jsonl``
"""
from __future__ import annotations

from . import checker, decisions, engine, retrieval, templates
from .checker import CheckResult, LightRunner, RunResult, find_light_root
from .decisions import (
    DecisionRecord,
    DecisionWriter,
    OBSERVABLE_VALIDATION_CODES,
    map_check_payload,
    validate_record,
)
from .engine import RuntimeEngine, TaskOutcome, TaskSpec, load_tasks
from .retrieval import BM25Index, extract_slots
from .templates import TEMPLATES, render

__all__ = [
    "checker", "decisions", "engine", "retrieval", "templates",
    "CheckResult", "RunResult", "LightRunner", "find_light_root",
    "DecisionRecord", "DecisionWriter", "OBSERVABLE_VALIDATION_CODES",
    "map_check_payload", "validate_record",
    "RuntimeEngine", "TaskOutcome", "TaskSpec", "load_tasks",
    "BM25Index", "extract_slots", "TEMPLATES", "render",
]

__version__ = "0.1.0"
