# -*- coding: utf-8 -*-
"""LightPrism demo 子包：4096 实词表的构建与校验。

64K 词表是目标形态（见 :mod:`lightprism.vocab`）；demo 只需要 4096 的实词表，
理由见 ``docs/Demo开发计划.md`` §3。这里不训练 BPE，只做**从语料统计出 token
清单**这一件事，并保证与 :class:`lightprism.tokens.TokenSpace` 的控制 token 逐条对齐。

典型用法::

    PYTHONPATH=src python -m lightprism.demo.vocab_build --build --light-root <light 仓>
    PYTHONPATH=src python -m lightprism.demo.vocab_build --check data/vocab_4096.json

不在包初始化时 import ``vocab_build``：``python -m lightprism.demo.vocab_build``
会经 runpy 再次执行该模块，预导入会触发 ``found in sys.modules`` 警告。
需要 API 时用惰性 ``__getattr__``（PEP 562）。
"""

from typing import Any

_EXPORTS = frozenset({
    "VOCAB_LIMIT", "MAX_SIZE", "HEADROOM_RATIO", "UNK_LIMIT",
    "CHAR_COVERAGE_TARGET", "build_vocab", "check_vocab", "unk_rate",
})

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        from . import vocab_build
        return getattr(vocab_build, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
