# -*- coding: utf-8 -*-
"""``lightprism chat`` 的推理薄层：checkpoint → DemoGenerator → 一段文本。

这里只做参数校验、计时和结果整形，**不另写推理**：真正的生成仍走
:mod:`lightprism.demo.smoke` 的 :class:`DemoGenerator`（带 KV cache 的那条路径）。

两条口径（卡面 T-0338）：

* 缺 checkpoint 是**错误**，调用方必须拿到异常并以非零退出——chat 是推理入口，
  没模型就没有任何东西可输出；这与 ``demo.sh``「缺 checkpoint 不算执行错误」
  的构建口径不同，不能混。
* 生成出来的文本有没有可见字符，单独用 ``ok`` / ``empty_output`` 表达。
  **不把空输出当成功**：32 个空格不会被说成「有内容」。但命令本身只要完整跑完
  推理就算执行成功（退出码 0），是否可用交给 ``ok`` 字段判断——验收脚本的
  V1/V3 只看「入口在不在、输出能不能解析」，内容质量是 V2 的事。

本模块**不在导入时**拉 torch：只在真正加载 checkpoint 时才 import
:mod:`lightprism.demo.smoke`。这样缺 torch 的环境仍能执行 ``--help``、参数校验、
缺 checkpoint 的错误路径和全部纯 CLI 测试。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

__all__ = ["ChatError", "DEFAULT_CKPT", "resolve_ckpt", "run_chat"]

#: 仓库根目录（``src/lightprism/demo/chat.py`` 往上三层）。
ROOT = Path(__file__).resolve().parents[3]

#: 默认 checkpoint，与 demo 训练脚本的落点一致。
#:
#: **相对当前工作目录**而不是模块所在仓库：`lightprism chat` 在仓库根运行就命中
#: `checkpoints/demo-3m/model.pt`；在别的目录运行则明确报「checkpoint 不存在」，
#: 不会悄悄用别的仓库里的模型顶替（验收脚本的 tmp_path 用例依赖这个语义）。
DEFAULT_CKPT = Path("checkpoints") / "demo-3m" / "model.pt"

#: 默认生成 token 数。
DEFAULT_MAX_TOKENS = 64

#: 环境变量覆盖默认 checkpoint。
CKPT_ENV = "LIGHTPRISM_CKPT"


class ChatError(RuntimeError):
    """chat 入口的可读错误（参数、checkpoint、推理失败）。"""


def resolve_ckpt(ckpt: Optional[str] = None) -> Path:
    """按 ``--ckpt`` → ``$LIGHTPRISM_CKPT`` → 默认路径的顺序解析 checkpoint。"""
    raw = ckpt or os.environ.get(CKPT_ENV) or DEFAULT_CKPT
    return Path(raw).expanduser()


def _load_generator(ckpt: Path, max_tokens: int) -> Any:
    """加载 demo 生成器（延迟 import，缺 torch 时转成可读错误）。"""
    try:
        from .smoke import load_demo_generator
    except ImportError as exc:  # pragma: no cover - 取决于环境是否装 torch
        raise ChatError(
            "加载推理依赖失败（需要 torch）："
            f"{exc}；请安装 `pip install 'light-prism[train]'` 或改用 --format json "
            "读取错误对象"
        ) from exc
    return load_demo_generator(ckpt, max_new_tokens=max_tokens)


def _model_info(generator: Any) -> Dict[str, Any]:
    """尽力给出模型规模；假生成器没有 model 就留空，不影响主流程。"""
    info: Dict[str, Any] = {}
    model = getattr(generator, "model", None)
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        for key in ("dim", "n_layers", "n_heads", "vocab_size", "max_seq"):
            value = getattr(cfg, key, None)
            if value is not None:
                info[key] = value
    vocab = getattr(generator, "vocab", None)
    vocab_size = getattr(vocab, "vocab_size", None)
    if vocab_size is not None:
        info["vocab_size"] = vocab_size
    return info


def run_chat(
    prompt: str,
    *,
    ckpt: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    loader: Optional[Callable[[Path, int], Any]] = None,
) -> Dict[str, Any]:
    """加载 checkpoint 生成一次，返回结构化结果。

    参数
    ----
    prompt
        任务提示词；空串直接报错（没有提示词就没有可生成的任务）。
    ckpt
        checkpoint 路径；``None`` 走 :func:`resolve_ckpt`。
    max_tokens
        最多生成 token 数，必须为正。
    loader
        生成器工厂，签名 ``(ckpt_path, max_tokens) -> generator``。测试注入假生成器
        用，默认走 :func:`_load_generator`。

    返回字段至少包含卡面要求的 ``text`` / ``tokens`` / ``tok_s``：

    * ``ok``：文本是否有**可见字符**；空输出为 ``False``（不冒充成功）。
    * ``empty_output``：``not ok`` 的显式副本，方便调用方直接读。
    * ``elapsed_s``：仅生成阶段的耗时；``tok_s`` 按生成 token 数 / 该耗时算。
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ChatError("--prompt 不能为空")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ChatError(f"--max-tokens 必须是正整数，收到 {max_tokens!r}")

    path = resolve_ckpt(ckpt)
    if not path.is_file():
        raise ChatError(
            f"checkpoint 不存在：{path}；先用 demo 训练脚本产出模型，或用 --ckpt 指定"
            f"（也可设 ${CKPT_ENV}）"
        )

    generator = (loader or _load_generator)(path, max_tokens)

    started = time.perf_counter()
    try:
        out = generator.generate(prompt, max_new_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 - 推理异常一律转成可读错误
        raise ChatError(f"推理失败：{type(exc).__name__}: {exc}") from exc
    elapsed = max(time.perf_counter() - started, 1e-9)

    text = str(out.get("text") or "")
    ids = list(out.get("ids") or [])
    tokens = int(out.get("generated_tokens") or len(ids))
    visible = text.strip()
    payload: Dict[str, Any] = {
        "ok": bool(visible),
        "empty_output": not visible,
        "prompt": prompt,
        "text": text,
        "tokens": tokens,
        "tok_s": round(tokens / elapsed, 2),
        "elapsed_s": round(elapsed, 4),
        "ckpt": str(path),
        "max_tokens": max_tokens,
        "truncated": bool(out.get("truncated")),
        "visible_chars": len(visible),
    }
    payload["model"] = _model_info(generator)
    return payload