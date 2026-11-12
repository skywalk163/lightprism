#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建 train_openi.py 的内嵌 payload（zlib+base64）。

用法::

    python scripts/embed_payload.py              # 重写 _PAYLOAD_B64 区块
    python scripts/embed_payload.py --check      # 只比对、不改文件（CI 用）

设计要点
--------
* 读 ``data/vocab_4096.json`` + ``data/sft/{json,code,err}_tasks.jsonl``
* 按 _load_payload() 同样的结构打包成 JSON，zlib 压缩，base64 编码
* 重写 ``train_openi.py`` 里的 ``_PAYLOAD_B64 = r'''...'''`` 区块
* 打印每份数据的 SHA-256，便于核对
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import zlib
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_OPENI = _REPO_ROOT / "train_openi.py"
VOCAB_PATH = _REPO_ROOT / "data" / "vocab_4096.json"
SFT_DIR = _REPO_ROOT / "data" / "sft"

#: _PAYLOAD_B64 区块的起止标记
PAYLOAD_START = "_PAYLOAD_B64 = r'''"
PAYLOAD_END = "'''"

#: base64 每行宽度（与原文件风格一致）
B64_LINE_WIDTH = 80


def load_payload() -> Dict[str, Any]:
    """从 data/ 目录加载完整 payload（与 _load_payload 外部模式同结构）。"""
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))

    payload: Dict[str, Any] = {
        "vocab": {
            k: vocab[k] for k in (
                "schema_version", "kind", "size", "tokens",
                "kinds", "control_layout", "counts",
            ) if k in vocab
        },
        "meta": {
            "source": "embed_payload.py",
        },
    }

    # 加载各 SFT 数据集
    for name in ("json_tasks", "code_tasks", "err_tasks"):
        path = SFT_DIR / f"{name}.jsonl"
        if path.is_file():
            payload[name] = path.read_text(encoding="utf-8")
        else:
            # err_tasks 不存在时该 key 为空，不报错
            payload[name] = ""

    # 计算各文件的 SHA-256
    payload["meta"]["json_tasks_sha256"] = hashlib.sha256(
        payload["json_tasks"].encode("utf-8")).hexdigest()
    payload["meta"]["code_tasks_sha256"] = hashlib.sha256(
        payload["code_tasks"].encode("utf-8")).hexdigest()
    payload["meta"]["vocab_sha256"] = hashlib.sha256(
        json.dumps(payload["vocab"], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if payload.get("err_tasks"):
        payload["meta"]["err_tasks_sha256"] = hashlib.sha256(
            payload["err_tasks"].encode("utf-8")).hexdigest()

    return payload


def encode_payload(payload: Dict[str, Any]) -> str:
    """把 payload 编码成 base64 字符串（与原文件格式一致）。"""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    b64 = base64.b64encode(compressed).decode("ascii")
    # 按每行 80 字符折行
    lines = [b64[i:i + B64_LINE_WIDTH] for i in range(0, len(b64), B64_LINE_WIDTH)]
    return "\n".join(lines)


def decode_payload(b64_text: str) -> Dict[str, Any]:
    """从 base64 文本解码回 payload dict。"""
    cleaned = "".join(b64_text.split())
    raw = zlib.decompress(base64.b64decode(cleaned))
    return json.loads(raw.decode("utf-8"))


def extract_current_b64(text: str) -> str:
    """从 train_openi.py 源码里提取当前的 _PAYLOAD_B64 内容。"""
    start_idx = text.find(PAYLOAD_START)
    if start_idx < 0:
        raise ValueError("找不到 _PAYLOAD_B64 起始标记")
    start_idx += len(PAYLOAD_START)
    end_idx = text.find(PAYLOAD_END, start_idx)
    if end_idx < 0:
        raise ValueError("找不到 _PAYLOAD_B64 结束标记")
    return text[start_idx:end_idx].strip()


def rebuild_train_openi(new_b64: str) -> str:
    """重新生成 train_openi.py 的内容（替换 _PAYLOAD_B64 区块）。"""
    text = TRAIN_OPENI.read_text(encoding="utf-8")
    current_b64 = extract_current_b64(text)

    # 构建新的区块
    old_block = PAYLOAD_START + "\n" + current_b64 + "\n" + PAYLOAD_END
    new_block = PAYLOAD_START + "\n" + new_b64 + "\n" + PAYLOAD_END

    return text.replace(old_block, new_block)


def print_sha_summary(payload: Dict[str, Any]) -> None:
    """打印各数据的 SHA-256 摘要。"""
    print("数据 SHA-256 摘要：")
    print(f"  vocab:      {payload['meta']['vocab_sha256']}")
    print(f"  json_tasks: {payload['meta']['json_tasks_sha256']}")
    print(f"  code_tasks: {payload['meta']['code_tasks_sha256']}")
    if "err_tasks_sha256" in payload["meta"]:
        print(f"  err_tasks:  {payload['meta']['err_tasks_sha256']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="重建 train_openi.py 的内嵌 payload",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="只比对、不改文件（CI 用）",
    )
    args = parser.parse_args()

    # 加载新 payload
    payload = load_payload()
    print_sha_summary(payload)

    # 编码成 base64
    new_b64 = encode_payload(payload)
    new_compressed_len = len(base64.b64decode(new_b64))
    print(f"\n新 payload 压缩后大小: {new_compressed_len} bytes")

    # 读取当前 train_openi.py
    current_text = TRAIN_OPENI.read_text(encoding="utf-8")
    current_b64 = extract_current_b64(current_text)

    # 解码当前 payload，对比内容
    try:
        current_payload = decode_payload(current_b64)
        # 比较关键字段
        same_vocab = current_payload.get("vocab") == payload["vocab"]
        same_json = current_payload.get("json_tasks", "") == payload["json_tasks"]
        same_code = current_payload.get("code_tasks", "") == payload["code_tasks"]
        same_err = current_payload.get("err_tasks", "") == payload.get("err_tasks", "")

        print(f"\n与当前 payload 对比：")
        print(f"  vocab:     {'一致' if same_vocab else '不一致'}")
        print(f"  json_tasks: {'一致' if same_json else '不一致'}")
        print(f"  code_tasks: {'一致' if same_code else '不一致'}")
        print(f"  err_tasks:  {'一致' if same_err else '不一致'}")

        all_same = same_vocab and same_json and same_code and same_err
    except Exception as exc:
        print(f"\n当前 payload 解码失败: {exc}")
        all_same = False

    if args.check:
        if all_same:
            print("\n✓ --check 通过：内嵌数据与文件一致")
            return 0
        else:
            print("\n✗ --check 失败：内嵌数据与文件不一致")
            return 1

    # 写入新的 train_openi.py
    new_text = rebuild_train_openi(new_b64)
    TRAIN_OPENI.write_text(new_text, encoding="utf-8")
    print(f"\n✓ 已重写 {TRAIN_OPENI} 的 _PAYLOAD_B64 区块")
    return 0


if __name__ == "__main__":
    sys.exit(main())
