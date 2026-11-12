#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# T-0335 · LightPrism demo 的一键入口。
#
#   ./scripts/demo.sh "把这三个文件按大小排序"
#   ./scripts/demo.sh --format json "报告磁盘使用率"
#
# 约定（写在这里，别在调用方靠猜）：
#   * 默认 checkpoint = checkpoints/demo-3m/model.pt（T-0332 产出，不入库）；
#   * **没有 checkpoint 不是执行错误**：打印人能读懂的提示，退出码 0。
#     这样 `make demo` 在骨架阶段就是一条可用命令，而不是 traceback；
#     JSON 里用 ok=false / error=checkpoint_missing 明确区分「没跑」与「跑砸了」。
#   * 有 checkpoint 时才做真实生成；生成失败一律非零退出，不吞异常。
#   * 输出五项：生成 token 数、耗时、tok/s、内存峰值、light ai check 状态。
#     拿不到的项写 null / skipped 并给原因，**绝不填推测值**。
#
# 允许改动的路径只有本文件与 Makefile/README/docs，故生成编排内联在此，
# 不新增 src/ 下的模块；推理复用已交付的 lightprism.demo.smoke 生成器。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONIOENCODING=utf-8
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    PY=python
fi
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "未找到 Python 解释器（可用 PYTHON=/path/to/python 指定）" >&2
    exit 127
fi

DEFAULT_CKPT="checkpoints/demo-3m/model.pt"
CKPT="$DEFAULT_CKPT"
FORMAT="text"
PROMPT=""
MAX_NEW_TOKENS=32

usage() {
    cat <<'USAGE'
用法: scripts/demo.sh [选项] ["prompt"]

  --format text|json    输出格式，默认 text（--json 是 --format json 的别名）
  --ckpt PATH           checkpoint 路径，默认 checkpoints/demo-3m/model.pt
  --max-new-tokens N    生成 token 数，默认 32
  --prompt STR          也可直接把 prompt 作为最后一个位置参数
  -h, --help            显示本帮助

  PYTHON=/path/to/python make demo   可指定解释器
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --format)
            [ $# -ge 2 ] || { echo "--format 缺少取值" >&2; exit 64; }
            FORMAT="$2"; shift 2 ;;
        --format=*) FORMAT="${1#*=}"; shift ;;
        --json)     FORMAT="json"; shift ;;
        --ckpt)     [ $# -ge 2 ] || { echo "--ckpt 缺少取值" >&2; exit 64; }
                    CKPT="$2"; shift 2 ;;
        --ckpt=*)   CKPT="${1#*=}"; shift ;;
        --max-new-tokens)
                    [ $# -ge 2 ] || { echo "--max-new-tokens 缺少取值" >&2; exit 64; }
                    MAX_NEW_TOKENS="$2"; shift 2 ;;
        --prompt)   [ $# -ge 2 ] || { echo "--prompt 缺少取值" >&2; exit 64; }
                    PROMPT="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        --)         shift; [ $# -eq 0 ] || PROMPT="$*"; break ;;
        -*)         echo "未知参数: $1（用 --help 看用法）" >&2; exit 64 ;;
        *)          PROMPT="${PROMPT:+$PROMPT }$1"; shift ;;
    esac
done

case "$FORMAT" in
    text|json) ;;
    *) echo "--format 只接受 text 或 json，收到: $FORMAT" >&2; exit 64 ;;
esac
case "$MAX_NEW_TOKENS" in
    ''|*[!0-9]*) echo "--max-new-tokens 必须是正整数" >&2; exit 64 ;;
esac
[ -n "$PROMPT" ] || PROMPT="把这三个文件按大小排序"

# ---- 缺 checkpoint：可读提示，不抛 traceback，不假装生成成功 -----------------
if [ ! -f "$CKPT" ]; then
    MISSING_MSG="未找到 $CKPT，请先跑 make train 或从 releases 下载"
    "$PY" - "$FORMAT" "$CKPT" "$PROMPT" "$MISSING_MSG" <<'PY'
import json
import sys

fmt, ckpt, prompt, msg = sys.argv[1:5]
if fmt == "json":
    print(json.dumps({
        "ok": False,
        "error": "checkpoint_missing",
        "ckpt": ckpt,
        "message": msg,
        "prompt": prompt,
        "format": "json",
        "generated_tokens": None,
        "elapsed_s": None,
        "tok_per_s": None,
        "peak_rss_mb": None,
        "light_ai_check": {"status": "skipped", "reason": "checkpoint_missing"},
    }, ensure_ascii=False, indent=2))
else:
    print("LightPrism demo · 3.4M 参数 CPU 推理（T-0335 门面）")
    print(f"  checkpoint : {ckpt}")
    print(f"  prompt     : {prompt}")
    print("  format     : text")
    print()
    print(msg)
    print()
    print("本次未执行生成（缺 checkpoint）：")
    print("  生成 token 数 : —")
    print("  耗时          : —")
    print("  tok/s         : —")
    print("  内存峰值      : —")
    print("  light ai check: 未执行（缺 checkpoint）")
PY
    exit 0
fi

# ---- 有 checkpoint：真实生成 + 实测计时 / 内存 / 验证器状态 -----------------
"$PY" - "$FORMAT" "$CKPT" "$PROMPT" "$MAX_NEW_TOKENS" <<'PY'
import json
import shutil
import sys
import time

fmt, ckpt, prompt, n_new = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])


def peak_rss_mb():
    """峰值常驻内存。跨平台取法，取不到就如实返回 None，不猜。"""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM"):
                    return round(int(line.split()[1]) / 1024.0, 1), "VmHWM(/proc)"
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            c = PMC()
            c.cb = ctypes.sizeof(c)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            if ok:
                return round(c.PeakWorkingSetSize / 1048576.0, 1), "PeakWorkingSet(win32)"
        except Exception:
            pass
    return None, "unavailable"


# light ai check：只做能力探测，不擅自推断 light CLI 的参数形态。
# 没有 CLI 就明确 skipped；串联校验待 T-0332 出 checkpoint 后另开卡补齐。
light_bin = shutil.which("light")
if light_bin:
    light_check = {
        "status": "not_implemented",
        "reason": "生成结果的 light ai check 串联待 T-0332 出 checkpoint 后补；"
                  "本卡不推断 light CLI 的参数形态",
    }
else:
    light_check = {"status": "skipped", "reason": "light CLI 不在 PATH，未执行验证"}

try:
    from lightprism.demo.smoke import load_demo_generator

    t0 = time.perf_counter()
    gen = load_demo_generator(ckpt, max_new_tokens=n_new)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    out = gen.generate(prompt, max_new_tokens=n_new)
    gen_s = time.perf_counter() - t1

    n = out["generated_tokens"]
    rss_mb, rss_src = peak_rss_mb()
    payload = {
        "ok": n > 0,
        "error": None,
        "ckpt": ckpt,
        "prompt": prompt,
        "format": fmt,
        "generated_tokens": n,
        "elapsed_s": round(gen_s, 4),
        "checkpoint_load_s": round(load_s, 4),
        "tok_per_s": round(n / gen_s, 1) if gen_s > 0 else None,
        "peak_rss_mb": rss_mb,
        "rss_source": rss_src,
        "truncated": out["truncated"],
        "text": out["text"],
        "light_ai_check": light_check,
    }
except Exception as exc:                      # noqa: BLE001 - 不吞异常，转成结构化失败
    payload = {
        "ok": False,
        "error": "generation_failed",
        "reason": f"{type(exc).__name__}: {exc}",
        "ckpt": ckpt, "prompt": prompt, "format": fmt,
        "generated_tokens": None, "elapsed_s": None, "tok_per_s": None,
        "peak_rss_mb": None, "light_ai_check": light_check,
    }

if fmt == "json":
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    def show(v):
        return "—" if v is None else str(v)
    print("LightPrism demo · 3.4M 参数 CPU 推理")
    print(f"  checkpoint : {ckpt}")
    print(f"  prompt     : {prompt}")
    print("  format     : text")
    print()
    if not payload["ok"]:
        print(f"生成失败：{payload.get('reason') or payload.get('error')}")
    else:
        print(f"  生成 token 数 : {show(payload['generated_tokens'])}"
              f"（截断={payload['truncated']}）")
        print(f"  耗时          : {show(payload['elapsed_s'])} s"
              f"（加载 checkpoint 另计 {show(payload['checkpoint_load_s'])} s）")
        print(f"  tok/s         : {show(payload['tok_per_s'])}")
        print(f"  内存峰值      : {show(payload['peak_rss_mb'])} MB"
              f"（{payload['rss_source']}）")
        print(f"  light ai check: {payload['light_ai_check']['status']}"
              f" — {payload['light_ai_check']['reason']}")
        print()
        print("  生成文本：")
        print(f"    {payload['text']!r}")

sys.exit(0 if payload["ok"] else 1)
PY
