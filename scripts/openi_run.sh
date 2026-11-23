#!/usr/bin/env bash
# =============================================================================
# LightPrism · OpenI 调试任务 一键实验脚本
#
# 设计前提（改脚本前先读）：
#   1. OpenI 只给网页终端，没有外部 SSH —— 所以本脚本必须在 JupyterLab 的
#      Terminal 里执行，不要在 notebook cell 里跑（断了不会自动重连）。
#   2. 调试任务单次最长 4 小时。默认预算 200 分钟，留 40 分钟给下载产物。
#   3. train_openi.py 自包含：词表 + 371 条 SFT 样本以 zlib+base64 内嵌，
#      不需要上传 data/，也不需要 pip install 一堆依赖（只要有 torch）。
#   4. 本脚本不做断点续训 —— train_openi.py 只存 best 权重、不存优化器状态
#      （docs/训练与推理.md §6 第 5 条）。所以策略是「预算内跑完一档，
#      超时前由 timeout 收尸，已完成的组产物保留」。
#   5. **CUDA 是硬前提**：run/bg 启动前强制预检，torch 看不到 GPU 就中止，
#      不再静默回落 --device cpu（2026-11-23 那次就是「显存 0、进度不动」被
#      当成没跑起来，实际是 CPU 在啃）。确实要纯 CPU 跑必须显式 LP_ALLOW_CPU=1。
#
# 用法（在 OpenI 终端里、仓库根目录执行；本脚本位于 scripts/）:
#   bash scripts/openi_run.sh env          # 环境体检
#   bash scripts/openi_run.sh selfcheck    # 纯标准库自检（秒级）
#   bash scripts/openi_run.sh smoke        # GPU 冒烟（2 步前向/反向）
#   bash scripts/openi_run.sh run          # 正式实验（前台，带超时收尸）
#   bash scripts/openi_run.sh bg           # 同上，但后台跑，安全断线
#   bash scripts/openi_run.sh status       # 一条命令看全：进程/显存/日志/产物
#   bash scripts/openi_run.sh watch        # 盯日志（可反复执行）
#   bash scripts/openi_run.sh sum          # 打印 verdict 摘要（贴回给助手）
#   bash scripts/openi_run.sh pack         # 打包产物，供网页下载
#   bash scripts/openi_run.sh all          # env+selfcheck+smoke+run 一条龙
#
# 可调环境变量:
#   LP_ROOT     仓库根目录（默认当前目录；日志与产物都落在这里）
#   LP_PY       训练脚本路径（默认自动找 ./train_openi.py 或 scripts/../train_openi.py）
#   LP_OUT      产物目录（默认 $LP_ROOT/lightprism_out）
#   LP_LOGS     日志目录（默认 $LP_ROOT/logs）
#   LP_BUDGET_MIN  前台/后台运行的时间预算（分钟，默认 200）
#   LP_PRESET   demo|small|300m（默认 demo）
#   LP_EXTRA    追加给 train_openi.py 的参数（如 "--max-new-tokens 512"）
#   LP_C2NET=1  训练任务场景：追加 --c2net --c2net-upload
#   LP_AMP      auto|fp16|bf16|off（默认 auto；Volta 请用 fp16，bf16 不可用）
#   LP_ALLOW_CPU=1  显式允许无 GPU 时纯 CPU 跑（数字不能当 V100 结论）
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LP_ROOT:-$(pwd)}"

# 训练脚本位置：LP_PY > 当前目录 > 脚本目录 > 上一级（scripts/ 布局的兜底）
resolve_py() {
    if [ -n "${LP_PY:-}" ]; then echo "$LP_PY"; return; fi
    local c
    for c in "$ROOT/train_openi.py" "$SCRIPT_DIR/train_openi.py" "$SCRIPT_DIR/../train_openi.py"; do
        if [ -f "$c" ]; then echo "$c"; return; fi
    done
    echo "$ROOT/train_openi.py"
}

PY_FILE="$(resolve_py)"
OUT_DIR="${LP_OUT:-$ROOT/lightprism_out}"
LOG_DIR="${LP_LOGS:-$ROOT/logs}"
BUDGET_MIN="${LP_BUDGET_MIN:-200}"
PRESET="${LP_PRESET:-demo}"
AMP="${LP_AMP:-auto}"

# Python 解释器：$PYTHON > 能用的 python3 > 能用的 python > python3
# （Windows 上 PATH 里的 python3 常是 Microsoft Store 的占位 stub，一调用就 rc=49；
#  没有这段兜底，`bash scripts/openi_run.sh sum` 在 Windows 上会假失败）
resolve_py_interp() {
    if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
    local c
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then
            echo "$c"; return
        fi
    done
    echo python3
}
PY="$(resolve_py_interp)"

mkdir -p "$LOG_DIR"

ts() { date +%Y%m%d_%H%M%S; }

latest_log() { ls -1t "$LOG_DIR"/run_*.log 2>/dev/null | head -1; }

have_timeout() { command -v timeout >/dev/null 2>&1; }

# torch 是否真的看得到 CUDA。注意：只看 torch.cuda.is_available()，
# 不用 nvidia-smi 单独判断（镜像有驱动但 torch 是 cpu 版时两者会打架）。
cuda_ok() {
    "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

# 强制 CUDA 预检：不可用则中止并给三条排查提示；LP_ALLOW_CPU=1 才放行。
require_cuda() {
    if cuda_ok; then return 0; fi
    if [ "${LP_ALLOW_CPU:-0}" = "1" ]; then
        echo "【警告】CUDA 不可用，但 LP_ALLOW_CPU=1 —— 显式允许纯 CPU 跑。"
        echo "         CPU 结果只能验流程/口径，不能当 V100 结论贴进报告。"
        return 0
    fi
    {
        echo "【中止】CUDA 不可用，正式实验拒绝启动（不再静默回落 --device cpu）。"
        echo "  排查 1 · torch 是否 CPU 版：$PY -c 'import torch;print(torch.__version__)'"
        echo "           应看到 +cuXXX 后缀；只有 2.x.x+cpu 说明镜像装的是 CPU 包。"
        echo "  排查 2 · 任务是否分到 GPU：nvidia-smi 能列出显卡；torch 里 False 多为"
        echo "           选成 CPU 集群 / 云脑二 NPU 集群，重建任务并选「云脑一 GPU」。"
        echo "  排查 3 · 镜像是否带 CUDA 运行时：换 ubuntu22.04-cudaXX-pyXX-torchX 镜像"
        echo "           重建任务；不要在错误镜像上 pip install 另一个 torch 来「修复」CUDA。"
        echo "  确实要纯 CPU 跑（只验流程，不看数字）：LP_ALLOW_CPU=1 bash $0 ${1:-bg}"
    } >&2
    return 3
}

# -----------------------------------------------------------------------------
# env —— 环境体检。任何一项不对，先解决它再往下跑。
# -----------------------------------------------------------------------------
cmd_env() {
    echo "==================== LightPrism · OpenI 环境体检 ===================="
    echo "当前目录   : $(pwd)"
    echo "仓库根目录 : $ROOT"
    echo "脚本目录   : $SCRIPT_DIR"
    echo "Python     : $PY -> $($PY -V 2>&1)"
    echo
    if [ -f "$PY_FILE" ]; then
        echo "训练脚本   : $PY_FILE (OK, $(wc -c < "$PY_FILE") 字节)"
    else
        echo "训练脚本   : $PY_FILE 【缺失】—— 请在仓库根目录执行，或用 LP_PY 指定路径"
    fi
    echo

    echo "---- torch / CUDA ----"
    "$PY" - <<'PYEOF'
import os, sys, platform
print(f"platform : {platform.platform()}")
print(f"python   : {sys.version.split()[0]}")
try:
    import torch
except Exception as e:
    print(f"torch    : 【缺失】{e}")
    print("           -> 该镜像没装 torch。换带 torch 的镜像（如 ubuntu22.04-cuda11.8.0-py310-torch2.1.0），")
    print("              或先执行: pip install torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple")
    raise SystemExit(0)
print(f"torch    : {torch.__version__}")
print(f"cuda可用 : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu[{i}] : {p.name}  {p.total_memory/1024**3:.1f}GiB  cc={p.major}.{p.minor}")
        if p.major < 8:
            print(f"           -> cc={p.major}.{p.minor}（Volta 及更早）无 bf16 tensorcore，"
                  f"即使 is_bf16_supported() 返回 True 也不要用 --amp bf16，用 fp16")
else:
    print("  -> 无 GPU。能跑 smoke/selfcheck，正式实验会极慢；run/bg 会拒绝启动。")
print(f"bf16支持 : {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'n/a'}"
      "（注意：Ampere+ 才有 bf16 tensorcore，Volta 上这个 True 是假绿灯）")
PYEOF
    echo

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "---- nvidia-smi ----"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version \
                   --format=csv,noheader 2>/dev/null || nvidia-smi 2>&1 | head -15
        echo
    fi

    echo "---- 磁盘 / 内存 ----"
    df -h . 2>/dev/null | tail -2
    if command -v free >/dev/null 2>&1; then free -g | head -2; fi
    echo

    echo "---- 写权限 ----"
    if mkdir -p "$OUT_DIR" 2>/dev/null && [ -w "$OUT_DIR" ]; then
        echo "产物目录   : $OUT_DIR (可写)"
    else
        echo "产物目录   : $OUT_DIR 【不可写】—— 换到工作目录下，或设 LP_OUT"
    fi
    echo
    echo "---- CUDA 预检 ----"
    if cuda_ok; then echo "CUDA 预检 : 通过（run/bg 可直接启动）"; else echo "CUDA 预检 : 不通过（run/bg 会拒绝启动；要硬跑 CPU 请 LP_ALLOW_CPU=1）"; fi
    echo "==================================================================="
    echo "提示：调试任务 4 小时上限；产物请落在工作目录（如 /home/ma-user/work）再下载。"
}

# -----------------------------------------------------------------------------
# selfcheck —— 纯标准库自检，不需要 torch，秒级。先过这一关再谈 GPU。
# -----------------------------------------------------------------------------
cmd_selfcheck() {
    echo "---- self-check（纯标准库，不需要 GPU）----"
    "$PY" "$PY_FILE" --self-check
    local rc=$?
    echo "self-check rc=$rc"
    return $rc
}

# 设备参数：CUDA 可用 → cuda + AMP；LP_ALLOW_CPU=1 且无 GPU → cpu/off。
device_args() {
    if cuda_ok; then
        echo "--device cuda --amp $AMP"
    else
        echo "--device cpu --amp off"
    fi
}

# -----------------------------------------------------------------------------
# smoke —— 2 步前向/反向 + 少量生成。目的不是看效果，是确认环境能跑。
# 无 GPU 时给醒目告警，但仍允许跑（它本身就是用来暴露「假 GPU」的）。
# -----------------------------------------------------------------------------
cmd_smoke() {
    local log="$LOG_DIR/smoke_$(ts).log"
    if ! cuda_ok; then
        echo "【警告】CUDA 不可用：smoke 只会跑 CPU，不能证明 GPU 通路正常。"
        if [ "${LP_ALLOW_CPU:-0}" != "1" ]; then
            echo "         正式 run/bg 仍会拒绝启动；要连 smoke 一起跑 CPU 请 LP_ALLOW_CPU=1。"
        fi
    fi
    echo "---- smoke（最小步数验环境）---- 日志: $log"
    # shellcheck disable=SC2046
    "$PY" "$PY_FILE" --smoke $(device_args) 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    echo "smoke rc=$rc"
    return $rc
}

# 组装正式实验参数。300m 在 16G V100 上必须降 batch/seq，见 PRESETS 注释。
train_args() {
    local a="--preset $PRESET --out-dir $OUT_DIR"
    case "$PRESET" in
        300m) a="$a --batch-size 4 --seq-len 512" ;;
    esac
    a="$a $(device_args)"
    if [ "${LP_C2NET:-0}" = "1" ]; then a="$a --c2net --c2net-upload"; fi
    if [ -n "${LP_EXTRA:-}" ]; then a="$a $LP_EXTRA"; fi
    echo "$a"
}

# -----------------------------------------------------------------------------
# run —— 前台跑，带时间预算。超时由 timeout 收尸（退出码 124），
#        已完成组的 checkpoints 与 train_log.jsonl 会保留。
# -----------------------------------------------------------------------------
cmd_run() {
    require_cuda run || return $?
    local log="$LOG_DIR/run_$(ts).log"
    local args; args="$(train_args)"
    echo "---- 正式实验 ----"
    echo "参数   : $args"
    echo "日志   : $log"
    echo "预算   : ${BUDGET_MIN} 分钟（调试任务 4h 上限，超时会自动停）"
    echo

    local runner
    if have_timeout; then
        runner="timeout --preserve-status -s INT ${BUDGET_MIN}m"
    else
        echo "【警告】没有 timeout 命令，时间预算不生效；请自己盯时钟，剩 40 分钟时 Ctrl-C。"
        runner=""
    fi

    # shellcheck disable=SC2086
    A=$args
    if [ -n "$runner" ]; then
        $runner "$PY" "$PY_FILE" $A 2>&1 | tee "$log"
    else
        "$PY" "$PY_FILE" $A 2>&1 | tee "$log"
    fi
    local rc=${PIPESTATUS[0]}
    echo
    case "$rc" in
        0)   echo "run 完成 rc=0" ;;
        124|130) echo "run 达到 ${BUDGET_MIN} 分钟预算被停下（rc=$rc）。产物已保留，先 sum 再看要不要续。" ;;
        *)   echo "run 异常退出 rc=$rc，先看日志末 40 行：tail -40 $log" ;;
    esac
    return 0
}

# -----------------------------------------------------------------------------
# bg —— 后台跑。网页终端断开也不会停（进程与 terminal 解绑）。
# -----------------------------------------------------------------------------
cmd_bg() {
    require_cuda bg || return $?
    local stamp; stamp=$(ts)
    local log="$LOG_DIR/run_${stamp}.log"
    local pidfile="$LOG_DIR/run_${stamp}.pid"
    local args; args="$(train_args)"
    echo "---- 后台启动 ----"
    echo "参数   : $args"
    echo "日志   : $log"
    echo "预算   : ${BUDGET_MIN} 分钟"

    if have_timeout; then
        nohup timeout --preserve-status -s INT "${BUDGET_MIN}m" \
            "$PY" "$PY_FILE" $args >>"$log" 2>&1 &
    else
        nohup "$PY" "$PY_FILE" $args >>"$log" 2>&1 &
    fi
    echo $! > "$pidfile"
    echo "PID    : $(cat "$pidfile")"
    echo
    echo "下一步："
    echo "  bash $0 status     # 进程/显存/日志/产物 一条命令看全"
    echo "  bash $0 watch      # 盯日志"
    echo "  kill \$(cat $pidfile)   # 手动停"
}

# -----------------------------------------------------------------------------
# status —— 一条命令回答「到底在跑没有」：
#   进程存活、nvidia-smi 显存、最新日志头 3 行 + 末 5 行、metrics.json 是否落盘。
# -----------------------------------------------------------------------------
cmd_status() {
    echo "==================== LightPrism · 运行状态 ===================="
    echo "仓库根目录 : $ROOT"
    echo "产物目录   : $OUT_DIR"
    echo "日志目录   : $LOG_DIR"
    echo
    echo "---- 训练进程 ----"
    if command -v pgrep >/dev/null 2>&1; then
        local hits; hits="$(pgrep -af train_openi.py 2>/dev/null)"
        if [ -n "$hits" ]; then echo "$hits"; else echo "无 train_openi.py 进程在跑"; fi
    else
        local hits; hits="$(ps -ef 2>/dev/null | grep '[t]rain_openi.py')"
        if [ -n "$hits" ]; then echo "$hits"; else echo "无 train_openi.py 进程在跑（无 pgrep，已退回 ps | grep）"; fi
    fi
    echo
    echo "---- GPU 显存 ----"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
                   --format=csv,noheader 2>/dev/null \
            || echo "nvidia-smi 调用失败"
        echo "（memory.used 为 0 且进程在跑 → 多半是 CPU 在算，不是 GPU）"
    else
        echo "无 nvidia-smi（CPU 环境或镜像缺驱动）"
    fi
    echo
    echo "---- 最新日志 ----"
    local log; log="$(latest_log)"
    if [ -n "$log" ]; then
        echo "文件 : $log ($(wc -l < "$log") 行)"
        echo "头 3 行:"
        head -3 "$log"
        echo "末 5 行:"
        tail -5 "$log"
    else
        echo "还没有 run 日志（$LOG_DIR/run_*.log）。先执行：bash $0 bg"
    fi
    echo
    echo "---- 产物 ----"
    if [ -f "$OUT_DIR/metrics.json" ]; then
        echo "metrics.json : 已生成（$(wc -c < "$OUT_DIR/metrics.json") 字节）→ 跑 bash $0 sum"
    else
        echo "metrics.json : 未生成（只有跑完才写；中途被打断只会留下 train_log.jsonl / checkpoints）"
    fi
    if [ -d "$OUT_DIR/checkpoints" ]; then echo "checkpoints  : $(ls -1 "$OUT_DIR/checkpoints" 2>/dev/null | wc -l) 个文件"; fi
    echo "=============================================================="
}

# -----------------------------------------------------------------------------
# watch —— 盯最新一次运行。断开网页就重新执行它。
# -----------------------------------------------------------------------------
cmd_watch() {
    local log; log="$(latest_log)"
    if [ -z "$log" ]; then
        echo "还没有 run 日志（$LOG_DIR/run_*.log）。先执行：bash $0 bg"
        return 1
    fi
    echo "盯日志: $log  （Ctrl-C 退出不影响后台进程）"
    tail -n 60 -f "$log"
}

# -----------------------------------------------------------------------------
# sum —— 只打印该贴回来的部分：verdict / 判据 / 生成口径（含截断分母）/ 速度。
# -----------------------------------------------------------------------------
cmd_sum() {
    local m="$OUT_DIR/metrics.json"
    echo "---- 摘要 ----"
    if [ ! -f "$m" ]; then
        echo "没有 $m —— 实验可能还没跑完，或被超时打断在中途。"
        echo "先看：bash $0 status；已完成的部分在 $OUT_DIR/train_log.jsonl 与 checkpoints/"
        return 1
    fi
    "$PY" - "$m" <<'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
v = d.get("verdict") or {}
print("verdict       :", v.get("verdict") if isinstance(v, dict) else d.get("verdict"))
print("evidence_level:", v.get("evidence_level") if isinstance(v, dict) else None)
print("gated         :", v.get("gated") if isinstance(v, dict) else None)
for c in (v.get("checks") or []):
    print(f"  - {c.get('criterion')}: {c.get('result')}  {str(c.get('detail',''))[:140]}")
print()
for tag in ("C", "B"):
    g = (d.get(tag) or {}).get("gen") or {}
    fr = g.get("free") or {}
    js = g.get("json") or {}
    if not fr:
        continue
    print(f"[{tag}] 生成口径: n_all={fr.get('n_all')} n_counted={fr.get('n_counted')} "
          f"n_truncated={fr.get('n_truncated')} "
          f"structure_rate_all={fr.get('structure_rate_all')} "
          f"structure_rate(非截断)={fr.get('structure_rate')} "
          f"tok/s={fr.get('tokens_per_s')}")
    if js:
        print(f"     JSON 代理: n={js.get('n')} valid={js.get('valid_rate')} "
              f"exact={js.get('exact')} ({js.get('exact_rate')})")
    print(f"     决策密度: {g.get('decision_density')}")
if isinstance(v, dict) and v.get("gate_blockers"):
    print("gate_blockers :", "；".join(v["gate_blockers"]))
print("速度下降(C vs B):", ((d.get("C") or {}).get("gen") or {}).get("speed_drop_vs_B"))
print("\n报告全文:", p.with_name("report.md"))
PYEOF
    echo
    echo "---- report.md 末尾 ----"
    tail -n 40 "$OUT_DIR/report.md" 2>/dev/null
    return 0
}

# -----------------------------------------------------------------------------
# pack —— 打包，供 JupyterLab 文件树右键下载。
# -----------------------------------------------------------------------------
cmd_pack() {
    local name="lightprism_openi_$(ts).tar.gz"
    if [ ! -d "$OUT_DIR" ]; then
        echo "没有产物目录 $OUT_DIR"
        return 1
    fi
    tar czf "$ROOT/$name" \
        -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")" \
        && echo "已打包: $ROOT/$name  ($(du -h "$ROOT/$name" | cut -f1))" \
        && echo "在左侧文件树里右键该文件 → Download 即可拿到本地。"
}

case "${1:-help}" in
    env)       cmd_env ;;
    selfcheck) cmd_selfcheck ;;
    smoke)     cmd_smoke ;;
    run)       cmd_run ;;
    bg)        cmd_bg ;;
    status)    cmd_status ;;
    watch)     cmd_watch ;;
    sum)       cmd_sum ;;
    pack)      cmd_pack ;;
    all)       cmd_env && cmd_selfcheck && cmd_smoke && cmd_run ;;
    *)         sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
