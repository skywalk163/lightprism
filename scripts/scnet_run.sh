#!/usr/bin/env bash
# =============================================================================
# LightPrism · SCNet（曙光异算平台）一键脚本
#
# 适用资源（SCNet 资源列表实测，2026-11-28）：
#   * NVIDIA GPU 资源组：L20 48GB / RTX 4090 24GB（CUDA 12.4）
#   * 曙光 DCU 资源组：异构加速卡AI 64GB（DTK 26.04）/ 异构加速卡1 16GB（DTK 25.04）
#   * CPU 资源组：X86 7185 32C（只够跑 selfcheck / 极小 smoke）
#
# 设计前提（改脚本前先读）：
#   1. SCNet 作业是**非交互批处理**：命令必须自己退出，不依赖网页终端。
#      本脚本所有子命令都写日志到 $LP_OUT/logs，rc 非零即作业失败。
#   2. DCU 的 torch 由 DTK 提供，走 HIP，但 API 仍是 torch.cuda.*，
#      设备字符串也仍写 cuda —— 不要写 hip，也不要用 nvidia-smi 判断死活。
#   3. 训练入口是 src 里的 demo 训练器（不是 train_openi.py）；作业里
#      PYTHONPATH=$ROOT/src，且必须 cd 到仓库根（configs/ 与 data/ 都是相对路径）。
#   4. 没有 checkpoint 也能跑 doctor / selfcheck；正式 train 会先做加速卡预检。
#
# 用法（仓库根目录）：
#   bash scripts/scnet_run.sh doctor      # 设备 / AMP / DCU 诊断（可贴回给助手）
#   bash scripts/scnet_run.sh selfcheck   # 纯标准库自检（秒级）
#   bash scripts/scnet_run.sh smoke       # 2 步训练 + checkpoint，验通路
#   bash scripts/scnet_run.sh train       # 正式训练（默认 demo 档）
#   bash scripts/scnet_run.sh status      # 进程 / 显卡 / 最新日志 / 产物
#   bash scripts/scnet_run.sh watch       # 跟踪最新训练日志
#   bash scripts/scnet_run.sh pack        # 打包产物，供客户端下载
#   bash scripts/scnet_run.sh all         # doctor + smoke + train 一条龙（推荐作业命令）
#
# 可调环境变量：
#   LP_ROOT        仓库根目录（默认脚本上一级）
#   LP_PYTHON      Python 解释器（默认 python3，再退 python）
#   LP_OUT         产物根目录（默认 $LP_ROOT/lightprism_out/scnet）
#   LP_LOGS        日志目录（默认 $LP_OUT/logs）
#   LP_PRESET      demo | 300m（默认 demo）
#   LP_STEPS       训练步数（默认 demo 档 200；300m 档 2000）
#   LP_DEVICE      auto | cpu | cuda | cuda:N | gpu | dcu（默认 auto）
#   LP_AMP         auto | bf16 | fp16 | off（默认 auto）
#   LP_EXTRA       追加给训练器的参数（如 "--seq-len 256 --batch-size 4"）
#   LP_BUDGET_MIN  训练时间预算（分钟，默认 0 = 不限制；作业墙钟请用平台配置）
#   LP_ALLOW_CPU=1 显式允许无加速卡时纯 CPU 跑（只验流程，数字不能当平台结论）
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LP_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT="${LP_OUT:-$ROOT/lightprism_out/scnet}"
LOGS="${LP_LOGS:-$OUT/logs}"
PRESET="${LP_PRESET:-demo}"
DEVICE="${LP_DEVICE:-auto}"
AMP="${LP_AMP:-auto}"
BUDGET_MIN="${LP_BUDGET_MIN:-0}"
EXTRA="${LP_EXTRA:-}"

resolve_python() {
    if [ -n "${LP_PYTHON:-}" ]; then echo "$LP_PYTHON"; return; fi
    local c
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then
            echo "$c"; return
        fi
    done
    echo python3
}
PY="$(resolve_python)"

mkdir -p "$OUT" "$LOGS"

ts() { date +%Y%m%d_%H%M%S; }

latest_log() { ls -1t "$LOGS"/train_*.log 2>/dev/null | head -1; }

have_timeout() { command -v timeout >/dev/null 2>&1; }

# 加速卡是否被 torch 看见。DCU 也走 torch.cuda，不要换成 nvidia-smi 判断。
accel_ok() {
    "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

show_smi() {
    if command -v hy-smi >/dev/null 2>&1; then
        echo "---- hy-smi（DCU）----"
        hy-smi 2>/dev/null | head -24
    elif command -v rocm-smi >/dev/null 2>&1; then
        echo "---- rocm-smi（DCU / ROCm）----"
        rocm-smi 2>/dev/null | head -24
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "---- nvidia-smi（NVIDIA）----"
        nvidia-smi 2>/dev/null | head -20
    else
        echo "---- 未找到 hy-smi / rocm-smi / nvidia-smi ----"
        echo "     可能是纯 CPU 资源组，或容器没挂载管理工具；以 doctor 的 torch 结果为准。"
    fi
}

require_accel() {
    if accel_ok; then return 0; fi
    if [ "${LP_ALLOW_CPU:-0}" = "1" ]; then
        echo "【警告】torch 看不到加速卡，但 LP_ALLOW_CPU=1 —— 显式允许纯 CPU 跑。"
        echo "         CPU 结果只能验流程/口径，不能当 GPU/DCU 结论贴进报告。"
        return 0
    fi
    {
        echo "【中止】torch.cuda.is_available() = False，正式训练拒绝启动。"
        echo "  排查 1 · 作业是否分到加速卡：资源组要选 GPU / DCU，而不是 X86 CPU。"
        echo "  排查 2 · torch 是否对应版本：DCU 必须是 DTK 镜像自带的 torch"
        echo "           （torch.version.hip 非空）；NVIDIA 应为 +cuXXX 轮子。"
        echo "  排查 3 · 可见设备变量：CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES 是否为空，"
        echo "           ROCR_VISIBLE_DEVICES 是否把卡屏蔽了。"
        echo "  先跑诊断：bash $0 doctor"
        echo "  确实要纯 CPU 验流程：LP_ALLOW_CPU=1 bash $0 ${1:-train}"
    } >&2
    return 3
}

# -----------------------------------------------------------------------------
# doctor —— 设备 / 后端 / AMP 方案。torch 缺失时 doctor 返回非零，这里不让它
# 带崩调用方（env 本身就是排障入口）。
# -----------------------------------------------------------------------------
cmd_doctor() {
    echo "==================== LightPrism · SCNet 环境诊断 ===================="
    echo "仓库根   : $ROOT"
    echo "解释器   : $PY（$("$PY" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)）"
    echo "作业号   : ${PJM_JOBID:-${SLURM_JOB_ID:-未检测到}}"
    echo
    show_smi
    echo
    PYTHONPATH="$ROOT/src" "$PY" -m lightprism doctor --device "$DEVICE" --amp "$AMP" || true
    echo
    echo "提示：DCU 上 torch.version.hip 非空、device 仍写 cuda 是正常的；"
    echo "      选设备/精度交给 auto，不要在脚本里硬编码 nvidia-smi 判断。"
}

# -----------------------------------------------------------------------------
# selfcheck —— 纯标准库，不需要 torch / 显卡，秒级。
# -----------------------------------------------------------------------------
cmd_selfcheck() {
    echo "---- self-check（纯标准库）----"
    ( cd "$ROOT" && PYTHONPATH="$ROOT/src" "$PY" -m lightprism tokens >/dev/null )
    local rc1=$?
    ( cd "$ROOT" && PYTHONPATH="$ROOT/src" "$PY" -m lightprism budget >/dev/null )
    local rc2=$?
    ( cd "$ROOT" && PYTHONPATH="$ROOT/src" "$PY" -m lightprism.demo.vocab_build \
        --check data/vocab_4096.json >/dev/null )
    local rc3=$?
    echo "tokens rc=$rc1  budget rc=$rc2  vocab rc=$rc3"
    [ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ] && [ "$rc3" -eq 0 ]
}

# -----------------------------------------------------------------------------
# smoke / train 的公共参数
# -----------------------------------------------------------------------------
common_args() {
    local a="--device $DEVICE --amp $AMP --out-dir $OUT"
    if [ -n "$EXTRA" ]; then a="$a $EXTRA"; fi
    echo "$a"
}

# demo 档：3.4M 参数，2 步即可验证前后向 + 保存；300m 档放大 seq/batch。
cmd_smoke() {
    local log="$LOGS/smoke_$(ts).log"
    local args="--config configs/demo-3m.json --steps 2 --seq-len 128 --batch-size 2"
    args="$args --eval-every 0 --log-every 1 $(common_args)"
    echo "---- smoke（2 步训练，验通路）----"
    echo "参数 : $args"
    echo "日志 : $log"
    ( cd "$ROOT" && PYTHONPATH="$ROOT/src" "$PY" -m lightprism.demo.train_demo $args ) \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    echo "smoke rc=$rc"
    return "$rc"
}

cmd_train() {
    require_accel train || return $?
    local log="$LOGS/train_$(ts).log"
    local steps="${LP_STEPS:-200}"
    local args
    case "$PRESET" in
        300m)
            steps="${LP_STEPS:-2000}"
            args="--config configs/300m.json --steps $steps --seq-len 512 --batch-size 4"
            ;;
        demo|*)
            args="--config configs/demo-3m.json --steps $steps --seq-len 512 --batch-size 8"
            ;;
    esac
    args="$args $(common_args)"
    echo "---- 正式训练（$PRESET）----"
    echo "参数 : $args"
    echo "日志 : $log"
    echo "预算 : ${BUDGET_MIN} 分钟（0 = 不限制；作业墙钟请在 SCNet 作业里配置）"

    local runner=""
    if have_timeout && [ "$BUDGET_MIN" -gt 0 ] 2>/dev/null; then
        runner="timeout --preserve-status -s INT ${BUDGET_MIN}m"
    fi

    if [ -n "$runner" ]; then
        ( cd "$ROOT" && PYTHONPATH="$ROOT/src" $runner "$PY" -m lightprism.demo.train_demo $args ) \
            2>&1 | tee "$log"
    else
        ( cd "$ROOT" && PYTHONPATH="$ROOT/src" "$PY" -m lightprism.demo.train_demo $args ) \
            2>&1 | tee "$log"
    fi
    local rc=${PIPESTATUS[0]}
    echo
    case "$rc" in
        0) echo "train 完成 rc=0；checkpoint 在 $OUT/model.pt" ;;
        124|130) echo "train 达到预算被停下（rc=$rc）；已完成的 checkpoint 保留。" ;;
        *) echo "train 异常退出 rc=$rc；先看日志末 40 行：tail -40 $log" ;;
    esac
    return "$rc"
}

cmd_status() {
    echo "==================== LightPrism · SCNet 作业状态 ===================="
    echo "仓库根 : $ROOT"
    echo "产物   : $OUT"
    echo "日志   : $LOGS"
    echo
    echo "---- 进程 ----"
    pgrep -af "lightprism.demo.train_demo" 2>/dev/null || echo "没有正在跑的训练进程"
    echo
    show_smi
    echo
    echo "---- 产物 ----"
    ls -lh "$OUT"/model.pt "$OUT"/train_log.json 2>/dev/null \
        || echo "还没有 model.pt / train_log.json"
    echo
    local log; log="$(latest_log)"
    if [ -n "$log" ]; then
        echo "---- 最新日志：$log（头 3 行 + 末 5 行）----"
        head -3 "$log"
        echo "…"
        tail -5 "$log"
    else
        echo "还没有训练日志"
    fi
}

cmd_watch() {
    local log; log="$(latest_log)"
    if [ -z "$log" ]; then echo "没有训练日志；先跑 bash $0 train" >&2; return 2; fi
    echo "跟踪 $log（Ctrl-C 退出，不影响作业）"
    tail -n 40 -f "$log"
}

cmd_pack() {
    local stamp; stamp="$(ts)"
    local tgz="$ROOT/lightprism_scnet_${stamp}.tar.gz"
    if [ ! -d "$OUT" ]; then echo "产物目录不存在：$OUT" >&2; return 2; fi
    tar czf "$tgz" -C "$OUT" .
    echo "已打包：$tgz"
    echo "下载后在本地解包：tar xzf $(basename "$tgz")"
}

cmd_all() {
    cmd_doctor
    echo
    if ! cmd_selfcheck; then
        echo "【失败】selfcheck 不通过，后续训练不启动。" >&2
        return 1
    fi
    echo
    if ! cmd_smoke; then
        echo "【失败】smoke 不通过：先修设备/数据，别直接上正式训练。" >&2
        return 1
    fi
    echo
    cmd_train
}

usage() {
    sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-help}" in
    doctor|env)  cmd_doctor ;;
    selfcheck)   cmd_selfcheck ;;
    smoke)       cmd_smoke ;;
    train|run)   cmd_train ;;
    status|st)   cmd_status ;;
    watch)       cmd_watch ;;
    pack)        cmd_pack ;;
    all)         cmd_all ;;
    help|-h|--help) usage ;;
    *)
        echo "未知子命令：$1" >&2
        echo "可用：doctor | selfcheck | smoke | train | status | watch | pack | all" >&2
        exit 2
        ;;
esac
