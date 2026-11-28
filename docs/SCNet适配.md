# SCNet（曙光异算平台）适配

> 平台：SCNet · 华东一区【昆山】集群（clusterId `11250`）。
> 本文的命令都在**仓库根目录**执行；设备 / 精度由 `lightprism.device` 统一判定，
> 不在脚本里硬编码 `nvidia-smi` 或 `cuda`。

## 1. 平台资源与后端对应

2026-11-28 用 SCNet 资源接口实测到的授权资源组（免费卡数会变，以客户端为准）：

| 资源组（queue） | 资源 | 类型 | 运行时 | LightPrism 后端 |
|---|---|---|---|---|
| `nvl20normal` | NVIDIA L20 48GB PCIE | GPU | CUDA 12.4 | `cuda`（bf16 可用） |
| `nv4090normalfea13c83` | NVIDIA RTX 4090 24GB | GPU | CUDA 12.4 | `cuda`（bf16 可用） |
| `hgk100_ainormal93b0dc03` | 异构加速卡AI 64GB | DCU | DTK 26.04 | `dcu`（HIP，默认 fp16） |
| `hgk100_ainormal018b690441` | 异构加速卡AI 64GB | DCU | DTK 26.04 | `dcu` |
| `hgz100lnormalc58590ec` | 异构加速卡1 16GB | DCU | DTK 25.04 | `dcu` |
| `hg7185normal227b7c31` | X86 7185 32C 128GB | CPU | — | `cpu`（只够自检） |

关键事实：

- **DCU 不是「另一种 torch API」**。DTK 版 PyTorch 走 HIP/ROCm，但对外仍是
  `torch.cuda.*`，`torch.cuda.is_available()` 为 `True`，`torch.device("cuda:0")`
  也照用。判断是不是海光卡看 **`torch.version.hip` 非空**，不要写 `hip` 设备类型。
- **DCU 上没有 `nvidia-smi`**。管理工具通常是 `hy-smi`（部分镜像为 `rocm-smi`）。
  脚本里用 `torch.cuda.is_available()` 做唯一硬判据，SMI 只用于人读展示。
- **AMP 不能照抄 NVIDIA 的经验**：V100（cc7.0）的 `is_bf16_supported()` 可能返回
  `True`，但没有 bf16 Tensor Core；DCU 的 bf16 支持随型号 / DTK 版本变化。

## 2. 上手步骤

```bash
# 1) 诊断：torch 版本、HIP/CUDA、卡数、显存、最终 device/amp 方案
bash scripts/scnet_run.sh doctor

# 2) 纯标准库自检（不需要 torch / 显卡，秒级）
bash scripts/scnet_run.sh selfcheck

# 3) 2 步训练，验证前向 / 反向 / checkpoint 落盘
bash scripts/scnet_run.sh smoke

# 4) 正式训练（默认 demo 档 200 步；产物在 lightprism_out/scnet/）
bash scripts/scnet_run.sh train

# 5) 打包产物，供客户端下载
bash scripts/scnet_run.sh pack
```

`all` = `doctor` + `selfcheck` + `smoke` + `train`，推荐直接作为作业命令。

常用覆盖：

```bash
LP_PRESET=300m LP_STEPS=2000 bash scripts/scnet_run.sh train
LP_DEVICE=cpu LP_ALLOW_CPU=1 bash scripts/scnet_run.sh smoke   # 只验流程
LP_AMP=fp16 bash scripts/scnet_run.sh train                    # 强制 fp16
LP_DEVICE=cuda:1 bash scripts/scnet_run.sh train                # 多卡机指定第二张
```

跑完可以直接用 CLI 复生成，不用记得 checkpoint 路径：

```bash
PYTHONPATH=src python -m lightprism doctor --json
PYTHONPATH=src python -m lightprism chat --prompt "把这三个文件按大小排序" \
    --ckpt lightprism_out/scnet/model.pt
```

## 3. 提交 SCNet 作业

用 SCNet 客户端的脚本作业（`scnet_script_job_prepare` 对应的表单）填：

| 字段 | 建议值 |
|---|---|
| `jobName` | `lightprism-demo` / `lightprism-300m` |
| `clusterId` | `11250` |
| `queue` | 资源组名，如 `nvl20normal`、`hgk100_ainormal93b0dc03`、`hgz100lnormalc58590ec` |
| `accelerator` | NVIDIA 资源组选 `{"type":"GPU","count":1}`；DCU 资源组选 `{"type":"DCU","count":1}` |
| `nprocs` | 8（demo）/ 16—32（300m，按卡的 CPU 配比） |
| `memoryMb` | 32768（demo）/ 131072（300m，DCU 64GB 卡可再放） |
| `wallTime` | `00:30:00`（demo）/ `04:00:00`（300m 起） |
| `storagePath` | 仓库所在的 SCNet 存储目录（脚本的 `LP_OUT` 默认落在其下） |
| `command` | `cd <仓库目录> && bash scripts/scnet_run.sh all` |

要点：

1. **命令必须自己结束**。不要用 `tail -f`、交互式 shell 或等待 stdin 的写法；
   `scnet_run.sh` 每个子命令都会退出并返回 rc。
2. **工作目录要在仓库根**：`configs/demo-3m.json`、`data/` 都是相对路径。
   脚本内部会 `cd "$ROOT"`，`LP_ROOT` 默认取脚本上一级。
3. **产物写到存储目录**：默认 `lightprism_out/scnet/`，含 `model.pt`、
   `train_log.json`、`logs/*.log`；`pack` 生成
   `lightprism_scnet_<时间戳>.tar.gz` 放在仓库根，便于下载。
4. **不要把 `queue` 写死成示例值**：以客户端资源组列表里当前可见、`maxFreeNum > 0`
   的资源组为准。接口返回的 `resourceGroupName` 就是 queue 名。
5. 多卡要真数据并行目前还不在本仓能力内（见 `docs/训练与推理.md` §6），
   `count > 1` 时只有第一张卡会被使用。

## 4. 设备与 AMP 判定规则

`lightprism.device` 的规则（`tests/test_scnet_device.py` 全覆盖，不需要真卡）：

| 后端 | `device` 写法 | `amp=auto` | 说明 |
|---|---|---|---|
| NVIDIA cc ≥ 8.0（L20 / 4090 / A100） | `cuda:0` | `bf16` | 无 GradScaler |
| NVIDIA cc < 8.0（V100 / T4） | `cuda:0` | `fp16` + GradScaler | 即使 `is_bf16_supported()` 为真也不用 bf16 |
| 海光 DCU（HIP） | `cuda:0` | `fp16` + GradScaler | bf16 需显式 `--amp bf16`，且标记为实验路径 |
| 其他 AMD ROCm | `cuda:0` | `fp16` + GradScaler | 设备名不含 DCU 标识时按通用 ROCm |
| CPU | `cpu` | `off` | 半精度直接报错，不静默降级 |

CLI 行为：

- `--device auto`（默认）：有加速卡用 `cuda:0`，否则 CPU；
- `--device gpu` / `dcu`：要求确实有加速卡 / HIP 后端，否则**报错退出**，不偷偷回落；
- `--device cuda:N`：越界直接报错；
- `--amp bf16` 在 V100 或 `is_bf16_supported()=False` 时**报错退出**，避免「软件模拟 bf16」把速度拖死；
- `--amp fp16` 自动启用 `GradScaler`；`--amp off` 走 fp32。

## 5. 与 OpenI 路径的差异

| | OpenI（`train_openi.py`） | SCNet（本仓 `src` + `scnet_run.sh`） |
|---|---|---|
| 训练入口 | 单文件自包含脚本 | `lightprism.demo.train_demo`（src 包） |
| 设备判定 | 脚本内 `--device/--amp` | 统一 `lightprism.device`，两个训练器共用 |
| DCU 支持 | 无（按 CUDA 处理） | 识别 HIP，默认 fp16，bf16 显式实验 |
| 结果目录 | `lightprism_out/` | `lightprism_out/scnet/`（可 `LP_OUT` 覆盖） |
| 断点续训 | 不支持 | 仍不支持（同 `docs/训练与推理.md` §6 第 5 条） |

## 6. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `torch.cuda.is_available()=False`，但作业在 DCU 资源组 | 用了 PyPI 官方 CUDA 轮子，覆盖了 DTK 的 torch | 不要在 DCU 镜像里 `pip install torch`；用镜像自带版本，确认 `torch.version.hip` 非空 |
| `doctor` 显示 `backend=cpu`，`nvidia-smi` 有卡 | `CUDA_VISIBLE_DEVICES` 为空串 / 作业选到 CPU 资源组 | 取消可见设备变量，或重建作业选 GPU/DCU 资源组 |
| `HIP error: invalid device function` / `no kernel image` | 二进制与 DCU 架构不匹配 | 用 DTK 镜像自带 torch；不要混装不同 DTK 版本的 wheel |
| `CUDA out of memory`（DCU 同样报 CUDA OOM） | batch × seq 太大 | 降 `--batch-size` / `--seq-len`，或换 64GB DCU 资源组 |
| 首步特别慢 | DCU MIOpen 首次算子编译 / CUDA context 初始化 | 正常；把 warmup 步数算进预算，不要用第一步计时 |
| `bf16` 跑出 NaN | DCU 型号或 DTK 版本 bf16 不完整 | 改 `--amp fp16`（默认 auto 已是 fp16） |
| 训练进程被作业系统杀掉 | 超过 `wallTime` | 提高墙钟或降 `LP_STEPS`；当前不支持断点续训，重跑会从头开始 |
| `smoke` 通过但 `train` 失败 | 数据量 / 显存 / 步数配置不同 | 看 `logs/train_*.log` 末 40 行；`smoke` 只证明通路 |

## 7. 已验证与未验证（诚实边界）

**已在本机验证（无 GPU / DCU）**：

- `lightprism.device` 的后端识别与 AMP 判定：15 项单测覆盖 NVIDIA L20 / V100 /
  DCU / 通用 ROCm / CPU / 无 torch；
- `lightprism doctor` 在无 torch 环境下输出可解析 JSON 并以非零退出；
- 训练入口的 `--device` / `--amp` 与推理入口的 `--device` 参数接线、checkpoint
  `runtime` 字段；CPU 上真实跑通 2 步训练；
- 全量 pytest：无 torch 控制节点 **231 passed / 15 skipped / 2 failed**，
  装入 CPU 版 torch 后 **320 passed / 6 skipped / 2 failed**（两项失败为改造前既有，
  与设备层无关）。

**尚未在 SCNet 真机验证**：

- DCU（DTK 25.04 / 26.04）上的实际前向 / 反向、bf16 可用性、MIOpen 首次编译耗时；
- L20 / 4090 上的 bf16 吞吐与显存占用；
- 作业系统对 `wallTime`、`PBS`/Slurm 环境变量的具体命名（脚本只读常见的
  `PJM_JOBID` / `SLURM_JOB_ID`，读不到也不影响运行）。

真机第一件事是 `bash scripts/scnet_run.sh doctor`，把输出贴回即可确认后端与 AMP
选择是否符合预期；再跑 `smoke`，最后才上 `train`。在拿到真机输出前，不要引用
本文档给出任何「DCU 上 tok/s / loss」数字。
