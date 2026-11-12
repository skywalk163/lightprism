# Demo 使用说明（T-0335）

> 本文只写**当前仓库里真能跑的东西**和**实测过的数字**。
> 没跑出来的输出一律标「待 T-0332 完成后补」，不贴示例、不填推测值。
> 计划事实源是 [`Demo开发计划.md`](Demo开发计划.md)（本卡不改它）。

## 0. 先说清楚这是什么

**这是 3.4M 参数的 demo 模型，不是 README 顶部表格里的 1B 版本。**
demo 档 `configs/demo-3m.json` 是 `d=192 / L=6 / H=6 / V=4096 / max_seq=512`，
参数量 3,443,520（非嵌入 2,657,088），见 [`../reports/推理性能基线.md`](../reports/推理性能基线.md) §1。
它用 4096 词表，不是方案书里的 64K 词表；理由见计划 §3。

模型权重（`checkpoints/demo-3m/model.pt`）由 T-0332 产出，被 `.gitignore` 的
`checkpoints/` 与 `*.pt` 挡住，**不入库**。所以从干净 clone 跑 `make demo`，
当前拿到的是一条可读提示，不是生成结果 —— 这是设计好的状态，不是失败。

## 1. 环境要求

| 项 | 要求 | 依据 / 实测 |
|---|---|---|
| Python | **≥ 3.10** | `pyproject.toml` 的 `requires-python`；实测 Windows Python 3.14.7 |
| GPU | **不需要** | 3.44M 参数、CPU 上贪心 decode；`torch` CPU 版即可 |
| 依赖 | 门面 / 词表 / 预算**只需标准库**；训练与生成需 `pip install -e ".[train]"`（`torch>=2.2`、`numpy>=1.26`） | `pyproject.toml` |
| 内存 | 峰值 RSS 实测 **283.0–283.1 MB**；计划上限 ≤ 8 GB | `reports/推理性能基线.md` §4.1；计划 §二 V5 |
| 平台 | `make bench` **需要 POSIX**（Linux / FreeBSD）：`scripts/bench.py` 顶层 `import resource`，Windows 标准库没有该模块 | 本机实测 `ModuleNotFoundError: No module named 'resource'` |
| 工具 | `make`（跑 Makefile）、`bash`（跑 `scripts/demo.sh`）、可选 `light` CLI（`light ai check`） | 本机实测：**`make` 与 `light` 均不在 PATH** |

Windows 上 `make` 不在 PATH 时，可以直接跑 Makefile 目标背后的命令，见 §3.3。
所有 Python 命令都显式带 `PYTHONPATH=src`，不要求先 `pip install -e .`。

## 2. 五分钟：三条命令

```bash
make check     # pytest + 4096 词表校验 + 64K 预算核算（不需要 checkpoint）
make demo      # 一条命令跑通（无 checkpoint 时给可读提示）
make bench     # tok/s 与峰值内存，POSIX 平台
```

`make check` 的三条底层命令在本机（Windows / Python 3.14.7）**实测全绿**：

```console
$ PYTHONPATH=src python -m pytest -q
........................................................................ [ 57%]
......................................................                   [100%]
# 退出码 0（126 项）

$ PYTHONPATH=src python -m lightprism.demo.vocab_build --check data/vocab_4096.json
[ OK ] UNK 率 < 1% — 0.1997% (3814/1909931 字符，口径：513 个唯一 md5 语料文件的字符频次，保守单字覆盖上界)
结论：全绿

$ PYTHONPATH=src python -m lightprism budget
  合计                65536  余量 +0
预算检查：通过
```

## 3. 三种跑法

| 跑法 | 当前状态 | 说明 |
|---|---|---|
| `make demo` | ✅ 可用 | 需要 `make`；内部转调 `scripts/demo.sh` |
| `python -m lightprism.demo` | ⚠️ **入口未落地** | 见 §3.2，需另开卡 |
| `scripts/demo.sh` | ✅ 可用 | 本卡交付，`text` / `json` 双格式 |

### 3.1 `make demo`

```bash
make demo
make demo PROMPT="报告磁盘使用率"
make demo FORMAT=json PROMPT="把这三个文件按大小排序"
PYTHON=C:/Python314/python.exe make demo     # Windows 指定解释器
```

`make` 变量：`PROMPT`（默认「把这三个文件按大小排序」）、`FORMAT`（`text` / `json`）、
`CKPT`（默认 `checkpoints/demo-3m/model.pt`）、`PYTHON`、`BENCH_TOKENS`、`TRAIN_STEPS`。

**没有 checkpoint 时的真实输出**（本机实跑，rc=0；提示语逐字来自脚本）：

```console
$ make demo
LightPrism demo · 3.4M 参数 CPU 推理（T-0335 门面）
  checkpoint : checkpoints/demo-3m/model.pt
  prompt     : 把这三个文件按大小排序
  format     : text

未找到 checkpoints/demo-3m/model.pt，请先跑 make train 或从 releases 下载

本次未执行生成（缺 checkpoint）：
  生成 token 数 : —
  耗时          : —
  tok/s         : —
  内存峰值      : —
  light ai check: 未执行（缺 checkpoint）
```

> **为什么这里退出码是 0**：缺权重是**预期状态**，不是执行错误，所以门面打印提示后正常退出；
> 机器判据在 JSON 里是 `ok=false / error=checkpoint_missing`，不会假装生成成功。
> 一旦 checkpoint 存在，生成失败会**非零退出**，脚本不吞异常。

**带 checkpoint 的生成输出样例：待 T-0332 完成后补**（本卡没有权重，不编造输出）。

### 3.2 `python -m lightprism.demo` ⚠️ 当前不可用

该跑法要求 `src/lightprism/demo/__main__.py`，而该文件**当前不存在**；`src/` 不在本卡
允许改动路径（`Makefile` / `README.md` / `docs/Demo使用.md` / `scripts/demo.sh`）内，
因此本卡未新增入口。本机实跑的真实报错：

```console
$ PYTHONPATH=src python -m lightprism.demo
C:\Python314\python.exe: No module named lightprism.demo.__main__; 'lightprism.demo' is a package and cannot be directly executed
```

**当前可用的等价入口**（已交付、T-0332 验证过）：

```bash
# 冒烟：加载 checkpoint，贪心生成 32 token，输出 JSON
PYTHONPATH=src python -m lightprism.demo.smoke --ckpt checkpoints/demo-3m/model.pt
# 词表校验
PYTHONPATH=src python -m lightprism.demo.vocab_build --check data/vocab_4096.json
# 训练（产 checkpoint）
PYTHONPATH=src python -m lightprism.demo.train_demo --config configs/demo-3m.json --steps 50
```

补 `__main__.py` 需另开卡，见 §7「待补」。

### 3.3 `scripts/demo.sh`

```bash
./scripts/demo.sh "把这三个文件按大小排序"
./scripts/demo.sh --format json "报告磁盘使用率"
```

| 选项 | 作用 |
|---|---|
| `--format text\|json` | 人读格式 / 机器读 JSON；`--json` 是 `--format json` 的别名 |
| `--ckpt PATH` | 换 checkpoint，默认 `checkpoints/demo-3m/model.pt` |
| `--max-new-tokens N` | 生成 token 数，默认 32 |
| `--prompt STR` | 也支持把 prompt 直接作为位置参数 |
| `-h` / `--help` | 用法 |
| `PYTHON=/path/to/python` | 环境变量指定解释器 |

脚本打印五项：生成 token 数、耗时、tok/s、内存峰值、`light ai check` 状态。
拿不到的项写 `null` / `skipped` 并给原因，**不填推测值**。

**真实 JSON 输出**（本机实跑；`json.loads` 实测可解析）：

```console
$ ./scripts/demo.sh --format json "报告磁盘使用率"
{
  "ok": false,
  "error": "checkpoint_missing",
  "ckpt": "checkpoints/demo-3m/model.pt",
  "message": "未找到 checkpoints/demo-3m/model.pt，请先跑 make train 或从 releases 下载",
  "prompt": "报告磁盘使用率",
  "format": "json",
  "generated_tokens": null,
  "elapsed_s": null,
  "tok_per_s": null,
  "peak_rss_mb": null,
  "light_ai_check": {
    "status": "skipped",
    "reason": "checkpoint_missing"
  }
}
```

`make` 不在 PATH 时，直接跑 Makefile `demo` 目标背后的命令即可：

```bash
PYTHON=C:/Python314/python.exe bash scripts/demo.sh \
    --ckpt checkpoints/demo-3m/model.pt --format text "把这三个文件按大小排序"
```

## 4. 换 prompt / 换 format 速查

| 想做的事 | 命令 |
|---|---|
| 换 prompt（make） | `make demo PROMPT="报告磁盘使用率"` |
| 换 prompt（脚本） | `./scripts/demo.sh "报告磁盘使用率"` |
| 换 format（make） | `make demo FORMAT=json PROMPT="把这三个文件按大小排序"` |
| 换 format（脚本） | `./scripts/demo.sh --format json "报告磁盘使用率"` |
| 换 checkpoint | `./scripts/demo.sh --ckpt /path/to/model.pt "..."` |
| 多生成一些 token | `./scripts/demo.sh --max-new-tokens 64 "..."` |
| 机器消费 | `./scripts/demo.sh --format json "..." \| python -c "import json,sys;print(json.load(sys.stdin)['ok'])"` |

`--format` 只接受 `text` 或 `json`，别的值会以退出码 64 拒绝并提示。

## 5. 已知限制（含真实失败样例）

### 5.1 语料只有 3.1 MB / 508 个文件 —— **不要指望它做没见过的任务**

计划 §1.2：`light/` 仓下 `.light` 文件 514 个、3.13 MB，去重后 508 个，这是**全部**可用语料。
3.4M 参数在 3MB 语料上会严重过拟合。demo 目标只是「语法合法的光明代码 / 合法 JSON」
与「见过的任务类型（排序、文件处理、CLI 工具）」，**不是通用编程能力**。
计划 §7 明确把「通用编程能力」排除在 demo 范围外。

### 5.2 真实失败样例（T-0332 训练报告 §5，全部实测，非编造）

[`../reports/训练报告.md`](../reports/训练报告.md) §5.1 用 SFT 前 8 条任务文本做 prompt、
贪心生成 64 token，**8 条全部退化为同一个空格 token（id 1923）的重复**：

| 任务（截断） | 生成文本 | 去重 token 数 | 含 `<TASK>` | 含 `<CODE>` | 含 `<V:` |
|---|---|---:|:---:|:---:|:---:|
| 已知目录内容为：__init__.py 1381… | 64 个空格 | 1 | 否 | 否 | 否 |
| 已知目录内容为：model.py 8650 字节… | 64 个空格 | 1 | 否 | 否 | 否 |
| 已知目录内容为：vocab.py 8313 字节… | 64 个空格 | 1 | 否 | 否 | 否 |
| 已知目录内容为：设计方案.md 7005 字节… | 64 个空格 | 1 | 否 | 否 | 否 |
| 已知目录内容为：300m.json 337 字节… | 64 个空格 | 1 | 否 | 否 | 否 |
| 已知目录内容为：T-0332_训练demo模型… | 64 个空格 | 1 | 否 | 否 | 否 |
| 已知目录内容为：data.py 12634 字节… | 64 个空格 | 1 | 否 | 否 | 否 |
| 已知目录内容为：sequence.py 8334… | 64 个空格 | 1 | 否 | 否 | 否 |

- **结构合法率 0 / 8 = 0%**；8 × 64 = 512 token 只出现 1 个不同 token，无任何代码片段或结构标记。
- 原因（报告 §5.2）：空格占训练流 **30.69%**（331,854 / 1,081,448 token），是最高频 token；
  50 步 / 3.4M 参数 / CPU 训练只学到边缘分布，argmax 被空格持续占据，loss 仍在 5.5–6.0。
- 报告 §5.3 的诚实边界：该 checkpoint 定位是**管线 demo**，验证词表→模型核→数据装配→
  训练闭环→持久化→推理加载整条链路可跑通，**不承诺生成语义正确或语法完整的程序**。

**换成超出训练分布的任务（如多线程网络服务、数据库事务、GUI 编程）必然失败**，
这不是配置问题，是 3MB 语料 + 3.4M 参数的物理上限。别调 prompt 试图绕过。

### 5.3 速度也没达标

速度数字**全部引用** [`../reports/推理性能基线.md`](../reports/推理性能基线.md)，本文不另给估计值：

- §4.1 单线程（验收口径）：**74.0–96.5 tok/s**，单 token 10.36–13.51 ms，峰值 RSS 283.0–283.1 MB；
- §4.2 多线程（12 线程，仅对比）：**7.7 tok/s**（130.680 ms/token），比单线程更慢；
- §6 结论：**未达到 ≥250 tok/s 目标**。四条差异解释（合成表是规模上限估算、本机 CPU 无 AVX2、
  decode 张量极小导致小算子固定开销占比高、多线程无增益）见报告 §6。

### 5.4 本卡尚未验证的部分

- 带 checkpoint 的真实生成输出、`light ai check` 串联结果：本机无权重、无 `light` CLI，未实跑；
- `scripts/demo.sh` 的「有 checkpoint」分支：逻辑已实现（复用 `lightprism.demo.smoke`
  的 `load_demo_generator` 计时，跨平台取峰值 RSS），但**未经端到端实跑**；
- 目标机（第 5 代酷睿）V6 实测：计划 §二标注为人工项，本卡不涉及。

## 6. 常见报错怎么查

### 6.1 找不到 checkpoint

现象与完整输出见 §3.1。三件事按顺序查：

1. `ls checkpoints/demo-3m/model.pt` —— 权重不入库，clone 后本来就没有；
2. 本地训练：`make train`（等价 `PYTHONPATH=src python -m lightprism.demo.train_demo
   --config configs/demo-3m.json --steps 50`）；或从 releases 下载后放到该路径；
3. 用了自定义路径就显式传 `--ckpt` / `CKPT=`。

**注意**：`make demo` 此时退出码是 0（预期状态）；要看机器判据请用
`./scripts/demo.sh --format json`，检查 `error == "checkpoint_missing"`。

### 6.2 内存不足

- 先看基线：报告 §4.1 峰值 RSS **283.0–283.1 MB**，计划上限 8 GB，正常推理远不到上限；
- `make bench` 会打印 `峰值内存: … MB（VmHWM(/proc) 或 ru_maxrss(KB)）`，用它确认是不是模型本身；
- 若真 OOM，优先怀疑**数据装配**而不是模型：训练侧的 `--batch-size` / `--seq-len`、
  `data/corpus/` 快照体积；报告 §7 也提醒 KV cache 若退回全量重算会 O(T²) 退化；
- `scripts/demo.sh` 在拿不到峰值内存时会写 `null` + `rss_source="unavailable"`，不会填假数。

### 6.3 速度不达标

1. **别拿多线程数字当验收**：报告 §4.2 的 12 线程只有 7.7 tok/s，同步开销主导；
   口径写死在 `scripts/bench.py`（单线程 / 贪心 / prompt 32 / 预热 10 个不计入），
   `make bench` 不接受命令行改口径；
2. 复跑 `make bench BENCH_TOKENS=100`，对照报告 §4.1 的 74.0–96.5 tok/s；
3. 检查 `torch.get_num_threads()` 是否为 1、CPU 是否有 AVX2（报告 §6 第 2 条：本机
   Ivy Bridge EP 无 AVX2，torch CPU 后端比 numpy 同形状慢约 2.3 倍）；
4. 看报告 §5 的单步耗时拆解：矩阵乘下限约 2.7 ms/步，其余是算子调度与拷贝；
5. Windows 上 `make bench` 会因 `import resource` 直接失败 → 换 Linux / FreeBSD 跑，
   或直接用报告里已有的实测数字（Makefile 已给出可读提示）。

### 6.4 `python -m lightprism.demo` 报 `__main__` 缺失

见 §3.2，入口未落地，用 `scripts/demo.sh` 或 `lightprism.demo.smoke` 代替。

### 6.5 Windows 上 `make` / `light` 不在 PATH

实测两者都不在 PATH。Makefile 目标可直接展开为底层命令：

```bash
# make demo
PYTHON=C:/Python314/python.exe bash scripts/demo.sh --ckpt checkpoints/demo-3m/model.pt --format text "把这三个文件按大小排序"
# make check
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m lightprism.demo.vocab_build --check data/vocab_4096.json
PYTHONPATH=src python -m lightprism budget
```

`light ai check` 未执行时，JSON 里是 `light_ai_check.status = "skipped"` 并附原因，
**不会报「通过」**。

## 7. 待补（诚实边界）

| 项 | 阻塞 | 归属 |
|---|---|---|
| 带 checkpoint 的真实生成输出样例 | T-0332 权重未入库 | T-0332 / 收口卡 |
| `light ai check` 与生成结果的串联校验 | 本机无 `light` CLI；本卡不推断 CLI 参数形态 | 另开卡 |
| `src/lightprism/demo/__main__.py`（`python -m lightprism.demo`） | `src/` 不在本卡允许路径内 | 另开卡 |
| V6 目标机（5 代酷睿）实测 | 需真机，计划 §二标为人工 | 验收卡 |
| ≥250 tok/s 达标 | 报告 §6 结论未达标，需 MKL/oneDNN/AVX2 构建或算子融合 | 另开卡 |
