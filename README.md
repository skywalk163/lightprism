# LightPrism（光棱）

> 一束白光穿过棱镜，分解成不同颜色的光谱；模型面对同一个序列，分解出**决策**、**代码**、**验证**。

**LightPrism** 用 Light（光明）的 64K 词表，把「做选择」和「写代码」统一成同一种序列语言。
一个 1B 级小模型，在**同一个自回归过程**里交替产出决策 token、光明代码和验证反馈——
不靠外挂决策分类器，也不靠后置脚本，让普通电脑也能跑起懂光明语言的编程助手。

| 层级 | 取值 |
|---|---|
| 英文技术名 | **LightPrism**（延续 `lightharness` / `lightmem` / `lighthive` 的 Light + 功能词风格） |
| 中文名 | **光棱** |
| 仓库 / 包名 | `lightprism`（PyPI 发布名 `light-prism`） |
| 模型发布名 | `light-prism-300m` / `light-prism-1b`（按参数量区分） |

> 备选名 LightDecide（明策）只说了「决策」，说不全本方案真正的贡献——
> 决策、代码、验证**共享同一个词表和同一个损失函数**。Prism 同时容纳「统一输入」与
> 「分解输出」两层含义，且与 Light（光）语义连贯，故取 LightPrism。

## 这个仓库现在是什么

**v0.2.0 = 设计定稿 + 可运行骨架 + 训练/推理管线**。不含模型权重，不含真实语料。

管线是**能真跑**的（5 条命令跑完训练→生成→探针），但默认配置是「CPU 几分钟跑完」的微缩模型；
1B 的真实训练需要换词表、加数据并行、上 bf16 —— 清单见 [docs/训练与推理.md](docs/训练与推理.md) §6。

它交付四件可验证的东西：

1. **控制 token 规范**：7 个决策点 + 8 个验证码的封闭集合，命名空间与代码 token 隔离，
   序列化形态为 `<D:EDIT=FR>` / `<V:ERR=SYNTAX>`。
2. **统一序列的解析与校验器**：`<TASK>` / `<D:…>` / `<CODE>` / `<V:…>` 的严格解析、
   结构合法性检查、决策密度统计。
3. **64K 词表预算核算**：八个槽位显式写死、合计恰好 65536，含 L0 核心字「一字一 token」检查。
4. **训练/推理/探针管线**：`model`（QK-Norm + RoPE + SwiGLU，320M/950M 两档配置）→
   `tokenizer`（控制 token 占词表末尾连续区间 + 结构状态机约束解码）→ `data`（样本 schema 与
   定长批次）→ `train`（阶段一/二）→ `infer` → `probe`（四条通过标准判定）。

```bash
python -m pytest -q                          # 72 项测试（44 项纯标准库 + 28 项需 torch）

# 纯标准库部分
PYTHONPATH=src python -m lightprism tokens     # 列出全部决策 / 验证 token（19 + 8）
PYTHONPATH=src python -m lightprism budget      # 预算核算（合计必须 = 65536）
PYTHONPATH=src python -m lightprism parse examples/demo.prism.txt
PYTHONPATH=src python -m lightprism map-check check.json    # light ai check --json → <V:…>
PYTHONPATH=src python -m lightprism config configs/1b.json  # 参数量：950M
PYTHONPATH=src python -m lightprism export --source-root DIR   # 从 light ai check 归档导样本

# 需要 torch（pip install -e ".[train]"）
PYTHONPATH=src python -m lightprism.train --data data/samples.jsonl \
    --steps 4 --seq-len 128 --batch-size 3 --log-every 1 --out-dir outputs/smoke
```

`--json` 在子命令前后等价，便于脚本消费。训练→生成→探针的完整命令见
[docs/训练与推理.md](docs/训练与推理.md)。

## 五分钟跑通 demo

**这是 3.4M 参数的 demo 模型，不是 1B 版本。** 顶部表格里的 `light-prism-1b` 是目标形态；
当前能跑的是 `configs/demo-3m.json`（4096 词表，d=192 / L=6 / H=6，参数 3,443,520）。
**不需要 GPU**，CPU 即可，Python ≥ 3.10。权重 `checkpoints/demo-3m/model.pt` 由 T-0332
产出且不入库，所以从干净 clone 跑 `make demo` 拿到的是一条可读提示，不是生成结果。

```bash
make check     # pytest + 4096 词表校验 + 64K 预算核算（不需要 checkpoint）
make demo      # 一条命令跑通
make bench     # tok/s 与峰值内存（需 POSIX 平台）
```

**真实输出样例**（本机 Windows / Python 3.14.7 实跑，非构造示例）：

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

换 prompt、换成机器可读格式：

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

`make check` 实测全绿（`pytest -q` 126 项、退出码 0；`vocab_build --check` 结论：全绿；
`lightprism budget` 预算检查：通过）。**带 checkpoint 的生成输出样例：待 T-0332 完成后补** ——
本卡不编造输出。

**速度数字一律引用 [reports/推理性能基线.md](reports/推理性能基线.md)，此处不另给估计值**：
该报告 §4.1 单线程 74.0–96.5 tok/s、峰值 RSS 283.0–283.1 MB，§6 结论是
**未达到 ≥250 tok/s 目标**（12 线程反而只有 7.7 tok/s）。

**能力边界**：语料只有 3.1 MB / 508 个文件，3.4M 参数在 3MB 语料上会严重过拟合；demo 只求
语法合法与见过的任务类型（排序、文件处理、CLI 工具），**不是通用编程能力**
（[docs/Demo开发计划.md](docs/Demo开发计划.md) §1.2 / §7）。真实失败样例：
[reports/训练报告.md](reports/训练报告.md) §5 中 8 条 prompt 贪心生成 64 token
**全部退化为空格重复，结构合法率 0/8 = 0%**。完整跑法、选项与报错排查见
[docs/Demo使用.md](docs/Demo使用.md)。

## 核心设计

### 1. 一条序列，三种功能

```text
<TASK>把测试文件按大小排序并输出前10名</TASK>
<D:PLAN=DECOMPOSE>
<D:EDIT=FR>
<CODE>
设 文件列表 为 列目录(".")。
</CODE>
<V:PASS>
<D:NEXT=TEST>
```

三类 token 共享**同一个词表**和**同一个下一 token 预测目标**：

| 类别 | 形态 | 作用 |
|---|---|---|
| 决策 | `<D:POINT=OPTION>` | 把「做选择」变成自回归过程里的一个词汇选择 |
| 代码 | `<CODE>…</CODE>` | 光明代码 token 序列 |
| 验证 | `<V:PASS>` / `<V:ERR=…>` | 把 `light ai check` 的确定性结果直接喂回模型上下文 |

### 2. 64K 词表预算（本仓库已锁死）

| 槽位 | 建议值 | 区间 | 说明 |
|---|---:|---|---|
| L0 核心字 | 44 | 精确 44 | 一字一 token，永不被 BPE 拆碎 |
| L1/L2 高频语法组合 | 8192 | 4096—8192 | `遍历`/`设`/`段`/`若…则…` |
| 标准库名/函数名 | 10240 | 6144—10240 | 109 个标准库模块的高频名字 |
| 中文自然语言 BPE | 34804 | 30720—36864 | 指令、注释、错误说明；取整余量落这里 |
| 通用代码 token | 10240 | 8192—12288 | 数字、运算符、标点、标识符片段 |
| **决策 token** | 320 | 128—512 | 硬上限：占全词表 ≤ 2% |
| **验证 token** | 160 | 64—256 | `light ai check` 错误码的确定性映射 |
| 保留/扩展 | 1536 | 1024—2048 | 后续决策点、实验 token、控制符 |
| **合计** | **65536** | — | 不许多也不许少，余量必须显式进「保留」 |

硬约束（违反即失败，不是建议）：

- 决策 token / 验证 token **从训练第一步就参与嵌入学习**，不后期注入；
- L0 核心字**一字一 token**，任何 L0 字被拆碎都破坏最底层稳定性；
- 决策 token 命名空间与代码 token 明确隔离，不允许模型把 `<D:…>` 当普通标识符生成。

### 3. 验证 token 由真实验证器生成，不由模型自判

`<V:…>` 在训练时**由 `light ai check --json` 的真实返回映射而来**，推理时模型生成后
lightharness 仍要实际调用验证器做最终确认。模型学的是「预测验证结果并据此调整后续生成」，
不是「替代验证器」。映射逻辑见 [`src/lightprism/tokens.py`](src/lightprism/tokens.py)
的 `DIAGNOSTIC_TO_VALIDATION` 与 `map_diagnostics()`。

## 与 lightharness 的边界

LightPrism 不替代 lightharness。分工是清楚的：

| | lightharness（编排层） | LightPrism（模型层） |
|---|---|---|
| 项目级上下文检索 | ✅ | — |
| 工具调用 / 文件读写 / 子进程 | ✅ | — |
| 多轮任务循环与人工确认 | ✅ | — |
| 决策 = 序列里的一个 token | — | ✅ |
| 决策 / 代码 / 验证共享损失 | — | ✅ |

lightharness 只需把模型输出中的 `<D:…>` 解析成结构化动作、把 `<V:…>` 作为下一轮输入。

## 口径对齐（三处与方案文档 v0.1 的差异，均已核对 light 仓现状）

| 项 | 方案文档 v0.1 | 本仓库 | 依据 |
|---|---|---|---|
| L0 核心字数量 | 「30 字」 | **44 字** | `light/docs/language/l0-core.md`（44 字权威表）；`light/src/keywords.py` 头部明确说明旧「30 字」是错数字 |
| `light ai check` 错误类型 | `MISSING_TYPE` / `UNDEFINED_VAR` 等语义名 | 语义名保留，但映射源是**真实诊断码** `L1001`—`L5003` | `light/src/diagnostics/registry.py`（19 条规则）+ `light/cli/light.py` 的 `_ai_check_json` |
| `<D:EDIT=NEW_FILE>` | 出现在 §2.1 示例 | **不在**首批 7 个决策点里，解析即报错 | §4.2 首批决策点表只有 `FR/RW/PATCH`；缺口需显式补进词表而非静默接受 |

## 路线图

- **阶段一 · 词表共存**（约 20% 训练量）：三类 token 随机交错，只做标准下一 token 预测；
  结束后跑线性探针实验（1M 参数以下）回答「通不通」。
- **阶段二 · 序列内协同**（约 50%）：严格「决策 → 代码 → 验证」交错，决策为后续生成设定策略上下文。
- **阶段三 · RLVR 微调**（约 30%）：奖励对齐到 `light ai check` 真实通过率、安全等级落地情况。

细节见 [`docs/设计方案.md`](docs/设计方案.md) 与
[`docs/方案一_统一决策Token词表_v0.1.md`](docs/方案一_统一决策Token词表_v0.1.md)（原始技术方案）。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/设计方案.md`](docs/设计方案.md) | 总体设计：序列形态、三阶段课程、评估与风险 |
| [`docs/序列规范.md`](docs/序列规范.md) | `<D:…>` / `<CODE>` / `<V:…>` 的语法与校验规则（唯一事实源） |
| [`docs/词表预算.md`](docs/词表预算.md) | 64K 预算表的推导与核算口径 |
| [`docs/探针实验设计.md`](docs/探针实验设计.md) | 20% 训练成本回答「通不通」的实验设计 |
| [`docs/方案一_统一决策Token词表_v0.1.md`](docs/方案一_统一决策Token词表_v0.1.md) | 原始技术方案（v0.1，2026-11-11） |
| [`docs/训练与推理.md`](docs/训练与推理.md) | 训练/推理/探针五条命令、模块职责、放大前必补清单 |
| [`docs/样本导出.md`](docs/样本导出.md) | 从 `light ai check` 归档导出样本；**为什么导不出修复样本** |
| [`docs/Demo开发计划.md`](docs/Demo开发计划.md) | 可跑通 demo 的开发计划：规模上限、验收标准、6 张任务卡 |
| [`docs/Demo使用.md`](docs/Demo使用.md) | 一键 demo 的三种跑法、prompt/format 选项、真实失败样例与报错排查 |
| [`docs/远端与发布.md`](docs/远端与发布.md) | 三远端清单、推送复核纪律、CI 现状 |

## 许可

MIT。见 [LICENSE](LICENSE)。

需要有人陪着跑通第一门课 / 第一个内部工具，或要定制 L3 领域语法，可以找我们：
[quye.com](https://quye.com/) · hello@quye.com
