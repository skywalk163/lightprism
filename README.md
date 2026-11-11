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

**v0.1.0 = 设计定稿 + 可运行骨架**，不含训练代码、不含模型权重。它交付三件可验证的东西：

1. **控制 token 规范**：7 个决策点 + 8 个验证码的封闭集合，命名空间与代码 token 隔离，
   序列化形态为 `<D:EDIT=FR>` / `<V:ERR=SYNTAX>`。
2. **统一序列的解析与校验器**：`<TASK>` / `<D:…>` / `<CODE>` / `<V:…>` 的严格解析、
   结构合法性检查、决策密度统计。
3. **64K 词表预算核算**：八个槽位显式写死、合计恰好 65536，含 L0 核心字「一字一 token」检查。

```bash
python -m pytest -q                          # 20 项测试，纯标准库
PYTHONPATH=src python -m lightprism tokens   # 列出全部决策 / 验证 token（19 + 8）
PYTHONPATH=src python -m lightprism budget   # 预算核算（合计必须 = 65536）
PYTHONPATH=src python -m lightprism parse examples/demo.prism.txt
PYTHONPATH=src python -m lightprism map-check check.json   # light ai check --json → <V:…>
```

`--json` 在子命令前后等价，便于脚本消费。

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

## 许可

MIT。见 [LICENSE](LICENSE)。

需要有人陪着跑通第一门课 / 第一个内部工具，或要定制 L3 领域语法，可以找我们：
[quye.com](https://quye.com/) · hello@quye.com
