# CHANGELOG

> 版本号真源：本文件首条 `## [vX.Y.Z]` 必须与 `pyproject.toml` 的 `version` 一致。

## [v0.2.0] — 2026-11-11

训练 / 推理 / 探针管线：**能真跑，不是能产出 1B 模型的训练脚本**（差别见 docs/训练与推理.md §6）。

### 新增

- **`model_cfg.py`**：`PrismConfig` 与参数量公式（纯标准库）。`configs/300m.json`
  = 320M，`configs/1b.json` = 950M（非嵌入 849M），由 `lightprism config` 校验。
  公式与实际建模结果由 `test_formula_matches_actual_model` 强制相等。
- **`model.py`**：decoder-only 主体 —— pre-norm + **QK-Norm**（方案 §8.1 的必要条件）+
  RoPE + SwiGLU + 权重绑定。`param_groups()` 把嵌入单独成组，倍数上限卡在 2.0。
- **`tokenizer.py`**：底座词表 + 控制 token 拼装。控制 token 占词表**末尾连续 id 区间**
  （4 个结构标记 + 19 个决策 + 8 个验证 = 31 个），区间有空洞直接报错。
  `allowed_next(history)` 是约束解码用的**结构状态机**。
- **`data.py`**：样本 schema（`Sample`/`Step`，来源限 `light_ai_check` /
  `harness_tool_log` / `human_edit` / `task_tree`）、JSONL 往返（脏行计数不中断）、
  定长批次（pad 权重恒 0）、**id 口径**的决策密度统计。
- **`train.py`**：阶段一/二训练循环（AdamW + 余弦退火 + 梯度裁剪 + checkpoint）、
  `evaluate()`；阶段三 RLVR 明确抛 `NotImplementedError`（需 lightharness 侧环境）。
- **`infer.py`**：约束解码 + `load_generator()` + `verify_with_checker()` 后置复核挂点。
- **`probe.py`**：线性探针（每决策点一个头）+ 方案 §7.3 四条通过标准的 `judge()`。
- **CLI**：新增 `config` 子命令（不需要 torch）。
- **数据/配置**：`data/samples.jsonl`（3 条示例）、`configs/300m.json`、`configs/1b.json`。
- **文档**：`docs/训练与推理.md`。

### 修掉的三个实测 bug

1. **梯度范数恒为 0**：原先在 `optimizer.zero_grad()` 之后取值，梯度已被清空。
   改为在 clip/step 之前取。
2. **CLI 被迫依赖 torch**：`PrismConfig` 原先在 `model.py` 里，`lightprism config`
   一执行就 import torch，纯标准库环境直接崩。拆出 `model_cfg.py` 切断耦合。
3. **checkpoint 恢复后解码全是 `<padNNN>`**：原先只存 `base_vocab_size`，恢复时用占位
   token 凑数。改为存 `base_vocab` 本体，缺失时明确报错而不是静默降级。

### 约束解码的两道修正（均由实测暴露）

- 只看紧邻一个 id 的约束**拦不住**未训练模型连续吐 `</TASK>` / 连续吐 `<D:EDIT=FR>`；
  改为按完整历史重构状态的 `allowed_next(history)`。
- 生成被 `max_new_tokens` 截断时，尾部裸露文本不该算模型的结构错误：
  `generate()` 增加 `truncated` 与 `counts_toward_structure_rate` 字段。

### 探针的诚实性约束

- 样本不足无法切分测试集时，用训练集自评的准确率会标 `eval_split=train(...)`，
  **不计入宏平均**；`macro_accuracy_is_meaningful` 显式报出来。
- `judge()` 对缺失指标返回 `incomplete`，不会因为缺数据就判通过。

### 测试

48 项（20 项纯标准库 + 28 项需 torch，后者用 `importorskip` 整体跳过）。

### 未包含

- 阶段三 RLVR、真实 BPE 词表、数据并行、bf16/激活重计算、断点续训；
- 从 `light ai check` 日志与 lightharness 工具记录**自动导出**样本的解析器
  （`data.py` 只定义了 schema 与校验，不解析真实日志）。

## [v0.1.0] — 2026-11-11

首个骨架版本：设计定稿 + 可运行骨架，**不含训练代码与模型权重**。

### 新增

- **控制 token 规范**（`src/lightprism/tokens.py`）：7 个决策点（`EDIT`/`SAFE`/`RETRY`/`FIX`/
  `NEXT`/`CTX`/`PLAN`）共 19 个决策 token；8 个验证码；`TokenSpace` 提供封闭性校验与
  序列化/反序列化。未登记的决策点、选项、验证码一律抛 `ParseError`，不静默兜底。
- **`light ai check --json` → 验证 token 的确定性映射**（`map_diagnostics()`）：
  按 `light/src/diagnostics/registry.py` 的 19 条诊断码建表，未登记码落 `ERR=CHECKER_FAIL`
  并在 detail 中暴露原码。
- **统一序列解析与校验**（`src/lightprism/sequence.py`）：`<TASK>` / `<D:…>` / `<CODE>` /
  `<V:…>` 严格解析（控制 token 之间的裸露文本即报错）、结构校验、渲染往返一致、
  决策密度统计。
- **64K 词表预算核算**（`src/lightprism/vocab.py`）：八个槽位建议值合计精确 65536；
  `VocabBudget.problems()` 校验区间、总额、控制 token 占比；`check_l0_atomicity()`
  校验 L0 核心字「一字一 token」。
- **CLI**（`src/lightprism/cli.py`）：`tokens` / `budget` / `map-check` / `parse` / `l0`
  五个只读子命令，`--json` 放在子命令前后均等价（`budget --json` ≡ `--json budget`）。
- **文档**：设计方案、序列规范（唯一事实源）、词表预算、探针实验设计，以及原始技术方案
  v0.1 存档。

### 口径对齐

- L0 核心字从原始方案的「30 字」改为 **44 字**，依据 `light/docs/language/l0-core.md`
  与 `light/src/keywords.py` 文件头说明。
- `<D:EDIT=NEW_FILE>` 出现在原始方案 §2.1 示例但不在 §4.2 首批表，本版本**解析即报错**，
  缺口记录在 `docs/序列规范.md`「已知缺口」。
- 验证 token 的映射源改为真实诊断码 `Lxxxx`，语义名仅作为 token 文本保留。

### 未包含（v0.2 候选）

- 训练/推理脚本与数据 schema；
- `tokenizers` 侧的真词表构建；
- `<V:…>` 携带附加字段（`line=…` / `expected=…`）的语法；
- 探针实验的实际运行结果。

### 测试

20 项 pytest，纯标准库，无第三方依赖。
