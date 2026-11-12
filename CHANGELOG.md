# CHANGELOG

> 版本号真源：本文件首条 `## [vX.Y.Z]` 必须与 `pyproject.toml` 的 `version` 一致。

## [v0.3.1] — 2026-11-12

`train_openi.py` v3：把「测量无效」的 fail 降级为 incomplete，并补齐数据/口径门禁。
依据 `reports/探针实验_v2_评审_2026-11-12.md`：v2 在 OpenI/V100 的 `verdict=fail`
经复核是四条判据无一具备判别力（1/7 决策点、7 条 test 且与 train 模板 100% 重合、
判据 ③ 分母排除截断、判据 ② 只差 1 条），不是方案被证伪。

### 新增

- **门禁（P0）**：决策点覆盖 <7 / 探针 test <30 / train-test 模板重合 / 无 ERR 类验证 /
  pos-1 特征退化为常数，任一命中 → 四条判据逐条记 `unknown`、整体 `incomplete`（不再报 fail）。
  条件补齐后真失败仍判 `fail`，全绿才 `pass`（`tests/test_openi_v3_judge.py` 4 个用例锁定）。
- **分组切分 `--group-split`**：按「类别+归一化 task 模板簇」整体切分，
  实测 `train_test_template_overlap` 从 27 降到 **0**（v2 decision 类 6 条 test 的 task
  在 train 里一字不差出现过，A 组 1.000 是记忆不是泛化）。
- **`--decision-position scattered`**：决策 token 不再永远紧跟 `</TASK>`，
  使 pos-1 探针特征从 50/50 的常数变成任务文本上下文（方案 §7「随机交错三类 token」）。
- **判据 ③ 全分母**：截断生成 = 未吐 `</CODE>`/`<V:...>` 的失败输出，计入分母；
  `--max-new-tokens` 160 → **384**（实测生成段 p95=330）。
- **判据 ② 显著性 `--min-decisive-gap`**：C/B 精确匹配差 <3 条记 unknown，
  堵住 v2 的 8/42 vs 7/42 擦边 pass；真实 `light ai check --run` 口径改列 caveat，
  不再占第五条判据导致 pass 不可达。
- **A' 对等对照 `--a-group-rival`**：冻结主干 + 同 pos-1 特征 + MLP，
  与 390419 参数的 A 组文本分类器并列报，分离「线性不可分」与「共享表征没编码」。
- **阶段一几何检查点（P2）**：`embedding_geometry` 报决策/文本嵌入余弦分布与
  「从未训练的决策 token 个数」（实测 **16/19** 是死权重）；§8.6 梯度范数比值摘要。
- **§8 产出物（P3）**：训练结束落 `reports/探针实验_<日期>.{md,json}`（`--no-report-to-repo` 可关）。
- `tests/test_openi_v3_judge.py`：11 项纯标准库回归（门禁四态、分组切分零重合、
  scattered 打破常数特征、全分母、覆盖率诊断）。

### 验证

- `python train_openi.py --self-check` PASS；`--group-split` / `--decision-position scattered` 各 PASS。
- 全量 `pytest -q`：109 项全绿（torch 路径用 importorskip，CI 纯标准库门禁可跑）。
- CPU `--smoke --a-group-rival` 端到端跑通，产出 `metrics.json` 含 coverage /
  embedding_geometry / grad_norm_summary / A_rival / structure_rate_all。
- **真训练仍需上传 OpenI/V100 重跑**；当前数据在门禁下如实报 `incomplete`，
  解锁条件是补齐 7 个决策点与 ERR 类正样本、模板零重合、生成预算覆盖 p95。

## [v0.3.0] — 2026-11-11

样本导出器：把 `light ai check` 的报错归档转成训练样本。

### 新增

- **`src/lightprism/exporters/`**：`read_corpus`（读 `<YYYY-MM>.jsonl` + `index.json`）、
  `SourceIndex`（按 `source_hash` 回找当前源文件）、`LightChecker`（现跑 `light ai check`）、
  `export_samples`（主流程）、`ExportStats`（每类跳过均计数）。
- **`export` CLI 子命令**：`--corpus-dir` / `--source-root` / `--light-root` /
  `--min-hits` / `--limit` / `--out`。
- **`docs/样本导出.md`**：数据源真实形状（含源码出处）、结构性限制、下一来源。

### 核心结论：本导出器**只产出失败样本**

`source_hash` 是对出错那一刻的源码算的，代码一改 hash 就变。所以「同一文件从坏到好」
的配对在 corpus 里结构上不存在：

- 代码没改 → hash 匹配 → 现跑必然仍失败；
- 代码改好 → 现跑通过 → hash 已变，记录匹配不上。

实测：把越界代码改成合法后，hash 从 `3f8e1a3acc644768` → `1284d1e5c731b464`，
该记录直接落 `skipped_no_source`。

修复样本只能从 lightharness 的**编辑日志**（记录文件路径、编辑方式、编辑前后 check 结果）
导出，而该日志格式目前不存在。CLI 输出里显式带 `produces_repair_samples: false`。

### 修掉两个真实映射错误（导出真实 corpus 时发现）

| 诊断码 | 原映射 | 改为 | 理由 |
|---|---|---|---|
| `L3001` 索引越界 | `ERR=UNDEFINED_VAR` | `ERR=RUNTIME` | 越界不是「变量不存在」 |
| `L5001` 后端不支持 | `ERR=CHECKER_FAIL` | `ERR=RUNTIME` | 编译期能力边界 ≠ 验证器故障 |

验证码从 8 个增至 9 个（新增 `ERR=RUNTIME`）。这两个错误会让模型学到反向的因果关联。

### `light ai check` 必须带 `--run`（实现层面的坑）

不带 `--run` 时只做语法检查，运行时错误一律返回 `ok: true`。实测同一份越界代码：
不带 → `{"ok": true}`，带 → `{"ok": false, "code": "L3001"}`。若默认不跑，
会把「根本没修」标成「已通过」。`LightChecker` 因此默认 `run=True`。

另一个坑：`--timeout` 在 argparse 侧是 `int`，传 `30.0` 会让子进程 `rc=2`、stdout 为空，
被误判成「检查器崩溃」。已在构造时校验。

### 测试

72 项（44 项纯标准库 + 28 项需 torch）。新增 `tests/test_exporters.py` 24 项，
含两条真实 corpus 回归（本机无归档时 skip）。

### 未包含

- lightharness 编辑日志解析器（格式尚不存在，见 docs/样本导出.md §4）；
- 从 `source_snippet` 还原代码（±2 行不构成可编译样本，明确不做）。

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
