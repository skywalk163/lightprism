# CHANGELOG

> 版本号真源：本文件首条 `## [vX.Y.Z]` 必须与 `pyproject.toml` 的 `version` 一致。

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

18 项 pytest，纯标准库，无第三方依赖。
