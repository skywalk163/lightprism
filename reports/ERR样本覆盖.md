# ERR 类验证样本覆盖报告

> 生成时间：2026-11-12
> 依据：`light ai check --run --json --no-corpus` 实测（Windows 环境）

## 可达性矩阵总览

| 验证码 | 诊断码 | 可达性 | 条数 | 触发证据 |
|---|---|---|---|---|
| `PASS` | — | ✓ 已有 | 363+ | json_tasks + code_tasks 全部 PASS |
| `ERR=RUNTIME` | L3001 | ✓ 可达 | 待生成 | 列表索引越界（唯一稳定复现的具体码） |
| `ERR=CHECKER_FAIL` | L0000 | ✓ 可达（兜底） | 待生成 | 大多数运行期错误都回落到此码 |
| `ERR=SYNTAX` | L1001..L1007 | ✗ 未覆盖 | 0 | 括号不闭合等语法错误实际回落到 L0000 |
| `ERR=TYPE_MISMATCH` | L2001/L2003 | ✗ 未覆盖 | 0 | 类型不匹配实际回落到 L0000 |
| `ERR=MISSING_TYPE` | L2002 | ✗ 未覆盖 | 0 | 缺类型标注实际回落到 L0000 |
| `ERR=UNDEFINED_VAR` | L4001 | ✗ 未覆盖 | 0 | 未定义变量实际回落到 L0000 |
| `ERR=IMPORT_FAIL` | L4002 | ✗ 未覆盖 | 0 | 导入失败实际回落到 L0000 |
| `WARN=NON_CANONICAL` | L1005/L1007/L3002.. | ✗ 未覆盖 | 0 | 警告类诊断未实测到 |

## 已知事实（本机 2026-11-12 实测）

`light ai check --run --json --no-corpus` 可用，但**大多数运行期错误都回落到 `L0000`**：

| 尝试的片段 | 实际诊断 | 备注 |
|---|---|---|
| `设 甲 为 (`（括号不闭合） | `L0000` | 语法错误→兜底 |
| `设 甲 为 [1, 2]` + `打印(甲[9])` | **`L3001`** | 唯一稳定复现的具体码 ✅ |
| `打印(甲)`（未定义） | `L0000` | 未定义变量→兜底 |
| `设 甲 为 1 + "文本"` | `L0000` | 类型不匹配→兜底 |
| `段落 加一(甲)` 无类型标注 | `L0000` | 缺类型标注→兜底 |
| `从《不存在的模块》导入《甲》。` | `L0000` | 导入失败→兜底 |

## 结论

1. **L3001 → ERR=RUNTIME** 是唯一能稳定触发的具体诊断码，可用于生成真实样本。
2. **L0000 → ERR=CHECKER_FAIL** 是兜底码，虽然可达，但语义上是「检查器没细分」，不应当作具体错误类型的监督信号。
3. 其余 7 个验证码（SYNTAX/TYPE_MISMATCH/MISSING_TYPE/UNDEFINED_VAR/IMPORT_FAIL/NON_CANONICAL）在当前 light 版本下**无法稳定触发**，都回落到 L0000。
4. 这不是样本生成器的问题，而是 light 解释器本身的诊断粒度问题——大多数错误还没细分到具体码。

## 下一步

* 等 light 解释器细化诊断码后，再跑 `gen_err_samples.py --check` 更新可达性矩阵。
* 当前样本集只包含 L3001（ERR=RUNTIME）的变体，足以让 `has_err_validation` 门禁过。
* 其余码记为「未覆盖 + 原因」，不拿 L0000 冒充。

---

## 2026-11 定因复测（T-0370，2026-11-16）

> 复测方式：`PYTHONPATH=src python -m lightprism.runtime.err_samples --check --json-out err_probe2.jsonl`（28 个候选片段，逐条真跑 `light ai check --json --no-corpus --run --timeout 60`），
> 后端：`light` gitcode/main `89948c47a`（只读 worktree `main-T0370`），Windows 10 / Python 3.13.16。
> 上面表格与结论保持原样不改写，本节只追加新实测。

### 复测结果（28 → 25 可达）

| 目标码 | 可达 | test 侧 | 结论 |
|---|---|---|---|
| `L3001` | 13/14 | 6 | 稳定可达，运行期 `列表索引越界` |
| `L0000` | 12/14 | 5 | 兜底码本身可达（语法检查 3 条 + 运行 9 条） |

3 条被剔除：`设 空 为 []`（实得 `L2002`）、`段落 加一(数)：`（直接 PASS）、`设 甲 为 真。\n打印(甲 加 1)。`（直接 PASS）。

### 两处对历史结论的修正

1. **`L2002` 其实是可达的**（原表记「✗ 未覆盖」）：`设 空 为 []` → `L2002 / ERR=CHECKER_FAIL / stage=语法检查`，报错文本明确说「`空`是关键字（表示空值/None），不能作变量名」，并建议改用 `空内容/空值/占位`。它映射到 `ERR=MISSING_TYPE`，不在可观测验证码集合，所以按现口径仍只能当 `ERR=CHECKER_FAIL` 用。
2. **`L0000` 的成因不是「规则未接线」**，而是**码表正则与真实报错文本对不上**（见下）。

### 定因：链路已接线，是 pattern 与报错文本不匹配

- 接线证据：`cli/light.py:1950`（边框块路径）与 `cli/light.py:1996`（单行路径）都会调用 `_ai_explain(message)` → `src/diagnostics/registry.py:412 explain(message)`；`explain()` 依次用每条 `Rule.pattern` 正则匹配**原始报错文本**，**匹配不上才**由 `cli/light.py:1961/2012` 发 `L0000`。码表里**没有任何兜底优先级规则**，所以不存在「priority 被兜底吃掉」。
- 逐码实测（5 条新片段，2026-11-16）：

| 目标码 | 片段 | 实测码 | light 实际报错文本 | 为什么没命中 |
|---|---|---|---|---|
| `L1001` | `def 加一(甲)：\n  返回 甲 加 1。` | `L0000` | 无法识别的语法元素：`'：'` | L1001 pattern 只认 `def`/`function` 关键字名，实际文本说的是语法元素 |
| `L2001` | `段落 加一 接收 甲：\n  返回 甲。\n打印(加一(1, 2))。` | `L0000` | ⚠ 警告：参数 `甲` 缺少类型标注 | L2001 pattern 认「参数数量错误」文案，实际报的是缺类型警告 |
| `L2003` | `段落 加一 接收 甲 为 1：\n  返回 甲。\n打印(加一())。` | `L0000` | 期望 冒号「:」，但得到 数字 | L2003 pattern 认默认参数写法，实际是 parser 冒号报错 |
| `L4001` | `设 甲 为 1。\n打印(乙)。` | `L0000` | 名称错误 `name '乙' is not defined` | L4001 pattern 认中文「缺导入/该函数来自『…』模块」，实际是英文 NameError 文案 |
| `L4002` | `导入《不存在的模块》。` | `L0000` | 模块未找到 `No module named '不存在的模块'` | L4002 pattern 认「从 math 导入 / print( / 没有 math 模块」 |

- 另一个能反向证明链路通的反例：`L3001`（「列表索引越界」文本）和 `L2002`（关键字作变量名的提示）**都对上了**，说明 `explain()` 在工作，只是码表 pattern 覆盖面不足。

### 定因结论

`light` 侧缺陷，位置在 `src/diagnostics/registry.py` 的 `Rule.pattern`（与 `cli/light.py` 上报的文本契约）：**规则已接线、但 pattern 只覆盖了少数中文措辞，parser/运行时的英文文案与「缺类型警告」等实际措辞没被覆盖**。修它属于 `light` 仓，**不在 T-0370 的 paths**，另开卡处理；本卡在 `lightprism` 侧的做法是：

- 只给实测可达的码造样本（`L3001` 13 条 / `L0000` 12 条，test 侧 6 + 5）；
- 把 20 个已登记码的定因结论固化成 `lightprism.runtime.err_samples.DIAGNOSTIC_REACHABILITY`，并加回归用例 `tests/test_err_samples.py::test_probe_matrix_against_real_light`——**哪天 light 把某条规则接对了，用例会变红提醒更新矩阵**；
- 详见 `reports/数据补强报告.md` 第六节。
