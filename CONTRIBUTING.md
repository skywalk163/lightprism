# 贡献指南

## 环境

只需 Python ≥ 3.10，测试与 CLI 全部依赖标准库：

```bash
python -m pytest -q
PYTHONPATH=src python -m lightprism budget
```

训练/推理侧依赖在 `pyproject.toml` 的 `train` extras 里，改词表或序列规范的改动**不需要**装。

## 改这几样东西时必须做的事

| 改动 | 必须同步 |
|---|---|
| 新增/删除/重排决策点或选项 | `TokenSpace.schema_version` 升版本 + `docs/序列规范.md` + `docs/词表预算.md` 重新核算 |
| 新增验证码 | `VALIDATION_CODES` + `DIAGNOSTIC_TO_VALIDATION` + 序列规范第三节 |
| 改 `<D>`/`<CODE>`/`<V>` 语法 | `docs/序列规范.md`（唯一事实源）+ `tests/test_sequence.py` |
| 改词表预算 | `DEFAULT_BUDGET` 合计必须仍等于 65536，且 `test_default_budget_sums_to_64k` 绿 |

## 硬规矩

1. **不静默兜底**：未登记的决策点/选项/验证码一律抛 `ParseError`，不要加 `default=` 参数。
2. **验证 token 的来源只有 `light ai check`**：任何"让模型自己判断验证结果"的改动都不接受。
3. **文档与代码冲突时以文档为准，然后改代码**；不许为了迁就实现去改规范。
4. 提交信息用中文，首行写清"改了什么 + 为什么"。

## 推送

三个远端（origin=Gitea / github / gitcode）用 `sh scripts/push-all.sh` 一次推齐并逐个复核，
细节见 [docs/远端与发布.md](docs/远端与发布.md)。**不要用 `git push | tail` 当复核。**

## 提交前自检

```bash
python -m pytest -q
PYTHONPATH=src python -m lightprism budget
PYTHONPATH=src python -m lightprism tokens --json > /dev/null
```
