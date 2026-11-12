# T-0335 · LightPrism demo 的一键门面
#
#   make demo                                   # 一条命令跑通（无 checkpoint 时给可读提示）
#   make demo PROMPT="报告磁盘使用率" FORMAT=json
#   make bench                                  # 速度基准：tok/s 与峰值内存
#   make check                                  # pytest + 4096 词表校验 + 64K 预算核算
#   make clean                                  # 清运行产物
#
# 口径说明：
#   * 所有 Python 命令都显式带 PYTHONPATH=src，不依赖 `pip install -e .`；
#   * PYTHON 可覆盖（Windows 常需 PYTHON=C:/Python314/python.exe）；
#   * `make demo` 在没有 checkpoint 时**有意返回 0** 并打印提示 —— 那是预期状态，
#     不是假装生成成功。JSON 里的 ok=false / error=checkpoint_missing 才是机器判据。
#     有 checkpoint 时生成失败会非零退出，绝不吞异常。
#   * `make bench` 的口径写死在 scripts/bench.py（单线程 / 贪心 / prompt 32 /
#     预热 10 个不计入），本文件只传 --config 与 --tokens，不改口径。

PYTHON        ?= python3
PROMPT        ?= 把这三个文件按大小排序
FORMAT        ?= text
CKPT          ?= checkpoints/demo-3m/model.pt
BENCH_CONFIG  ?= configs/demo-3m.json
BENCH_TOKENS  ?= 100
TRAIN_CONFIG  ?= configs/demo-3m.json
TRAIN_STEPS   ?= 50

.DEFAULT_GOAL := help
.PHONY: help demo bench check clean train

help:
	@echo "LightPrism make 目标（T-0335）"
	@echo "  make demo    [PROMPT=...] [FORMAT=text|json] [CKPT=...]  一键 demo"
	@echo "  make bench   [BENCH_TOKENS=100]                           速度基准"
	@echo "  make check                                                pytest + 词表 + 预算"
	@echo "  make train   [TRAIN_STEPS=50]                             训练 demo checkpoint"
	@echo "  make clean                                                清 outputs/ 与 __pycache__"
	@echo "  PYTHON=... 可指定解释器；详见 docs/Demo使用.md"

demo:
	@PYTHON="$(PYTHON)" bash scripts/demo.sh --ckpt "$(CKPT)" --format "$(FORMAT)" "$(PROMPT)"

train:
	PYTHONPATH=src $(PYTHON) -m lightprism.demo.train_demo --config $(TRAIN_CONFIG) --steps $(TRAIN_STEPS)

bench:
	@$(PYTHON) -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('resource') else 1)" \
		|| { echo "make bench 需要 POSIX 平台：scripts/bench.py 顶层依赖标准库 resource（Windows 无此模块）。请在 Linux / FreeBSD 上运行；已有实测数字见 reports/推理性能基线.md。" >&2; exit 1; }
	PYTHONPATH=src $(PYTHON) scripts/bench.py --config $(BENCH_CONFIG) --tokens $(BENCH_TOKENS)

check:
	PYTHONPATH=src $(PYTHON) -m pytest -q
	PYTHONPATH=src $(PYTHON) -m lightprism.demo.vocab_build --check data/vocab_4096.json
	PYTHONPATH=src $(PYTHON) -m lightprism budget

clean:
	rm -rf outputs/ .pytest_cache/
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
