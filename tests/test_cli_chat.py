# -*- coding: utf-8 -*-
"""T-0338：``lightprism chat`` 与 ``--format json`` 的 CLI 测试。

覆盖卡面四件事：

1. 参数解析：``chat --prompt`` / ``--max-tokens`` / ``--ckpt`` / ``--format json``；
2. ``--format json`` 的输出可被 ``json.loads``，且字段至少含 ``text`` / ``tokens`` / ``tok_s``；
3. 缺 checkpoint 语义：可读错误 + **非零退出**（chat 是推理入口，没模型就是失败）；
4. 空输出不许当成功：``ok`` 为 False，但推理完整跑完时退出码仍为 0。

**不依赖 torch 与真模型**：生成器通过 ``lightprism.demo.chat._load_generator``
注入假实现；缺 checkpoint 的用例在加载之前就返回，也不会碰 torch。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lightprism import cli
from lightprism.demo import chat as chat_mod

ROOT = Path(__file__).resolve().parents[1]


class FakeGenerator:
    """最小假生成器：只回固定文本，记录收到的参数。"""

    def __init__(self, text: str = "print(1 + 1)") -> None:
        self.text = text
        self.calls: list[dict] = []

    def generate(self, task: str, max_new_tokens: int | None = None):
        n = int(max_new_tokens or 0)
        self.calls.append({"task": task, "max_new_tokens": max_new_tokens})
        return {
            "ids": [0] * n,
            "text": self.text,
            "generated_tokens": n,
            "truncated": n > 0,
        }


def _write_dummy_ckpt(tmp_path: Path) -> Path:
    """假生成器不读 checkpoint；只要文件存在就能通过前置校验。"""
    ckpt = tmp_path / "fake-model.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    return ckpt


def _install_fake(monkeypatch, text: str = "print(1 + 1)") -> FakeGenerator:
    fake = FakeGenerator(text)
    monkeypatch.setattr(chat_mod, "_load_generator", lambda path, n: fake)
    return fake


def _run_main(argv: list[str], capsys) -> tuple[int, str, str]:
    rc = cli.main(argv)
    out, err = capsys.readouterr()
    return rc, out, err


# --------------------------------------------------------------------------- #
# 1. 参数解析
# --------------------------------------------------------------------------- #
def test_chat_parser_registers_prompt_and_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["chat", "--prompt", "写一个加法程序"])
    assert args.command == "chat"
    assert args.func is cli.cmd_chat
    assert args.prompt == "写一个加法程序"
    assert args.ckpt is None
    assert args.max_tokens == chat_mod.DEFAULT_MAX_TOKENS
    assert args.format == "text"


def test_chat_parser_accepts_max_tokens_and_ckpt():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["chat", "--prompt", "x", "--max-tokens", "8", "--ckpt", "model.pt"]
    )
    assert args.max_tokens == 8
    assert args.ckpt == "model.pt"


def test_prompt_is_required():
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["chat"])
    assert exc.value.code == 2


def test_format_json_works_before_and_after_subcommand():
    parser = cli.build_parser()
    after = parser.parse_args(["chat", "--prompt", "x", "--format", "json"])
    before = parser.parse_args(["--format", "json", "chat", "--prompt", "x"])
    assert after.format == "json"
    assert before.format == "json"


def test_legacy_json_flag_still_means_json_for_other_commands(capsys):
    rc, out, _ = _run_main(["--json", "tokens"], capsys)
    assert rc == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["decision_tokens"]


def test_format_json_is_equivalent_for_other_commands(capsys):
    rc, out, _ = _run_main(["tokens", "--format", "json"], capsys)
    assert rc == 0
    assert json.loads(out)["counts"]["decision"] > 0


# --------------------------------------------------------------------------- #
# 2. JSON 可解析与字段
# --------------------------------------------------------------------------- #
def test_chat_format_json_is_parseable(tmp_path, monkeypatch, capsys):
    fake = _install_fake(monkeypatch, text="print(1 + 1)")
    ckpt = _write_dummy_ckpt(tmp_path)

    rc, out, err = _run_main(
        ["chat", "--prompt", "写一个加法程序", "--ckpt", str(ckpt),
         "--max-tokens", "8", "--format", "json"],
        capsys,
    )
    assert rc == 0
    payload = json.loads(out.strip())  # 卡面判据：json.loads 必能解析
    assert payload["ok"] is True
    assert payload["text"] == "print(1 + 1)"
    assert payload["tokens"] == 8
    assert isinstance(payload["tok_s"], float)
    assert payload["tok_s"] >= 0.0
    assert payload["ckpt"].endswith("fake-model.pt")
    assert payload["max_tokens"] == 8
    assert payload["empty_output"] is False
    assert fake.calls == [{"task": "写一个加法程序", "max_new_tokens": 8}]


def test_chat_json_flag_alias_is_parseable(tmp_path, monkeypatch, capsys):
    _install_fake(monkeypatch)
    ckpt = _write_dummy_ckpt(tmp_path)

    rc, out, _ = _run_main(
        ["chat", "--prompt", "x", "--ckpt", str(ckpt), "--json"],
        capsys,
    )
    assert rc == 0
    assert json.loads(out.strip())["tokens"] == chat_mod.DEFAULT_MAX_TOKENS


def test_chat_text_mode_prints_generated_text(tmp_path, monkeypatch, capsys):
    _install_fake(monkeypatch, text="print(42)")
    ckpt = _write_dummy_ckpt(tmp_path)

    rc, out, _ = _run_main(
        ["chat", "--prompt", "写个程序", "--ckpt", str(ckpt), "--max-tokens", "4"],
        capsys,
    )
    assert rc == 0
    assert "print(42)" in out
    assert "tok/s" in out


# --------------------------------------------------------------------------- #
# 3. 缺 checkpoint：可读错误 + 非零退出
# --------------------------------------------------------------------------- #
def test_missing_checkpoint_text_mode_is_nonzero(tmp_path, capsys):
    missing = tmp_path / "nope.pt"
    rc, out, err = _run_main(
        ["chat", "--prompt", "x", "--ckpt", str(missing)], capsys
    )
    assert rc != 0
    assert "checkpoint 不存在" in err
    assert out.strip() == ""


def test_missing_checkpoint_json_mode_is_nonzero_and_parseable(tmp_path, capsys):
    missing = tmp_path / "nope.pt"
    rc, out, _ = _run_main(
        ["chat", "--prompt", "x", "--ckpt", str(missing), "--format", "json"],
        capsys,
    )
    assert rc != 0
    payload = json.loads(out.strip())
    assert payload["ok"] is False
    assert "checkpoint 不存在" in payload["error"]


def test_blank_prompt_is_rejected(tmp_path, monkeypatch, capsys):
    _install_fake(monkeypatch)
    ckpt = _write_dummy_ckpt(tmp_path)
    rc, out, _ = _run_main(
        ["chat", "--prompt", "   ", "--ckpt", str(ckpt), "--format", "json"],
        capsys,
    )
    assert rc != 0
    assert "--prompt 不能为空" in json.loads(out.strip())["error"]


# --------------------------------------------------------------------------- #
# 4. 空输出不许当成功
# --------------------------------------------------------------------------- #
def test_empty_output_is_not_reported_as_success(tmp_path, monkeypatch, capsys):
    _install_fake(monkeypatch, text=" " * 32)
    ckpt = _write_dummy_ckpt(tmp_path)

    rc, out, err = _run_main(
        ["chat", "--prompt", "x", "--ckpt", str(ckpt),
         "--max-tokens", "32", "--format", "json"],
        capsys,
    )
    # 推理完整跑完 → 退出码 0；但内容空 → ok=False，绝不冒充成功
    assert rc == 0
    payload = json.loads(out.strip())
    assert payload["ok"] is False
    assert payload["empty_output"] is True
    assert payload["visible_chars"] == 0
    assert payload["tokens"] == 32


def test_empty_output_text_mode_warns_on_stderr(tmp_path, monkeypatch, capsys):
    _install_fake(monkeypatch, text=" " * 8)
    ckpt = _write_dummy_ckpt(tmp_path)

    rc, out, err = _run_main(
        ["chat", "--prompt", "x", "--ckpt", str(ckpt), "--max-tokens", "8"],
        capsys,
    )
    assert rc == 0
    assert "未产出可见字符" in err
    assert "tok/s" in out  # 表头仍在，入口可用性可判


# --------------------------------------------------------------------------- #
# 5. 不因缺 torch 整体 skip：模块导入必须惰性
# --------------------------------------------------------------------------- #
def test_chat_module_does_not_import_torch_at_import_time():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import lightprism.demo.chat; "
         "print('torch' in sys.modules); print('ok')"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "False", "chat 模块顶层不应 import torch"
    assert lines[1] == "ok"


def test_default_checkpoint_is_relative_to_cwd(tmp_path, monkeypatch):
    """默认路径按当前工作目录解析：在不含 checkpoint 的目录里必须明确报错。"""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(chat_mod.ChatError) as exc:
        chat_mod.run_chat("x")
    assert "checkpoint 不存在" in str(exc.value)


def test_chat_help_exits_zero():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-m", "lightprism", "chat", "--help"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0
    assert "--prompt" in proc.stdout
    assert "--format" in proc.stdout
    assert "--max-tokens" in proc.stdout