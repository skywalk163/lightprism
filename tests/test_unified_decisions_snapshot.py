# -*- coding: utf-8 -*-
"""T-0493：``data/unified`` 决策快照必须与 ``data/runtime`` 真实日志对齐。

守卫目的：``--build-unified`` 曾经把 unified/decisions.jsonl 留在 29 行旧快照，
而 runtime 已补到 241 行（T-0370）。快照悄悄落后时，下游（T-0494 的
train_openi 接线）会拿到过期标签却毫无察觉，所以这里逐项对账：

1. 行数一致；
2. CRLF 归一后的内容 sha256 一致；
3. manifest 的 ``counts.decisions`` / ``decisions_snapshot`` 与实测一致；
4. manifest 的 vocab 段（size / control_span / segment_markers）来自实际词表；
5. manifest 的 err_tasks 条数与 unified 文件一致。

失败时的修复动作写进断言消息：重跑 ``--build-unified``。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ROOT / "data" / "unified"
RUNTIME = ROOT / "data" / "runtime"


def _lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _content_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _manifest() -> dict:
    path = UNIFIED / "manifest.json"
    assert path.is_file(), f"缺少 {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_unified_decisions_match_runtime_bytes():
    unified = UNIFIED / "decisions.jsonl"
    runtime = RUNTIME / "decisions.jsonl"
    assert unified.is_file() and runtime.is_file()
    assert _lines(unified) == _lines(runtime), (
        f"unified 决策快照 {_lines(unified)} 行 != runtime {_lines(runtime)} 行；"
        "跑 python -m lightprism.demo.train_demo --build-unified --build-only")
    assert _content_sha(unified) == _content_sha(runtime), (
        "unified 决策快照内容与 runtime 不一致；"
        "跑 python -m lightprism.demo.train_demo --build-unified --build-only")


def test_manifest_decisions_counts_and_snapshot():
    manifest = _manifest()
    unified = UNIFIED / "decisions.jsonl"
    runtime = RUNTIME / "decisions.jsonl"
    n = _lines(unified)

    assert manifest["counts"]["decisions"] == n, (
        f"manifest counts.decisions={manifest['counts']['decisions']} 与实测 {n} 不一致")
    snap = manifest.get("decisions_snapshot") or {}
    assert snap.get("lines") == n, f"decisions_snapshot.lines 不是 {n}"
    assert snap.get("sha256") == _content_sha(runtime), "decisions_snapshot.sha256 与 runtime 不一致"
    assert "对齐" in str(snap.get("note") or ""), (
        f"快照 note 未标注已对齐：{snap.get('note')!r}")


def test_manifest_vocab_section_derives_from_actual_vocab():
    manifest = _manifest()
    vocab = json.loads((UNIFIED / "vocab_4096.json").read_text(encoding="utf-8"))
    layout = vocab["control_layout"]
    markers = (vocab.get("segment_markers") or {}).get("ids") or {}

    assert manifest["vocab"]["size"] == vocab["size"] == 2902
    assert manifest["vocab"]["control_span"] == [layout["start"], layout["end"]]
    assert manifest["vocab"]["segment_markers"] == markers
    assert markers == {"<PY>": 2900, "<LIGHT>": 2901}


def test_manifest_err_tasks_matches_unified_file():
    manifest = _manifest()
    unified_err = UNIFIED / "err_tasks.jsonl"
    n = _lines(unified_err)

    assert manifest["counts"]["err_tasks"] == n, (
        f"manifest counts.err_tasks={manifest['counts']['err_tasks']} 与实测 {n} 不一致")
    split = manifest["counts"].get("err_tasks_split") or {}
    assert split.get("train", 0) + split.get("test", 0) == n
    assert manifest["sha256"]["err_tasks.jsonl"] == _content_sha(unified_err), (
        "manifest 的 err_tasks sha 与文件不一致")


def test_manifest_translation_sha_tracks_written_rows():
    manifest = _manifest()
    path = UNIFIED / "translation_tasks.jsonl"
    assert manifest["sha256"]["translation_tasks.jsonl"] == _content_sha(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert manifest["counts"]["translation_rows"] == len(rows)
    assert all(r.get("sequence_marked") for r in rows), "存在缺 sequence_marked 的行"
