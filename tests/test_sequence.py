# -*- coding: utf-8 -*-
"""序列结构、解析、预算、映射的一致性测试。"""

from __future__ import annotations

import json

import pytest

from lightprism import (
    DEFAULT_BUDGET,
    DEFAULT_DECISION_POINTS,
    LIGHT_L0_CORE,
    VOCAB_TOTAL,
    DecisionToken,
    ParseError,
    TokenSpace,
    VocabBudget,
    ValidationToken,
    check_l0_atomicity,
    decision_density,
    map_diagnostics,
    parse_sequence,
    render_sequence,
)


# ---- 决策 / 验证 token ------------------------------------------------
def test_decision_point_options_are_closed_and_unique():
    for point in DEFAULT_DECISION_POINTS:
        assert len(set(point.options)) == len(point.options)
        assert all(point.options)


def test_unknown_option_is_rejected_not_defaulted():
    space = TokenSpace()
    with pytest.raises(ParseError):
        space.decision("EDIT", "MAGIC")
    with pytest.raises(ParseError):
        space.decision("NOPE", "FR")


def test_decision_token_text_is_namespaced():
    assert TokenSpace().decision("EDIT", "FR").text == "<D:EDIT=FR>"
    assert str(ValidationToken("PASS")) == "<V:PASS>"


def test_unknown_validation_code_is_rejected():
    with pytest.raises(ParseError):
        ValidationToken("ERR=NOT_A_CODE")


# ---- 预算 -------------------------------------------------------------
def test_default_budget_sums_to_64k():
    budget = VocabBudget()
    assert budget.total == VOCAB_TOTAL == 65536
    assert budget.total_allocated() == VOCAB_TOTAL
    assert budget.remaining() == 0
    assert budget.problems(TokenSpace()) == []


def test_budget_flags_wrong_total_and_out_of_range_slot():
    # 少分 1 个额度 → 必须报出来，不允许"悄悄空着"
    shifted = VocabBudget(alloc={"reserved": 1023})
    assert any("≠ 词表总量" in p for p in shifted.problems())
    # 突破槽位上限 → 也必须报出来（不只是总额对得上就行）
    over = VocabBudget(alloc={"decision": 600, "chinese_bpe": 30720})
    assert any("槽位 decision=600" in p for p in over.problems())


def test_budget_rejects_unknown_slot():
    with pytest.raises(ParseError):
        VocabBudget(alloc={"nope": 1})


def test_l0_core_is_one_token_per_char():
    assert len(LIGHT_L0_CORE) == 44
    assert len(set(LIGHT_L0_CORE)) == 44
    assert check_l0_atomicity(list(LIGHT_L0_CORE)) == []


def test_l0_atomicity_catches_fused_characters():
    # "若则" 被粘成两个 token 时不算违规（各 2 字但都不含 L0 字）——
    # 但 "若果" / "设置" 这类把 L0 字并入多字 token 的情况必须抓住。
    issues = check_l0_atomicity(["若果", "设置", "列表", "若", "设"])
    assert len(issues) == 2


# ---- 序列解析 ---------------------------------------------------------
SAMPLE = """<TASK>把测试文件按大小排序并输出前10名</TASK>
<D:PLAN=DECOMPOSE>
<D:EDIT=FR>
<CODE>
设 文件列表 为 列目录(".")。
</CODE>
<V:PASS>
<D:NEXT=TEST>
<CODE>
印 文件列表。
</CODE>
<V:ERR=SYNTAX>
<D:FIX=ADD_TYPE>
"""


def test_parse_and_render_roundtrip():
    parsed = parse_sequence(SAMPLE)
    assert parsed.task.startswith("把测试文件")
    assert [t.text for t in parsed.decisions] == [
        "<D:PLAN=DECOMPOSE>",
        "<D:EDIT=FR>",
        "<D:NEXT=TEST>",
        "<D:FIX=ADD_TYPE>",
    ]
    assert [t.text for t in parsed.validations] == ["<V:PASS>", "<V:ERR=SYNTAX>"]
    assert parsed.validate() == []
    assert parse_sequence(parsed.render()).render() == parsed.render()


def test_validation_token_without_code_is_flagged():
    parsed = parse_sequence("<TASK>t</TASK>\n<D:EDIT=FR>\n<V:PASS>\n"
                            "<CODE>\n设 甲 为 1。\n</CODE>\n<V:PASS>\n")
    problems = parsed.validate()
    assert any("没有 <CODE> 段" in p for p in problems)


def test_two_decisions_in_a_row_allowed_by_default_flagged_on_demand():
    # 方案文档 §2.1 的示例本身就是 PLAN 紧跟 EDIT —— 复合决策合法，默认不判
    parsed = parse_sequence("<TASK>t</TASK>\n<D:PLAN=DECOMPOSE>\n<D:EDIT=FR>\n"
                            "<CODE>\n设 甲 为 1。\n</CODE>\n<V:PASS>\n")
    assert parsed.validate() == []
    assert any("连续决策 token" in p
               for p in parsed.validate(forbid_consecutive_decisions=True))


def test_bare_text_between_tokens_is_an_error():
    with pytest.raises(ParseError):
        parse_sequence("<TASK>t</TASK>\n这里不该有文本\n<D:EDIT=FR>\n")


def test_unregistered_option_in_sequence_is_an_error():
    # 方案文档 §2.1 的示例用了 <D:EDIT=NEW_FILE>，但 §4.2 首批表只有 FR/RW/PATCH。
    # 本骨架的选择是**明确报错**，而不是静默接受一个词表里没有的 token。
    with pytest.raises(ParseError):
        parse_sequence("<TASK>t</TASK>\n<D:EDIT=NEW_FILE>\n")
    # 合法序列仍能正常通过
    parsed = parse_sequence("<TASK>t</TASK>\n<CODE>\n设 甲 为 1。\n</CODE>\n<V:PASS>\n")
    assert parsed.validate() == []


def test_decision_density_counts_decisions_only():
    parsed = parse_sequence(SAMPLE)
    # 9 个序列项（Task + 4 决策 + 2 代码 + 2 验证）里有 4 个决策
    assert decision_density(parsed.items) == pytest.approx(4 / 9)


# ---- light ai check → 验证 token --------------------------------------
def test_pass_and_warning_mapping():
    assert map_diagnostics([]).code == "PASS"
    assert map_diagnostics([{"code": "L1005", "severity": "warning"}]).code == "WARN=NON_CANONICAL"


def test_first_error_wins_and_unknown_codes_are_visible():
    diags = [
        {"code": "L1005", "severity": "warning"},
        {"code": "L2002", "severity": "error"},
        {"code": "L1001", "severity": "error"},
    ]
    assert map_diagnostics(diags).code == "ERR=MISSING_TYPE"
    unknown = map_diagnostics([{"code": "L9999", "severity": "error"}])
    assert unknown.code == "ERR=CHECKER_FAIL"
    assert "L9999" in unknown.detail


def test_check_json_end_to_end(tmp_path):
    from lightprism.cli import main

    payload = {
        "schema_version": "1",
        "ok": False,
        "file": "demo.light",
        "diagnostics": [{"code": "L4002", "severity": "error", "message": "用英文模块名"}],
    }
    path = tmp_path / "check.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert main(["--json", "map-check", str(path)]) == 0


# ---- CLI 参数位置（fresh-clone 自检抓到的回归）-------------------------
def test_json_flag_works_before_and_after_subcommand(capsys):
    from lightprism.cli import main

    for argv in (["budget", "--json"], ["--json", "budget"]):
        capsys.readouterr()
        assert main(argv) == 0
        out = capsys.readouterr().out
        # 两种写法都必须真的吐 JSON，而不是被子命令默认值抹掉后回落到文本
        assert out.lstrip().startswith("{"), (argv, out[:80])
        assert '"total_allocated": 65536' in out

    capsys.readouterr()
    assert main(["budget"]) == 0
    assert capsys.readouterr().out.startswith("词表预算")


def test_json_flag_works_on_every_subcommand(capsys):
    from lightprism.cli import main

    assert main(["tokens", "--json"]) == 0
    assert capsys.readouterr().out.lstrip().startswith("{")
    assert main(["parse", "examples/demo.prism.txt", "--json"]) == 0
    assert capsys.readouterr().out.lstrip().startswith("{")


# ---- T-0371：段标记词表预算与序列 --------------------------------------
def test_segment_markers_are_carved_from_reserved_with_fixed_ids():
    from lightprism.vocab import (
        SEGMENT_MARKERS,
        SEGMENT_TOKEN_IDS,
        slot_offsets,
    )

    budget = VocabBudget()
    assert tuple(SEGMENT_MARKERS) == ("<PY>", "<LIGHT>")
    ids = budget.segment_token_ids()
    assert ids == {"<PY>": 64000, "<LIGHT>": 64001}
    assert ids == dict(SEGMENT_TOKEN_IDS)
    # 段标记从 reserved 头部切出，总额仍精确 65536
    offs = slot_offsets(budget)
    assert offs["reserved"] == 64000
    assert budget.total_allocated() == VOCAB_TOTAL == 65536
    assert budget.problems(TokenSpace()) == []
    assert budget.segment_id_range() == (64000, 64002)


def test_segment_tokens_do_not_consume_decision_or_validation_quota():
    from lightprism.vocab import slot_offsets

    budget = VocabBudget()
    space = TokenSpace()
    offs = slot_offsets(budget)
    seg_ids = set(budget.segment_token_ids().values())
    for name in ("decision", "validation"):
        lo = offs[name]
        hi = lo + budget.value_of(name)
        assert not (seg_ids & set(range(lo, hi))), f"段标记占用了 {name} 额度"
        assert budget.value_of(name) >= len(
            space.decision_tokens() if name == "decision" else space.validation_tokens())
    # 决策 token 占比仍 ≤ 2%，没有被两个新 token 挤占
    assert len(space.decision_tokens()) / VOCAB_TOTAL <= 0.02


MARKED_SAMPLE = """<TASK>把 Python 翻译成光明</TASK>
<PY>
x = 1
print(x)
<LIGHT>
设 x 为 1。
印 x。
<V:PASS>
"""


def test_marked_translation_sequence_parse_render_roundtrip():
    from lightprism.sequence import MarkedSegment

    parsed = parse_sequence(MARKED_SAMPLE)
    assert [i.lang for i in parsed.marked_segments] == ["py", "light"]
    assert isinstance(parsed.marked_segments[0], MarkedSegment)
    assert parsed.segment("py") == "x = 1\nprint(x)"
    assert parsed.segment("light") == "设 x 为 1。\n印 x。"
    assert parsed.validate() == []
    assert parsed.render() == MARKED_SAMPLE
    assert parsed.counts()["py_segment"] == 1
    assert parsed.counts()["light_segment"] == 1


def test_marked_segment_validation_has_teeth():
    # PY 后没有 LIGHT
    p1 = parse_sequence("<TASK>t</TASK>\n<PY>\nx = 1\n")
    assert any("缺少 `<LIGHT>`" in p for p in p1.validate())
    # LIGHT 先于 PY（PY 在后面，属于段乱序；单独的 light_only 序列合法）
    p2 = parse_sequence("<TASK>t</TASK>\n<LIGHT>\n设 x 为 1。\n<PY>\nx = 1\n<V:PASS>\n")
    assert any("`<LIGHT>` 段出现在 `<PY>` 之前" in p for p in p2.validate())
    standalone = parse_sequence("<TASK>t</TASK>\n<LIGHT>\n设 x 为 1。\n<V:PASS>\n")
    assert standalone.validate() == []
    # 空光明段
    p3 = parse_sequence("<TASK>t</TASK>\n<PY>\nx = 1\n<LIGHT>\n<V:PASS>\n")
    assert any("光明段为空" in p for p in p3.validate())
    # 验证 token 可以跟在 LIGHT 段之后（翻译序列没有 <CODE>）
    p4 = parse_sequence("<TASK>t</TASK>\n<PY>\nx = 1\n<LIGHT>\n设 x 为 1。\n<V:PASS>\n")
    assert p4.validate() == []


def test_legacy_dual_code_sequence_still_parses():
    legacy = ("<TASK>翻译</TASK>\n<CODE>\nx = 1\n</CODE>\n"
              "<CODE>\n设 x 为 1。\n</CODE>\n<V:PASS>\n")
    parsed = parse_sequence(legacy)
    assert parsed.validate() == []
    assert parsed.segment("light") == ""
    assert len(parsed.codes) == 2
