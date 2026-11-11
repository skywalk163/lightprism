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
