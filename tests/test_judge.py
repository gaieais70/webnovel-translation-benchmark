import json

import pytest

from ati.config import AXES
from ati.judge import (
    Judgment,
    JudgeSchemaError,
    build_judge_prompt,
    judge_with_retry,
    parse_judgment,
    rationale_quote_violations,
)
from ati.runners.fake import FakeRunner


def valid_payload(**overrides):
    payload = {
        "scores": {axis: 4 for axis in AXES},
        "rationales": {axis: "reads well" for axis in AXES},
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parse_judgment_reads_scores_and_rationales():
    j = parse_judgment(valid_payload())
    assert j.axes["accuracy"] == 4
    assert j.rationales["fluency"] == "reads well"
    assert j.term_table is None


def test_parse_judgment_tolerates_markdown_fenced_json():
    raw = "Here you go:\n```json\n" + valid_payload() + "\n```\n"
    assert parse_judgment(raw).axes["register"] == 4


def test_parse_judgment_rejects_missing_axis():
    payload = json.loads(valid_payload())
    del payload["scores"]["fidelity"]
    with pytest.raises(JudgeSchemaError, match="missing score for fidelity"):
        parse_judgment(json.dumps(payload))


def test_parse_judgment_rejects_out_of_range_score():
    payload = json.loads(valid_payload())
    payload["scores"]["accuracy"] = 9
    with pytest.raises(JudgeSchemaError, match="out of range"):
        parse_judgment(json.dumps(payload))


def test_parse_judgment_rejects_non_json():
    with pytest.raises(JudgeSchemaError, match="no JSON object"):
        parse_judgment("I'd rather not score this.")


def test_parse_judgment_requires_term_table_when_expected():
    with pytest.raises(JudgeSchemaError, match="term_table"):
        parse_judgment(valid_payload(), expect_term_table=True)


def test_parse_judgment_reads_term_table():
    raw = valid_payload(term_table={"云澈": "Yun Che", "元气": "Profound Energy"})
    j = parse_judgment(raw, expect_term_table=True)
    assert j.term_table == {"云澈": "Yun Che", "元气": "Profound Energy"}


def test_build_judge_prompt_includes_source_translation_and_reference():
    prompt = build_judge_prompt(
        template="TEMPLATE",
        source_text="SOURCE",
        translation="TRANSLATION",
        reference="REFERENCE",
        expect_term_table=False,
    )
    assert "TEMPLATE" in prompt
    assert "SOURCE" in prompt
    assert "TRANSLATION" in prompt
    assert "REFERENCE" in prompt


def test_build_judge_prompt_omits_reference_section_when_absent():
    prompt = build_judge_prompt("T", "S", "X", None, False)
    assert "REFERENCE TRANSLATION" not in prompt


def test_build_judge_prompt_requests_term_table_for_consistency_tasks():
    prompt = build_judge_prompt("T", "S", "X", None, expect_term_table=True)
    assert "term_table" in prompt


def test_judge_with_retry_succeeds_after_one_bad_response():
    runner = FakeRunner(responses=["garbage", valid_payload()])
    j = judge_with_retry(runner, "prompt", expect_term_table=False)
    assert j.axes["accuracy"] == 4
    assert len(runner.calls) == 2


def test_judge_with_retry_raises_after_three_failures():
    runner = FakeRunner(responses=["a", "b", "c"])
    with pytest.raises(JudgeSchemaError):
        judge_with_retry(runner, "prompt", expect_term_table=False)
    assert len(runner.calls) == 3


def test_rationale_quote_violations_flags_long_quoted_spans():
    j = Judgment(
        axes={axis: 3 for axis in AXES},
        rationales={
            **{axis: "fine" for axis in AXES},
            "accuracy": 'renders it as "one two three four five six seven eight nine"',
        },
        term_table=None,
    )
    assert rationale_quote_violations(j) == ["accuracy"]


def test_rationale_quote_violations_allows_short_quotes():
    j = Judgment(
        axes={axis: 3 for axis in AXES},
        rationales={**{axis: "fine" for axis in AXES}, "accuracy": 'uses "sect elder"'},
        term_table=None,
    )
    assert rationale_quote_violations(j) == []
