from ati.config import ModelConfig
from ati.fingerprint import (
    input_fingerprints,
    judge_fingerprint,
    judge_route,
    pairwise_prompt_frame,
    rubric_prompt_frame,
)
from ati.runners.agy import NO_TOOLS_PREAMBLE

ENDPOINT = "https://example.invalid/v1"


def _agy(schema=None):
    return ModelConfig(id="gemini-judge", runner="agy", model="gemini-x", json_schema=schema)


def test_the_same_judge_and_prompt_give_the_same_fingerprint():
    route = judge_route("gemini-judge", _agy(), ENDPOINT)
    frame = rubric_prompt_frame("TEMPLATE")
    assert judge_fingerprint("gemini-judge", route, frame) == judge_fingerprint(
        "gemini-judge", judge_route("gemini-judge", _agy(), ENDPOINT), frame
    )


def test_a_prompt_edit_changes_the_fingerprint():
    route = judge_route("gemini-judge", _agy(), ENDPOINT)
    assert judge_fingerprint("j", route, rubric_prompt_frame("A")) != judge_fingerprint(
        "j", route, rubric_prompt_frame("B")
    )


def test_a_different_model_behind_the_same_id_changes_the_fingerprint():
    a = judge_route("j", _agy(), ENDPOINT)
    b = judge_route(
        "j", ModelConfig(id="j", runner="agy", model="gemini-y"), ENDPOINT
    )
    assert judge_fingerprint("j", a, "F") != judge_fingerprint("j", b, "F")


def test_the_antigravity_route_carries_the_no_tools_line_and_the_schema(tmp_path):
    schema = tmp_path / "s.json"
    schema.write_text('{"a": 1}', encoding="utf-8")
    route = judge_route("j", _agy(), ENDPOINT, json_schema=str(schema))
    assert route["preamble"] == NO_TOOLS_PREAMBLE
    assert route["schema"] == '{"a": 1}'
    before = judge_fingerprint("j", route, "F")
    schema.write_text('{"a": 2}', encoding="utf-8")
    after = judge_fingerprint("j", judge_route("j", _agy(), ENDPOINT, str(schema)), "F")
    assert before != after


def test_an_undeclared_judge_is_a_plain_call_at_temperature_zero():
    route = judge_route("vendor/judge", None, ENDPOINT)
    assert route == {
        "runner": "api",
        "model": "vendor/judge",
        "endpoint": ENDPOINT,
        "temperature": 0.0,
    }


def test_timeouts_do_not_change_the_fingerprint():
    """How long to wait changes whether an answer arrives, not what it says."""
    fast = ModelConfig(id="j", runner="agy", model="gemini-x", timeout_s=60)
    slow = ModelConfig(id="j", runner="agy", model="gemini-x", timeout_s=900)
    assert judge_route("j", fast, ENDPOINT) == judge_route("j", slow, ENDPOINT)


def test_prompt_frames_hold_placeholders_not_text():
    frame = pairwise_prompt_frame("TEMPLATE")
    assert "{translation_a}" in frame and "{translation_b}" in frame
    rubric = rubric_prompt_frame("TEMPLATE")
    assert "{translation}" in rubric and "term_table" in rubric


def test_input_fingerprints_mark_unreadable_files(tmp_path):
    present = tmp_path / "a.txt"
    present.write_text("x", encoding="utf-8")
    prints = input_fingerprints({"a": present, "b": tmp_path / "missing"})
    assert len(prints["a"]) == 64
    assert prints["b"] is None
