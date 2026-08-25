import json

import pytest

from ati.pairwise import (
    build_pair_prompt,
    comparison_path,
    compare,
    load_comparisons,
    parse_verdict,
    resolve_outcome,
    run_pairwise,
    select_pairs,
)
from ati.runners.fake import FakeRunner
from ati.runstage import run_output_path


def verdict(winner):
    return json.dumps({"winner": winner, "reason": "clearer phrasing"})


def test_parse_verdict_reads_winner():
    assert parse_verdict(verdict("A")) == "A"
    assert parse_verdict(verdict("B")) == "B"
    assert parse_verdict(verdict("tie")) == "tie"


def test_parse_verdict_is_case_insensitive():
    assert parse_verdict(json.dumps({"winner": "a"})) == "A"


def test_parse_verdict_rejects_garbage():
    with pytest.raises(ValueError):
        parse_verdict("the first one I suppose")


def test_resolve_outcome_agrees():
    # forward: A=a wins. reverse: A=b, and B (which is a) wins.
    assert resolve_outcome("A", "B") == "a"
    assert resolve_outcome("B", "A") == "b"


def test_resolve_outcome_treats_a_flip_as_a_tie():
    """If swapping the order swaps the verdict, the judge has no real preference."""
    assert resolve_outcome("A", "A") == "tie"
    assert resolve_outcome("B", "B") == "tie"


def test_resolve_outcome_propagates_explicit_ties():
    assert resolve_outcome("tie", "tie") == "tie"
    assert resolve_outcome("A", "tie") == "tie"


def test_build_pair_prompt_contains_both_translations_and_source():
    p = build_pair_prompt("TEMPLATE", "SOURCE", "ALPHA", "BETA")
    assert "TEMPLATE" in p and "SOURCE" in p
    assert "ALPHA" in p and "BETA" in p


def test_build_pair_prompt_does_not_name_the_models():
    p = build_pair_prompt("T", "S", "ALPHA", "BETA")
    assert "gpt" not in p.lower() and "claude" not in p.lower()


def test_compare_runs_both_directions():
    runner = FakeRunner(responses=[verdict("A"), verdict("B")])
    forward, reverse, outcome = compare(runner, "T", "SRC", "TEXT_A", "TEXT_B")
    assert forward == "A"
    assert reverse == "B"
    assert outcome == "a"
    assert len(runner.calls) == 2


def test_compare_swaps_order_on_the_second_call():
    runner = FakeRunner(responses=[verdict("A"), verdict("B")])
    compare(runner, "T", "SRC", "AAA", "BBB")
    first_prompt = runner.calls[0][1]
    second_prompt = runner.calls[1][1]
    assert first_prompt.index("AAA") < first_prompt.index("BBB")
    assert second_prompt.index("BBB") < second_prompt.index("AAA")


def test_select_pairs_prefers_close_and_rarely_compared():
    models = ["a", "b", "c"]
    ratings = {"a": 1000.0, "b": 1005.0, "c": 1600.0}
    counts = {}
    pairs = select_pairs(models, ratings, counts, k=1)
    assert pairs == [("a", "b")]


def test_select_pairs_deprioritizes_already_compared_pairs():
    models = ["a", "b", "c"]
    ratings = {"a": 1000.0, "b": 1005.0, "c": 1010.0}
    counts = {("a", "b"): 20}
    pairs = select_pairs(models, ratings, counts, k=1)
    assert pairs != [("a", "b")]


def test_select_pairs_is_deterministic():
    models = ["a", "b", "c", "d"]
    ratings = {m: 1000.0 for m in models}
    assert select_pairs(models, ratings, {}, k=3) == select_pairs(
        models, ratings, {}, k=3
    )


def seed_run(tmp_path, model_id, text, passage_id="cn-01"):
    payload = {
        "model_id": model_id,
        "task_id": "t",
        "task_type": "passage",
        "passage_id": passage_id,
        "iteration": 0,
        "text": text,
        "source_chars": 10,
        "latency_s": 1.0,
        "tokens_in": 1,
        "tokens_out": 1,
        "cost_usd": 0.0,
        "refused": False,
        "unavailable": False,
        "error": None,
    }
    p = run_output_path(tmp_path / "runs", "r1", model_id, f"t.{passage_id}.0")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def pairwise_args(tmp_path, runner, **kw):
    args = dict(
        judge_runner=runner,
        run_id="r1",
        runs_root=tmp_path / "runs",
        pairwise_root=tmp_path / "pairwise",
        corpus_root=tmp_path / "corpus",
        passages={},
        template="TEMPLATE",
        models=["m1", "m2"],
        max_comparisons=10,
    )
    args.update(kw)
    return args


def test_run_pairwise_writes_one_file_per_comparison(tmp_path):
    seed_run(tmp_path, "m1", "alpha")
    seed_run(tmp_path, "m2", "beta")
    runner = FakeRunner(responses=[verdict("A"), verdict("B")])
    count = run_pairwise(**pairwise_args(tmp_path, runner))
    assert count == 1
    path = comparison_path(tmp_path / "pairwise", "r1", "m1", "m2", "cn-01")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["outcome"] == "a"
    assert data["forward"] == "A"
    assert data["reverse"] == "B"


def test_run_pairwise_skips_completed_comparisons(tmp_path):
    seed_run(tmp_path, "m1", "alpha")
    seed_run(tmp_path, "m2", "beta")
    run_pairwise(**pairwise_args(tmp_path, FakeRunner(responses=[verdict("A"), verdict("B")])))
    second = FakeRunner(responses=[verdict("B"), verdict("A")])
    count = run_pairwise(**pairwise_args(tmp_path, second))
    assert count == 0
    assert second.calls == []


def test_run_pairwise_stops_cleanly_on_judge_error(tmp_path):
    """A usage limit must lose at most the comparison in flight."""
    seed_run(tmp_path, "m1", "alpha")
    seed_run(tmp_path, "m2", "beta")
    runner = FakeRunner(error="quota exceeded")
    count = run_pairwise(**pairwise_args(tmp_path, runner))
    assert count == 0
    assert not comparison_path(
        tmp_path / "pairwise", "r1", "m1", "m2", "cn-01"
    ).exists()


def test_run_pairwise_respects_max_comparisons(tmp_path):
    for model in ("m1", "m2", "m3"):
        seed_run(tmp_path, model, f"text-{model}")
    runner = FakeRunner(responses=[verdict("A"), verdict("B")] * 10)
    count = run_pairwise(
        **pairwise_args(tmp_path, runner, models=["m1", "m2", "m3"], max_comparisons=2)
    )
    assert count == 2


def test_run_pairwise_ignores_models_with_no_output(tmp_path):
    seed_run(tmp_path, "m1", "alpha")
    runner = FakeRunner(responses=[verdict("A"), verdict("B")])
    count = run_pairwise(**pairwise_args(tmp_path, runner))
    assert count == 0


def test_load_comparisons_reads_back_outcomes(tmp_path):
    seed_run(tmp_path, "m1", "alpha")
    seed_run(tmp_path, "m2", "beta")
    run_pairwise(**pairwise_args(tmp_path, FakeRunner(responses=[verdict("A"), verdict("B")])))
    comparisons = load_comparisons(tmp_path / "pairwise", "r1")
    assert comparisons == [("m1", "m2", "a")]
