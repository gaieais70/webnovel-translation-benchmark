import json

import pytest

from ati.config import AXES
from ati.report import (
    ANCHOR_DRIFT_THRESHOLD,
    ModelResult,
    aggregate,
    anchor_drift,
    fmt,
    is_comparable,
    render_markdown,
)
from ati.judgestage import judgment_output_path
from ati.runstage import run_output_path

WEIGHTS = {
    "accuracy": 0.25,
    "terminology": 0.20,
    "register": 0.15,
    "fluency": 0.15,
    "fidelity": 0.15,
    "sanitization": 0.10,
}


def seed(tmp_path, model_id, instance_id, *, scores=None, run=None, judgment=None):
    run_payload = {
        "model_id": model_id,
        "task_id": "t",
        "task_type": "passage",
        "passage_id": "cn-01",
        "iteration": 0,
        "text": "he drew his sword and cut the man down",
        "source_chars": 800,
        "latency_s": 8.0,
        "tokens_in": 500,
        "tokens_out": 400,
        "cost_usd": 0.004,
        "refused": False,
        "unavailable": False,
        "error": None,
    }
    run_payload.update(run or {})
    rp = run_output_path(tmp_path / "runs", "r1", model_id, instance_id)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(run_payload), encoding="utf-8")

    judgment_payload = {
        "model_id": model_id,
        "task_id": run_payload["task_id"],
        "task_type": run_payload["task_type"],
        "passage_id": run_payload["passage_id"],
        "iteration": run_payload["iteration"],
        "judge_id": "judge",
        "scores": scores if scores is not None else {axis: 4 for axis in AXES},
        "rationales": {axis: "ok" for axis in AXES},
        "term_table": None,
        "refused": run_payload["refused"],
        "skipped": False,
        "judge_error": None,
        "quote_violations": [],
    }
    judgment_payload.update(judgment or {})
    jp = judgment_output_path(tmp_path / "judgments", "r1", model_id, instance_id)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(judgment_payload), encoding="utf-8")


def run_aggregate(tmp_path, anchors=None):
    return aggregate(
        run_id="r1",
        runs_root=tmp_path / "runs",
        judgments_root=tmp_path / "judgments",
        weights=WEIGHTS,
        slop_list=["a testament to"],
        anchors=anchors or {},
    )


def test_fmt_renders_none_as_na():
    assert fmt(None) == "N/A"
    assert fmt(3.14159, 2) == "3.14"


def test_aggregate_computes_composite_from_judgments(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    assert len(results) == 1
    assert results[0].model_id == "m1"
    assert results[0].composite == pytest.approx(75.0)


def test_aggregate_averages_iterations_and_reports_stdev(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", scores={axis: 5 for axis in AXES})
    seed(
        tmp_path,
        "m1",
        "t.cn-01.1",
        scores={axis: 3 for axis in AXES},
        run={"iteration": 1},
        judgment={"iteration": 1},
    )
    results = run_aggregate(tmp_path)
    assert results[0].axis_means["accuracy"] == pytest.approx(4.0)
    assert results[0].axis_stdevs["accuracy"] == pytest.approx(1.0)


def test_aggregate_excludes_judge_errored_axes_and_counts_them(tmp_path):
    seed(
        tmp_path,
        "m1",
        "t.cn-01.0",
        judgment={"scores": {}, "judge_error": "schema failure"},
    )
    seed(tmp_path, "m1", "t.cn-01.1", run={"iteration": 1}, judgment={"iteration": 1})
    results = run_aggregate(tmp_path)
    assert results[0].judge_errors == 1
    assert results[0].composite == pytest.approx(75.0)


def test_aggregate_reports_cost_none_for_subscription_models(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", run={"cost_usd": None})
    results = run_aggregate(tmp_path)
    assert results[0].cost_per_1k is None
    assert results[0].value is None


def test_aggregate_never_treats_unknown_cost_as_free(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", run={"cost_usd": None})
    results = run_aggregate(tmp_path)
    assert results[0].cost_per_1k != 0.0


def test_aggregate_computes_refusal_rate(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", run={"refused": True}, judgment={"refused": True, "scores": {}})
    seed(tmp_path, "m1", "t.cn-01.1", run={"iteration": 1}, judgment={"iteration": 1})
    results = run_aggregate(tmp_path)
    assert results[0].refusal_rate == pytest.approx(50.0)


def test_aggregate_marks_model_incomplete_above_failure_threshold(tmp_path):
    for i in range(4):
        seed(
            tmp_path,
            "m1",
            f"t.cn-01.{i}",
            run={"iteration": i, "error": "provider down", "text": ""},
            judgment={"iteration": i, "skipped": True, "scores": {}},
        )
    seed(tmp_path, "m1", "t.cn-01.9", run={"iteration": 9}, judgment={"iteration": 9})
    results = run_aggregate(tmp_path)
    assert results[0].failed_tasks == 4
    assert results[0].incomplete is True


def test_aggregate_leaves_healthy_model_complete(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    assert run_aggregate(tmp_path)[0].incomplete is False


def test_aggregate_sorts_by_composite_descending(tmp_path):
    seed(tmp_path, "low", "t.cn-01.0", scores={axis: 2 for axis in AXES})
    seed(tmp_path, "high", "t.cn-01.0", scores={axis: 5 for axis in AXES})
    assert [r.model_id for r in run_aggregate(tmp_path)] == ["high", "low"]


def test_aggregate_labels_anchor_models(tmp_path):
    seed(tmp_path, "anchor-weak", "t.cn-01.0")
    results = run_aggregate(tmp_path, anchors={"anchor-weak": "weak"})
    assert results[0].anchor == "weak"


def test_anchor_drift_reports_difference_from_previous_run():
    results = [
        ModelResult(
            model_id="a",
            composite=70.0,
            axis_means={},
            axis_stdevs={},
            refusal_rate=0.0,
            cost_per_1k=None,
            speed_per_1k=None,
            value=None,
            slop=0.0,
            repetition=0.0,
            judge_errors=0,
            failed_tasks=0,
            total_tasks=1,
            anchor="strong",
            incomplete=False,
        )
    ]
    assert anchor_drift(results, {"a": 62.0}) == {"a": 8.0}


def test_is_comparable_is_false_above_threshold():
    assert is_comparable({"a": ANCHOR_DRIFT_THRESHOLD + 0.1}) is False
    assert is_comparable({"a": 1.0}) is True
    assert is_comparable({}) is True


def test_render_markdown_contains_ranking_and_na_for_unknown_cost(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", run={"cost_usd": None})
    results = run_aggregate(tmp_path)
    md = render_markdown(
        run_id="r1",
        results=results,
        drift={},
        comparable=True,
        metadata={"judge_id": "judge", "date": "2026-08-23"},
    )
    assert "m1" in md
    assert "N/A" in md
    assert "Artificial Translation Index" in md
    assert "judge" in md


def test_render_markdown_warns_when_not_comparable(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown(
        run_id="r1",
        results=results,
        drift={"anchor-strong": 9.0},
        comparable=False,
        metadata={"judge_id": "judge", "date": "2026-08-23"},
    )
    assert "not comparable" in md.lower()


def test_render_markdown_lists_incomplete_models_separately(tmp_path):
    for i in range(4):
        seed(
            tmp_path,
            "broken",
            f"t.cn-01.{i}",
            run={"iteration": i, "error": "down", "text": ""},
            judgment={"iteration": i, "skipped": True, "scores": {}},
        )
    seed(tmp_path, "broken", "t.cn-01.9", run={"iteration": 9}, judgment={"iteration": 9})
    seed(tmp_path, "healthy", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown("r1", results, {}, True, {"judge_id": "j", "date": "d"})
    assert "Incomplete" in md
    assert "broken" in md


def test_fmt_cost_labels_subscription_zero():
    from ati.report import fmt_cost

    assert fmt_cost(None) == "N/A"
    assert fmt_cost(0.0) == "sub"
    assert fmt_cost(0.0031) == "0.0031"


def test_render_markdown_includes_pairwise_ratings_when_present(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown(
        run_id="r1",
        results=results,
        drift={},
        comparable=True,
        metadata={"judge_id": "gemini-judge", "date": "2026-08-25"},
        pairwise={"m1": 1240.0},
        pairwise_ci={"m1": (1180.0, 1300.0)},
    )
    assert "Head-to-head" in md
    assert "1240" in md
    assert "1180" in md


def test_render_markdown_omits_pairwise_section_when_absent(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown(
        "r1", results, {}, True, {"judge_id": "j", "date": "d"}
    )
    assert "Head-to-head" not in md


def test_render_markdown_warns_when_intervals_overlap(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    seed(tmp_path, "m2", "t.cn-01.0", scores={axis: 3 for axis in AXES})
    results = run_aggregate(tmp_path)
    md = render_markdown(
        "r1",
        results,
        {},
        True,
        {"judge_id": "j", "date": "d"},
        pairwise={"m1": 1200.0, "m2": 1190.0},
        pairwise_ci={"m1": (1100.0, 1300.0), "m2": (1090.0, 1290.0)},
    )
    assert "overlap" in md.lower()


def test_render_markdown_labels_a_pinned_anchor_interval(tmp_path):
    """A zero-width interval is the pinned scale, not certainty about the anchor."""
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown(
        "r1",
        results,
        {},
        True,
        {"judge_id": "j", "date": "d"},
        pairwise={"m1": 1000.0},
        pairwise_ci={"m1": (1000.0, 1000.0)},
        pairwise_anchor="m1",
    )
    assert "pinned" in md
    assert "1000 – 1000" not in md


def test_zero_width_interval_without_an_anchor_says_too_few(tmp_path):
    """Zero width for a non-anchor means every resample drew the same handful
    of comparisons, not that the rating is certain."""
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown(
        "r1",
        results,
        {},
        True,
        {"judge_id": "j", "date": "d"},
        pairwise={"m1": 1191.0},
        pairwise_ci={"m1": (1191.0, 1191.0)},
        pairwise_anchor="other-model",
    )
    assert "too few comparisons" in md
    assert "pinned" not in md


def test_notes_distinguish_subscription_from_unknown_cost(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown("r1", results, {}, True, {"judge_id": "j", "date": "d"})
    assert "`sub`" in md
    assert "billed" in md.lower()


def test_the_ranking_shows_how_many_tasks_each_model_was_scored_on():
    from ati.report import ModelResult, render_markdown

    r = ModelResult(
        model_id="m",
        composite=80.0,
        axis_means={},
        axis_stdevs={},
        refusal_rate=0.0,
        cost_per_1k=None,
        speed_per_1k=1.0,
        value=None,
        slop=0.1,
        repetition=0.1,
        judge_errors=0,
        failed_tasks=2,
        total_tasks=11,
        anchor=None,
        incomplete=False,
    )
    md = render_markdown("r", [r], {}, True, {"judge_id": "j", "date": "d"})
    assert "| Scored |" in md
    assert "| 9/11 |" in md


def test_a_refused_chapter_does_not_count_as_scored():
    from ati.report import ModelResult, render_markdown

    r = ModelResult(
        model_id="m",
        composite=80.0,
        axis_means={},
        axis_stdevs={},
        refusal_rate=9.1,
        cost_per_1k=None,
        speed_per_1k=1.0,
        value=None,
        slop=0.1,
        repetition=0.1,
        judge_errors=0,
        failed_tasks=1,
        total_tasks=11,
        refused_tasks=1,
    )
    md = render_markdown("r", [r], {}, True, {"judge_id": "j", "date": "d"})
    assert "| 9/11 |" in md


def test_pairwise_records_count_wins_losses_and_ties():
    from ati.report import pairwise_records

    records = pairwise_records([("x", "y", "a"), ("x", "z", "b"), ("y", "z", "tie")])
    assert records["x"] == (1, 1, 0)
    assert records["y"] == (0, 1, 1)
    assert records["z"] == (1, 0, 1)


def test_the_head_to_head_table_shows_each_record():
    from ati.report import render_markdown

    md = render_markdown(
        "r", [], {}, True, {"judge_id": "j", "date": "d"},
        {"x": 1100.0, "y": 900.0}, {"x": (1000.0, 1200.0), "y": (800.0, 1000.0)},
        None, {"x": (3, 1, 2), "y": (1, 3, 2)},
    )
    assert "W–L–T" in md
    assert "| x | 1100 | 1000 – 1200 | 3–1–2 |" in md


# --- anchor baseline ----------------------------------------------------------


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_the_baseline_is_the_run_that_wrote_anchors_json(tmp_path):
    from ati.report import read_anchor_baseline

    path = tmp_path / "anchors.json"
    _write(path, {"run_id": "r0", "anchors": {"hy3": 70.0}, "baseline": None})
    assert read_anchor_baseline(path, "r1") == ("r0", {"hy3": 70.0})


def test_a_second_report_of_a_run_keeps_its_real_baseline(tmp_path):
    """anchors.json is rewritten by every report; read naively, the second
    report of a run compared the run with itself and drift was always zero."""
    from ati.report import read_anchor_baseline

    path = tmp_path / "anchors.json"
    _write(
        path,
        {"run_id": "r1", "anchors": {"hy3": 65.0},
         "baseline": {"run_id": "r0", "anchors": {"hy3": 71.0}}},
    )
    assert read_anchor_baseline(path, "r1") == ("r0", {"hy3": 71.0})
    _write(path, {"run_id": "r1", "anchors": {"hy3": 65.0}, "baseline": None})
    assert read_anchor_baseline(path, "r1") == (None, {})


def test_a_legacy_file_written_by_this_run_is_not_a_baseline(tmp_path):
    from ati.report import read_anchor_baseline

    _write(tmp_path / "anchors.json", {"hy3": 65.05987292277615})
    _write(
        tmp_path / "r1.json",
        {"run_id": "r1", "results": [
            {"model_id": "hy3", "anchor": "weak", "composite": 65.05987292277615},
            {"model_id": "x", "anchor": None, "composite": 80.0},
        ]},
    )
    assert read_anchor_baseline(tmp_path / "anchors.json", "r1") == (None, {})


def test_a_legacy_file_is_labelled_with_the_run_beside_it(tmp_path):
    """The shape of archive/2026-08-ghost-commute/results/."""
    from ati.report import read_anchor_baseline

    _write(tmp_path / "anchors.json", {"hy3": 71.36})
    _write(
        tmp_path / "2026-08.json",
        {"run_id": "2026-08", "results": [
            {"model_id": "hy3", "anchor": "weak", "composite": 71.36}]},
    )
    assert read_anchor_baseline(tmp_path / "anchors.json", "2026-09") == (
        "2026-08", {"hy3": 71.36},
    )
    (tmp_path / "2026-08.json").unlink()
    assert read_anchor_baseline(tmp_path / "anchors.json", "2026-09") == (
        "unknown", {"hy3": 71.36},
    )


@pytest.mark.parametrize("content", ["", "not json", "[1]", '{"hy3": "x"}'])
def test_an_unusable_anchors_file_means_no_baseline(tmp_path, content):
    from ati.report import read_anchor_baseline

    path = tmp_path / "anchors.json"
    path.write_text(content, encoding="utf-8")
    assert read_anchor_baseline(path, "r1") == (None, {})


def _one_result(**overrides):
    base = dict(
        model_id="m1", composite=70.0, axis_means={a: 3.8 for a in AXES},
        axis_stdevs={a: 0.1 for a in AXES}, refusal_rate=0.0, cost_per_1k=None,
        speed_per_1k=1.0, value=None, slop=0.1, repetition=0.1, judge_errors=0,
        failed_tasks=0, total_tasks=11,
    )
    base.update(overrides)
    return ModelResult(**base)


def test_the_leaderboard_says_when_drift_was_not_checked():
    md = render_markdown("r1", [_one_result()], {}, True, {"anchor_baseline": None})
    assert "Anchor drift: not checked" in md


def test_the_leaderboard_names_the_baseline_run_and_the_drift():
    md = render_markdown(
        "r1", [_one_result()], {"hy3": -6.3}, False, {"anchor_baseline": "2026-08"}
    )
    assert "Anchor drift against run `2026-08`: hy3 -6.3." in md
    assert "Not comparable" in md


def test_the_docs_links_follow_where_the_leaderboard_is_written():
    md = render_markdown("r1", [_one_result()], {}, True, {}, docs_prefix="../../../docs/")
    assert "(../../../docs/rubric.md)" in md and "(../../../docs/methodology.md)" in md


def test_the_leaderboard_states_pairwise_coverage():
    md = render_markdown(
        "r1", [_one_result()], {}, True,
        {"pairwise_coverage": {"comparisons": 30, "pairs": 12, "possible_pairs": 15}},
        {"m1": 1000.0, "m2": 990.0}, None, None, {"m1": (1, 0, 0), "m2": (0, 1, 0)},
    )
    assert "Based on 30 comparisons covering 12 of 15 model pairs." in md


def test_judge_errors_and_unjudged_items_are_surfaced():
    md = render_markdown(
        "r1", [_one_result(judge_errors=2)], {}, True, {"pending_judgments": 5}
    )
    assert "**Judge errors**: m1 (2)" in md
    assert "**Not yet judged**: 5 outputs" in md
    clean = render_markdown("r1", [_one_result()], {}, True, {"pending_judgments": 0})
    assert "Judge errors" not in clean and "Not yet judged" not in clean


def test_a_torn_or_stray_record_is_skipped_not_fatal(tmp_path, capsys):
    """A report read while a stage is writing must not die on one file."""
    seed(tmp_path, "m1", "t.cn-01.0")
    torn = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.1")
    torn.write_text('{"model_id": "m1", "sco', encoding="utf-8")
    stray = run_output_path(tmp_path / "runs", "r1", "m1", "notes")
    stray.write_text('{"comment": "copied in by hand"}', encoding="utf-8")
    results = run_aggregate(tmp_path)
    assert results[0].composite == pytest.approx(75.0)
    assert results[0].total_tasks == 1
    out = capsys.readouterr().out
    assert "skipping unreadable" in out and "missing model_id" in out


# --- judge panel ------------------------------------------------------------


def seed_panel_judgment(tmp_path, judge, model_id, instance_id, scores, **extra):
    """A second judge's record of the same item, under judgments/<judge>/."""
    passage = instance_id.split(".")[1]
    payload = {
        "model_id": model_id,
        "task_id": instance_id.split(".")[0],
        "task_type": extra.pop("task_type", "passage"),
        "passage_id": passage,
        "iteration": 0,
        "judge_id": judge,
        "scores": scores,
        "rationales": {axis: "ok" for axis in AXES},
        "term_table": extra.pop("term_table", None),
        "refused": False,
        "skipped": False,
        "judge_error": None,
        "quote_violations": [],
    }
    jp = judgment_output_path(tmp_path / "panel" / judge, "r1", model_id, instance_id)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload), encoding="utf-8")


def run_panel(tmp_path, judges):
    return aggregate(
        run_id="r1",
        runs_root=tmp_path / "runs",
        judgments_root=None,
        weights=WEIGHTS,
        slop_list=[],
        anchors={},
        judge_roots={j: tmp_path / "panel" / j for j in judges},
    )


def test_a_panel_averages_its_judges_item_by_item(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")  # the run record; judgments go under panel/
    seed_panel_judgment(tmp_path, "ja", "m1", "t.cn-01.0", {a: 5 for a in AXES})
    seed_panel_judgment(tmp_path, "jb", "m1", "t.cn-01.0", {a: 3 for a in AXES})
    [result] = run_panel(tmp_path, ["ja", "jb"])
    assert result.axis_means["accuracy"] == pytest.approx(4.0)
    assert result.composite == pytest.approx(75.0)
    assert result.judge_composites == {
        "ja": pytest.approx(100.0), "jb": pytest.approx(50.0),
    }


def test_a_panel_of_one_matches_the_single_judge_report(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", scores={a: 4 for a in AXES})
    seed_panel_judgment(tmp_path, "ja", "m1", "t.cn-01.0", {a: 4 for a in AXES})
    [single] = run_aggregate(tmp_path)
    [panel] = run_panel(tmp_path, ["ja"])
    assert panel.composite == pytest.approx(single.composite)
    assert panel.axis_means == pytest.approx(single.axis_means)


def test_terminology_stability_is_computed_per_judge_then_averaged(tmp_path):
    """Two judges extract a term table differently. Pooled, the difference
    would read as the translator rendering one term two ways."""
    for ch in ("ch01", "ch02"):
        seed(tmp_path, "m1", f"c.{ch}.0", run={"task_id": "c", "passage_id": ch})
        seed_panel_judgment(
            tmp_path, "ja", "m1", f"c.{ch}.0", {a: 4 for a in AXES},
            task_type="consistency", term_table={"청명": "Chung Myung"},
        )
        seed_panel_judgment(
            tmp_path, "jb", "m1", f"c.{ch}.0", {a: 4 for a in AXES},
            task_type="consistency", term_table={"청명": "Cheongmyeong"},
        )
    [result] = run_panel(tmp_path, ["ja", "jb"])
    # Each judge alone sees one rendering throughout: perfectly stable.
    assert result.axis_means["terminology"] == pytest.approx(5.0)


def test_the_leaderboard_shows_each_panel_judge(tmp_path):
    r = ModelResult(
        model_id="m1", composite=75.0, axis_means={a: 4.0 for a in AXES},
        axis_stdevs={a: 0.0 for a in AXES}, refusal_rate=0.0, cost_per_1k=None,
        speed_per_1k=1.0, value=None, slop=0.0, repetition=0.0, judge_errors=0,
        failed_tasks=0, total_tasks=1,
        judge_composites={"haiku-judge": 70.0, "gemini-judge": 80.0},
    )
    md = render_markdown("r1", [r], {}, True, {"judge_id": "haiku-judge + gemini-judge"})
    assert "| haiku-judge | gemini-judge |" in md
    assert "| 70.0 | 80.0 |" in md


# --- list-price cost, the basis every route shares ---------------------------


def test_list_cost_and_tokens_per_chapter_come_from_the_roster_price(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0",
         run={"cost_usd": None, "tokens_in": 1_000_000, "tokens_out": 100_000})
    [r] = aggregate(
        run_id="r1", runs_root=tmp_path / "runs", judgments_root=tmp_path / "judgments",
        weights=WEIGHTS, slop_list=[], anchors={},
        prices={"m1": {"input": 1.0, "output": 10.0}},
    )
    assert r.list_cost_per_chapter == pytest.approx(2.0)
    assert r.tokens_out_per_chapter == pytest.approx(100_000)
    # Value is composite points per list dollar per 1k source characters:
    # $2.00 over 800 characters is $2.50 per 1k.
    assert r.value == pytest.approx(r.composite / 2.5)


def test_a_model_with_unpriced_chapters_has_no_list_cost(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", run={"tokens_in": None, "tokens_out": None})
    [r] = aggregate(
        run_id="r1", runs_root=tmp_path / "runs", judgments_root=tmp_path / "judgments",
        weights=WEIGHTS, slop_list=[], anchors={}, prices={"m1": {"input": 1, "output": 1}},
    )
    assert r.list_cost_per_chapter is None and r.tokens_out_per_chapter is None


def test_the_ranking_shows_list_cost_and_output_tokens(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", run={"tokens_in": 1000, "tokens_out": 12_345})
    results = aggregate(
        run_id="r1", runs_root=tmp_path / "runs", judgments_root=tmp_path / "judgments",
        weights=WEIGHTS, slop_list=[], anchors={}, prices={"m1": {"input": 1, "output": 1}},
    )
    md = render_markdown("r1", results, {}, True, {"judge_id": "j", "date": "d"})
    assert "List $ /ch" in md and "Out tok /ch" in md
    assert "| $0.0133 | 12,345 |" in md
    assert "list price" in md.lower()
