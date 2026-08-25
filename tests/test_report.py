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
