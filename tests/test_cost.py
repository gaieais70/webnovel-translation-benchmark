from ati.cost import render_cost, run_cost
from ati.fileio import write_json_atomic


def _record(root, model, instance, **fields):
    rec = {"model_id": model, "passage_id": instance, "text": "t", "error": None}
    rec.update(fields)
    write_json_atomic(root / "r1" / model / f"{instance}.json", rec)


def test_cost_prices_recorded_tokens_and_projects_the_rest(tmp_path):
    _record(tmp_path, "m", "a", tokens_in=1_000_000, tokens_out=100_000)
    _record(tmp_path, "m", "b", tokens_in=1_000_000, tokens_out=300_000)
    # A failed call that still reports usage was paid for.
    _record(tmp_path, "m", "c", tokens_in=0, tokens_out=500_000, error="finish_reason=length")
    rows = run_cost(
        "r1", tmp_path, {"m": {"input": 1.0, "output": 10.0}}, ["a", "b", "c", "d"]
    )
    row = rows["m"]
    assert row.spent == 1.0 + 1.0 + 10 * 0.9
    assert row.remaining == 2  # c failed, d never ran
    # Projection uses completed chapters only: (2.0 + 4.0) / 2 = 3.0 each.
    assert row.projected == 6.0


def test_cost_leaves_unpriced_and_unmetered_rows_unknown(tmp_path):
    _record(tmp_path, "m", "a", tokens_in=None, tokens_out=None)
    rows = run_cost("r1", tmp_path, {"m": None}, ["a"])
    assert rows["m"].spent is None and rows["m"].projected is None
    assert "?" in render_cost("r1", rows)
