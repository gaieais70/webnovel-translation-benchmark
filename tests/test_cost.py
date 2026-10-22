import pytest

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


def test_record_cost_prices_cached_input_at_the_cached_rate():
    from ati.cost import record_cost

    price = {"input": 2.0, "output": 10.0, "cached_input": 0.1}
    rec = {"tokens_in": 1_000_000, "cached_tokens_in": 600_000, "tokens_out": 100_000}
    assert record_cost(rec, price) == pytest.approx(0.4 * 2.0 + 0.6 * 0.1 + 0.1 * 10.0)
    # No cached rate declared: cached input is priced as ordinary input.
    assert record_cost(rec, {"input": 2.0, "output": 10.0}) == pytest.approx(2.0 + 1.0)


def test_record_cost_prefers_the_routes_own_list_cost():
    from ati.cost import record_cost

    rec = {"tokens_in": 10, "tokens_out": 10, "list_cost_usd": 0.42}
    assert record_cost(rec, {"input": 1.0, "output": 1.0}) == 0.42
    assert record_cost(rec, None) == 0.42
    assert record_cost({"tokens_in": None, "tokens_out": None}, {"input": 1, "output": 1}) is None
    assert record_cost({"tokens_in": 5, "tokens_out": 5}, None) is None


def test_a_row_priced_only_by_its_route_still_counts(tmp_path):
    _record(tmp_path, "m", "a", tokens_in=1, tokens_out=1, list_cost_usd=0.5)
    rows = run_cost("r1", tmp_path, {"m": None}, ["a", "b"])
    assert rows["m"].spent == pytest.approx(0.5)
    assert rows["m"].projected == pytest.approx(0.5)
