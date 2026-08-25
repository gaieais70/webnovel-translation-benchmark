import pytest

from ati.config import AXES
from ati.scoring import (
    aggregate_axis,
    composite,
    cost_per_1k_chars,
    refusal_rate,
    speed_per_1k_chars,
    value_score,
)


def test_aggregate_axis_computes_mean_and_stdev():
    stats = aggregate_axis([4, 4, 5, 3])
    assert stats.mean == 4.0
    assert stats.n == 4
    # population stdev, matching aggregate_axis: with 2-3 iterations the
    # sample form inflates spread (by ~41% at n=2) and this number describes
    # the runs actually performed, not an inference about a wider population
    assert stats.stdev == pytest.approx(0.7071067812, rel=1e-6)


def test_aggregate_axis_of_single_score_has_zero_stdev():
    stats = aggregate_axis([5])
    assert stats.mean == 5.0
    assert stats.stdev == 0.0


def test_aggregate_axis_with_no_scores_returns_none_mean():
    stats = aggregate_axis([])
    assert stats.mean is None
    assert stats.n == 0
    assert stats.stdev == 0.0


def test_composite_maps_all_fives_to_one_hundred():
    weights = {axis: 1 / len(AXES) for axis in AXES}
    assert composite({axis: 5.0 for axis in AXES}, weights) == pytest.approx(100.0)


def test_composite_maps_all_ones_to_zero():
    weights = {axis: 1 / len(AXES) for axis in AXES}
    assert composite({axis: 1.0 for axis in AXES}, weights) == pytest.approx(0.0)


def test_composite_respects_weights():
    weights = {
        "accuracy": 0.25,
        "terminology": 0.20,
        "register": 0.15,
        "fluency": 0.15,
        "fidelity": 0.15,
        "sanitization": 0.10,
    }
    means = {axis: 3.0 for axis in AXES}
    means["accuracy"] = 5.0
    # 0.25 axis at 1.0 normalized, rest at 0.5 -> 0.25 + 0.75*0.5 = 0.625
    assert composite(means, weights) == pytest.approx(62.5)


def test_composite_renormalizes_when_an_axis_is_excluded():
    weights = {
        "accuracy": 0.25,
        "terminology": 0.20,
        "register": 0.15,
        "fluency": 0.15,
        "fidelity": 0.15,
        "sanitization": 0.10,
    }
    means = {axis: 5.0 for axis in AXES}
    means["sanitization"] = None  # judge errored on this axis
    # Remaining axes all perfect -> still 100, not 90
    assert composite(means, weights) == pytest.approx(100.0)


def test_composite_returns_none_when_every_axis_is_missing():
    weights = {axis: 1 / len(AXES) for axis in AXES}
    assert composite({axis: None for axis in AXES}, weights) is None


def test_refusal_rate_is_a_percentage():
    assert refusal_rate([True, False, False, False]) == 25.0
    assert refusal_rate([]) == 0.0


def test_cost_per_1k_chars():
    assert cost_per_1k_chars(0.02, 4000) == pytest.approx(0.005)


def test_cost_per_1k_chars_is_none_when_cost_unknown():
    assert cost_per_1k_chars(None, 4000) is None


def test_cost_per_1k_chars_is_none_when_no_chars():
    assert cost_per_1k_chars(0.02, 0) is None


def test_speed_per_1k_chars_uses_median():
    assert speed_per_1k_chars([10.0, 20.0, 30.0], [1000, 1000, 1000]) == 20.0


def test_speed_per_1k_chars_is_none_without_data():
    assert speed_per_1k_chars([], []) is None


def test_value_score_is_composite_per_dollar():
    assert value_score(80.0, 0.004) == pytest.approx(20000.0)


def test_value_score_is_none_when_cost_unknown():
    assert value_score(80.0, None) is None


def test_value_score_is_none_when_composite_unknown():
    assert value_score(None, 0.004) is None


def test_value_score_is_none_when_cost_is_zero():
    assert value_score(80.0, 0.0) is None
