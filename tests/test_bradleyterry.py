import pytest

from ati.bradleyterry import bootstrap_ci, fit, to_elo, win_probability


def test_fit_returns_empty_for_no_comparisons():
    assert fit([]) == {}


def test_stronger_model_gets_higher_strength():
    comparisons = [("a", "b", "a")] * 8 + [("a", "b", "b")] * 2
    s = fit(comparisons)
    assert s["a"] > s["b"]


def test_equal_records_give_equal_strength():
    comparisons = [("a", "b", "a")] * 5 + [("a", "b", "b")] * 5
    s = fit(comparisons)
    assert s["a"] == pytest.approx(s["b"], rel=1e-6)


def test_all_ties_give_equal_strength():
    s = fit([("a", "b", "tie")] * 6)
    assert s["a"] == pytest.approx(s["b"], rel=1e-6)


def test_ties_count_as_half_a_win_each():
    """A tie must sit strictly between a win and a loss."""
    tie_only = fit([("a", "b", "tie")] * 10)
    a_wins = fit([("a", "b", "a")] * 10)
    assert a_wins["a"] > tie_only["a"]


def test_transitivity_is_recovered():
    # outcome letters are positional: "a" = first-listed model won.
    comparisons = (
        [("a", "b", "a")] * 9
        + [("b", "c", "a")] * 9
        + [("a", "c", "a")] * 9
    )
    s = fit(comparisons)
    assert s["a"] > s["b"] > s["c"]


def test_undefeated_model_does_not_diverge():
    """Without regularization BT has no finite MLE for an undefeated model."""
    s = fit([("a", "b", "a")] * 20)
    assert s["a"] < float("inf")
    assert s["a"] > s["b"] > 0


def test_fit_is_order_independent():
    forward = [("a", "b", "a"), ("b", "c", "b"), ("a", "c", "a")] * 3
    backward = list(reversed(forward))
    assert fit(forward) == pytest.approx(fit(backward), rel=1e-9)


def test_strengths_normalize_to_geometric_mean_one():
    import math

    s = fit([("a", "b", "a")] * 4 + [("b", "c", "tie")] * 4)
    logs = [math.log(v) for v in s.values()]
    assert sum(logs) / len(logs) == pytest.approx(0.0, abs=1e-9)


def test_to_elo_centres_on_mean_without_anchor():
    elo = to_elo({"a": 2.0, "b": 0.5})
    assert sum(elo.values()) / 2 == pytest.approx(1000.0)


def test_to_elo_pins_the_anchor():
    elo = to_elo({"a": 2.0, "weak": 0.5}, anchor_id="weak", anchor_rating=1000.0)
    assert elo["weak"] == pytest.approx(1000.0)
    assert elo["a"] > elo["weak"]


def test_elo_gap_of_400_is_ten_to_one_odds():
    elo = to_elo({"a": 10.0, "b": 1.0})
    assert elo["a"] - elo["b"] == pytest.approx(400.0)


def test_win_probability_is_symmetric_and_calibrated():
    assert win_probability(1000, 1000) == pytest.approx(0.5)
    assert win_probability(1400, 1000) == pytest.approx(10 / 11, rel=1e-6)


def test_bootstrap_ci_brackets_the_point_estimate():
    comparisons = [("a", "b", "a")] * 15 + [("a", "b", "b")] * 5
    point = to_elo(fit(comparisons))
    ci = bootstrap_ci(comparisons, rounds=80, seed=1)
    lo, hi = ci["a"]
    assert lo <= point["a"] <= hi


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    comparisons = [("a", "b", "a")] * 10 + [("a", "b", "b")] * 4
    assert bootstrap_ci(comparisons, rounds=50, seed=7) == bootstrap_ci(
        comparisons, rounds=50, seed=7
    )


def test_bootstrap_interval_is_wider_with_less_data():
    few = [("a", "b", "a")] * 3 + [("a", "b", "b")] * 1
    many = [("a", "b", "a")] * 30 + [("a", "b", "b")] * 10
    w_few = bootstrap_ci(few, rounds=80, seed=3)["a"]
    w_many = bootstrap_ci(many, rounds=80, seed=3)["a"]
    assert (w_few[1] - w_few[0]) > (w_many[1] - w_many[0])


def test_fit_rejects_unknown_outcome():
    with pytest.raises(ValueError, match="unknown outcome"):
        fit([("a", "b", "maybe")])
