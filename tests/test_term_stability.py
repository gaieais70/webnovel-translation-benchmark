import pytest

from ati.scoring import stability_to_axis_score, term_stability


def test_term_stability_is_one_when_every_term_is_stable():
    tables = [
        {"云澈": "Yun Che", "元气": "Profound Energy"},
        {"云澈": "Yun Che", "元气": "Profound Energy"},
    ]
    assert term_stability(tables) == 1.0


def test_term_stability_counts_only_terms_seen_more_than_once():
    tables = [
        {"云澈": "Yun Che", "只出现一次": "Once Only"},
        {"云澈": "Yun Che"},
    ]
    # the singleton term cannot be inconsistent, so it is excluded
    assert term_stability(tables) == 1.0


def test_term_stability_penalizes_drift():
    tables = [
        {"云澈": "Yun Che", "元气": "Profound Energy"},
        {"云澈": "Cloud Clear", "元气": "Profound Energy"},
    ]
    assert term_stability(tables) == 0.5


def test_term_stability_ignores_case_and_surrounding_space():
    tables = [{"元气": "Profound Energy"}, {"元气": " profound energy "}]
    assert term_stability(tables) == 1.0


def test_term_stability_is_none_without_shared_terms():
    assert term_stability([{"a": "x"}]) is None
    assert term_stability([]) is None


def test_stability_maps_onto_the_five_point_axis_scale():
    assert stability_to_axis_score(1.0) == 5.0
    assert stability_to_axis_score(0.0) == 1.0
    assert stability_to_axis_score(0.5) == 3.0
    assert stability_to_axis_score(None) is None
