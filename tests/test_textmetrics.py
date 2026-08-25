import pytest

from ati.textmetrics import load_slop_list, repetition_score, slop_score


def test_load_slop_list_strips_comments_and_blanks(tmp_path):
    p = tmp_path / "slop.txt"
    p.write_text(
        "# comment\n\na testament to\ncouldn't help but\n", encoding="utf-8"
    )
    assert load_slop_list(p) == ["a testament to", "couldn't help but"]


def test_slop_score_counts_hits_per_1000_words():
    text = " ".join(["word"] * 998) + " a testament to"
    assert slop_score(text, ["a testament to"]) == pytest.approx(1.0, rel=1e-3)


def test_slop_score_is_case_insensitive():
    assert slop_score("A Testament To greatness", ["a testament to"]) > 0


def test_slop_score_counts_repeats():
    text = "a testament to a testament to " + " ".join(["x"] * 994)
    assert slop_score(text, ["a testament to"]) == pytest.approx(2.0, rel=1e-2)


def test_slop_score_is_zero_for_clean_text():
    assert slop_score("he drew his sword", ["a testament to"]) == 0.0


def test_slop_score_is_zero_for_empty_text():
    assert slop_score("", ["a testament to"]) == 0.0


def test_repetition_score_is_higher_for_repetitive_text():
    repetitive = "the sect elder said " * 50
    varied = " ".join(f"word{i}" for i in range(200))
    assert repetition_score(repetitive) > repetition_score(varied)


def test_repetition_score_is_bounded():
    assert 0.0 <= repetition_score("the the the the the") <= 1.0


def test_repetition_score_is_zero_for_empty_text():
    assert repetition_score("") == 0.0
