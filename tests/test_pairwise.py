import pytest

from ati.pairwise import PairwiseNotImplemented, rank_pairwise


def test_rank_pairwise_is_a_documented_stub():
    with pytest.raises(PairwiseNotImplemented, match="deferred"):
        rank_pairwise(judge_runner=None, run_id="r1", runs_root=None, top_n=5)


def test_stub_documents_the_intended_design():
    doc = rank_pairwise.__doc__ or ""
    for expected in ("bidirectional", "truncat", "Glicko", "anchor"):
        assert expected.lower() in doc.lower()
