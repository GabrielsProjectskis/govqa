import numpy as np
import pytest

from govqa.retrieve import Hit, Retriever, rrf, tokenize


def test_tokenize_lowercases_and_keeps_dutch_characters():
    assert tokenize("Coöperatie, DigiD & BSN-nummer!") == ["coöperatie", "digid", "bsn", "nummer"]


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([[1, 2, 3], [2, 3, 1]])
    assert fused[0][0] == 2  # second and first place beats first and third
    assert {i for i, _ in fused} == {1, 2, 3}


def test_rrf_includes_items_seen_by_one_ranker_only():
    assert [i for i, _ in rrf([[1], [2]])] == [1, 2]


def test_bm25_finds_the_right_document(retriever):
    hits = retriever.search("paspoort aanvragen gemeente", method="bm25", k=3)
    assert retriever.doc_ranking(hits)[0] == "paspoort"


def test_bm25_returns_nothing_for_unknown_words(retriever):
    assert retriever.search("zzzz qqqq", method="bm25") == []


def test_doc_ranking_deduplicates_chunks(retriever):
    idx = [i for i, d in enumerate(retriever.doc_ids) if d == "huur"]
    assert len(idx) == 2
    assert retriever.doc_ranking([Hit(idx[0], 2.0), Hit(idx[1], 1.0)]) == ["huur"]


def test_dense_without_embeddings_is_a_clear_error(retriever):
    assert not retriever.has_dense
    with pytest.raises(RuntimeError, match="without embeddings"):
        retriever.search("huur", method="dense")


def test_hybrid_with_fake_embeddings(index_dir):
    r = Retriever(index_dir)
    n = len(r.texts)
    r.embeddings = np.eye(n, dtype=np.float32)
    target = r.doc_ids.index("school")
    hits = r.search("tropenrooster temperaturen", method="hybrid", k=2, q_emb=r.embeddings[target])
    assert r.doc_ranking(hits)[0] == "school"


def test_unknown_method(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", method="magic")
