import numpy as np
import pytest

from govqa.metrics import (
    bootstrap_ci,
    cohen_kappa,
    gold_rank,
    ndcg_at_k,
    paired_bootstrap_diff,
    per_query_metrics,
    recall_at_k,
    reciprocal_rank,
)


def test_gold_rank():
    assert gold_rank(["a", "b", "c"], "b") == 2
    assert gold_rank(["a", "b"], "z") == 0


def test_recall_mrr_ndcg_on_known_ranks():
    ranks = np.array([1, 2, 0, 11])  # hit@1, hit@2, miss, outside top 10
    assert recall_at_k(ranks, 1).tolist() == [1, 0, 0, 0]
    assert recall_at_k(ranks, 5).tolist() == [1, 1, 0, 0]
    assert reciprocal_rank(ranks).tolist() == [1.0, 0.5, 0.0, 0.0]
    assert ndcg_at_k(ranks) == pytest.approx([1.0, 1 / np.log2(3), 0.0, 0.0])


def test_per_query_metrics_shapes():
    out = per_query_metrics(np.array([1, 0, 3]))
    assert set(out) == {"recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"}
    assert all(len(v) == 3 for v in out.values())


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = np.array([1.0] * 70 + [0.0] * 30)
    mean, low, high = bootstrap_ci(values)
    assert mean == pytest.approx(0.7)
    assert low < 0.7 < high
    assert 0.55 < low and high < 0.85
    assert bootstrap_ci(values) == (mean, low, high)


def test_bootstrap_ci_of_constant_is_a_point():
    assert bootstrap_ci(np.ones(20)) == (1.0, 1.0, 1.0)


def test_paired_diff_detects_a_real_difference():
    a = np.array([1.0] * 90 + [0.0] * 10)
    b = np.array([1.0] * 50 + [0.0] * 50)
    diff, low, high = paired_bootstrap_diff(a, b)
    assert diff == pytest.approx(0.4)
    assert low > 0


def test_cohen_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cohen_kappa([], [])
