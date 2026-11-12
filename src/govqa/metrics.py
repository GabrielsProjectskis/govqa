"""Ranking metrics for the one-relevant-document case, plus bootstrap confidence intervals.

`ranks` is an array with, per query, the 1-based rank of the correct document, or 0 when it was
not retrieved at all.
"""

from __future__ import annotations

import numpy as np


def gold_rank(ranking: list[str], gold: str) -> int:
    try:
        return ranking.index(gold) + 1
    except ValueError:
        return 0


def recall_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    return ((ranks > 0) & (ranks <= k)).astype(float)


def reciprocal_rank(ranks: np.ndarray, k: int = 10) -> np.ndarray:
    hit = (ranks > 0) & (ranks <= k)
    return np.where(hit, 1.0 / np.maximum(ranks, 1), 0.0)


def ndcg_at_k(ranks: np.ndarray, k: int = 10) -> np.ndarray:
    # With a single relevant document the ideal DCG is 1, so nDCG = 1 / log2(rank + 1).
    hit = (ranks > 0) & (ranks <= k)
    return np.where(hit, 1.0 / np.log2(np.maximum(ranks, 1) + 1.0), 0.0)


def per_query_metrics(ranks: np.ndarray) -> dict[str, np.ndarray]:
    ranks = np.asarray(ranks)
    return {
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        "mrr@10": reciprocal_rank(ranks, 10),
        "ndcg@10": ndcg_at_k(ranks, 10),
    }


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap confidence interval: (mean, low, high)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) on the same queries with a bootstrap CI. CI excluding 0 = real difference."""
    return bootstrap_ci(
        np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n_boot, alpha, seed
    )


def cohen_kappa(a: list[int], b: list[int]) -> float:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if len(a_arr) == 0 or len(a_arr) != len(b_arr):
        raise ValueError("need two equally long, non-empty label lists")
    observed = float((a_arr == b_arr).mean())
    labels = np.union1d(a_arr, b_arr)
    expected = float(sum((a_arr == lab).mean() * (b_arr == lab).mean() for lab in labels))
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)
