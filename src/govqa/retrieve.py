"""Retrieval: BM25, dense, hybrid (reciprocal rank fusion) and hybrid + cross-encoder rerank."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from govqa import config

METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank")
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion. Each ranking is a list of item ids, best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL, device="cpu", max_length=512)


@dataclass
class Hit:
    chunk_idx: int
    score: float


class Retriever:
    def __init__(self, index_dir: Path):
        self.chunks = pd.read_parquet(index_dir / "chunks.parquet")
        self.texts: list[str] = self.chunks.text.tolist()
        self.doc_ids: list[str] = self.chunks.doc_id.tolist()
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        emb_path = index_dir / "embeddings.npy"
        self.embeddings = np.load(emb_path) if emb_path.exists() else None

    @property
    def has_dense(self) -> bool:
        return self.embeddings is not None

    def _bm25(self, query: str, n: int) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:n]
        return [Hit(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def _dense(self, query: str, n: int, q_emb: np.ndarray | None) -> list[Hit]:
        if self.embeddings is None:
            raise RuntimeError("This index was built without embeddings (--no-dense)")
        if q_emb is None:
            from govqa.index import embed_queries

            q_emb = embed_queries([query])[0]
        scores = self.embeddings @ q_emb
        top = np.argsort(-scores, kind="stable")[:n]
        return [Hit(int(i), float(scores[i])) for i in top]

    def search(
        self,
        query: str,
        method: str = "hybrid",
        k: int = 10,
        q_emb: np.ndarray | None = None,
        candidates: int = 50,
        rerank_top: int = 30,
    ) -> list[Hit]:
        """Return the top-k chunks. `q_emb` lets the evaluation pass precomputed embeddings."""
        if method == "bm25":
            return self._bm25(query, k)
        if method == "dense":
            return self._dense(query, k, q_emb)
        if method not in ("hybrid", "hybrid_rerank"):
            raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")

        fused = rrf(
            [
                [h.chunk_idx for h in self._bm25(query, candidates)],
                [h.chunk_idx for h in self._dense(query, candidates, q_emb)],
            ]
        )
        if method == "hybrid":
            return [Hit(i, s) for i, s in fused[:k]]

        pool = [i for i, _ in fused[:rerank_top]]
        scores = _reranker().predict([(query, self.texts[i]) for i in pool], batch_size=32)
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [Hit(pool[j], float(scores[j])) for j in order[:k]]

    def doc_ranking(self, hits: list[Hit]) -> list[str]:
        """Collapse chunk hits to a ranking of distinct documents, best first."""
        seen: dict[str, None] = {}
        for h in hits:
            seen.setdefault(self.doc_ids[h.chunk_idx], None)
        return list(seen)

    def row(self, chunk_idx: int) -> dict:
        return self.chunks.iloc[chunk_idx].to_dict()
