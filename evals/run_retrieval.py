"""Retrieval experiments: every (chunk size, method) pair on one split.

    uv run python -m evals.run_retrieval --split dev --chunk-words 100 200 400
    uv run python -m evals.run_retrieval --split test --chunk-words 200 \
        --methods bm25 dense hybrid hybrid_rerank
    uv run python -m evals.run_retrieval --sample          # BM25 on the committed sample (CI)

Writes results/retrieval_<split>.json (means with 95% bootstrap CIs) and
results/ranks_<split>.parquet (per-query ranks, used for paired significance tests).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from govqa import config
from govqa.metrics import bootstrap_ci, gold_rank, per_query_metrics
from govqa.retrieve import Retriever

CHUNKS_PER_QUERY = 30  # enough chunks to yield 10 distinct documents


def evaluate(
    retriever: Retriever, queries: pd.DataFrame, method: str, q_embs: np.ndarray | None
) -> tuple[np.ndarray, float]:
    ranks = np.zeros(len(queries), dtype=int)
    start = time.perf_counter()
    for i, q in enumerate(queries.itertuples(index=False)):
        hits = retriever.search(
            q.question,
            method=method,
            k=CHUNKS_PER_QUERY,
            q_emb=None if q_embs is None else q_embs[i],
        )
        ranks[i] = gold_rank(retriever.doc_ranking(hits), q.doc_id)
    ms = (time.perf_counter() - start) * 1000 / len(queries)
    return ranks, ms


def summarise(ranks: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for name, values in per_query_metrics(ranks).items():
        mean, low, high = bootstrap_ci(values)
        out[name] = {"mean": round(mean, 4), "low": round(low, 4), "high": round(high, 4)}
    return out


def run(
    queries: pd.DataFrame,
    index_names: list[str],
    methods: list[str],
    rerank_only_on: str | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    results, rank_table = [], queries[["doc_id"]].copy()
    q_embs = None
    for name in index_names:
        retriever = Retriever(config.INDEX_DIR / name)
        needs_dense = any(m != "bm25" for m in methods)
        if needs_dense and q_embs is None:
            from govqa.index import embed_queries

            q_embs = embed_queries(queries.question.tolist(), show_progress=True)
        for method in methods:
            if method == "hybrid_rerank" and rerank_only_on and name != rerank_only_on:
                continue  # the reranker is slow on CPU, so it runs on one index only
            ranks, ms = evaluate(retriever, queries, method, q_embs if method != "bm25" else None)
            # Query embedding time is excluded from ms_per_query for dense methods; it is
            # reported separately in the README.
            row = {
                "index": name,
                "method": method,
                "n_queries": len(queries),
                "n_chunks": len(retriever.texts),
                "ms_per_query": round(ms, 1),
                "metrics": summarise(ranks),
            }
            results.append(row)
            rank_table[f"{name}|{method}"] = ranks
            m = row["metrics"]
            print(
                f"{name:>12} {method:<14} R@1={m['recall@1']['mean']:.3f} "
                f"R@5={m['recall@5']['mean']:.3f} MRR={m['mrr@10']['mean']:.3f} {ms:6.1f} ms/q"
            )
    return results, rank_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="dev", help="dev, test or test_paraphrased")
    ap.add_argument("--chunk-words", type=int, nargs="+", default=[config.DEFAULT_CHUNK_WORDS])
    ap.add_argument("--methods", nargs="+", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--rerank-chunk-words", type=int, default=None)
    ap.add_argument("--tag", default=None, help="output file suffix, defaults to the split name")
    args = ap.parse_args()

    if args.sample:
        queries = pd.read_parquet(config.SAMPLE_DOCS_PATH)[["doc_id", "question"]]
        names, methods, tag = [f"sample-w{w}" for w in args.chunk_words], ["bm25"], "sample"
    else:
        queries = pd.read_parquet(config.TESTSET_DIR / f"{args.split}.parquet")
        names, methods, tag = [f"w{w}" for w in args.chunk_words], args.methods, args.split
    tag = args.tag or tag

    rerank_on = f"w{args.rerank_chunk_words}" if args.rerank_chunk_words else None
    results, ranks = run(queries, names, methods, rerank_on)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / f"retrieval_{tag}.json").write_text(json.dumps(results, indent=2))
    ranks.to_parquet(config.RESULTS_DIR / f"ranks_{tag}.parquet", index=False)


if __name__ == "__main__":
    main()
