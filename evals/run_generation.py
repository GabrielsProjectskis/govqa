"""End-to-end generation evaluation. Needs LLM_MODEL (and its API key) in `.env`.

    uv run python -m evals.run_generation --n 150

Answerable questions come from the test split; out-of-scope questions (evals/out_of_scope.txt)
check that the assistant refuses instead of making something up.

Writes results/generation.json (summary) and results/generation_rows.jsonl (every answer).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evals.judge import judge_correctness, judge_faithfulness
from govqa import config
from govqa.generate import answer_question
from govqa.metrics import bootstrap_ci
from govqa.retrieve import Retriever

OUT_OF_SCOPE = Path(__file__).with_name("out_of_scope.txt")


def run_one(retriever: Retriever, item: dict, reference: str | None) -> dict:
    ans = answer_question(retriever, item["question"])
    row = {
        "kind": item["kind"],
        "doc_id": item.get("doc_id"),
        "question": item["question"],
        "answer": ans.text,
        "refused": ans.refused,
        "url": next((s.url for s in ans.sources if s.doc_id == item.get("doc_id")), None),
        "gold_in_context": any(s.doc_id == item.get("doc_id") for s in ans.sources),
        "cited_gold": any(ans.sources[n - 1].doc_id == item.get("doc_id") for n in ans.cited),
        "n_cited": len(ans.cited),
        "latency_s": ans.llm.latency_s,
        "cost_usd": ans.llm.cost_usd,
        "input_tokens": ans.llm.input_tokens,
        "output_tokens": ans.llm.output_tokens,
        "faithful": None,
        "correctness": None,
    }
    if item["kind"] == "answerable" and not ans.refused:
        try:
            row["faithful"] = judge_faithfulness(ans.text, [s.text for s in ans.sources])[
                "faithful"
            ]
            row["correctness"] = judge_correctness(item["question"], ans.text, reference)["score"]
        except Exception as err:  # a malformed judge reply should not sink the whole run
            row["judge_error"] = str(err)
    return row


def _ci(values) -> dict:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return {"mean": None, "low": None, "high": None, "n": 0}
    mean, low, high = bootstrap_ci(values)
    return {"mean": round(mean, 4), "low": round(low, 4), "high": round(high, 4), "n": len(values)}


def summarise(df: pd.DataFrame) -> dict:
    ans = df[df.kind == "answerable"]
    oos = df[df.kind == "out_of_scope"]
    answered = ans[~ans.refused]
    return {
        "llm_model": config.LLM_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "retrieval": {
            "method": config.DEFAULT_METHOD,
            "chunk_words": config.DEFAULT_CHUNK_WORDS,
            "top_k": config.DEFAULT_TOP_K,
        },
        "n_answerable": len(ans),
        "n_out_of_scope": len(oos),
        "gold_in_context": _ci(ans.gold_in_context),
        "faithfulness": _ci(answered.faithful),
        "fully_correct": _ci([None if c is None else c == 2 for c in answered.correctness]),
        "cited_gold_page": _ci(answered.cited_gold),
        "false_refusal_rate": _ci(ans.refused),
        "correct_refusal_rate": _ci(oos.refused),
        "latency_p50_s": round(float(df.latency_s.median()), 2),
        "latency_p95_s": round(float(df.latency_s.quantile(0.95)), 2),
        "cost_per_answer_usd": round(float(df.cost_usd.mean()), 6),
        "mean_input_tokens": int(df.input_tokens.mean()),
        "mean_output_tokens": int(df.output_tokens.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if not config.LLM_MODEL:
        raise SystemExit("LLM_MODEL is not set. Copy .env.example to .env and fill it in.")

    docs = pd.read_parquet(config.DOCS_PATH).set_index("doc_id")
    test = pd.read_parquet(config.TESTSET_DIR / "test.parquet").head(args.n)
    items = [
        {"kind": "answerable", "doc_id": r.doc_id, "question": r.question}
        for r in test.itertuples(index=False)
    ]
    for q in OUT_OF_SCOPE.read_text(encoding="utf-8").splitlines():
        if q.strip():
            items.append({"kind": "out_of_scope", "question": q.strip()})

    retriever = Retriever(config.index_path(config.DEFAULT_CHUNK_WORDS))

    def work(item: dict) -> dict:
        ref = docs.loc[item["doc_id"], "text"] if item.get("doc_id") else None
        return run_one(retriever, item, ref)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(work, items))

    config.RESULTS_DIR.mkdir(exist_ok=True)
    with (config.RESULTS_DIR / "generation_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarise(pd.DataFrame(rows))
    (config.RESULTS_DIR / "generation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
