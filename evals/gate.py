"""Quality gate for CI: BM25 retrieval on the committed sample must not get worse.

    uv run python -m evals.gate            # exit code 1 on a regression
    uv run python -m evals.gate --update   # accept the current numbers as the new baseline

Runs without any model download or API key, so it is cheap enough for every push.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from evals.run_retrieval import run
from govqa import config
from govqa.index import build_index

BASELINE = Path(__file__).with_name("baseline.json")
TOLERANCE = 0.02
GATED = ("recall@5", "mrr@10")


def current_scores(chunk_words: int = config.DEFAULT_CHUNK_WORDS) -> dict[str, float]:
    docs = pd.read_parquet(config.SAMPLE_DOCS_PATH)
    name = f"sample-w{chunk_words}"
    build_index(docs, chunk_words, config.INDEX_DIR / name, dense=False)
    results, _ = run(docs[["doc_id", "question"]], [name], ["bm25"])
    return {m: results[0]["metrics"][m]["mean"] for m in GATED}


def check(scores: dict[str, float], baseline: dict[str, float]) -> list[str]:
    return [
        f"{m}: {scores[m]:.3f} is below baseline {baseline[m]:.3f} - {TOLERANCE}"
        for m in GATED
        if scores[m] < baseline[m] - TOLERANCE
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()

    scores = current_scores()
    if args.update:
        BASELINE.write_text(json.dumps(scores, indent=2))
        print(f"baseline updated: {scores}")
        return
    failures = check(scores, json.loads(BASELINE.read_text()))
    if failures:
        print("QUALITY GATE FAILED\n" + "\n".join(failures))
        sys.exit(1)
    print(f"quality gate passed: {scores}")


if __name__ == "__main__":
    main()
