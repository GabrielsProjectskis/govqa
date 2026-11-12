"""Create the evaluation splits.

    uv run python -m evals.build_testset                  # dev + test splits
    uv run python -m evals.build_testset --make-sample    # also refresh data/sample/docs.parquet
    uv run python -m evals.build_testset --paraphrase 200 # harder queries, needs LLM_MODEL

Every page title is a real citizen question and the page is its official answer, so a
(question, doc_id) pair is a free retrieval label. Splits are disjoint and seeded:
dev is for choosing the configuration, test is only for the final numbers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from govqa import config
from govqa.llm import complete

SEED = 42
DEV_SIZE, TEST_SIZE, SAMPLE_SIZE = 500, 1000, 200

PARAPHRASE_PROMPT = """Herschrijf de vraag hieronder zoals een gewone burger hem in een zoekbalk \
of chat zou typen: informeel, korter, met andere woorden dan het origineel, eventueel vanuit een \
persoonlijke situatie. De betekenis moet gelijk blijven. Geef alleen de herschreven vraag.

Vraag: {question}"""


def make_splits(docs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = docs[["doc_id", "question"]].sample(frac=1.0, random_state=SEED)
    dev_n = min(DEV_SIZE, len(shuffled) // 5)
    test_n = min(TEST_SIZE, (len(shuffled) - dev_n) // 2)
    dev = shuffled.iloc[:dev_n].reset_index(drop=True)
    test = shuffled.iloc[dev_n : dev_n + test_n].reset_index(drop=True)
    return dev, test


def paraphrase(questions: list[str]) -> list[str]:
    def one(q: str) -> str:
        msg = [{"role": "user", "content": PARAPHRASE_PROMPT.format(question=q)}]
        return complete(msg, temperature=0.7, max_tokens=100).text.strip().strip('"')

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, questions))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--paraphrase", type=int, default=0, metavar="N")
    args = ap.parse_args()

    docs = pd.read_parquet(config.DOCS_PATH)
    config.TESTSET_DIR.mkdir(parents=True, exist_ok=True)
    dev, test = make_splits(docs)
    dev.to_parquet(config.TESTSET_DIR / "dev.parquet", index=False)
    test.to_parquet(config.TESTSET_DIR / "test.parquet", index=False)
    print(f"dev={len(dev)} test={len(test)} of {len(docs)} docs")

    if args.make_sample:
        sample = docs.sample(n=min(SAMPLE_SIZE, len(docs)), random_state=SEED)
        config.SAMPLE_DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
        sample.sort_values("doc_id").to_parquet(config.SAMPLE_DOCS_PATH, index=False)
        print(f"sample of {len(sample)} docs -> {config.SAMPLE_DOCS_PATH}")

    if args.paraphrase:
        subset = test.head(args.paraphrase).copy()
        subset["original"] = subset["question"]
        subset["question"] = paraphrase(subset["original"].tolist())
        subset.to_parquet(config.TESTSET_DIR / "test_paraphrased.parquet", index=False)
        print(f"paraphrased {len(subset)} test questions")


if __name__ == "__main__":
    main()
