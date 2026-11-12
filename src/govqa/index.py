"""Build a search index: chunks.parquet plus (optionally) a matrix of chunk embeddings.

    uv run python -m govqa.index --chunk-words 100 200 400
    uv run python -m govqa.index --sample --no-dense      # what CI does

The page title (the citizen's question) is deliberately NOT indexed. The evaluation uses those
titles as queries, so indexing them would leak the answer key into the index.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from govqa import config
from govqa.chunk import chunk_text


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


def _encode(texts: list[str], prefix: str, show_progress: bool = False) -> np.ndarray:
    # E5 models expect "query: " / "passage: " prefixes.
    emb = _embedder().encode(
        [prefix + t for t in texts],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_passages(texts: list[str], show_progress: bool = False) -> np.ndarray:
    return _encode(texts, "passage: ", show_progress)


def embed_queries(texts: list[str], show_progress: bool = False) -> np.ndarray:
    return _encode(texts, "query: ", show_progress)


def make_chunks(docs: pd.DataFrame, chunk_words: int) -> pd.DataFrame:
    rows = []
    for doc in docs.itertuples(index=False):
        for i, c in enumerate(chunk_text(doc.text, chunk_words)):
            rows.append(
                {
                    "chunk_id": f"{doc.doc_id}-{i}",
                    "doc_id": doc.doc_id,
                    "url": doc.url,
                    "question": doc.question,
                    "heading": c["heading"],
                    "text": c["text"],
                }
            )
    return pd.DataFrame(rows)


def build_index(docs: pd.DataFrame, chunk_words: int, out_dir: Path, dense: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = make_chunks(docs, chunk_words)
    chunks.to_parquet(out_dir / "chunks.parquet", index=False)
    print(f"[w{chunk_words}] {len(docs)} docs -> {len(chunks)} chunks")
    if dense:
        emb = embed_passages(chunks.text.tolist(), show_progress=True)
        np.save(out_dir / "embeddings.npy", emb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunk-words", type=int, nargs="+", default=[config.DEFAULT_CHUNK_WORDS])
    ap.add_argument("--sample", action="store_true", help="index the small committed sample")
    ap.add_argument("--no-dense", action="store_true", help="skip embeddings (BM25 only)")
    args = ap.parse_args()

    docs = pd.read_parquet(config.SAMPLE_DOCS_PATH if args.sample else config.DOCS_PATH)
    for words in args.chunk_words:
        name = f"sample-w{words}" if args.sample else f"w{words}"
        build_index(docs, words, config.INDEX_DIR / name, dense=not args.no_dense)


if __name__ == "__main__":
    main()
