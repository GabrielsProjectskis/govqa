"""Paths and settings. Everything is overridable through environment variables or `.env`."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("GOVQA_DATA_DIR", ROOT / "data"))
DOCS_PATH = DATA_DIR / "docs.parquet"
SAMPLE_DOCS_PATH = ROOT / "data" / "sample" / "docs.parquet"
INDEX_DIR = DATA_DIR / "index"
TESTSET_DIR = DATA_DIR / "testset"
RESULTS_DIR = ROOT / "results"

EMBED_MODEL = os.getenv("GOVQA_EMBED_MODEL", "intfloat/multilingual-e5-small")
RERANK_MODEL = os.getenv("GOVQA_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# Index and retrieval configuration served by the API. Chosen on the dev split, see README.
DEFAULT_CHUNK_WORDS = int(os.getenv("GOVQA_CHUNK_WORDS", "200"))
DEFAULT_METHOD = os.getenv("GOVQA_METHOD", "dense")
DEFAULT_TOP_K = int(os.getenv("GOVQA_TOP_K", "5"))

# Any LiteLLM model string, e.g. "gemini/gemini-2.5-flash" or "openai/gpt-4o-mini".
# Empty means: no generation, the app answers with the retrieved passages only.
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "").strip() or LLM_MODEL


def index_path(chunk_words: int) -> Path:
    return INDEX_DIR / f"w{chunk_words}"
