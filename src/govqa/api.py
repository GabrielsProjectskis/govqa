"""HTTP API.

    uv run uvicorn govqa.api:app --port 8000

Uses the full index when it exists (data/index/w<N>). On a fresh clone it falls back to a
BM25-only index over the committed sample, so the service always starts.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from govqa import config
from govqa.generate import answer_question
from govqa.retrieve import METHODS, Retriever

REQUESTS = Counter("govqa_requests_total", "Questions answered", ["mode"])
REFUSALS = Counter(
    "govqa_refusals_total", "Answers where the model said the sources do not cover it"
)
COST = Counter("govqa_llm_cost_usd_total", "Cumulative LLM spend in USD")
LATENCY = Histogram(
    "govqa_request_seconds", "End-to-end /ask latency", buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8)
)

state: dict = {}


def load_retriever() -> tuple[Retriever, str]:
    full = config.index_path(config.DEFAULT_CHUNK_WORDS)
    if (full / "chunks.parquet").exists():
        return Retriever(full), "full"
    from govqa.index import build_index

    sample = config.INDEX_DIR / f"sample-w{config.DEFAULT_CHUNK_WORDS}"
    if not (sample / "chunks.parquet").exists():
        docs = pd.read_parquet(config.SAMPLE_DOCS_PATH)
        build_index(docs, config.DEFAULT_CHUNK_WORDS, sample, dense=False)
    return Retriever(sample), "sample"


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["retriever"], state["corpus"] = load_retriever()
    if state["retriever"].has_dense and config.DEFAULT_METHOD != "bm25":
        from govqa.index import embed_queries

        embed_queries(["warm-up"])  # load the embedding model now, not on the first question
    yield
    state.clear()


app = FastAPI(title="govqa", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    method: str | None = Field(default=None, pattern="^(" + "|".join(METHODS) + ")$")
    k: int | None = Field(default=None, ge=1, le=10)


class SourceOut(BaseModel):
    n: int
    title: str
    url: str
    text: str
    score: float
    cited: bool


class AskResponse(BaseModel):
    answer: str | None
    refused: bool
    mode: str  # "generated" or "retrieval_only"
    model: str | None
    sources: list[SourceOut]
    latency_ms: int
    cost_usd: float


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.perf_counter()
    ans = answer_question(state["retriever"], req.question, method=req.method, k=req.k)
    elapsed = time.perf_counter() - start

    mode = "generated" if ans.llm else "retrieval_only"
    REQUESTS.labels(mode=mode).inc()
    LATENCY.observe(elapsed)
    if ans.refused:
        REFUSALS.inc()
    if ans.llm:
        COST.inc(ans.llm.cost_usd)

    return AskResponse(
        answer=ans.text,
        refused=ans.refused,
        mode=mode,
        model=config.LLM_MODEL or None,
        sources=[
            SourceOut(
                n=s.n,
                title=s.title,
                url=s.url,
                text=s.text,
                score=round(s.score, 4),
                cited=s.n in ans.cited,
            )
            for s in ans.sources
        ],
        latency_ms=int(elapsed * 1000),
        cost_usd=ans.llm.cost_usd if ans.llm else 0.0,
    )


@app.get("/health")
def health() -> dict:
    retriever: Retriever = state["retriever"]
    return {
        "status": "ok",
        "corpus": state["corpus"],
        "chunks": len(retriever.texts),
        "dense": retriever.has_dense,
        "llm_model": config.LLM_MODEL or None,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
