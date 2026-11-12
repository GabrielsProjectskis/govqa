"""Streamlit front end. Talks to the API; start that first.

uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("GOVQA_API_URL", "http://localhost:8000")
RESULTS = Path(__file__).resolve().parents[1] / "results"
METHOD_LABELS = {
    "dense": "Embeddings (best on the test split)",
    "hybrid": "Hybrid (BM25 + embeddings)",
    "hybrid_rerank": "Hybrid + reranker (slow)",
    "bm25": "BM25 keywords only",
}

st.set_page_config(page_title="govqa", page_icon="🏛️", layout="centered")


@st.cache_data(ttl=30)
def health() -> dict | None:
    try:
        return httpx.get(f"{API_URL}/health", timeout=5).json()
    except httpx.HTTPError:
        return None


def ask_page(info: dict) -> None:
    st.title("Vraag het de Rijksoverheid")
    st.caption(
        "Answers come only from the question-and-answer pages of Rijksoverheid.nl (CC0). "
        "Portfolio project, not an official government service."
    )
    methods = list(METHOD_LABELS) if info["dense"] else ["bm25"]
    with st.sidebar:
        method = st.selectbox("Retrieval", methods, format_func=METHOD_LABELS.get)
        k = st.slider("Passages given to the model", 1, 10, 5)
        st.divider()
        st.write(f"**Corpus:** {info['corpus']} ({info['chunks']:,} passages)")
        st.write(f"**LLM:** {info['llm_model'] or 'none, showing passages only'}")

    question = st.chat_input("Bijvoorbeeld: Wanneer heb ik recht op huurtoeslag?")
    if not question:
        return
    st.chat_message("user").write(question)
    with st.spinner("Zoeken..."):
        try:
            resp = httpx.post(
                f"{API_URL}/ask", json={"question": question, "method": method, "k": k}, timeout=90
            )
            resp.raise_for_status()
        except httpx.HTTPError as err:
            st.error(f"The API returned an error: {err}")
            return
    data = resp.json()

    with st.chat_message("assistant"):
        if data["mode"] == "retrieval_only":
            st.info("No LLM is configured, so these are the best matching passages.")
        elif data["refused"]:
            st.warning(data["answer"] or "De bronnen behandelen deze vraag niet.")
        else:
            st.write(data["answer"])
        cost = f" · ${data['cost_usd']:.5f}" if data["cost_usd"] else ""
        st.caption(f"{data['latency_ms']} ms{cost}")

    st.subheader("Bronnen")
    for s in data["sources"]:
        mark = " ✅ cited" if s["cited"] else ""
        with st.expander(f"[{s['n']}] {s['title']}{mark}", expanded=data["mode"] != "generated"):
            st.markdown(s["text"].replace("\n", "  \n"))  # keep line breaks
            st.markdown(f"[Open on Rijksoverheid.nl]({s['url']})")


def _table(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    return pd.DataFrame(
        [
            {
                "index": r["index"],
                "method": r["method"],
                **{m: v["mean"] for m, v in r["metrics"].items()},
                "ms/query": r["ms_per_query"],
            }
            for r in rows
        ]
    )


def eval_page() -> None:
    st.title("Evaluation")
    st.caption("Numbers produced by the scripts in evals/. See the README for how to read them.")
    test = RESULTS / "retrieval_test.json"
    if test.exists():
        st.subheader("Retrieval on the held-out test split")
        st.dataframe(_table(test), hide_index=True, use_container_width=True)
    for name, title in [
        ("retrieval_test.png", "Method comparison"),
        ("chunk_size_dev.png", "Chunk size, dev split"),
    ]:
        if (RESULTS / name).exists():
            st.subheader(title)
            st.image(str(RESULTS / name))
    gen = RESULTS / "generation.json"
    st.subheader("Generation")
    if gen.exists():
        st.json(json.loads(gen.read_text()))
    else:
        st.info("Not run yet: needs an LLM key. Run `python -m evals.run_generation`.")


info = health()
if info is None:
    st.error(f"Cannot reach the API at {API_URL}. Start it with `uvicorn govqa.api:app`.")
    st.stop()
page = st.sidebar.radio("Page", ["Ask", "Evaluation"], label_visibility="collapsed")
if page == "Ask":
    ask_page(info)
else:
    eval_page()
