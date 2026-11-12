"""Grounded answer generation: retrieved passages in, cited answer (or a refusal) out."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from govqa import config
from govqa.llm import LLMResult, complete
from govqa.retrieve import Retriever

REFUSAL_TAG = "[GEEN_ANTWOORD]"

SYSTEM_PROMPT = f"""Je bent een assistent die vragen van burgers beantwoordt met informatie van \
Rijksoverheid.nl.

Regels:
- Gebruik uitsluitend de genummerde bronnen hieronder. Gebruik geen eigen kennis.
- Zet na elke bewering het nummer van de bron tussen vierkante haken, bijvoorbeeld [2].
- Staat het antwoord niet in de bronnen? Begin je antwoord dan met {REFUSAL_TAG} en leg in één \
zin uit dat de bronnen dit niet behandelen. Ga niet gokken.
- Antwoord kort en in gewone taal, in de taal van de vraag. Maximaal 150 woorden."""


@dataclass
class Source:
    n: int
    chunk_id: str
    doc_id: str
    url: str
    title: str
    text: str
    score: float


@dataclass
class Answer:
    question: str
    sources: list[Source]
    text: str | None = None
    refused: bool = False
    cited: list[int] = field(default_factory=list)
    llm: LLMResult | None = None


def retrieve_sources(
    retriever: Retriever, question: str, method: str | None = None, k: int | None = None
) -> list[Source]:
    method = method or config.DEFAULT_METHOD
    if not retriever.has_dense and method != "bm25":
        method = "bm25"
    hits = retriever.search(question, method=method, k=k or config.DEFAULT_TOP_K)
    sources = []
    for n, hit in enumerate(hits, start=1):
        row = retriever.row(hit.chunk_idx)
        sources.append(
            Source(
                n,
                row["chunk_id"],
                row["doc_id"],
                row["url"],
                row["question"],
                row["text"],
                hit.score,
            )
        )
    return sources


def build_messages(question: str, sources: list[Source]) -> list[dict[str, str]]:
    blocks = "\n\n".join(f"[{s.n}] (pagina: {s.title})\n{s.text}" for s in sources)
    user = f"Bronnen:\n\n{blocks}\n\nVraag: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_citations(text: str, n_sources: int) -> list[int]:
    """Source numbers the answer cites, in order of first use, ignoring out-of-range numbers."""
    cited: dict[int, None] = {}
    for group in re.findall(r"\[([\d,\s]+)\]", text):
        for num in re.findall(r"\d+", group):
            if 1 <= int(num) <= n_sources:
                cited.setdefault(int(num), None)
    return list(cited)


def answer_question(
    retriever: Retriever,
    question: str,
    method: str | None = None,
    k: int | None = None,
    model: str | None = None,
) -> Answer:
    sources = retrieve_sources(retriever, question, method, k)
    answer = Answer(question=question, sources=sources)
    if not (model or config.LLM_MODEL):
        return answer  # retrieval-only mode
    result = complete(build_messages(question, sources), model=model)
    answer.llm = result
    answer.refused = result.text.startswith(REFUSAL_TAG)
    answer.text = result.text.removeprefix(REFUSAL_TAG).strip()
    answer.cited = parse_citations(answer.text, len(sources))
    return answer
