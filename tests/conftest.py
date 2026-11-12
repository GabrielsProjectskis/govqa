import pandas as pd
import pytest

from govqa import config
from govqa.index import build_index
from govqa.retrieve import Retriever

DOCS = pd.DataFrame(
    [
        {
            "doc_id": "huur",
            "url": "https://example.test/huur",
            "question": "Wanneer heb ik recht op huurtoeslag?",
            "text": "U krijgt huurtoeslag als uw huur en inkomen niet te hoog zijn.\n"
            "## Voorwaarden\nU bent 18 jaar of ouder en huurt een zelfstandige woning.",
        },
        {
            "doc_id": "paspoort",
            "url": "https://example.test/paspoort",
            "question": "Hoe vraag ik een paspoort aan?",
            "text": "Een paspoort vraagt u aan bij de gemeente waar u staat ingeschreven.\n"
            "## Kosten\nEen paspoort kost in 2026 een vast bedrag dat de gemeente bepaalt.",
        },
        {
            "doc_id": "school",
            "url": "https://example.test/school",
            "question": "Wanneer krijgen scholen een tropenrooster?",
            "text": "Bij extreem hoge temperaturen kan een school een tropenrooster instellen.",
        },
    ]
)


@pytest.fixture
def docs() -> pd.DataFrame:
    return DOCS.copy()


@pytest.fixture
def index_dir(tmp_path, docs):
    out = tmp_path / "idx"
    build_index(docs, chunk_words=15, out_dir=out, dense=False)
    return out


@pytest.fixture
def retriever(index_dir) -> Retriever:
    return Retriever(index_dir)


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "")
