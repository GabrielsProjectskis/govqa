import pytest
from fastapi.testclient import TestClient

from govqa import api
from govqa.retrieve import Retriever


@pytest.fixture
def client(index_dir, no_llm, monkeypatch):
    monkeypatch.setattr(api, "load_retriever", lambda: (Retriever(index_dir), "test"))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["dense"] is False and body["llm_model"] is None


def test_ask_retrieval_only(client):
    resp = client.post("/ask", json={"question": "Hoe vraag ik een paspoort aan?", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "retrieval_only" and body["answer"] is None
    assert body["sources"][0]["url"].endswith("/paspoort")
    assert len(body["sources"]) <= 2


@pytest.mark.parametrize(
    "payload",
    [{"question": "a"}, {"question": "paspoort", "k": 99}, {"question": "paspoort", "method": "x"}],
)
def test_ask_validates_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_metrics_counts_requests(client):
    client.post("/ask", json={"question": "huurtoeslag voorwaarden"})
    text = client.get("/metrics").text
    assert 'govqa_requests_total{mode="retrieval_only"}' in text
    assert "govqa_request_seconds_bucket" in text
