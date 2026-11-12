from govqa import config, generate
from govqa.generate import REFUSAL_TAG, answer_question, build_messages, parse_citations
from govqa.llm import LLMResult, parse_json


def test_parse_citations_order_ranges_and_bounds():
    assert parse_citations("Ja [2]. Ook [1, 3] en nog eens [2]. Niet [9].", n_sources=3) == [
        2,
        1,
        3,
    ]
    assert parse_citations("Geen verwijzingen.", n_sources=3) == []


def test_prompt_contains_numbered_sources_and_question(retriever, no_llm):
    sources = generate.retrieve_sources(retriever, "huurtoeslag voorwaarden", method="bm25", k=2)
    messages = build_messages("Krijg ik huurtoeslag?", sources)
    assert messages[0]["role"] == "system" and REFUSAL_TAG in messages[0]["content"]
    assert (
        "[1]" in messages[1]["content"] and "Vraag: Krijg ik huurtoeslag?" in messages[1]["content"]
    )


def test_retrieval_only_mode_without_llm(retriever, no_llm):
    ans = answer_question(retriever, "paspoort aanvragen")
    assert ans.text is None and ans.llm is None
    assert ans.sources[0].doc_id == "paspoort"


def _fake_llm(text):
    def fake(messages, model=None, **kwargs):
        return LLMResult(text, 100, 20, 0.0001, 0.5)

    return fake


def test_generated_answer_with_citations(retriever, monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "fake/model")
    monkeypatch.setattr(generate, "complete", _fake_llm("Bij de gemeente [1]."))
    ans = answer_question(retriever, "paspoort aanvragen", method="bm25")
    assert ans.text == "Bij de gemeente [1]." and ans.cited == [1] and not ans.refused


def test_refusal_is_detected_and_tag_removed(retriever, monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "fake/model")
    monkeypatch.setattr(generate, "complete", _fake_llm(f"{REFUSAL_TAG} Dat staat er niet in."))
    ans = answer_question(retriever, "paspoort appeltaart", method="bm25")
    assert ans.refused and ans.text == "Dat staat er niet in."


def test_parse_json_tolerates_code_fences():
    assert parse_json('```json\n{"faithful": true, "unsupported": ""}\n```')["faithful"] is True


def test_llm_wrapper_against_litellm_mock():
    from govqa.llm import complete

    out = complete(
        [{"role": "user", "content": "hoi"}], model="gpt-4o-mini", mock_response="Hallo [1]."
    )
    assert out.text == "Hallo [1]." and out.latency_s >= 0 and out.cost_usd >= 0
