from evals.build_testset import make_splits
from evals.gate import check

from govqa.index import make_chunks
from govqa.ingest import doc_id, parse_page

URL = "https://www.rijksoverheid.nl/documenten/vraag-en-antwoord/schooltijden/tropenrooster"
HTML = """<html><head><title>Wanneer krijgen scholen een tropenrooster? | Rijksoverheid.nl</title>
<meta name="OVERHEID.authority" content="Ministerie van OCW"/>
<meta name="DCTERMS.modified" content="2026-11-11T17:34:38.437Z"/></head><body>
<nav><p>Menu dat niet in de tekst hoort</p></nav>
<main><h1>Wanneer krijgen scholen een tropenrooster?</h1>
<div class="intro"><p>Bij extreem hoge temperaturen kan een school zelf besluiten om een
tropenrooster in te stellen en op welke manier dat gebeurt.</p></div>
<h2>Andere maatregelen</h2><ul><li><p>meer pauzes;</p></li><li>veel drinken.</li></ul>
<div class="block links-list-block navigation-block"><h2>Meer informatie</h2><ul><li>Hoe regelen basisscholen de schooltijden?</li></ul></div>
<svg><path d="M1"/></svg><script>var x = 1;</script></main></body></html>"""


def test_parse_page_extracts_clean_structured_text():
    rec = parse_page(HTML, URL)
    assert rec["question"] == "Wanneer krijgen scholen een tropenrooster?"
    assert rec["topic"] == "schooltijden"
    assert rec["authority"] == "Ministerie van OCW" and rec["modified"] == "2026-11-11"
    lines = rec["text"].splitlines()
    assert lines[1] == "## Andere maatregelen"
    assert lines[2:] == ["- meer pauzes;", "- veel drinken."]  # <p> inside <li> not duplicated
    assert "Menu" not in rec["text"] and "var x" not in rec["text"]
    assert "Meer informatie" not in rec["text"] and "basisscholen" not in rec["text"]
    assert rec["question"] not in rec["text"]  # the title must not leak into the indexed body


def test_parse_page_rejects_pages_without_body():
    assert (
        parse_page("<html><body><main><h1>Titel?</h1><p>Kort.</p></main></body></html>", URL)
        is None
    )
    assert parse_page("<html><body>geen main</body></html>", URL) is None


def test_doc_id_is_stable():
    assert doc_id(URL) == doc_id(URL) and len(doc_id(URL)) == 12


def test_chunks_do_not_contain_the_question(docs):
    chunks = make_chunks(docs, chunk_words=50)
    assert set(chunks.doc_id) == set(docs.doc_id)
    for row in chunks.itertuples():
        assert row.question not in row.text


def test_splits_are_disjoint_and_reproducible(docs):
    import pandas as pd

    big = pd.concat([docs.assign(doc_id=docs.doc_id + str(i)) for i in range(20)])
    dev, test = make_splits(big)
    assert set(dev.doc_id).isdisjoint(test.doc_id)
    assert make_splits(big)[0].doc_id.tolist() == dev.doc_id.tolist()


def test_gate_flags_only_real_regressions():
    base = {"recall@5": 0.90, "mrr@10": 0.80}
    assert check({"recall@5": 0.89, "mrr@10": 0.80}, base) == []
    assert len(check({"recall@5": 0.80, "mrr@10": 0.80}, base)) == 1
