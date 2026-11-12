import pytest

from govqa.chunk import chunk_text, split_sections

TEXT = "Intro zin een.\nIntro zin twee.\n## Kop A\nRegel a1.\n- punt a2\n## Kop B\nRegel b1."


def test_split_sections_keeps_headings_and_order():
    sections = split_sections(TEXT)
    assert [h for h, _ in sections] == ["", "Kop A", "Kop B"]
    assert sections[1][1] == ["Regel a1.", "- punt a2"]


def test_short_sections_are_merged_into_one_chunk():
    chunks = chunk_text(TEXT, max_words=100)
    assert len(chunks) == 1
    assert chunks[0]["text"].splitlines()[2:4] == ["Kop A", "Regel a1."]


def test_heading_is_never_left_dangling_and_is_repeated_after_a_split():
    text = "## Kop\n" + "\n".join(f"regel {i} met vijf woorden" for i in range(6))
    chunks = chunk_text("Intro van vier woorden.\n" + text, max_words=12)
    assert all(not c["text"].endswith("Kop") for c in chunks)
    assert all(c["text"].startswith("Kop\n") for c in chunks[1:])
    assert all(c["heading"] == "Kop" for c in chunks[1:])


def test_chunks_respect_word_limit():
    text = "\n".join(f"zin nummer {i} heeft zes woorden" for i in range(40))
    chunks = chunk_text(text, max_words=30)
    assert len(chunks) > 1
    assert all(len(c["text"].split()) <= 30 for c in chunks)


def test_overlong_line_is_split():
    chunks = chunk_text(" ".join(["woord"] * 95), max_words=40)
    assert [len(c["text"].split()) for c in chunks] == [40, 40, 15]


def test_no_text_is_lost():
    chunks = chunk_text(TEXT, max_words=3)
    joined = " ".join(c["text"] for c in chunks)
    for token in ["Intro", "twee.", "a1.", "a2", "b1."]:
        assert token in joined


def test_empty_and_invalid_input():
    assert chunk_text("") == []
    assert chunk_text("## Alleen een kop") == []
    with pytest.raises(ValueError):
        chunk_text("x", max_words=0)
