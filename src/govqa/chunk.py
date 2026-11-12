"""Split a page body into chunks of at most `max_words` words.

The body uses "## heading" lines for sections (see ingest.parse_page). Lines are packed greedily
in reading order. A chunk may span several short sections, but a heading is never left dangling
at the end of a chunk, and a chunk that starts in the middle of a section repeats that section's
heading so it can be read on its own.
"""

from __future__ import annotations


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Return (heading, lines) pairs. Text before the first heading gets an empty heading."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, lines) for h, lines in sections if lines]


def _split_long(line: str, max_words: int) -> list[str]:
    words = line.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _words(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def chunk_text(text: str, max_words: int = 200) -> list[dict[str, str]]:
    """Return chunks as {"heading": first section heading in the chunk, "text": chunk text}."""
    if max_words < 1:
        raise ValueError("max_words must be positive")
    chunks: list[dict[str, str]] = []
    current: list[str] = []
    current_heading = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append({"heading": current_heading, "text": "\n".join(current)})
            current = []

    for heading, lines in split_sections(text):
        head = [heading] if heading else []
        head_words = _words(head)
        budget = max(1, max_words - head_words)
        pieces = [p for line in lines for p in _split_long(line, budget)]
        for i, piece in enumerate(pieces):
            need = len(piece.split()) + (head_words if i == 0 else 0)
            if current and _words(current) + need > max_words:
                flush()
            if not current:
                current_heading = heading
                current.extend(head)  # also repeats the heading on a mid-section split
            elif i == 0:
                current.extend(head)
            current.append(piece)
    flush()
    return chunks
