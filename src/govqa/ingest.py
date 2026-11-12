"""Download the "vraag en antwoord" pages of rijksoverheid.nl (licence: CC0) into one Parquet file.

    uv run python -m govqa.ingest             # download what is missing, then parse
    uv run python -m govqa.ingest --offline   # only re-parse the cached pages

The site publishes a sitemap index; every Q&A page lives under /documenten/vraag-en-antwoord/.
Pages are cached under data/raw/, so the crawl is resumable and never fetches a page twice.
The crawler is slow on purpose and stops as soon as the site answers 403 or 429.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
from pathlib import Path

import httpx
import pandas as pd
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from govqa import config

SITEMAP_INDEX = "https://www.rijksoverheid.nl/sitemap.xml"
FAQ_MARKER = "/documenten/vraag-en-antwoord/"
HEADERS = {"User-Agent": "govqa-portfolio-project (educational, low volume)"}
CONCURRENCY = 2
DELAY_S = 0.4  # be a polite guest


def doc_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:12]


def parse_page(html: str, url: str) -> dict | None:
    """Turn one Q&A page into a record. Returns None when the page has no usable body."""
    tree = HTMLParser(html)
    main = tree.css_first("main")
    if main is None:
        return None
    # Drop page furniture. The "Meer informatie" block lists the titles of related questions;
    # leaving it in would put other pages' questions into this page's text.
    for node in main.css(
        "script, style, svg, nav, form, figure, .links-list-block, .navigation-block, .feedback-bar"
    ):
        node.decompose()

    h1 = tree.css_first("h1")
    title = h1.text(strip=True) if h1 else ""
    if not title:
        t = tree.css_first("title")
        title = t.text(strip=True).split("|")[0].strip() if t else ""

    lines: list[str] = []
    for node in main.css("h2, h3, p, li"):
        if node.tag == "p" and node.parent is not None and node.parent.tag == "li":
            continue
        text = re.sub(r"\s+", " ", node.text(separator=" ", strip=True)).strip()
        if not text or text == title:
            continue
        if node.tag in ("h2", "h3"):
            lines.append(f"## {text}")
        elif node.tag == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    body = "\n".join(lines).strip()
    if len(body.split()) < 15 or not title:
        return None

    def meta(name: str) -> str:
        node = tree.css_first(f'meta[name="{name}"]')
        return node.attributes.get("content", "") if node else ""

    return {
        "doc_id": doc_id(url),
        "url": url,
        "question": title,
        "topic": url.split(FAQ_MARKER)[1].split("/")[0],
        "authority": meta("OVERHEID.authority"),
        "modified": meta("DCTERMS.modified")[:10],
        "text": body,
    }


class Blocked(Exception):
    """The site answered 403/429: stop instead of hammering it."""


async def _get(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> str | None:
    async with sem:
        for attempt in range(3):
            try:
                r = await asyncio.wait_for(client.get(url), timeout=40)
                if r.status_code == 200:
                    await asyncio.sleep(DELAY_S)
                    return r.text
                if r.status_code == 404:
                    return None
                if r.status_code in (403, 429):
                    raise Blocked(url)
            except (httpx.HTTPError, TimeoutError):
                pass
            await asyncio.sleep(2.0 * (attempt + 1))
    return None


async def discover_urls(client: httpx.AsyncClient, sem: asyncio.Semaphore) -> list[str]:
    index = await _get(client, sem, SITEMAP_INDEX)
    if index is None:
        raise RuntimeError("Could not download the sitemap index")
    sitemaps = [u for u in re.findall(r"<loc>([^<]+)</loc>", index) if "/sitemap/" in u]
    pages = await asyncio.gather(*(_get(client, sem, u) for u in sitemaps))
    urls: set[str] = set()
    for xml in pages:
        if xml:
            urls.update(u for u in re.findall(r"<loc>([^<]+)</loc>", xml) if FAQ_MARKER in u)
    return sorted(urls)


def _cached_url(html: str) -> str | None:
    m = re.search(r'<meta name="DCTERMS.identifier" content="([^"]+)"', html)
    return m.group(1) if m else None


def parse_cache(raw_dir: Path) -> list[dict]:
    """Parse every page already on disk. Needs no network."""
    records = []
    for path in sorted(raw_dir.glob("*.html")):
        html = path.read_text(encoding="utf-8")
        url = _cached_url(html)
        if url and FAQ_MARKER in url:
            rec = parse_page(html, url)
            if rec:
                records.append(rec)
    return records


async def download(raw_dir: Path, limit: int | None) -> None:
    """Fetch pages that are not cached yet. Stops quietly when the site starts refusing."""
    sem = asyncio.Semaphore(CONCURRENCY)
    url_list = raw_dir / "_urls.txt"
    async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        try:
            if url_list.exists():
                urls = url_list.read_text().split()
            else:
                urls = await discover_urls(client, sem)
                url_list.write_text("\n".join(urls))
            todo = [u for u in urls if not (raw_dir / f"{doc_id(u)}.html").exists()][:limit]
            print(f"{len(urls)} Q&A urls in sitemap, {len(todo)} to download", flush=True)

            async def fetch(url: str) -> None:
                html = await _get(client, sem, url)
                if html:
                    (raw_dir / f"{doc_id(url)}.html").write_text(html, encoding="utf-8")

            for start in range(0, len(todo), 100):
                await asyncio.gather(*(fetch(u) for u in todo[start : start + 100]))
                print(f"  {min(start + 100, len(todo))}/{len(todo)}", flush=True)
        except Blocked:
            print(
                "The site is refusing requests (403/429). Keeping what is cached; "
                "run again later to continue.",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="download at most N new pages")
    ap.add_argument("--offline", action="store_true", help="only parse pages already cached")
    args = ap.parse_args()

    raw_dir = config.DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        asyncio.run(download(raw_dir, args.limit))

    df = pd.DataFrame(parse_cache(raw_dir))
    df = df.drop_duplicates("question").sort_values("doc_id").reset_index(drop=True)
    config.DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DOCS_PATH, index=False)
    words = df.text.str.split().str.len()
    print(f"saved {len(df)} docs to {config.DOCS_PATH} (median {int(words.median())} words)")


if __name__ == "__main__":
    main()
