"""Gate 0 - answer without launching a browser wherever possible.

A live browser is the most expensive and most dangerous way to read a page: it
costs seconds of startup, hundreds of MB of RAM, a round trip per observation,
and it executes whatever JavaScript the page ships. Most requests that sound
like browsing are really just reading, and a plain HTTP fetch answers them.

The ladder, cheapest first: RSS/Atom via feedparser, Trafilatura over static
HTML, Jina Reader (a third-party server-side render, which discloses the URL,
hence its own switch), then NEEDS_BROWSER.

Everything returned has been through `content_extractor.sanitize()`, whichever
rung produced it.

Run standalone to check a URL:
    python -m browser.no_browser_gate --url https://example.com/article
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

import config
from browser.content_extractor import SanitizedContent, sanitize

logger = logging.getLogger(__name__)

# Below this, an "extraction" is a nav bar and a cookie banner rather than an
# article, and the next rung is worth trying.
_MIN_USEFUL_CHARS = 200

# A plain, honest UA. This path does no stealth: a site that blocks it is the
# signal to escalate to the real engine, where the anti-bot work lives.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_FEED_TYPES = ("application/rss+xml", "application/atom+xml", "application/xml", "text/xml")


@dataclass
class GateResult:
    """Outcome of the no-browser ladder."""

    url: str
    source: str  # "rss" | "trafilatura" | "jina" | "none"
    content: SanitizedContent | None
    needs_browser: bool
    reason: str = ""

    @property
    def text(self) -> str:
        return self.content.text if self.content else ""


def _fetch(url: str) -> requests.Response | None:
    try:
        response = requests.get(
            url,
            headers=_HEADERS,
            timeout=config.NO_BROWSER_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        response.raise_for_status()
        return response
    except Exception as e:
        logger.info("Static fetch of %s failed (%s); escalating", url, e)
        return None


def _try_feed(url: str, response: requests.Response) -> SanitizedContent | None:
    content_type = response.headers.get("Content-Type", "").lower()
    looks_like_feed = any(t in content_type for t in _FEED_TYPES) or url.rstrip("/").endswith(
        (".rss", ".atom", "/feed", "/rss")
    )
    if not looks_like_feed:
        return None

    try:
        import feedparser
    except ImportError:
        return None

    parsed = feedparser.parse(response.content)
    if not parsed.entries:
        return None

    lines = [f"Feed: {parsed.feed.get('title', url)}", ""]
    for entry in parsed.entries[:25]:
        lines.append(f"## {entry.get('title', '(untitled)')}")
        if entry.get("published"):
            lines.append(f"Published: {entry['published']}")
        if entry.get("link"):
            lines.append(entry["link"])
        summary = entry.get("summary") or ""
        if summary:
            lines.append(summary)
        lines.append("")

    return sanitize("\n".join(lines))


def _try_trafilatura(response: requests.Response) -> SanitizedContent | None:
    try:
        import trafilatura
    except ImportError:
        return None

    extracted = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=True,
        # Markdown keeps structural cues that plain text loses; the image and
        # link constructs it introduces are what sanitize() strips.
        output_format="markdown",
        with_metadata=True,
    )
    if not extracted or len(extracted.strip()) < _MIN_USEFUL_CHARS:
        return None
    return sanitize(extracted)


def _try_jina(url: str) -> SanitizedContent | None:
    if not config.JINA_READER_ENABLED:
        return None

    try:
        response = requests.get(
            config.JINA_READER_ENDPOINT + url,
            headers={"Accept": "text/plain"},
            timeout=config.NO_BROWSER_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as e:
        logger.info("Jina Reader failed for %s (%s); escalating", url, e)
        return None

    if len(response.text.strip()) < _MIN_USEFUL_CHARS:
        return None
    return sanitize(response.text)


def try_without_browser(url: str) -> GateResult:
    """Walk the ladder. Returns extracted content, or needs_browser=True."""
    if not config.NO_BROWSER_GATE_ENABLED:
        return GateResult(url, "none", None, True, "no-browser gate disabled")

    response = _fetch(url)

    if response is not None:
        content = _try_feed(url, response)
        if content:
            return GateResult(url, "rss", content, False)

        content = _try_trafilatura(response)
        if content:
            return GateResult(url, "trafilatura", content, False)

    # Either the fetch failed or the HTML held no readable article - both of
    # which are what a server-side render is for.
    content = _try_jina(url)
    if content:
        return GateResult(url, "jina", content, False)

    return GateResult(
        url,
        "none",
        None,
        True,
        "no static extraction produced readable content",
    )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Test the no-browser gate against a URL.")
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = try_without_browser(args.url)
    print(f"source={result.source} needs_browser={result.needs_browser} {result.reason}")
    if result.content:
        c = result.content
        print(
            f"chars={len(c.text)} urls_redacted={len(c.urls)} "
            f"images_stripped={c.images_stripped} invisible_stripped={c.invisible_stripped}"
        )
        print("-" * 60)
        print(c.text[:2000])


if __name__ == "__main__":
    _main()
