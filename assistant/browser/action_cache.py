"""Stagehand-style DOM->action cache.

A workflow run every week costs two model calls per step the first time and
should cost none the second.

What is cached is the element's kind and visible LABEL, not a ref (assigned
fresh on every observation) or a CSS selector (sites regenerate class names,
and a stale selector fails silently by matching the wrong element). A label is
what a human would use to find the control again, and survives redesigns.

A hit skips both per-step model calls - the Actor and the Quarantined
extraction - because replay needs only the element list, which we build
ourselves. See `page_actions.observe_structure()`.

Self-healing: a label that no longer matches deletes its entry and falls
through to the Actor, so a stale cache degrades to first-run speed rather than
to a broken task.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()


@dataclass
class CachedAction:
    action: str  # click | fill | navigate | press
    element_kind: str = ""
    element_label: str = ""
    text: str = ""
    url: str = ""
    key_ref: str = ""  # cache key, so a failed replay can invalidate itself


def _connect_raw() -> sqlite3.Connection:
    """Bare connection to the shared browser DB. `plan_cache` uses this too."""
    os.makedirs(os.path.dirname(config.BROWSER_DB_PATH), exist_ok=True)
    return sqlite3.connect(config.BROWSER_DB_PATH, timeout=10)


def _connect() -> sqlite3.Connection:
    connection = _connect_raw()
    # v2 caches a sequence of actions rather than one, so a batched "fill the
    # box, then press the button" replays as a unit.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS action_cache_v2 (
            key         TEXT PRIMARY KEY,
            url_pattern TEXT NOT NULL,
            intent      TEXT NOT NULL,
            actions     TEXT NOT NULL,
            created_at  REAL NOT NULL,
            hits        INTEGER DEFAULT 0
        )
        """
    )
    connection.commit()
    return connection


def _normalize_url(url: str) -> str:
    """Drop query strings and fragments so `/search?q=shoes` and `/search?q=hats`
    share cache entries - the *page* is the same, and its controls are too."""
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    except Exception:
        return url


# Dropping these lets "Click the search button" and "click search button" share
# a cache entry.
_STOPWORDS = frozenset(
    """a an the of in on at to for from into with and or is are was be been by
    it its this that these those then than so as if just please now use using
    button field box link page site your you my our their his her""".split()
)

_WORD_RE = re.compile(r"[a-z0-9']+")

# How much two step descriptions must overlap to be treated as the same intent.
# A cache miss only costs time, while a wrong hit acts on the wrong element, so
# this is set conservatively.
_SIMILARITY_THRESHOLD = 0.6


def _intent_tokens(intent: str) -> frozenset[str]:
    """Reduce a step description to its distinguishing content words.

    The key is what the Planner said, and it rephrases the same step on every
    run - so keying on the raw string makes cache hits a matter of luck.
    """
    return frozenset(w for w in _WORD_RE.findall(intent.lower()) if w not in _STOPWORDS)


def _normalize_intent(intent: str) -> str:
    return " ".join(sorted(_intent_tokens(intent)))


def _similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard overlap. Deliberately not embeddings: a miss is merely slower,
    and a second sentence-transformers model in memory is not worth it."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _key(url: str, intent: str) -> str:
    raw = f"{_normalize_url(url)}||{_normalize_intent(intent)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def lookup(url: str, intent: str) -> CachedAction | None:
    """Return a cached action for this page+intent, or None."""
    if not config.ACTION_CACHE_ENABLED:
        return None

    key = _key(url, intent)
    cutoff = time.time() - config.ACTION_CACHE_TTL_DAYS * 86400

    try:
        with _lock, _connect() as connection:
            connection.execute(
                "DELETE FROM action_cache_v2 WHERE created_at < ?", (cutoff,)
            )

            row = connection.execute(
                "SELECT actions, key FROM action_cache_v2 WHERE key = ?", (key,)
            ).fetchone()

            if row is None:
                # Near-miss pass: same page, similar intent. An exact-key-only
                # cache almost never hits, because the wording drifts.
                wanted = _intent_tokens(intent)
                best, best_score = None, 0.0
                for candidate in connection.execute(
                    "SELECT actions, key, intent FROM action_cache_v2 WHERE url_pattern = ?",
                    (_normalize_url(url),),
                ):
                    score = _similarity(wanted, frozenset(candidate[2].split()))
                    if score > best_score:
                        best, best_score = candidate, score
                if best is None or best_score < _SIMILARITY_THRESHOLD:
                    return None
                logger.info(
                    "Action cache near-hit (%.2f) for %r -> %r", best_score, intent, best[2]
                )
                row = best

            connection.execute(
                "UPDATE action_cache_v2 SET hits = hits + 1 WHERE key = ?", (row[1],)
            )
    except sqlite3.Error:
        # A cache is an optimisation; never let it fail a task.
        logger.exception("Action cache lookup failed; treating as a miss")
        return None

    try:
        payload = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        logger.warning("Corrupt action cache entry; dropping")
        invalidate(row[1])
        return None

    return [
        CachedAction(
            action=item.get("action", ""),
            element_kind=item.get("element_kind", ""),
            element_label=item.get("element_label", ""),
            text=item.get("text", ""),
            url=item.get("url", ""),
            key_ref=row[1],
        )
        for item in payload
    ] or None


def store(url: str, intent: str, actions: list[CachedAction]) -> None:
    """Record the sequence of actions that accomplished this step on this page."""
    if not config.ACTION_CACHE_ENABLED or not actions:
        return

    payload = json.dumps(
        [
            {
                "action": a.action,
                "element_kind": a.element_kind,
                "element_label": a.element_label,
                "text": a.text,
                "url": a.url,
            }
            for a in actions
        ]
    )
    try:
        with _lock, _connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO action_cache_v2 "
                "(key, url_pattern, intent, actions, created_at, hits) VALUES (?,?,?,?,?,0)",
                (
                    _key(url, intent),
                    _normalize_url(url),
                    _normalize_intent(intent),
                    payload,
                    time.time(),
                ),
            )
    except sqlite3.Error:
        logger.exception("Action cache store failed; continuing uncached")


def invalidate(key: str) -> None:
    """Drop a cache entry whose element could not be found on replay."""
    if not key:
        return
    try:
        with _lock, _connect() as connection:
            connection.execute("DELETE FROM action_cache_v2 WHERE key = ?", (key,))
        logger.info("Invalidated stale action cache entry %s", key[:12])
    except sqlite3.Error:
        logger.exception("Action cache invalidation failed")


def match_element(cached: CachedAction, elements) -> int | None:
    """Find the ref of the element a cached action refers to, or None.

    Three passes, loosest last: exact label, case-insensitive, then substring.
    Kind must always agree - a "Submit" link and a "Submit" button are not
    interchangeable.
    """
    label = cached.element_label.strip()
    if not label:
        return None

    candidates = [e for e in elements if e.kind == cached.element_kind]

    for element in candidates:
        if element.label.strip() == label:
            return element.ref
    lowered = label.lower()
    for element in candidates:
        if element.label.strip().lower() == lowered:
            return element.ref
    for element in candidates:
        if lowered in element.label.strip().lower():
            return element.ref
    return None


def stats() -> dict:
    try:
        with _lock, _connect() as connection:
            total, hits = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM action_cache_v2"
            ).fetchone()
        return {"entries": total, "hits": hits}
    except sqlite3.Error:
        return {"entries": 0, "hits": 0}


def clear() -> None:
    try:
        with _lock, _connect() as connection:
            connection.execute("DELETE FROM action_cache_v2")
    except sqlite3.Error:
        logger.exception("Action cache clear failed")
