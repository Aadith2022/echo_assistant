"""Cache of task -> plan, so repeating a task doesn't re-plan it.

Directly, this removes one model call per task. Indirectly - and this is the
bigger win - the action cache is keyed on the Planner's step wording, and the
Planner rephrases every run. Caching the plan makes the step text
byte-identical next time, so every downstream action lookup is an exact hit
rather than a similarity guess.

No TTL, unlike the action cache: a plan is about the task rather than a page's
DOM, so it goes stale far more slowly. Still invalidated when a task fails.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time

import config
from browser.action_cache import (
    _SIMILARITY_THRESHOLD,
    _connect_raw,
    _intent_tokens,
    _normalize_intent,
    _similarity,
)

logger = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    connection = _connect_raw()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS plan_cache (
            key        TEXT PRIMARY KEY,
            task       TEXT NOT NULL,
            steps      TEXT NOT NULL,
            domains    TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL,
            hits       INTEGER DEFAULT 0
        )
        """
    )
    connection.commit()
    return connection


def lookup(task: str) -> tuple[list[str], list[str]] | None:
    """Return (steps, domains) for a previously-planned task, or None."""
    if not config.ACTION_CACHE_ENABLED:
        return None

    key = _normalize_intent(task)
    try:
        with _connect() as connection:
            row = connection.execute(
                "SELECT steps, domains, key FROM plan_cache WHERE key = ?", (key,)
            ).fetchone()

            if row is None:
                wanted = _intent_tokens(task)
                best, best_score = None, 0.0
                for candidate in connection.execute(
                    "SELECT steps, domains, key FROM plan_cache"
                ):
                    score = _similarity(wanted, frozenset(candidate[2].split()))
                    if score > best_score:
                        best, best_score = candidate, score
                # A plan commits the agent to a whole sequence of actions, so
                # the bar for reusing one is higher than for a single action.
                if best is None or best_score < max(_SIMILARITY_THRESHOLD, 0.75):
                    return None
                logger.info("Plan cache near-hit (%.2f) for %r", best_score, task)
                row = best

            connection.execute(
                "UPDATE plan_cache SET hits = hits + 1 WHERE key = ?", (row[2],)
            )
    except sqlite3.Error:
        logger.exception("Plan cache lookup failed; treating as a miss")
        return None

    return json.loads(row[0]), json.loads(row[1])


def store(task: str, steps: list[str], domains: list[str]) -> None:
    if not config.ACTION_CACHE_ENABLED:
        return
    try:
        with _connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO plan_cache "
                "(key, task, steps, domains, created_at, hits) VALUES (?,?,?,?,?,0)",
                (
                    _normalize_intent(task),
                    task,
                    json.dumps(steps),
                    json.dumps(domains),
                    time.time(),
                ),
            )
    except sqlite3.Error:
        logger.exception("Plan cache store failed; continuing uncached")


def invalidate(task: str) -> None:
    """Drop a plan whose task failed, so it is not replayed on every retry."""
    try:
        with _connect() as connection:
            connection.execute(
                "DELETE FROM plan_cache WHERE key = ?", (_normalize_intent(task),)
            )
    except sqlite3.Error:
        logger.exception("Plan cache invalidation failed")


def clear() -> None:
    try:
        with _connect() as connection:
            connection.execute("DELETE FROM plan_cache")
    except sqlite3.Error:
        logger.exception("Plan cache clear failed")
