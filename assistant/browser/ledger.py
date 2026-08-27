"""The task ledger - facts one browser task established, available to the next.

Without it every task is an island: flights, then hotels, then things to do
gives three correct answers with nothing to do with each other, and the hotel
search does not know which dates the flight landed on. A checkpoint is one
task's progress; this is the layer above.

Session-scoped, because a ledger entry is a fact with a shelf life. Persisting
it would need decay, expiry and re-verification - most of `memory/` rebuilt for
facts that do not warrant it. Still SQLite rather than a dict, because the
session outlives any single task.

One deterministic control. Findings are Quarantined output, so showing them to
the Planner is the reduction already accepted for re-planning - but the
Planner's `needs_domains` seeds the Origin Set, so a fact carrying a hostname
would be a page widening its successor's powers. `strip_locators()` removes
URLs and hostnames in code before anything reaches planning. The ledger carries
constraints, never destinations; provenance is recorded for the user and the
audit log, not fed back into planning.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass

import config
from guardrails.audit_log import log_event

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# One process is one session.
_session_id = uuid.uuid4().hex[:16]

# How many facts the Planner is shown. Pasting a long session's whole ledger
# into every plan buries the useful constraints.
_MAX_CONTEXT_FACTS = 25
_MAX_FACT_CHARS = 240


@dataclass(frozen=True)
class Entry:
    task: str
    fact: str
    source_domains: str
    created_at: float


# --- locator stripping --------------------------------------------------------
#
# Hardcoded and blunt for the same reason the critical-label list is: this runs
# before a plan is made and decides what can influence the task's scope, so a
# model deciding it would give an injected page a say in whether its own domain
# survives.

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# Lowercase-only, the conservative direction: real hostnames in extracted text
# are written lowercase essentially always, and requiring it avoids eating a
# sentence whose full stop lost its following space.
_HOST_RE = re.compile(r"\b(?:[a-z0-9][a-z0-9-]*\.)+[a-z]{2,24}\b")


def strip_locators(text: str) -> str:
    """Remove URLs and bare hostnames from a fact.

    The Planner gets the constraint, not the address - a fact does not need to
    name its source site for the Planner to plan around it.
    """
    cleaned = _URL_RE.sub("[site]", text or "")
    return _HOST_RE.sub("[site]", cleaned)


# --- storage ------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.BROWSER_DB_PATH), exist_ok=True)
    connection = sqlite3.connect(config.BROWSER_DB_PATH, timeout=10)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ledger_entries (
            session_id     TEXT NOT NULL,
            task_id        TEXT NOT NULL,
            task           TEXT NOT NULL,
            fact           TEXT NOT NULL,
            source_domains TEXT NOT NULL DEFAULT '',
            created_at     REAL NOT NULL
        )
        """
    )
    # A fact re-established by a later task is not new information.
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ledger_unique "
        "ON ledger_entries (session_id, fact)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS ledger_by_session ON ledger_entries (session_id)"
    )
    connection.commit()
    return connection


def session_id() -> str:
    return _session_id


def new_session() -> str:
    """Start a fresh ledger. Returns the new session id."""
    global _session_id
    _session_id = uuid.uuid4().hex[:16]
    return _session_id


def record(
    task_id: str,
    task: str,
    facts: list[str],
    source_domains: list[str] | None = None,
    tainted: bool = False,
) -> int:
    """Add what a finished task established. Returns how many facts were new.

    `tainted` is the deterministic refusal: a task whose pages tried to issue
    instructions leaves nothing behind for later tasks to plan from.
    """
    if not config.BROWSER_LEDGER_ENABLED:
        return 0

    if tainted:
        log_event(
            "ledger_write_refused",
            task_id=task_id,
            reason="the task's pages attempted prompt injection",
            facts=len(facts or []),
        )
        logger.warning("Not recording %d fact(s) from a task flagged for injection", len(facts or []))
        return 0

    usable = [f.strip() for f in (facts or []) if f and f.strip()]
    if not usable:
        return 0

    _prune_once()

    domains = ", ".join(sorted({d for d in (source_domains or []) if d}))
    now = time.time()
    written = 0
    try:
        with _lock, _connect() as connection:
            for fact in usable:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO ledger_entries "
                    "(session_id, task_id, task, fact, source_domains, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (_session_id, task_id, task, fact[:2000], domains, now),
                )
                written += cursor.rowcount
    except sqlite3.Error:
        # The task has already succeeded; failing to remember it must not undo
        # having done it.
        logger.exception("Ledger write failed for task %s", task_id)
        return 0

    if written:
        log_event(
            "ledger_recorded",
            task_id=task_id,
            session=_session_id,
            facts=written,
            domains=domains,
        )
    return written


def entries(limit: int = 200) -> list[Entry]:
    """This session's facts, oldest first."""
    try:
        with _lock, _connect() as connection:
            rows = connection.execute(
                "SELECT task, fact, source_domains, created_at FROM ledger_entries "
                "WHERE session_id = ? ORDER BY created_at ASC, rowid ASC LIMIT ?",
                (_session_id, limit),
            ).fetchall()
    except sqlite3.Error:
        logger.exception("Ledger read failed")
        return []
    return [Entry(task=r[0], fact=r[1], source_domains=r[2], created_at=r[3]) for r in rows]


def planner_context(exclude_task: str = "") -> str:
    """What this session has established, formatted for a planning prompt.

    Empty when there is nothing yet, so the caller can omit the section rather
    than tell the Planner about an empty ledger.

    `exclude_task` drops facts this same request established: re-showing a task
    its own findings invites it to treat work in progress as already finished.
    """
    if not config.BROWSER_LEDGER_ENABLED:
        return ""

    collected = [e for e in entries() if e.task != exclude_task]
    if not collected:
        return ""

    # Most recent last - models weight the end of a list more heavily.
    recent = collected[-_MAX_CONTEXT_FACTS:]
    lines = []
    for entry in recent:
        fact = strip_locators(entry.fact)
        if len(fact) > _MAX_FACT_CHARS:
            fact = fact[:_MAX_FACT_CHARS].rsplit(" ", 1)[0] + "..."
        lines.append(f"  - {fact}")
    return "\n".join(lines)


def clear() -> None:
    """Drop this session's entries."""
    try:
        with _lock, _connect() as connection:
            connection.execute("DELETE FROM ledger_entries WHERE session_id = ?", (_session_id,))
    except sqlite3.Error:
        logger.exception("Ledger clear failed")


_pruned = False


def _prune_once() -> None:
    """Clear out dead sessions, the first time this process writes anything.

    Once per process, which keeps pruning off the read path entirely.
    """
    global _pruned
    if _pruned:
        return
    _pruned = True
    prune()


def prune(max_age_hours: int = 48) -> None:
    """Drop entries from sessions that have long since ended.

    Nothing ever reads them again; they only take up space in a database the
    checkpoints share.
    """
    cutoff = time.time() - max_age_hours * 3600
    try:
        with _lock, _connect() as connection:
            connection.execute("DELETE FROM ledger_entries WHERE created_at < ?", (cutoff,))
    except sqlite3.Error:
        logger.debug("Ledger prune failed", exc_info=True)
