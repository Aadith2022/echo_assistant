"""Task checkpointing - the desktop version of save-state / re-queue / restore.

A fifteen-step browser task takes minutes, and the things that interrupt it -
a page timing out, Chrome crashing, the user closing the window - are normal.
Without checkpoints every interruption restarts from step one, which on a task
with side effects is worse than not retrying at all.

APScheduler is the worker pool, SQLite is both queue and state store, and the
restore is a row read. Screenshots taken during a task are ephemeral -
`cleanup_task()` wipes them when the task ends.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field

import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()

PENDING = "pending"
RUNNING = "running"
# Paused on a question for the user, distinct from failed: the work so far is
# intact and the task continues the moment an answer arrives.
WAITING = "waiting"
DONE = "done"
FAILED = "failed"


@dataclass
class Checkpoint:
    task_id: str
    task: str
    steps: list[str] = field(default_factory=list)
    last_step: int = 0
    status: str = PENDING
    allowed_domains: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    attempts: int = 0
    updated_at: float = 0.0
    pending_question: str = ""
    answers: list[str] = field(default_factory=list)
    # Steps assessed as not achieved and never recovered, so the final answer
    # can be reported as partial even when the last page reads well. Persisted,
    # or a resumed task would report a clean success it did not earn.
    unmet: list[str] = field(default_factory=list)
    # Persisted for the same reason: the budget must not reset on resume.
    replans: int = 0
    # Where the browser was, as opposed to how far the plan got. Those are only
    # the same thing inside one process - after a restart the engine launches
    # cold on about:blank, so a step written to continue where the last one
    # ended continues from nothing. Re-entered through the task's Origin Set,
    # like any other navigation.
    current_url: str = ""

    @property
    def is_resumable(self) -> bool:
        return self.status in (PENDING, RUNNING, WAITING) and self.last_step < len(self.steps)

    @property
    def context_from_answers(self) -> str:
        """The user's clarifications, for the Planner and Actor to work from."""
        if not self.answers:
            return ""
        return "\n".join(f"- {a}" for a in self.answers)


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.BROWSER_DB_PATH), exist_ok=True)
    connection = sqlite3.connect(config.BROWSER_DB_PATH, timeout=10)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS task_checkpoints (
            task_id         TEXT PRIMARY KEY,
            task            TEXT NOT NULL,
            steps           TEXT NOT NULL,
            last_step       INTEGER NOT NULL DEFAULT 0,
            status          TEXT NOT NULL,
            allowed_domains TEXT NOT NULL DEFAULT '[]',
            notes           TEXT NOT NULL DEFAULT '[]',
            attempts        INTEGER NOT NULL DEFAULT 0,
            updated_at      REAL NOT NULL,
            pending_question TEXT NOT NULL DEFAULT '',
            answers         TEXT NOT NULL DEFAULT '[]'
        )
        """
    )

    # CREATE TABLE IF NOT EXISTS does nothing to an existing table, so a
    # database written before these columns were added would fail every read
    # with "no such column". Add them in place instead.
    existing = {row[1] for row in connection.execute("PRAGMA table_info(task_checkpoints)")}
    for column, ddl in (
        ("pending_question", "TEXT NOT NULL DEFAULT ''"),
        ("answers", "TEXT NOT NULL DEFAULT '[]'"),
        ("unmet", "TEXT NOT NULL DEFAULT '[]'"),
        ("replans", "INTEGER NOT NULL DEFAULT 0"),
        ("current_url", "TEXT NOT NULL DEFAULT ''"),
    ):
        if column not in existing:
            connection.execute(f"ALTER TABLE task_checkpoints ADD COLUMN {column} {ddl}")
            logger.info("Migrated task_checkpoints: added %s", column)

    connection.commit()
    return connection


def new_task_id() -> str:
    return uuid.uuid4().hex[:16]


def save(checkpoint: Checkpoint) -> None:
    checkpoint.updated_at = time.time()
    try:
        with _lock, _connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO task_checkpoints "
                "(task_id, task, steps, last_step, status, allowed_domains, notes, "
                " attempts, updated_at, pending_question, answers, unmet, replans, "
                " current_url) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    checkpoint.task_id,
                    checkpoint.task,
                    json.dumps(checkpoint.steps),
                    checkpoint.last_step,
                    checkpoint.status,
                    json.dumps(checkpoint.allowed_domains),
                    json.dumps(checkpoint.notes[-50:]),
                    checkpoint.attempts,
                    checkpoint.updated_at,
                    checkpoint.pending_question,
                    json.dumps(checkpoint.answers[-20:]),
                    json.dumps(checkpoint.unmet[-20:]),
                    checkpoint.replans,
                    checkpoint.current_url,
                ),
            )
    except sqlite3.Error:
        # Losing a checkpoint costs a restart, not correctness.
        logger.exception("Checkpoint save failed for %s", checkpoint.task_id)


def load(task_id: str) -> Checkpoint | None:
    try:
        with _lock, _connect() as connection:
            row = connection.execute(
                "SELECT task_id, task, steps, last_step, status, allowed_domains, notes, "
                "attempts, updated_at, pending_question, answers, unmet, replans, "
                "current_url "
                "FROM task_checkpoints WHERE task_id = ?",
                (task_id,),
            ).fetchone()
    except sqlite3.Error:
        logger.exception("Checkpoint load failed for %s", task_id)
        return None

    if row is None:
        return None
    return Checkpoint(
        task_id=row[0],
        task=row[1],
        steps=json.loads(row[2]),
        last_step=row[3],
        status=row[4],
        allowed_domains=json.loads(row[5]),
        notes=json.loads(row[6]),
        attempts=row[7],
        updated_at=row[8],
        pending_question=row[9],
        answers=json.loads(row[10]),
        unmet=json.loads(row[11]),
        replans=row[12],
        current_url=row[13],
    )


def resumable_tasks(max_age_hours: int = 24) -> list[Checkpoint]:
    """Tasks that were interrupted recently enough to be worth resuming.

    Age-bounded: resuming a two-week-old half-finished checkout against a page
    that has since changed is not helpful.
    """
    cutoff = time.time() - max_age_hours * 3600
    try:
        with _lock, _connect() as connection:
            rows = connection.execute(
                "SELECT task_id FROM task_checkpoints WHERE status IN (?, ?, ?) "
                "AND updated_at > ?",
                (PENDING, RUNNING, WAITING, cutoff),
            ).fetchall()
    except sqlite3.Error:
        logger.exception("Listing resumable tasks failed")
        return []

    tasks = [load(row[0]) for row in rows]
    return [t for t in tasks if t and t.is_resumable]


def cleanup_task(task_id: str) -> None:
    """Wipe a finished task's ephemeral artefacts.

    The checkpoint row is kept as the audit trail of what ran; the screenshots
    are not, because they do not outlive the task that took them.
    """
    pattern = os.path.join(config.SCREENSHOTS_DIR, f"{task_id}_*")
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except OSError:
            logger.debug("Could not remove ephemeral file %s", path, exc_info=True)


def mark(task_id: str, status: str) -> None:
    checkpoint = load(task_id)
    if checkpoint:
        checkpoint.status = status
        save(checkpoint)
        if status in (DONE, FAILED):
            cleanup_task(task_id)
