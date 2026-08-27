"""Per-role LLM call timing.

Per-call latency, not our own code, is what a turn costs - and the roles differ
enough that extrapolating from one is useless, so each is measured separately.

Deliberately trivial: an in-memory counter, no persistence, no I/O on the hot
path. `summary()` is what the task runner logs when a task finishes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RoleStats:
    calls: int = 0
    total_seconds: float = 0.0
    durations: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.total_seconds / self.calls if self.calls else 0.0


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._roles: dict[str, RoleStats] = defaultdict(RoleStats)

    def record(self, role: str, seconds: float) -> None:
        with self._lock:
            stats = self._roles[role]
            stats.calls += 1
            stats.total_seconds += seconds
            stats.durations.append(seconds)

    @contextmanager
    def time(self, role: str):
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            self.record(role, elapsed)
            logger.info("%s call took %.2fs", role, elapsed)

    def snapshot(self) -> dict[str, RoleStats]:
        with self._lock:
            return dict(self._roles)

    def summary(self) -> str:
        snap = self.snapshot()
        if not snap:
            return "no model calls recorded"
        total_calls = sum(s.calls for s in snap.values())
        total_time = sum(s.total_seconds for s in snap.values())
        parts = [
            f"{role}={s.calls}x/{s.mean:.1f}s avg"
            for role, s in sorted(snap.items(), key=lambda kv: -kv[1].total_seconds)
        ]
        return f"{total_calls} model calls, {total_time:.1f}s total | " + ", ".join(parts)

    def reset(self) -> None:
        with self._lock:
            self._roles.clear()


metrics = Metrics()
