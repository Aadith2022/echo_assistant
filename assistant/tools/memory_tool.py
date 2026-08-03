import logging
from concurrent.futures import ThreadPoolExecutor

from memory.store import MemoryRouter

logger = logging.getLogger(__name__)

_router = MemoryRouter()
# Fire-and-forget writes: the caller doesn't need to wait for the embed +
# Chroma write to complete before the turn continues. Fast enough in practice
# (well under a second) that it lands before the next turn's auto-recall runs,
# but this is a best-effort ordering, not a guarantee — acceptable for a
# single-user desktop app; a hosted multi-user version would need a durable
# queue instead of a bare thread pool.
_write_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory-write")


def remember_fact(text: str) -> str:
    def _write():
        try:
            _router.remember(text)
        except Exception:
            logger.exception("Background memory write failed for: %r", text)

    _write_executor.submit(_write)
    return "Saved to memory."


def recall_memories(query: str, k: int = 5) -> str:
    hits = _router.recall(query, k=k)
    if not hits:
        return "No relevant memories found."
    return "\n".join(f"- {hit['text']}" for hit in hits)
