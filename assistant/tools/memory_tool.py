import logging
from concurrent.futures import ThreadPoolExecutor

from memory.store import MemoryRouter

logger = logging.getLogger(__name__)

_router = MemoryRouter()
# Fire-and-forget: the turn does not wait for the embed and Chroma write.
# Best-effort ordering rather than a guarantee - fine for a single-user
# desktop app, but a hosted version would need a durable queue.
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
