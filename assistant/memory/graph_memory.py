class GraphMemory:
    """Stub for a future Graphiti-backed temporal knowledge graph (Phase 10+).

    Kept as a no-op seam so MemoryRouter's call sites don't need to change
    when the real graph backend is added later.
    """

    def remember(self, text: str, metadata: dict | None = None) -> None:
        return None

    def recall(self, query: str, k: int = 5) -> list[dict]:
        return []
