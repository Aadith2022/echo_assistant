class GraphMemory:
    """Stub for a future Graphiti-backed temporal knowledge graph.

    A no-op seam, so MemoryRouter's call sites do not change when a real graph
    backend is added.
    """

    def remember(self, text: str, metadata: dict | None = None) -> None:
        return None

    def recall(self, query: str, k: int = 5) -> list[dict]:
        return []
