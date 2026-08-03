def categorize(text: str) -> str:
    """Classify a fact as "dense" (semantic store) or "relational" (graph store).

    Always returns "dense" today - relational/temporal-path routing is deferred
    until graph_memory.py is actually built (Phase 10+), so this stays a stub
    rather than fake-classifying with no real graph backend to route to.
    """
    return "dense"
