def categorize(text: str) -> str:
    """Classify a fact as "dense" (semantic store) or "relational" (graph).

    Always "dense" until graph_memory.py is real - classifying with nothing to
    route to would be worse than a stub.
    """
    return "dense"
