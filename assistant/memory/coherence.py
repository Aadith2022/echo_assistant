import time

# Half-life (in days) used for recency decay when re-ranking search hits.
DECAY_HALF_LIFE_DAYS = 30


def tag_new_memory(metadata: dict | None = None) -> dict:
    """Stamp a new memory with creation/update timestamps.

    Writes are ADD-only: a changed fact is stored alongside the old one rather
    than replacing it, and recency surfaces the current one at read time.
    """
    metadata = dict(metadata or {})
    now = time.time()
    metadata.setdefault("created_at", now)
    metadata["updated_at"] = now
    return metadata


def decayed_relevance(distance: float, metadata: dict, recency_weight: float = 0.3) -> float:
    """Combine semantic distance with recency decay into one rank score.

    Lower is better, mirroring Chroma's distance convention, so recency is
    subtracted - which is what lets a newer contradicting fact outrank an
    older one without either being deleted.
    """
    updated_at = metadata.get("updated_at", metadata.get("created_at", time.time()))
    age_days = max(0.0, (time.time() - updated_at) / 86400)
    recency_decay = 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)

    return distance - (recency_weight * recency_decay)


def rerank(hits: list[dict]) -> list[dict]:
    scored = [(decayed_relevance(hit["distance"], hit["metadata"]), hit) for hit in hits]
    scored.sort(key=lambda pair: pair[0])
    return [hit for _, hit in scored]
