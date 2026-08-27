from rank_bm25 import BM25Okapi

from memory import coherence
from memory.categories import categorize
from memory.graph_memory import GraphMemory
from memory.text_memory import TextMemory

# Reciprocal Rank Fusion constant (Cormack et al. 2009). Combines ranked lists
# from different scoring methods without normalizing their raw scores.
RRF_K = 60


class MemoryRouter:
    """Single entry point the rest of the assistant talks to for memory."""

    def __init__(self):
        self.text_memory = TextMemory()
        self.graph_memory = GraphMemory()

    def remember(self, text: str, metadata: dict | None = None) -> str:
        category = categorize(text)
        metadata = coherence.tag_new_memory(metadata)

        memory_id = self.text_memory.add(text, metadata)
        if category == "relational":
            self.graph_memory.remember(text, metadata)

        return memory_id

    def recall(self, query: str, k: int = 5) -> list[dict]:
        oversample = max(k * 3, 10)

        semantic_hits = coherence.rerank(self.text_memory.search(query, k=oversample))
        keyword_hits = self._keyword_search(query, k=oversample)

        fused = self._reciprocal_rank_fusion([semantic_hits, keyword_hits], k=k)
        graph_hits = self.graph_memory.recall(query, k=k)

        return fused + graph_hits

    def _keyword_search(self, query: str, k: int) -> list[dict]:
        docs = self.text_memory.get_all()
        if not docs:
            return []

        tokenized_corpus = [doc["text"].lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())

        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, score in ranked[:k] if score > 0]

    def _reciprocal_rank_fusion(self, ranked_lists: list[list[dict]], k: int) -> list[dict]:
        scores: dict[str, float] = {}
        items: dict[str, dict] = {}

        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list):
                items[item["id"]] = item
                scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [items[memory_id] for memory_id in ranked_ids[:k]]
