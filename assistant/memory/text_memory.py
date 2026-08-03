import time
import uuid

import chromadb
from chromadb.utils import embedding_functions

import config


class TextMemory:
    """Dense semantic memory store backed by ChromaDB + sentence-transformers."""

    def __init__(self):
        client = chromadb.PersistentClient(path=config.MEMORY_DB_PATH)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.MEMORY_EMBEDDING_MODEL
        )
        self.collection = client.get_or_create_collection(
            name="facts",
            embedding_function=embedding_fn,
        )

    def add(self, text: str, metadata: dict | None = None) -> str:
        metadata = dict(metadata or {})
        metadata.setdefault("created_at", time.time())
        metadata.setdefault("last_accessed", metadata["created_at"])
        memory_id = str(uuid.uuid4())
        self.collection.add(documents=[text], metadatas=[metadata], ids=[memory_id])
        return memory_id

    def update(self, memory_id: str, text: str, metadata: dict) -> None:
        self.collection.update(ids=[memory_id], documents=[text], metadatas=[metadata])

    def delete(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.collection.count() == 0:
            return []
        n_results = min(k, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=n_results)

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits

    def get_all(self) -> list[dict]:
        results = self.collection.get()
        return [
            {"id": results["ids"][i], "text": results["documents"][i], "metadata": results["metadatas"][i]}
            for i in range(len(results["ids"]))
        ]
