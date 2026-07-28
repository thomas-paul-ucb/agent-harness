"""Semantic cache for full agent task runs.

Before running a task through the agent loop — which can mean several
tool calls and several model calls — check whether a semantically similar
task has already been completed. If so, return the stored answer instantly
instead of re-running the loop, at ~0 additional tokens.

Uses chromadb because this is exactly the scenario a vector index is for:
searching a growing collection of past tasks, not just a handful of items
in one conversation (that's what budget.py's direct cosine similarity is
for — see the note there on why they're different tools).
"""

from __future__ import annotations

import uuid

import chromadb
from sentence_transformers import SentenceTransformer


class SemanticCache:
    """Caches (task -> final answer) pairs and retrieves by meaning, not
    exact text match."""

    _embedder: SentenceTransformer | None = None  # loaded once, shared across instances

    def __init__(
        self,
        persist_directory: str = ".cache/semantic_cache",
        similarity_threshold: float = 0.90,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name="task_cache",
            metadata={"hnsw:space": "cosine"},
        )

    def _get_embedder(self) -> SentenceTransformer:
        if SemanticCache._embedder is None:
            SemanticCache._embedder = SentenceTransformer(self.embedding_model_name)
        return SemanticCache._embedder

    def lookup(self, task: str) -> str | None:
        """Return a cached answer if a semantically similar task exists
        above the similarity threshold, otherwise None."""
        if self._collection.count() == 0:
            return None

        embedder = self._get_embedder()
        query_embedding = embedder.encode(task).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
        )

        if not results["ids"][0]:
            return None

        # chromadb returns cosine *distance* (0 = identical), so convert to similarity.
        distance = results["distances"][0][0]
        similarity = 1 - distance

        if similarity >= self.similarity_threshold:
            return results["metadatas"][0][0]["answer"]
        return None

    def store(self, task: str, answer: str) -> None:
        """Store a completed (task, answer) pair for future lookups."""
        embedder = self._get_embedder()
        embedding = embedder.encode(task).tolist()

        self._collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            metadatas=[{"task": task, "answer": answer}],
        )