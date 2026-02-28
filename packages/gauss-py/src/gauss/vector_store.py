"""
VectorStore — In-memory vector storage and similarity search via Rust.
Rust signature: vector_store_upsert(handle, chunks_json), vector_store_search(handle, embedding_json, top_k)
"""

from __future__ import annotations

import json
from typing import Any, Optional


class VectorStore:
    """
    In-memory vector store backed by Rust.

    Example:
        >>> store = VectorStore()
        >>> await store.upsert([{"id": "doc1", "embedding": [0.1, 0.2], "metadata": {"text": "Hello"}}])
        >>> results = await store.search([0.1, 0.2], top_k=5)
    """

    def __init__(self) -> None:
        from gauss._native import create_vector_store

        self._handle = create_vector_store()

    async def upsert(self, chunks: list[dict[str, Any]]) -> None:
        """Insert or update vectors. Each chunk: {id, embedding, metadata}."""
        from gauss._native import vector_store_upsert

        await vector_store_upsert(self._handle, json.dumps(chunks))

    async def search(self, query_embedding: list[float], top_k: int = 10) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        from gauss._native import vector_store_search

        result_json = await vector_store_search(
            self._handle, json.dumps(query_embedding), top_k
        )
        return json.loads(result_json) if result_json else []

    def destroy(self) -> None:
        """Release the underlying Rust vector store."""
        from gauss._native import destroy_vector_store

        destroy_vector_store(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using Rust."""
    from gauss._native import cosine_similarity as _cosine_similarity

    return _cosine_similarity(json.dumps(a), json.dumps(b))
