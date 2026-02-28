"""
VectorStore — In-memory vector storage and similarity search via Rust.
"""

from __future__ import annotations

import json
from typing import Any


class VectorStore:
    """
    In-memory vector store backed by Rust.

    Example:
        >>> store = VectorStore()
        >>> store.upsert("doc1", [0.1, 0.2, 0.3], {"text": "Hello"})
        >>> results = store.search([0.1, 0.2, 0.3], top_k=5)
    """

    def __init__(self) -> None:
        from gauss._native import create_vector_store

        self._handle = create_vector_store()

    def upsert(self, id: str, embedding: list[float], metadata: dict[str, Any] | None = None) -> None:
        """Insert or update a vector."""
        from gauss._native import vector_store_upsert

        embedding_json = json.dumps(embedding)
        meta_json = json.dumps(metadata) if metadata else "{}"
        vector_store_upsert(self._handle, id, embedding_json, meta_json)

    def search(self, query: list[float], top_k: int = 10) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        from gauss._native import vector_store_search

        query_json = json.dumps(query)
        result_json = vector_store_search(self._handle, query_json, top_k)
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
