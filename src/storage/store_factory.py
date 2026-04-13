from __future__ import annotations

from src.config.settings import Settings
from src.storage.vector_store_base import VectorStoreBase


def get_vector_store(settings: Settings) -> VectorStoreBase:
    """Factory that returns the correct vector store based on settings."""
    store_type = settings.vector_store

    if store_type == "faiss":
        from src.storage.faiss_store import FAISSStore
        return FAISSStore(settings)

    elif store_type == "qdrant":
        from src.storage.qdrant_store import QdrantStore
        return QdrantStore(settings)

    else:  # default: chroma
        from src.storage.chroma_store import ChromaStore
        return ChromaStore(settings)
