from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.config.settings import Settings
from src.core.schemas import Chunk
from src.storage.vector_store_base import VectorStoreBase
from src.core.logger import get_logger

logger = get_logger("QdrantStore")


def _to_uuid(hex_id: str) -> str:
    """Convert 32-char hex chunk_id to UUID format for Qdrant."""
    h = hex_id[:32].ljust(32, "0")
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class QdrantStore(VectorStoreBase):

    def __init__(self, settings: Settings) -> None:
        self.collection_name = settings.chroma_collection_name

        # Decide: Cloud or Local
        if settings.qdrant_url:
            # CLOUD MODE — connect to Qdrant Cloud cluster
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=settings.request_timeout_seconds,
                check_compatibility=False,
            )
            logger.info("Connected to Qdrant Cloud: %s", settings.qdrant_url)
        else:
            # LOCAL MODE — embedded, persisted to disk
            self.persist_dir = str(settings.artifacts_dir / "vector_index" / "qdrant")
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = QdrantClient(path=self.persist_dir)
            logger.info("Using local Qdrant at: %s", self.persist_dir)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created collection '%s' (dim=768, cosine)", self.collection_name)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        points = []
        for chunk, vector in zip(chunks, embeddings):
            safe_meta = {
                k: v for k, v in chunk.metadata.items()
                if isinstance(v, (str, int, float, bool))
            }
            payload = {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                **safe_meta,
            }
            points.append(
                PointStruct(
                    id=_to_uuid(chunk.chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info("Upserted %d points to Qdrant", len(points))

    def query(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        hits = []
        for result in results:
            payload = dict(result.payload) if result.payload else {}
            text = payload.pop("text", "")
            chunk_id = payload.pop("chunk_id", str(result.id))
            hits.append({
                "chunk_id": chunk_id,
                "text": text,
                "score": result.score,
                "metadata": payload,
            })
        return hits

    def count(self) -> int:
        """Equivalent to: POST /collections/rag_chunks/points/count"""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        logger.info("Deleted Qdrant collection '%s'", self.collection_name)
