from __future__ import annotations
from typing import Any
import chromadb
from src.config.settings import Settings
from src.core.schemas import Chunk
from src.storage.vector_store_base import VectorStoreBase
from src.core.logger import get_logger

logger = get_logger("Storage")

class ChromaStore(VectorStoreBase):
    def __init__(self, settings: Settings) -> None:
        self.client = chromadb.PersistentClient(
            path= str(settings.chroma_persist_dir)
        )
        self.collection = self.client.get_or_create_collection(
            name= settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ids = []
        documents = []
        metadatas = []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)

            safe_meta = {
                k:v for k,v in chunk.metadata.items()
                if isinstance(v, (str, int, float, bool))
            } 
            metadatas.append(safe_meta)
        self.collection.upsert(
            ids= ids,
            documents= documents,
            metadatas= metadatas,
            embeddings=embeddings,
        )
        logger.info("Upserted %d chunks into ChromaDB", len(ids))

    def query(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results = top_k,
            include= ["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": 1.0 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return hits

    def count(self) -> int:
        return self.collection.count()
    
    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection.name)
        logger.info("Deleted collection %s", self.collection.name)