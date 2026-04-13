from __future__ import annotations
import json
import os
from typing import Any
import faiss
import numpy as np
from src.config.settings import Settings
from src.core.schemas import Chunk
from src.storage.vector_store_base import VectorStoreBase
from src.core.logger import get_logger

logger = get_logger("FaissStore")

class FAISSStore(VectorStoreBase):
    def __init__(self, settings:Settings) -> None:
        self.persist_dir = str(settings.artifacts_dir / "vector_index" / "faiss")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_path = os.path.join(self.persist_dir, "index.faiss")
        self.meta_path = os.path.join(self.persist_dir, "metadata.jsonl")
        self.dimension = 768
        self.index = None
        self.metadata_score = {}
        self.id_list = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as  f:
                saved = json.load(f)
            self.metadata_score = saved["metadata_score"]
            self.id_list = saved["id_list"]
            logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("created new FAISS Index (dimensions = %d)", self.dimension)

    def _save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata_score": self.metadata_score,
                "id_list": self.id_list,
            }, f)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]])-> None:
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        for chunk in chunks:
            self.id_list.append(chunk.chunk_id)
            self.metadata_score[chunk.chunk_id] = {
                "text" : chunk.text,
                "metadata" : {
                    k: v for k, v in chunk.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
        self._save()
        logger.info("Added %d chunks to FAISS (total: %d)", len(chunks), self.index.ntotal)

    def query(self, query_embedding: list[float], top_k:int)-> list[dict[str, Any]]:
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = self.id_list[idx]
            stored = self.metadata_score[chunk_id]
            hits.append({
                "chunk_id": chunk_id,
                "text": stored["text"],
                "score": float(score),
                "metadata": stored["metadata"],
            })
        return hits
        
    def count(self) -> int:
        return self.index.ntotal
    
    def delete_collection(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_score = {}
        self.id_list = []
        self._save()
        logger.info("Deleted FAISS index")