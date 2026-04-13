from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from src.core.schemas import Chunk

class VectorStoreBase(ABC):

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        pass
    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        pass
    @abstractmethod
    def count(self) -> int:
        pass
    @abstractmethod
    def delete_collection(self) -> None:
        pass