from __future__ import annotations

from src.config.settings import Settings
from src.core.schemas import Chunk
from src.embedding.ollama_embedder import OllamaEmbedder
from src.storage.store_factory import get_vector_store
from src.core.logger import get_logger

logger = get_logger("EmbeddingPipeline")


def embed_and_store(chunks: list[Chunk], settings: Settings) -> int:
    embedder = OllamaEmbedder(settings)
    store = get_vector_store(settings)
    
    batch_size = settings.embedding_batch_size
    total_stored = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        texts = [chunk.text for chunk in batch]
        vectors = embedder.embed_batch(texts)
        store.add_chunks(batch, vectors)

        total_stored += len(batch)
        logger.info(
            "Progress: %d / %d chunks embedded and stored",
            total_stored,
            len(chunks),
        )
    return total_stored
