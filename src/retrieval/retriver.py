from __future__ import annotations
from src.config.settings import Settings
from src.core.schemas import RetrievalHit
from src.embedding.ollama_embedder import OllamaEmbedder
from src.storage.store_factory import get_vector_store
from src.preprocessing.query_preprocess import preprocess_query
from src.core.logger import get_logger

logger = get_logger("Retriver")

def retrieve(query: str, settings: Settings) -> list[RetrievalHit]:
    query = preprocess_query(query)
    embedder = OllamaEmbedder(settings)
    query_vector = embedder.embed_single(query)
    store = get_vector_store(settings)
    raw_results = store.query(query_vector, settings.retrieval_top_k)
    hits = []
    for result in raw_results:
        hits.append(
            RetrievalHit(
                chunk_id=result["chunk_id"],
                score = result["score"],
                text = result["text"],
                metadata=result["metadata"]
            )
        )
    logger.info("Retrieved %d hits for query: '%s'", len(hits), query[:50])
    return hits
    