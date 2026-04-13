from __future__ import annotations

from src.config.settings import Settings, load_settings, ensure_runtime_dirs
from src.ingestion.loaders import load_documents
from src.preprocessing.document_preprocess import preprocess_documents
from src.chunking.chunker import chunk_documents
from src.embedding.embed_pipeline import embed_and_store
from src.core.logger import get_logger

logger = get_logger("IngestPipeline")

def run_ingestion(paths: list[str], settings: Settings | None = None) -> dict:
    if settings is None:
        settings = load_settings()
    ensure_runtime_dirs(settings)

    #step-1 = load documents
    logger.info("Loading documents form %d paths...", len(paths))
    documents = load_documents(paths)
    #step-2 = preprocess documents
    logger.info("loaded %d documents", len(documents))
    documents = preprocess_documents(documents)
    logger.info("After preprocessing: %d documents", len(documents))
    #step-3 = chunk documents
    chunks = chunk_documents(documents, settings)
    logger.info("Created %d chunks", len(chunks))
    #step-4 = embed and store chunks
    stored = embed_and_store(chunks, settings)
    logger.info("Stored %d chunks in vector store (%s)", stored, settings.vector_store)

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": stored,
    }