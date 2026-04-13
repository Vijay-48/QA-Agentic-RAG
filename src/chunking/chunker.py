from __future__ import annotations

from src.core.schemas import Document, Chunk, build_chunk_id
from src.ingestion.metadata import build_chunk_metadata
from src.config.settings import Settings
from src.chunking.strategies import fixed_size_chunks, sliding_window_chunks, semantic_chunks
from src.core.logger import get_logger

logger = get_logger("chunking")

def chunk_document(document: Document, settings: Settings) -> list[Chunk]:
    strategy = settings.chunk_strategy

    if strategy == "fixed":
        raw_chunks = fixed_size_chunks(
            document.text,
            settings.chunk_size_tokens,
            settings.chunk_overlap_tokens,
        )
    elif strategy == "sliding":
        raw_chunks = sliding_window_chunks(
            document.text,
            settings.chunk_size_tokens,
            settings.chunk_overlap_tokens,
        )
    else:
        raw_chunks = semantic_chunks(
            document.text,
            settings.chunk_min_tokens,
            settings.chunk_hard_max_tokens,
        )
    chunks = []
    total = len(raw_chunks)
    for index, chunk_text in enumerate(raw_chunks):
        chunk_id = build_chunk_id(document.doc_id, index, chunk_text)
        metadata = build_chunk_metadata(document, index, total, chunk_text)
        chunks.append(
            Chunk(
                chunk_id = chunk_id,
                doc_id = document.doc_id,
                text = chunk_text,
                metadata = metadata,
            )
        )
    logger.info("Document '%s' -> %d chunks (strategy = %s)", document.doc_id[:8], total, strategy)
    return chunks

def chunk_documents(documents: list[Document], settings: Settings) -> list[Chunk]:
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, settings))
    logger.info("Total chunks from %d documents: %d", len(documents), len(all_chunks))
    return all_chunks

