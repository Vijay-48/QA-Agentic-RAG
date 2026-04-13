"""Context compression — extract only the relevant parts from retrieved chunks.

After retrieval, each chunk may be 200-500 tokens but only a few sentences
are actually relevant to the query.  Compression extracts just those
sentences, reducing noise and token usage.
"""
from __future__ import annotations

from src.config.settings import Settings
from src.core.schemas import RetrievalHit
from src.generation.ollama_generator import generate_answer
from src.generation.answer_postprocess import postprocess_answer
from src.core.logger import get_logger

logger = get_logger("ContextCompressor")

_COMPRESS_PROMPT = """Extract ONLY the sentences from the passage below that are directly relevant to answering the question.
If nothing is relevant, respond with "NOT_RELEVANT".
Do not add any new information. Only extract existing sentences.

Question: {question}

Passage:
{passage}

Relevant sentences:"""


def compress_hit(question: str, hit: RetrievalHit, settings: Settings) -> RetrievalHit | None:
    """Compress a single retrieval hit to only relevant content.

    Args:
        question: The user's question.
        hit: A retrieval hit with full chunk text.
        settings: Application settings.

    Returns:
        A new RetrievalHit with compressed text, or None if nothing was relevant.
    """
    prompt = _COMPRESS_PROMPT.format(question=question, passage=hit.text)
    try:
        raw = generate_answer(prompt, settings)
        compressed = postprocess_answer(raw)

        if "NOT_RELEVANT" in compressed.upper():
            return None

        # Only use compression if it actually reduced the text
        if len(compressed) < len(hit.text) and len(compressed) > 20:
            return RetrievalHit(
                chunk_id=hit.chunk_id,
                score=hit.score,
                text=compressed,
                metadata=hit.metadata,
            )

        return hit  # Return original if compression didn't help
    except Exception as e:
        logger.warning("Compression failed for chunk %s: %s", hit.chunk_id[:8], e)
        return hit  # Return original on failure


def compress_context(
    question: str,
    hits: list[RetrievalHit],
    settings: Settings,
) -> list[RetrievalHit]:
    """Compress all retrieval hits, removing irrelevant content.

    Args:
        question: The user's question.
        hits: List of retrieval hits.
        settings: Application settings.

    Returns:
        List of compressed hits with irrelevant ones removed.
    """
    compressed = []
    original_tokens = 0
    compressed_tokens = 0

    for hit in hits:
        original_tokens += len(hit.text.split())
        result = compress_hit(question, hit, settings)
        if result is not None:
            compressed.append(result)
            compressed_tokens += len(result.text.split())

    logger.info(
        "Context compression: %d tokens → %d tokens (%d%% reduction, %d/%d chunks kept)",
        original_tokens,
        compressed_tokens,
        (1 - compressed_tokens / max(original_tokens, 1)) * 100,
        len(compressed),
        len(hits),
    )
    return compressed
