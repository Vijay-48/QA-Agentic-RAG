"""Re-ranking retrieved results using the LLM as a cross-encoder.

Instead of a separate cross-encoder model, we use the existing Ollama
chat model to score (query, chunk) relevance.  This keeps the stack
simple — only Ollama is needed, no extra model downloads.
"""
from __future__ import annotations

import re

from src.config.settings import Settings
from src.core.schemas import RetrievalHit
from src.generation.ollama_generator import generate_answer
from src.core.logger import get_logger

logger = get_logger("Ranker")

_RERANK_PROMPT = """You are a relevance scoring system. Rate how relevant the following passage is to the given query.

Query: {query}

Passage: {passage}

Score the relevance from 0 to 10 where:
- 0 = completely irrelevant
- 5 = somewhat relevant
- 10 = perfectly relevant and directly answers the query

Respond with ONLY a single integer number, nothing else."""


def _parse_score(raw: str) -> float:
    """Extract a numeric score from the LLM response."""
    match = re.search(r"\d+", raw.strip())
    if match:
        score = int(match.group())
        return min(score, 10) / 10.0  # Normalize to 0.0–1.0
    return 0.5  # Default if parsing fails


def rerank(
    query: str,
    hits: list[RetrievalHit],
    settings: Settings,
    top_n: int | None = None,
) -> list[RetrievalHit]:
    """Re-rank retrieval hits using LLM-based relevance scoring.

    Args:
        query: The user's original query.
        hits: Retrieved chunks from the vector store.
        settings: Application settings.
        top_n: Number of top results to return.  None = return all, re-sorted.

    Returns:
        Re-ranked list of RetrievalHit with updated scores.
    """
    if not hits:
        return hits

    scored_hits: list[RetrievalHit] = []

    for hit in hits:
        prompt = _RERANK_PROMPT.format(
            query=query,
            passage=hit.text[:500],  # Limit passage length for speed
        )
        try:
            raw_score = generate_answer(prompt, settings)
            score = _parse_score(raw_score)
        except Exception as e:
            logger.warning("Rerank scoring failed for chunk %s: %s", hit.chunk_id[:8], e)
            score = hit.score  # Fall back to original score

        scored_hits.append(
            RetrievalHit(
                chunk_id=hit.chunk_id,
                score=score,
                text=hit.text,
                metadata=hit.metadata,
            )
        )

    # Sort by new score (descending)
    scored_hits.sort(key=lambda h: h.score, reverse=True)

    if top_n is not None:
        scored_hits = scored_hits[:top_n]

    logger.info("Re-ranked %d hits → top %d", len(hits), len(scored_hits))
    return scored_hits
