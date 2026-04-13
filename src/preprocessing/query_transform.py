"""Query transformation techniques to improve retrieval quality.

Three strategies:
1. Query Expansion — generate multiple query variations
2. Query Decomposition — break complex questions into sub-questions
3. HyDE — generate a hypothetical answer and search with that
"""
from __future__ import annotations

from src.config.settings import Settings
from src.generation.ollama_generator import generate_answer
from src.generation.answer_postprocess import postprocess_answer
from src.core.logger import get_logger

logger = get_logger("QueryTransform")

_EXPAND_PROMPT = """Generate 3 alternative versions of the following search query. 
Each version should express the same information need using different words and phrasing.
Return ONLY the 3 queries, one per line, numbered 1-3. No explanations.

Original query: {query}

Alternative queries:"""

_DECOMPOSE_PROMPT = """Break the following complex question into 2-4 simpler sub-questions 
that together would help answer the original question completely.
Return ONLY the sub-questions, one per line, numbered 1-4. No explanations.

Complex question: {query}

Sub-questions:"""

_HYDE_PROMPT = """Write a short paragraph that would be a perfect answer to the following question.
Write it as if you are quoting from a textbook or documentation.
Do not say "I don't know". Write a plausible, detailed answer.

Question: {query}

Answer paragraph:"""


def expand_query(query: str, settings: Settings) -> list[str]:
    """Generate multiple query variations for broader retrieval.

    Args:
        query: The original user query.
        settings: Application settings.

    Returns:
        List of query strings including the original.
    """
    prompt = _EXPAND_PROMPT.format(query=query)
    try:
        raw = generate_answer(prompt, settings)
        raw = postprocess_answer(raw)
        # Parse numbered lines
        variations = [query]  # Always include original
        for line in raw.strip().split("\n"):
            line = line.strip()
            # Remove numbering like "1.", "1)", "1:"
            if line and line[0].isdigit():
                line = line.lstrip("0123456789.):- ").strip()
            if line and line != query:
                variations.append(line)
        logger.info("Expanded query into %d variations", len(variations))
        return variations[:4]  # Cap at 4 total
    except Exception as e:
        logger.warning("Query expansion failed: %s. Using original.", e)
        return [query]


def decompose_query(query: str, settings: Settings) -> list[str]:
    """Break a complex question into simpler sub-questions.

    Args:
        query: The complex user query.
        settings: Application settings.

    Returns:
        List of sub-questions.
    """
    prompt = _DECOMPOSE_PROMPT.format(query=query)
    try:
        raw = generate_answer(prompt, settings)
        raw = postprocess_answer(raw)
        sub_questions = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                line = line.lstrip("0123456789.):- ").strip()
            if line:
                sub_questions.append(line)
        logger.info("Decomposed query into %d sub-questions", len(sub_questions))
        return sub_questions[:4] if sub_questions else [query]
    except Exception as e:
        logger.warning("Query decomposition failed: %s. Using original.", e)
        return [query]


def hyde_transform(query: str, settings: Settings) -> str:
    """Generate a Hypothetical Document Embedding (HyDE) for the query.

    Instead of searching with the question, we search with a hypothetical
    answer.  The answer embedding is closer in vector space to real answers
    than the question embedding is.

    Args:
        query: The original user query.
        settings: Application settings.

    Returns:
        A hypothetical answer text to embed and search with.
    """
    prompt = _HYDE_PROMPT.format(query=query)
    try:
        raw = generate_answer(prompt, settings)
        hypothesis = postprocess_answer(raw)
        logger.info("HyDE generated %d-char hypothetical answer", len(hypothesis))
        return hypothesis
    except Exception as e:
        logger.warning("HyDE generation failed: %s. Using original query.", e)
        return query
