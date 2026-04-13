"""RAG evaluation metrics — measure retrieval and generation quality.

Three core metrics:
1. Recall@k — Did we retrieve the right chunks?
2. Faithfulness — Is the answer grounded in the context (no hallucination)?
3. Answer Relevance — Does the answer address the question?
"""
from __future__ import annotations

from src.config.settings import Settings
from src.generation.ollama_generator import generate_answer
from src.generation.answer_postprocess import postprocess_answer
from src.core.logger import get_logger

logger = get_logger("Metrics")

_FAITHFULNESS_PROMPT = """You are an evaluation judge. Determine if the given answer is fully supported by the provided context.

Context:
{context}

Answer:
{answer}

Is every claim in the answer supported by the context? Respond with ONLY a number from 0 to 10:
- 0 = The answer contains entirely made-up information
- 5 = Some claims are supported, some are not
- 10 = Every claim is directly supported by the context

Score:"""

_RELEVANCE_PROMPT = """You are an evaluation judge. Determine if the answer actually addresses the question.

Question: {question}

Answer: {answer}

Does the answer fully and directly address the question? Respond with ONLY a number from 0 to 10:
- 0 = The answer is completely off-topic
- 5 = The answer partially addresses the question
- 10 = The answer fully and directly addresses the question

Score:"""


def recall_at_k(
    retrieved_texts: list[str],
    expected_keywords: list[str],
    k: int | None = None,
) -> float:
    """Measure what fraction of expected keywords appear in retrieved chunks.

    This is a proxy for true recall since we may not have labeled relevance
    judgments.  We check if the retrieved chunks contain the expected keywords.

    Args:
        retrieved_texts: Text content of retrieved chunks.
        expected_keywords: Keywords that should appear in relevant chunks.
        k: Only consider the first k results. None = use all.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not expected_keywords:
        return 1.0

    if k is not None:
        retrieved_texts = retrieved_texts[:k]

    # Combine all retrieved text into one blob for keyword search
    combined = " ".join(retrieved_texts).lower()

    found = sum(1 for kw in expected_keywords if kw.lower() in combined)
    score = found / len(expected_keywords)

    logger.info("Recall@%s: %d/%d keywords found = %.2f", k or "all", found, len(expected_keywords), score)
    return score


def faithfulness_score(
    answer: str,
    context: str,
    settings: Settings,
) -> float:
    """Use LLM to judge if the answer is grounded in the context.

    Args:
        answer: The generated answer.
        context: The retrieved context that was used.
        settings: Application settings.

    Returns:
        Score between 0.0 and 1.0 (1.0 = fully grounded).
    """
    prompt = _FAITHFULNESS_PROMPT.format(context=context[:2000], answer=answer)
    try:
        raw = generate_answer(prompt, settings)
        raw = postprocess_answer(raw)
        # Extract number
        import re
        match = re.search(r"\d+", raw.strip())
        score = int(match.group()) / 10.0 if match else 0.5
        score = min(max(score, 0.0), 1.0)
        logger.info("Faithfulness score: %.2f", score)
        return score
    except Exception as e:
        logger.warning("Faithfulness scoring failed: %s", e)
        return 0.5


def answer_relevance_score(
    question: str,
    answer: str,
    settings: Settings,
) -> float:
    """Use LLM to judge if the answer addresses the question.

    Args:
        question: The original question.
        answer: The generated answer.
        settings: Application settings.

    Returns:
        Score between 0.0 and 1.0 (1.0 = perfectly relevant).
    """
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    try:
        raw = generate_answer(prompt, settings)
        raw = postprocess_answer(raw)
        import re
        match = re.search(r"\d+", raw.strip())
        score = int(match.group()) / 10.0 if match else 0.5
        score = min(max(score, 0.0), 1.0)
        logger.info("Answer relevance score: %.2f", score)
        return score
    except Exception as e:
        logger.warning("Relevance scoring failed: %s", e)
        return 0.5
