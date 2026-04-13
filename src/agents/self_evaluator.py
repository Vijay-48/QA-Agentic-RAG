"""Self-evaluator — agent judges its own answer quality after generation.

If the self-evaluation score is too low, the agent can retry with
different search strategies or provide a caveat with its answer.
"""
from __future__ import annotations

import re

from src.config.settings import Settings
from src.agents.prompts import SELF_EVAL_PROMPT
from src.generation.ollama_generator import generate_answer
from src.generation.answer_postprocess import postprocess_answer
from src.core.logger import get_logger

logger = get_logger("SelfEvaluator")


def self_evaluate(
    question: str,
    answer: str,
    context: str,
    settings: Settings,
) -> dict[str, float]:
    """Have the LLM evaluate its own answer quality.

    Args:
        question: The original question.
        answer: The generated answer.
        context: The context/evidence used.
        settings: Application settings.

    Returns:
        Dict with 'completeness', 'accuracy', 'clarity' scores (0.0-1.0)
        and 'overall' average score.
    """
    prompt = SELF_EVAL_PROMPT.format(
        question=question,
        context=context[:2000],
        answer=answer,
    )

    default_scores = {
        "completeness": 0.5,
        "accuracy": 0.5,
        "clarity": 0.5,
        "overall": 0.5,
    }

    try:
        raw = generate_answer(prompt, settings)
        raw = postprocess_answer(raw)

        # Parse "7,8,9" format
        numbers = re.findall(r"\d+", raw)
        if len(numbers) >= 3:
            completeness = min(int(numbers[0]), 10) / 10.0
            accuracy = min(int(numbers[1]), 10) / 10.0
            clarity = min(int(numbers[2]), 10) / 10.0
            overall = (completeness + accuracy + clarity) / 3.0

            scores = {
                "completeness": completeness,
                "accuracy": accuracy,
                "clarity": clarity,
                "overall": overall,
            }

            logger.info(
                "Self-eval scores: completeness=%.1f, accuracy=%.1f, clarity=%.1f, overall=%.1f",
                completeness, accuracy, clarity, overall,
            )
            return scores

        logger.warning("Could not parse self-eval scores from: '%s'", raw[:100])
        return default_scores

    except Exception as e:
        logger.warning("Self-evaluation failed: %s", e)
        return default_scores


def should_retry(scores: dict[str, float], threshold: float = 0.4) -> bool:
    """Determine if the answer quality is too low and should be retried.

    Args:
        scores: Self-evaluation scores dict.
        threshold: Minimum acceptable overall score.

    Returns:
        True if the answer should be retried.
    """
    return scores.get("overall", 0.5) < threshold
