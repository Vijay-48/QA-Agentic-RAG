"""Evaluation pipeline — run a test dataset through the QA pipeline and measure quality."""
from __future__ import annotations

import json
from pathlib import Path

from src.config.settings import Settings, load_settings
from src.pipeline.qa_pipeline import ask
from src.evaluation.metrics import recall_at_k, faithfulness_score, answer_relevance_score
from src.prompting.prompt_builder import format_context
from src.core.logger import get_logger

logger = get_logger("Evaluator")


def load_eval_dataset(path: str | Path) -> list[dict]:
    """Load evaluation dataset from JSON file.

    Expected format:
    [
        {
            "question": "What is chunking?",
            "expected_answer": "Chunking is ...",
            "expected_keywords": ["chunk", "split", "overlap"],
            "expected_sources": ["Rag Docs.pdf"]
        },
        ...
    ]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info("Loaded %d evaluation examples", len(dataset))
    return dataset


def evaluate_single(
    example: dict,
    settings: Settings,
) -> dict:
    """Evaluate a single question against the QA pipeline.

    Returns:
        Dict with question, actual answer, and all metric scores.
    """
    question = example["question"]
    expected_keywords = example.get("expected_keywords", [])

    # Run the QA pipeline
    result = ask(question, settings)

    # Get retrieved texts for recall
    retrieved_texts = [hit.text for hit in result.hits]

    # Build context string for faithfulness
    context = format_context(result.hits)

    # Compute metrics
    recall = recall_at_k(retrieved_texts, expected_keywords)
    faithfulness = faithfulness_score(result.answer, context, settings)
    relevance = answer_relevance_score(question, result.answer, settings)

    return {
        "question": question,
        "answer": result.answer,
        "expected_keywords": expected_keywords,
        "recall_at_k": recall,
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "num_hits": len(result.hits),
        "citations": result.citations,
    }


def run_evaluation(
    dataset_path: str | Path | None = None,
    settings: Settings | None = None,
) -> dict:
    """Run full evaluation on the dataset and return aggregate results.

    Args:
        dataset_path: Path to eval dataset JSON. Defaults to data/eval/eval_dataset.json.
        settings: Application settings. Loaded if None.

    Returns:
        Dict with per-question results and aggregate metrics.
    """
    if settings is None:
        settings = load_settings()

    if dataset_path is None:
        dataset_path = settings.data_eval_dir / "eval_dataset.json"

    dataset = load_eval_dataset(dataset_path)

    results = []
    total_recall = 0.0
    total_faithfulness = 0.0
    total_relevance = 0.0

    for i, example in enumerate(dataset):
        logger.info("Evaluating %d/%d: '%s'", i + 1, len(dataset), example["question"][:50])
        try:
            result = evaluate_single(example, settings)
            results.append(result)
            total_recall += result["recall_at_k"]
            total_faithfulness += result["faithfulness"]
            total_relevance += result["answer_relevance"]
        except Exception as e:
            logger.error("Evaluation failed for question '%s': %s", example["question"][:50], e)
            results.append({
                "question": example["question"],
                "error": str(e),
                "recall_at_k": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
            })

    n = max(len(results), 1)
    summary = {
        "total_questions": len(dataset),
        "successful": sum(1 for r in results if "error" not in r),
        "avg_recall_at_k": total_recall / n,
        "avg_faithfulness": total_faithfulness / n,
        "avg_answer_relevance": total_relevance / n,
        "per_question": results,
    }

    logger.info(
        "Evaluation complete: Recall=%.2f, Faithfulness=%.2f, Relevance=%.2f",
        summary["avg_recall_at_k"],
        summary["avg_faithfulness"],
        summary["avg_answer_relevance"],
    )
    return summary
