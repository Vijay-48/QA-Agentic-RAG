"""Run the RAG evaluation pipeline."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.evaluator import run_evaluation

if __name__ == "__main__":
    dataset_path = None
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    print("\n📊 Running RAG Evaluation...\n")
    summary = run_evaluation(dataset_path=dataset_path)

    print(f"\n{'='*50}")
    print(f"  Evaluation Results")
    print(f"{'='*50}")
    print(f"  Questions evaluated : {summary['total_questions']}")
    print(f"  Successful          : {summary['successful']}")
    print(f"  Avg Recall@k        : {summary['avg_recall_at_k']:.3f}")
    print(f"  Avg Faithfulness    : {summary['avg_faithfulness']:.3f}")
    print(f"  Avg Relevance       : {summary['avg_answer_relevance']:.3f}")

    print(f"\n{'='*50}")
    print(f"  Per-Question Results")
    print(f"{'='*50}")
    for i, r in enumerate(summary["per_question"], 1):
        q = r["question"][:50]
        if "error" in r:
            print(f"  {i}. ❌ {q}... → Error: {r['error']}")
        else:
            print(
                f"  {i}. ✅ {q}... → "
                f"R={r['recall_at_k']:.2f} F={r['faithfulness']:.2f} A={r['answer_relevance']:.2f}"
            )
