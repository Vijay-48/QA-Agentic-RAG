import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline.qa_pipeline import ask

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask a question: ")
    
    result = ask(question)
    print(f"\nQuestion: {result.question}")
    print(f"\nAnswer: {result.answer}")
    print(f"\nSources:")
    for citations in result.citations:
        print(f"- {citations['source']} (score: {citations['score']:.3f})")