import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.ingest_pipeline import run_ingestion


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]

    else:
        paths = [os.path.join(os.path.dirname(__file__), "..", "data", "raw")]

    print(f"Ingesting documents from: {paths}")
    result = run_ingestion(paths)
    print(f"\nDone")
    print(f"Documents loaded: {result['documents_loaded']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"Chunks stored: {result['chunks_stored']}")