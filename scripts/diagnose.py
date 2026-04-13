"""
RAG Pipeline Diagnostic Script
Run this to test each component and see what's inside ChromaDB.
Usage: python scripts/diagnose.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def step1_check_ollama():
    """Check if Ollama is running and list available models."""
    separator("STEP 1: Ollama Connection")
    import requests

    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"  Status: CONNECTED (HTTP {r.status_code})")
        models = r.json().get("models", [])
        if models:
            print(f"  Models available:")
            for m in models:
                size_mb = m.get("size", 0) // 1024 // 1024
                print(f"    - {m['name']} ({size_mb} MB)")
        else:
            print("  WARNING: No models pulled! Run:")
            print("    ollama pull nomic-embed-text")
            print("    ollama pull qwen3:0.6b")
        return True
    except requests.ConnectionError:
        print("  ERROR: Cannot connect to Ollama!")
        print("  Fix: Open another terminal and run: ollama serve")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def step2_test_embedding():
    """Test the embedding model."""
    separator("STEP 2: Embedding Model Test")
    from src.config.settings import load_settings
    from src.embedding.ollama_embedder import OllamaEmbedder

    settings = load_settings()
    print(f"  Model: {settings.ollama_embedding_model}")

    try:
        embedder = OllamaEmbedder(settings)
        vector = embedder.embed_single("Hello, this is a test.")
        print(f"  Status: WORKING")
        print(f"  Vector dimension: {len(vector)}")
        print(f"  First 5 values: {vector[:5]}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Fix: ollama pull {settings.ollama_embedding_model}")
        return False


def step3_inspect_chromadb():
    """Inspect what's stored in ChromaDB."""
    separator("STEP 3: ChromaDB Inspection")
    from src.config.settings import load_settings
    from src.storage.chroma_store import ChromaStore

    settings = load_settings()
    print(f"  Persist dir: {settings.chroma_persist_dir}")
    print(f"  Collection: {settings.chroma_collection_name}")

    try:
        store = ChromaStore(settings)
        count = store.count()
        print(f"  Total chunks stored: {count}")

        if count == 0:
            print("  WARNING: ChromaDB is empty! Run ingestion first:")
            print("    python scripts/run_ingest.py")
            return False

        # Show a sample of stored data
        sample = store.collection.peek(limit=3)
        print(f"\n  --- Sample Chunks (first 3) ---")
        for i in range(min(3, len(sample["ids"]))):
            chunk_id = sample["ids"][i]
            text = sample["documents"][i][:100]
            meta = sample["metadatas"][i]
            has_embedding = sample["embeddings"] is not None
            print(f"\n  Chunk {i+1}:")
            print(f"    ID: {chunk_id[:16]}...")
            print(f"    Text: \"{text}...\"")
            print(f"    Metadata: {meta}")
            print(f"    Has embedding: {has_embedding}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def step4_test_retrieval():
    """Test retrieval with a sample query."""
    separator("STEP 4: Retrieval Test")
    from src.config.settings import load_settings
    from src.retrieval.retriver import retrieve

    settings = load_settings()
    test_query = "What is chunking?"

    try:
        hits = retrieve(test_query, settings)
        print(f"  Query: \"{test_query}\"")
        print(f"  Hits found: {len(hits)}")
        for i, hit in enumerate(hits):
            print(f"\n  Hit {i+1} (score: {hit.score:.4f}):")
            print(f"    Source: {hit.metadata.get('filename', 'unknown')}")
            print(f"    Text: \"{hit.text[:120]}...\"")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def step5_test_generation():
    """Test the LLM with a simple short prompt (not RAG)."""
    separator("STEP 5: LLM Generation Test")
    from src.config.settings import load_settings
    from src.generation.ollama_generator import generate_answer

    settings = load_settings()
    print(f"  Model: {settings.ollama_chat_model}")
    print(f"  Timeout: {settings.request_timeout_seconds}s")

    short_prompt = "Reply in one sentence: What is Python?"

    try:
        print(f"  Sending short test prompt...")
        print(f"  (waiting for response, this may take a minute on first load...)")
        answer = generate_answer(short_prompt, settings)
        print(f"  Status: WORKING")
        print(f"  Answer: {answer[:200]}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"\n  Possible fixes:")
        print(f"    1. Pull the model: ollama pull {settings.ollama_chat_model}")
        print(f"    2. Try a smaller model: set OLLAMA_CHAT_MODEL=qwen3:0.6b in .env")
        print(f"    3. Increase timeout: set REQUEST_TIMEOUT_SECONDS=300 in .env")
        return False


def step6_full_qa_test():
    """Test the full QA pipeline end-to-end."""
    separator("STEP 6: Full QA Pipeline Test")
    from src.pipeline.qa_pipeline import ask

    question = "What is chunking?"

    try:
        print(f"  Question: \"{question}\"")
        print(f"  Running full pipeline...")
        result = ask(question)
        print(f"  Status: WORKING")
        print(f"  Answer: {result.answer[:300]}")
        print(f"  Citations: {len(result.citations)}")
        for c in result.citations:
            print(f"    - {c['source']} (score: {c['score']:.3f})")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


if __name__ == "__main__":
    print("\n  RAG Pipeline Diagnostics")
    print("  " + "=" * 40)

    results = {}

    results["ollama"] = step1_check_ollama()
    if not results["ollama"]:
        print("\n  STOP: Fix Ollama connection first!")
        sys.exit(1)

    results["embedding"] = step2_test_embedding()
    results["chromadb"] = step3_inspect_chromadb()

    if results["embedding"] and results["chromadb"]:
        results["retrieval"] = step4_test_retrieval()
    else:
        print("\n  SKIPPING retrieval test (embedding or chromadb failed)")

    results["generation"] = step5_test_generation()

    if all(results.get(k, False) for k in ["retrieval", "generation"]):
        results["full_qa"] = step6_full_qa_test()
    else:
        print("\n  SKIPPING full QA test (prior steps failed)")

    separator("SUMMARY")
    for step, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {step:15s} : {status}")
