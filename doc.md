# 🛠️ Q&A RAG Pipeline — Step-by-Step Coding Guide

## Your Project at a Glance

You have a well-architected project with **12 modules** under `src/`. Here's the current status:

### ✅ Already Completed (7 files)
| File | Module | What It Does |
|------|--------|-------------|
| [schemas.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py) | `core` | Data classes: [Document](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#21-27), [Chunk](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#29-35), [RetrievalHit](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#37-43), [AnswerResult](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#45-51) + ID builders |
| [settings.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py) | `config` | All settings: Ollama URLs, chunk params, ChromaDB paths, loaded from [.env](file:///a:/Project/Q&A%20RAG%20Pipeline/.env) |
| [loaders.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/loaders.py) | `ingestion` | Reads PDF/TXT/MD files, cleans text, creates [Document](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#21-27) objects |
| [cleaners.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/cleaners.py) | `ingestion` | Normalizes newlines, removes hyphen breaks, collapses whitespace |
| [metadata.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/metadata.py) | `ingestion` | Builds metadata dicts for documents and chunks |
| [document_preprocess.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/preprocessing/document_preprocess.py) | `preprocessing` | Re-cleans document text, filters out empty docs |
| [query_preprocess.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/preprocessing/query_preprocess.py) | `preprocessing` | Strips and normalizes user query whitespace |

### 🔲 Empty — Need Your Code (19 files)
| File | Module | Purpose |
|------|--------|---------|
| [exceptions.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/exceptions.py) | `core` | Custom exception classes |
| [logger.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/logger.py) | `core` | Logging setup |
| [strategies.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/chunking/strategies.py) | `chunking` | Fixed-size, semantic, sliding-window chunking functions |
| [chunker.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/chunking/chunker.py) | `chunking` | Orchestrator that picks a strategy and chunks a document |
| [ollama_embedder.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/embedding/ollama_embedder.py) | `embedding` | Calls Ollama API to get embedding vectors |
| [embed_pipeline.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/embedding/embed_pipeline.py) | `embedding` | Batch-embeds a list of chunks |
| [vector_store_base.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/storage/vector_store_base.py) | `storage` | Abstract base class for any vector store |
| [in_memory_store.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/storage/in_memory_store.py) | `storage` | Simple numpy-based store (for testing) |
| [retriver.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/retrieval/retriver.py) | `retrieval` | Queries ChromaDB, returns [RetrievalHit](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#37-43) list |
| [rankers.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/retrieval/rankers.py) | `retrieval` | Re-ranks hits (BM25, cross-encoder) |
| [filters.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/retrieval/filters.py) | `retrieval` | Metadata-based filtering for results |
| [templates.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/prompting/templates.py) | `prompting` | RAG prompt templates as string constants |
| [prompt_builder.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/prompting/prompt_builder.py) | `prompting` | Assembles context + query into a final prompt |
| [ollama_generator.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/generation/ollama_generator.py) | `generation` | Calls Ollama chat API to generate the answer |
| [answer_postprocess.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/generation/answer_postprocess.py) | `generation` | Cleans/formats the LLM response |
| [ingest_pipeline.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/pipeline/ingest_pipeline.py) | `pipeline` | End-to-end: load → preprocess → chunk → embed → store |
| [qa_pipeline.py](file:///a:/Project/Q&A%20RAG%20Pipeline/tests/test_qa_pipeline.py) | `pipeline` | End-to-end: query → retrieve → prompt → generate → return |
| [evaluator.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/evaluation/evaluator.py) | `evaluation` | Runs eval datasets against the pipeline |
| [metrics.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/evaluation/metrics.py) | `evaluation` | Recall@k, faithfulness, relevance scoring |

---

## 📊 How the RAG Pipeline Works (Data Flow)

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                          │
│                                                                 │
│  PDF/TXT/MD ──→ loaders.py ──→ cleaners.py ──→ Document        │
│       │                                            │            │
│       │              document_preprocess.py ←───────┘            │
│       │                        │                                │
│       ▼                        ▼                                │
│  strategies.py ←──── chunker.py ──→ list[Chunk]                 │
│  (fixed/semantic/                       │                       │
│   sliding window)                       ▼                       │
│                        ollama_embedder.py ──→ vectors           │
│                                  │                              │
│                                  ▼                              │
│                        ChromaDB (storage) ──→ stored on disk    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      QA PIPELINE                                │
│                                                                 │
│  User Question ──→ query_preprocess.py ──→ clean query          │
│                              │                                  │
│                              ▼                                  │
│                     ollama_embedder.py ──→ query vector          │
│                              │                                  │
│                              ▼                                  │
│                    retriver.py (ChromaDB) ──→ top-k chunks      │
│                              │                                  │
│                              ▼                                  │
│                    prompt_builder.py ──→ final prompt            │
│                    (context + question)                          │
│                              │                                  │
│                              ▼                                  │
│                   ollama_generator.py ──→ raw answer             │
│                              │                                  │
│                              ▼                                  │
│                  answer_postprocess.py ──→ AnswerResult          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗺️ Coding Order — 10 Steps

The order follows the **dependency chain**: each step only depends on things you've already built. This is critical — you can test each step independently before moving forward.

---

### Step 1: [src/core/logger.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/logger.py) — Logging Setup

**Why first?** Every subsequent file will import the logger. Set it up once and you'll see clear output from every module as you build.

**What to code:**
- A function `get_logger(name: str)` that returns a configured `logging.Logger`
- Use Python's built-in `logging` module
- Set format to include timestamp, module name, and level: `"%(asctime)s | %(name)s | %(levelname)s | %(message)s"`
- Set default level to `INFO` (optionally read from env var `LOG_LEVEL`)
- Add a `StreamHandler` for console output

**How it works:** Python's `logging.getLogger(name)` returns a named logger. If you call it with the same name twice, you get the same logger instance (singleton pattern). The format string tells Python how to structure each log line. `StreamHandler` sends logs to your terminal.

**Test it:** After writing, open a Python shell and do:
```python
from src.core.logger import get_logger
log = get_logger("test")
log.info("Hello RAG!")  # Should print formatted line
```

---

### Step 2: [src/core/exceptions.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/exceptions.py) — Custom Exceptions

**Why now?** Clean error handling from day one. These are simple classes that later modules will raise.

**What to code:**
- `class RAGPipelineError(Exception)` — base for all your custom errors
- `class IngestionError(RAGPipelineError)` — file loading failures
- `class ChunkingError(RAGPipelineError)` — chunking issues
- `class EmbeddingError(RAGPipelineError)` — Ollama API failures
- `class RetrievalError(RAGPipelineError)` — ChromaDB query failures
- `class GenerationError(RAGPipelineError)` — LLM response failures

**How it works:** Custom exceptions let you catch specific error types. For example, `except EmbeddingError` will only catch embedding failures, not file loading issues. The inheritance from `RAGPipelineError` lets you also catch ALL pipeline errors with a single `except RAGPipelineError`.

---

### Step 3: [src/chunking/strategies.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/chunking/strategies.py) — Chunking Functions

**Why now?** This is the first major RAG concept you'll implement. Chunking determines the quality of your retrieval — too big and you get noisy context, too small and you lose meaning.

**What to code — 3 functions:**

#### 3a: `fixed_size_chunks(text, chunk_size, overlap)`
- Split text into words using `text.split()`
- Walk through the word list, taking `chunk_size` words at a time
- On each step, move forward by `chunk_size - overlap` words
- Join each group back into a string
- Return `list[str]`

**How it works:** If you have 100 words, chunk_size=30, overlap=5: Chunk 1 = words[0:30], Chunk 2 = words[25:55], Chunk 3 = words[50:80], Chunk 4 = words[75:100]. The overlap means each chunk shares 5 words with its neighbor, preserving context at boundaries.

#### 3b: `sliding_window_chunks(text, window_size, step_size)`
- Very similar to fixed-size, but you think in terms of a "window" sliding across the text
- Split on whitespace, slide a window of `window_size` words, moving `step_size` words each time
- The key difference from fixed-size: `step_size` is explicit, giving finer control

**How it works:** If window=40, step=10: you get chunks with 75% overlap. This is useful for dense technical documents where important information might appear at any boundary.

#### 3c: `semantic_chunks(text, min_tokens, max_tokens)`
- Split text on paragraph boundaries (`\n\n`)
- Merge small paragraphs until you reach `min_tokens` words
- If a single paragraph exceeds `max_tokens`, fall back to fixed-size chunking on that paragraph
- Return the list of semantically coherent chunks

**How it works:** Instead of blindly cutting at word counts, you respect natural paragraph breaks. A paragraph about "What is RAG?" stays together rather than being split mid-sentence. This gives the LLM more coherent context.

> [!TIP]
> Use `len(text.split())` as a simple word/token counter. It's not exact for real tokenizers, but good enough for this stage.

---

### Step 4: [src/chunking/chunker.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/chunking/chunker.py) — Chunking Orchestrator

**Why now?** Wires up the strategies to your document model using settings.

**What to code:**
- Import [Settings](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py#20-45) from `src.config.settings`
- Import [Document](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#21-27), [Chunk](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#29-35), [build_chunk_id](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#17-19) from `src.core.schemas`
- Import [build_chunk_metadata](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/metadata.py#27-43) from `src.ingestion.metadata`
- Import all 3 strategies from `src.chunking.strategies`
- A function `chunk_document(document: Document, settings: Settings) -> list[Chunk]`:
  1. Read `settings.chunk_strategy` (one of `"fixed"`, `"sliding"`, `"semantic"`)
  2. Call the matching strategy function with the document's text and settings params (`chunk_size_tokens`, `chunk_overlap_tokens`, etc.)
  3. For each chunk text returned, build a [Chunk](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#29-35) object with:
     - [chunk_id](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#17-19) from [build_chunk_id(doc.doc_id, index, chunk_text)](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#17-19)
     - `doc_id` from the document
     - [text](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/cleaners.py#26-32) = the chunk text
     - [metadata](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/metadata.py#27-43) from [build_chunk_metadata(document, index, total_chunks, chunk_text)](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/metadata.py#27-43)
  4. Return the list of [Chunk](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#29-35) objects
- A function `chunk_documents(documents: list[Document], settings: Settings) -> list[Chunk]`:
  - Loops over documents, calls `chunk_document` on each, collects all chunks flat

**How it works:** This is the **Strategy Pattern** — the `settings.chunk_strategy` string selects which algorithm runs. The orchestrator handles the mapping from raw text chunks (strings) to structured [Chunk](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#29-35) objects with IDs and metadata. This separation means you can add new strategies later without touching the orchestrator.

**Test it:**
```python
from src.config.settings import load_settings
from src.core.schemas import Document
from src.chunking.chunker import chunk_document

settings = load_settings()
doc = Document(doc_id="test1", source="test.txt", text="Your sample text here " * 100)
chunks = chunk_document(doc, settings)
print(f"Got {len(chunks)} chunks")
print(chunks[0].text[:100])
```

---

### Step 5: [src/embedding/ollama_embedder.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/embedding/ollama_embedder.py) — Embedding via Ollama

**Why now?** You need vectors to store in ChromaDB. This is the bridge between text and math.

**What to code:**
- Import `requests` for HTTP calls
- Import your logger, [Settings](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py#20-45), and `EmbeddingError`
- A class `OllamaEmbedder`:
  - `__init__(self, settings: Settings)` — store the base URL, model name, timeout
  - `embed_single(self, text: str) -> list[float]`:
    1. POST to `{base_url}/api/embed` with JSON body: `{"model": model_name, "input": text}`
    2. Parse response JSON — Ollama returns `{"embeddings": [[0.1, 0.2, ...]]}`
    3. Return `response["embeddings"][0]`
    4. Wrap in try/except, raise `EmbeddingError` on failures
  - `embed_batch(self, texts: list[str]) -> list[list[float]]`:
    1. POST to the same endpoint but with `"input": texts` (Ollama accepts a list)
    2. Return `response["embeddings"]`
    3. If the batch fails, fall back to calling `embed_single` for each text

**How it works:** Ollama runs locally as an HTTP server. When you call `/api/embed`, it passes the text through a neural network (like `nomic-embed-text`) that converts text into a fixed-size vector (e.g., 768 floats). Texts with similar meanings get vectors that point in similar directions. "What is Python?" and "Tell me about the Python language" would have vectors very close together in this 768-dimensional space.

> [!IMPORTANT]
> Make sure Ollama is running (`ollama serve`) and you've pulled the embedding model: `ollama pull nomic-embed-text`

---

### Step 6: [src/storage/vector_store_base.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/storage/vector_store_base.py) + ChromaDB Store

**Why now?** You need somewhere to put those embeddings. ChromaDB is your main store.

#### 6a: [vector_store_base.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/storage/vector_store_base.py) — Abstract Base Class

**What to code:**
- Import `abc.ABC, abstractmethod`
- Import [Chunk](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#29-35) from schemas
- A class `VectorStoreBase(ABC)` with abstract methods:
  - `add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None`
  - [query(self, query_embedding: list[float], top_k: int) -> list[dict]](file:///a:/Project/Q&A%20RAG%20Pipeline/src/preprocessing/query_preprocess.py#6-10) — returns list of `{"chunk_id", "text", "score", "metadata"}`
  - `count(self) -> int`
  - `delete_collection(self) -> None`

**How it works:** An abstract base class is a contract. Any store (ChromaDB, FAISS, Pinecone) must implement these 4 methods. Your retrieval module only talks to this interface, not to ChromaDB directly — so swapping stores later is trivial.

#### 6b: Create a new file `src/storage/chroma_store.py`

**What to code:**
- Import `chromadb`
- Import `VectorStoreBase`, [Chunk](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#29-35), [Settings](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py#20-45)
- A class `ChromaStore(VectorStoreBase)`:
  - `__init__(self, settings: Settings)`:
    1. Create a ChromaDB persistent client: `chromadb.PersistentClient(path=str(settings.chroma_persist_dir))`
    2. Get or create a collection: `client.get_or_create_collection(name=settings.chroma_collection_name, metadata={"hnsw:space": "cosine"})`
  - `add_chunks(self, chunks, embeddings)`:
    1. Prepare lists: `ids`, [documents](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/loaders.py#58-80), `metadatas`, `embeddings`
    2. For each chunk: `ids.append(chunk.chunk_id)`, `documents.append(chunk.text)`, `metadatas.append(chunk.metadata)`
    3. Call `self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)`
    4. Use `upsert` instead of `add` so re-running ingestion doesn't create duplicates
  - [query(self, query_embedding, top_k)](file:///a:/Project/Q&A%20RAG%20Pipeline/src/preprocessing/query_preprocess.py#6-10):
    1. Call `self.collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])`
    2. ChromaDB returns `{"ids": [[...]], "documents": [[...]], "distances": [[...]], "metadatas": [[...]]}`
    3. Zip the inner lists together and build a list of dicts with [chunk_id](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#17-19), [text](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/cleaners.py#26-32), `score` (convert distance to similarity: `1 - distance` for cosine), [metadata](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/metadata.py#27-43)
  - `count(self)`: return `self.collection.count()`
  - `delete_collection(self)`: `self.client.delete_collection(self.collection.name)`

**How it works:** ChromaDB stores your chunks as a combination of: the original text, a metadata dictionary, and the embedding vector. When you query, it computes **cosine similarity** between the query vector and all stored vectors using an HNSW index (approximate nearest neighbors). It returns the top-k most similar chunks. The `PersistentClient` saves everything to disk at `artifacts/vector_index/chroma/`, so data survives restarts.

> [!NOTE]
> ChromaDB metadata values must be [str](file:///a:/Project/Q&A%20RAG%20Pipeline/src/ingestion/cleaners.py#21-24), [int](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py#10-18), `float`, or `bool`. If any metadata field is a `list` or `None`, ChromaDB will throw an error. You may need to filter/convert metadata before storing.

---

### Step 7: [src/embedding/embed_pipeline.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/embedding/embed_pipeline.py) — Batch Embed + Store

**Why now?** Connects the embedder and the store — this is the "write" side of the pipeline.

**What to code:**
- Import `OllamaEmbedder`, `ChromaStore`, [Settings](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py#20-45), [Chunk](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#29-35), and your logger
- A function `embed_and_store(chunks: list[Chunk], settings: Settings) -> int`:
  1. Create an `OllamaEmbedder(settings)` and a `ChromaStore(settings)`
  2. Process chunks in batches of size `settings.embedding_batch_size`
  3. For each batch:
     - Extract texts: `[chunk.text for chunk in batch]`
     - Call `embedder.embed_batch(texts)` to get vectors
     - Call `store.add_chunks(batch, vectors)` to store them
     - Log progress: `"Embedded batch {i}/{total}"`
  4. Return total chunks stored

**How it works:** Batching is essential. Sending 1000 chunks one-by-one means 1000 HTTP calls to Ollama. With batch_size=16, that's ~63 calls. Each call sends multiple texts and gets back multiple vectors in one response. After getting vectors, we immediately store them in ChromaDB.

---

### Step 8: [src/prompting/templates.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/prompting/templates.py) + [prompt_builder.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/prompting/prompt_builder.py)

**Why now?** Before you can generate answers, you need to construct the right prompt.

#### 8a: [templates.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/prompting/templates.py)

**What to code:**
- Define a constant `QA_TEMPLATE` — a multi-line string like:
```
You are a helpful assistant answering questions based ONLY on the provided context.
If the context does not contain enough information to answer the question, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:
```
- Optionally define `QA_TEMPLATE_WITH_CITATIONS` that asks the LLM to cite which chunks it used

**How it works:** The template is the **heart of RAG quality**. By instructing the LLM to answer "ONLY based on context," you reduce hallucination. The `{context}` placeholder will be filled with your retrieved chunks, and `{question}` with the user's query. The LLM sees the context first, then the question — this primes it to draw from the context.

#### 8b: [prompt_builder.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/prompting/prompt_builder.py)

**What to code:**
- Import [RetrievalHit](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#37-43) from schemas, `QA_TEMPLATE` from templates
- A function `format_context(hits: list[RetrievalHit]) -> str`:
  1. For each hit, format as: `"[Source: {hit.metadata.get('filename', 'unknown')}]\n{hit.text}"`
  2. Join all with `"\n\n---\n\n"` separator
  3. Return the combined string
- A function `build_qa_prompt(question: str, hits: list[RetrievalHit]) -> str`:
  1. Call `format_context(hits)` to get the context string
  2. Return `QA_TEMPLATE.format(context=context, question=question)`

**How it works:** `format_context` turns a list of retrieved chunks into a single readable block with source annotations. `build_qa_prompt` plugs that block plus the question into your template. The result is a fully formed prompt ready to send to the LLM.

---

### Step 9: [src/retrieval/retriver.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/retrieval/retriver.py) — Query ChromaDB

**Why now?** This is the "read" side — given a question, find relevant chunks.

**What to code:**
- Import `OllamaEmbedder`, `ChromaStore`, [Settings](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py#20-45), [RetrievalHit](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#37-43)
- Import [preprocess_query](file:///a:/Project/Q&A%20RAG%20Pipeline/src/preprocessing/query_preprocess.py#6-10) from `src.preprocessing.query_preprocess`
- A function `retrieve(query: str, settings: Settings) -> list[RetrievalHit]`:
  1. Clean the query: `query = preprocess_query(query)`
  2. Embed the query: `embedder = OllamaEmbedder(settings)` → `query_vector = embedder.embed_single(query)`
  3. Search ChromaDB: `store = ChromaStore(settings)` → `results = store.query(query_vector, settings.retrieval_top_k)`
  4. Convert each result dict to a [RetrievalHit(chunk_id=r["chunk_id"], score=r["score"], text=r["text"], metadata=r["metadata"])](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#37-43)
  5. Return the list sorted by score (highest first)

**How it works:** The retriever converts the user's question into the same vector space as your stored chunks. ChromaDB then finds the chunks whose vectors are closest (most similar) to the query vector. If you stored a chunk about "Python data types" and the user asks "What types does Python have?", their vectors will be very close — even though the words are different. That's the power of semantic search.

---

### Step 10: [src/generation/ollama_generator.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/generation/ollama_generator.py) + [answer_postprocess.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/generation/answer_postprocess.py)

**Why now?** The final piece — take the prompt and generate the answer.

#### 10a: [ollama_generator.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/generation/ollama_generator.py)

**What to code:**
- Import `requests`, [Settings](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py#20-45), `GenerationError`, and your logger
- A function `generate_answer(prompt: str, settings: Settings) -> str`:
  1. POST to `{settings.ollama_base_url}/api/chat` with JSON:
     ```json
     {
       "model": settings.ollama_chat_model,
       "messages": [{"role": "user", "content": prompt}],
       "stream": false
     }
     ```
  2. Parse response: `response.json()["message"]["content"]`
  3. Return the raw answer text
  4. Wrap in try/except, raise `GenerationError` on failures

**How it works:** Ollama's `/api/chat` endpoint works like OpenAI's Chat API. You send a message with role "user" and the model responds. Setting `"stream": false` means you get the complete answer in one response (simpler for now; streaming can be added later). The model reads your entire prompt (context + question) and generates an answer token by token.

#### 10b: [answer_postprocess.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/generation/answer_postprocess.py)

**What to code:**
- A function `postprocess_answer(raw_answer: str) -> str`:
  1. Strip leading/trailing whitespace
  2. Remove any `<think>...</think>` tags that reasoning models might produce (regex: `re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)`)
  3. Collapse multiple newlines
  4. Return clean answer

**How it works:** Some Ollama models (like Qwen) include internal reasoning wrapped in `<think>` tags. Users shouldn't see these. The postprocessor cleans up the model's raw output into a presentation-ready answer.

---

## 🔗 Wiring It All Together

After Steps 1–10, you build two orchestrator pipelines:

### [src/pipeline/ingest_pipeline.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/pipeline/ingest_pipeline.py)

```
load_documents(paths) → preprocess_documents(docs) → chunk_documents(docs, settings)
    → embed_and_store(chunks, settings)
```

### [src/pipeline/qa_pipeline.py](file:///a:/Project/Q&A%20RAG%20Pipeline/src/pipeline/qa_pipeline.py)

```
preprocess_query(question) → retrieve(question, settings) → build_qa_prompt(question, hits)
    → generate_answer(prompt, settings) → postprocess_answer(raw) → AnswerResult(...)
```

### [scripts/run_ingest.py](file:///a:/Project/Q&A%20RAG%20Pipeline/scripts/run_ingest.py)
- Parse CLI args (path to documents folder)
- Call [load_settings()](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py#47-76) and [ensure_runtime_dirs(settings)](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py#78-91)
- Call the ingest pipeline

### [scripts/run_qa.py](file:///a:/Project/Q&A%20RAG%20Pipeline/scripts/run_qa.py)
- Accept a question from CLI
- Call [load_settings()](file:///a:/Project/Q&A%20RAG%20Pipeline/src/config/settings.py#47-76)
- Call the QA pipeline
- Print the [AnswerResult](file:///a:/Project/Q&A%20RAG%20Pipeline/src/core/schemas.py#45-51)

---

## ⚡ Prerequisites Before You Start Coding

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Install Ollama:** Download from [ollama.com](https://ollama.com)
3. **Pull required models:**
   ```
   ollama pull nomic-embed-text
   ollama pull qwen3.5:2b
   ```
4. **Start Ollama:** `ollama serve` (keep this terminal open)
5. **Add a test document:** Put any PDF or TXT file in `data/raw/`

---

## 📋 Summary — What to Code and In What Order

| Order | File | Depends On | You'll Learn |
|-------|------|-----------|-------------|
| **1** | `core/logger.py` | nothing | Python logging |
| **2** | `core/exceptions.py` | nothing | Exception hierarchy |
| **3** | `chunking/strategies.py` | nothing | Chunking algorithms |
| **4** | `chunking/chunker.py` | schemas, strategies, metadata | Strategy pattern |
| **5** | `embedding/ollama_embedder.py` | settings, exceptions | HTTP APIs, embeddings |
| **6** | `storage/vector_store_base.py` + **NEW** `storage/chroma_store.py` | schemas, settings | Abstract classes, ChromaDB |
| **7** | `embedding/embed_pipeline.py` | embedder, store | Batch processing |
| **8** | `prompting/templates.py` + `prompt_builder.py` | schemas | Prompt engineering |
| **9** | `retrieval/retriver.py` | embedder, store, preprocessing | Semantic search |
| **10** | `generation/ollama_generator.py` + `answer_postprocess.py` | settings, exceptions | LLM API calls |
| **11** | `pipeline/ingest_pipeline.py` | steps 1–7 | Pipeline orchestration |
| **12** | `pipeline/qa_pipeline.py` | steps 8–10 | End-to-end RAG |
| **13** | `scripts/run_ingest.py` + `scripts/run_qa.py` | pipelines | CLI integration |

> [!NOTE]
> The files `retrieval/rankers.py`, `retrieval/filters.py`, `evaluation/evaluator.py`, `evaluation/metrics.py`, and `storage/in_memory_store.py` are **advanced features** for later. Focus on the 13 steps above first to get a working pipeline.

---

## 🚀 Start Here → Step 1: `src/core/logger.py`

Once you've got your prerequisites set up, open `src/core/logger.py` and start coding. When you're ready, come back and I'll walk you through the exact code and explain every line.
