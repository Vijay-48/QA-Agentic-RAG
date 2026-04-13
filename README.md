# 🤖 Agentic RAG Pipeline

A production-ready **Retrieval-Augmented Generation (RAG)** system with full **agentic capabilities** — built from scratch using Python, Ollama, and Qdrant Cloud. The agent can reason, use tools, maintain memory across conversations, and self-evaluate its answers.

> **No LangChain. No LlamaIndex.** Every component is hand-written for deep understanding of RAG internals.

---

## ✨ Features

### Core RAG Pipeline
- 📄 **Multi-format ingestion** — PDF, TXT, Markdown
- ✂️ **Smart chunking** — Fixed, sliding window, and semantic strategies
- 🔢 **Embedding** — Ollama `nomic-embed-text` (768-dim vectors)
- 🗄️ **Vector storage** — Qdrant Cloud (primary), ChromaDB, FAISS
- 🔍 **Retrieval** — Dense vector search with configurable top-k
- 💬 **Generation** — Ollama local LLMs (qwen3, qwen3.5, etc.)

### Advanced Retrieval
- 🔀 **Hybrid search** — BM25 (keyword) + dense (semantic) with Reciprocal Rank Fusion
- 📊 **LLM re-ranking** — Cross-encoder style relevance scoring using the chat model
- 🏷️ **Metadata filters** — Narrow search by filename, document ID, or chunk range
- 🗜️ **Context compression** — Extract only relevant sentences from retrieved chunks

### Query Intelligence
- 🔄 **Query expansion** — Generate multiple query variations for broader recall
- 🧩 **Query decomposition** — Break complex questions into answerable sub-questions
- 🎯 **HyDE** — Hypothetical Document Embedding for better retrieval alignment

### Agentic System
- 🧠 **ReAct reasoning** — Think → Act → Observe loop with tool execution
- 🔧 **Tool system** — Pluggable tools with registry (search, calculator, summarizer)
- 💾 **Three-tier memory** — Conversation (short-term), working (scratchpad), long-term (Qdrant)
- ✅ **Self-evaluation** — Agent grades its own answers and retries if quality is low

### Evaluation
- 📈 **Recall@k** — Keyword-based retrieval quality measurement
- 🎯 **Faithfulness** — LLM-judged hallucination detection
- 📋 **Answer relevance** — LLM-judged topic adherence scoring

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agentic RAG Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Agent    │───▶│  Tool        │───▶│  search_knowledge_base│  │
│  │  (ReAct)  │    │  Registry    │    │  calculator           │  │
│  │          │◀───│              │◀───│  summarize            │  │
│  └──────────┘    └──────────────┘    └───────────────────────┘  │
│      │                                        │                 │
│      │                                        ▼                 │
│      │         ┌──────────────────────────────────────┐         │
│      │         │         Retrieval Layer               │         │
│      │         │  ┌────────┐  ┌───────┐  ┌─────────┐  │         │
│      │         │  │ Dense  │  │ BM25  │  │Reranker │  │         │
│      │         │  │ Search │  │Search │  │  (LLM)  │  │         │
│      │         │  └───┬────┘  └───┬───┘  └────┬────┘  │         │
│      │         │      └─────┬─────┘           │       │         │
│      │         │            ▼                 │       │         │
│      │         │     ┌──────────┐             │       │         │
│      │         │     │   RRF    │─────────────┘       │         │
│      │         │     │  Fusion  │                     │         │
│      │         │     └──────────┘                     │         │
│      │         └──────────────────────────────────────┘         │
│      │                        │                                 │
│      │                        ▼                                 │
│      │         ┌──────────────────────────────────────┐         │
│      │         │        Vector Store (Qdrant)          │         │
│      │         │     29 chunks from Rag Docs.pdf       │         │
│      │         └──────────────────────────────────────┘         │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Memory   │    │  Ollama LLM  │    │  Self-Evaluator      │  │
│  │ Short+Long│    │  (qwen3)     │    │  (optional retry)    │  │
│  └──────────┘    └──────────────┘    └───────────────────────┘  │
│      │                   │                                      │
│      ▼                   ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Final Answer                           │   │
│  │         + reasoning trace + citations + stats             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Agentic RAG/
│
├── app.py                              # Unified CLI — 7 commands
├── .env                                # API keys & config
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── src/
│   ├── core/                           # Abstractions & schemas
│   │   ├── schemas.py                  # Document, Chunk, RetrievalHit, AgentResult
│   │   ├── base_tool.py                # Abstract tool interface
│   │   ├── base_agent.py               # Abstract agent interface
│   │   ├── exceptions.py               # Custom exceptions
│   │   └── logger.py                   # Logging config
│   │
│   ├── config/
│   │   └── settings.py                 # Centralized settings (dataclass + .env)
│   │
│   ├── ingestion/
│   │   └── file_loader.py              # PDF, TXT, MD document loading
│   │
│   ├── chunking/
│   │   └── strategies.py               # Fixed, sliding window, semantic chunking
│   │
│   ├── embedding/
│   │   └── ollama_embedder.py          # Ollama embedding API wrapper
│   │
│   ├── storage/                        # Vector store backends
│   │   ├── vector_store_base.py        # Abstract vector store interface
│   │   ├── store_factory.py            # Factory pattern for backend selection
│   │   ├── qdrant_store.py             # Qdrant Cloud/local implementation
│   │   ├── chroma_store.py             # ChromaDB implementation
│   │   └── faiss_store.py              # FAISS implementation
│   │
│   ├── retrieval/
│   │   ├── retriver.py                 # Basic dense retrieval
│   │   ├── hybrid_retriever.py         # BM25 + dense with RRF fusion
│   │   ├── rankers.py                  # LLM-based re-ranking
│   │   └── filters.py                  # Qdrant metadata filters
│   │
│   ├── preprocessing/
│   │   ├── document_preprocess.py      # Text cleaning & normalization
│   │   ├── query_preprocess.py         # Query preprocessing
│   │   ├── query_transform.py          # Expansion, decomposition, HyDE
│   │   └── context_compressor.py       # Extract relevant sentences only
│   │
│   ├── prompting/
│   │   └── prompt_builder.py           # Context + question → prompt
│   │
│   ├── generation/
│   │   ├── ollama_generator.py         # Ollama chat API wrapper
│   │   └── answer_postprocess.py       # Clean up LLM output
│   │
│   ├── evaluation/
│   │   ├── metrics.py                  # Recall@k, Faithfulness, Relevance
│   │   └── evaluator.py               # End-to-end evaluation pipeline
│   │
│   ├── agents/                         # Agentic RAG core
│   │   ├── prompts.py                  # ReAct system prompts
│   │   ├── tool_registry.py            # Tool discovery & management
│   │   ├── output_parser.py            # Parse Thought/Action/Final Answer
│   │   ├── reasoning_loop.py           # Think → Act → Observe engine
│   │   ├── memory.py                   # Conversation + long-term + working
│   │   ├── self_evaluator.py           # Answer quality self-check
│   │   └── rag_agent.py               # Main agent orchestrator
│   │
│   ├── tools/                          # Agent tools
│   │   ├── search_tool.py              # Knowledge base search
│   │   ├── calculator_tool.py          # Safe math (AST-based)
│   │   └── summarize_tool.py           # Text summarization
│   │
│   └── pipeline/
│       ├── ingest_pipeline.py          # Document → chunks → vectors
│       ├── qa_pipeline.py              # Basic RAG: query → answer
│       └── agentic_pipeline.py         # Agentic RAG: query → reason → answer
│
├── data/
│   ├── raw/                            # Source documents (PDFs, etc.)
│   └── eval/
│       └── eval_dataset.json           # Evaluation test questions
│
└── scripts/
    ├── run_ingest.py                   # Standalone ingestion script
    ├── run_qa.py                       # Standalone QA script
    ├── run_eval.py                     # Standalone evaluation script
    └── diagnose.py                     # Pipeline diagnostics
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** — [Install from ollama.com](https://ollama.com)
- **Qdrant Cloud** account (free tier) — [cloud.qdrant.io](https://cloud.qdrant.io)

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd "Agentic RAG"
pip install -r requirements.txt
```

### 2. Pull Ollama Models

```bash
ollama pull nomic-embed-text       # Embedding model (768-dim)
ollama pull qwen3:0.6b             # Chat model (lightweight)
# OR for better quality:
ollama pull qwen3.5:2b             # Larger, follows instructions better
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
# Qdrant Cloud
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-api-key-here

# Ollama (defaults shown)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=qwen3:0.6b

# Optional overrides
VECTOR_STORE=qdrant
CHUNK_STRATEGY=semantic
RETRIEVAL_TOP_K=5
REQUEST_TIMEOUT_SECONDS=120
```

### 4. Start Ollama

```bash
ollama serve
```

### 5. Ingest Documents

```bash
python app.py ingest ./data/raw
```

### 6. Ask Questions

```bash
# Basic RAG
python app.py ask "What is RAG?"

# Agentic RAG (with reasoning trace)
python app.py agent "Explain how embedding works in RAG"

# Interactive chat with memory
python app.py agent-chat
```

---

## 💻 CLI Reference

| Command | Description | Example |
|---------|-------------|---------|
| `ingest` | Load documents into the vector store | `python app.py ingest ./data/raw` |
| `ask` | Ask a question (basic RAG pipeline) | `python app.py ask "What is chunking?"` |
| `chat` | Interactive Q&A loop (basic RAG) | `python app.py chat` |
| `agent` | Ask using the agentic pipeline | `python app.py agent "Compare chunking strategies"` |
| `agent-chat` | Interactive agentic chat with memory | `python app.py agent-chat` |
| `eval` | Run evaluation metrics on test dataset | `python app.py eval` |
| `diagnose` | Check pipeline health (Ollama, DB, etc.) | `python app.py diagnose` |

### Agent-Chat Commands

Inside `agent-chat` mode:
- Type any question and press Enter
- Type `reset` to clear conversation memory
- Type `quit`, `exit`, or `q` to stop

---

## 🧠 How the Agent Works

The agent follows the **ReAct** (Reasoning + Acting) pattern:

```
User: "What are the chunking strategies used in RAG?"

Agent Reasoning:
  Step 1:
    💭 Thought: I need to search the knowledge base to answer this question.
    🔧 Action: search_knowledge_base
    📥 Input: "What are the chunking strategies used in RAG?"
    👁️ Observation: [5 relevant chunks from Rag Docs.pdf]

  Step 2:
    💭 Thought: I have enough information from the search results to answer.

Final Answer:
  Based on the search results, chunking strategies include:
  1. PyPDF-based segmentation using predefined separators...
  2. Embedding-based chunking that transforms text into vectors...
  [with source citations]

📊 Stats: 1 LLM call, tools used: ['search_knowledge_base']
```

### Available Tools

| Tool | Purpose | Example Input |
|------|---------|---------------|
| `search_knowledge_base` | Search ingested documents for relevant information | `"What is semantic chunking?"` |
| `calculator` | Perform safe mathematical calculations | `"15 * 3 + 7"` or `"sqrt(144)"` |
| `summarize` | Condense long text into 2-4 sentences | `"<long text to summarize>"` |

### Adding Custom Tools

Create a new tool by implementing the `BaseTool` interface:

```python
# src/tools/my_tool.py
from src.core.base_tool import BaseTool

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Description the LLM reads to decide when to use this tool."

    def execute(self, input_text: str) -> str:
        # Your tool logic here
        return "result text"
```

Register it in `rag_agent.py`:
```python
from src.tools.my_tool import MyTool
self._registry.register(MyTool())
```

---

## 📊 Evaluation

### Running Evaluation

```bash
# Use default dataset
python app.py eval

# Use custom dataset
python app.py eval --dataset path/to/questions.json
```

### Evaluation Dataset Format

```json
[
    {
        "question": "What is RAG?",
        "expected_answer": "RAG is a technique that...",
        "expected_keywords": ["retrieval", "generation", "augmented"],
        "expected_sources": ["Rag Docs.pdf"]
    }
]
```

### Metrics Explained

| Metric | What It Measures | How It Works |
|--------|-----------------|--------------|
| **Recall@k** | Did we retrieve the right chunks? | Checks if expected keywords appear in retrieved text |
| **Faithfulness** | Is the answer grounded in context? | LLM judges if every claim is supported by evidence |
| **Answer Relevance** | Does the answer address the question? | LLM judges if the answer matches the intent |

---

## 🔧 Configuration

All settings are managed via `src/config/settings.py` and can be overridden with environment variables in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Model for generating embeddings |
| `OLLAMA_CHAT_MODEL` | `qwen3:0.6b` | Model for generation & reasoning |
| `VECTOR_STORE` | `qdrant` | Backend: `qdrant`, `chroma`, or `faiss` |
| `CHUNK_STRATEGY` | `semantic` | Strategy: `fixed`, `sliding`, or `semantic` |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `REQUEST_TIMEOUT_SECONDS` | `120` | LLM request timeout |

---

## 🧩 Design Principles

1. **No Frameworks** — Every component is hand-written. No LangChain, no LlamaIndex. This ensures deep understanding of how RAG works internally.

2. **Interface-Driven** — Abstract base classes (`VectorStoreBase`, `BaseTool`, `BaseAgent`) allow swapping implementations without changing pipeline logic.

3. **Modular Architecture** — Each module (chunking, embedding, retrieval, generation) is independent and can be tested, replaced, or upgraded individually.

4. **Local-First** — Runs entirely on your machine using Ollama for LLM inference. The only cloud dependency is Qdrant (which also supports local mode).

5. **Progressive Enhancement** — The system works as a basic RAG pipeline and progressively adds advanced features (hybrid search, re-ranking, agent reasoning, memory).

---

## 📈 Roadmap

- [x] Layer 1: Basic RAG Pipeline
- [x] Layer 2: Evaluation System (Recall@k, Faithfulness, Relevance)
- [x] Layer 2: Advanced Retrieval (Hybrid search, Re-ranking, Filters)
- [x] Layer 2: Query Intelligence (Expansion, Decomposition, HyDE)
- [x] Layer 3: Agent Foundations (Tools, Registry, Prompts)
- [x] Layer 3: Reasoning Loop (ReAct pattern)
- [x] Layer 3: Memory System (Conversation + Long-term)
- [x] Layer 3: Self-Evaluation & Retry
- [ ] Layer 4: Web Search Tool
- [ ] Layer 4: Streaming Responses
- [ ] Layer 4: FastAPI REST API
- [ ] Layer 4: Multi-document Conversations

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM Inference | [Ollama](https://ollama.com) (local) |
| Embedding Model | `nomic-embed-text` (768-dim) |
| Chat Model | `qwen3:0.6b` / `qwen3.5:2b` |
| Vector Database | [Qdrant Cloud](https://cloud.qdrant.io) |
| Keyword Search | BM25 via `rank-bm25` |
| PDF Parsing | `pypdf` |
| Config | `python-dotenv` + dataclasses |

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com) — Local LLM inference
- [Qdrant](https://qdrant.tech) — Vector search engine
- [ReAct Paper](https://arxiv.org/abs/2210.03629) — Reasoning + Acting pattern
- [RAG Paper](https://arxiv.org/abs/2005.11401) — Retrieval-Augmented Generation
