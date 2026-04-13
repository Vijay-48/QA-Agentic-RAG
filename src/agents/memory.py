"""Memory system for the agent — short-term, long-term, and working memory.

Short-term: Current conversation turns (cleared per session)
Long-term: Past interactions stored in Qdrant (persisted across sessions)
Working: Scratchpad for facts discovered during the current reasoning loop
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.config.settings import Settings
from src.core.logger import get_logger

logger = get_logger("Memory")


class ConversationMemory:
    """Short-term conversation buffer — list of role/content message dicts.

    Maintains the current conversation context so the agent can
    understand follow-up questions like "tell me more about that".
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._messages: list[dict[str, str]] = []
        self._max_turns = max_turns

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

    def get_history(self) -> list[dict[str, str]]:
        """Return all messages in chronological order."""
        return list(self._messages)

    def get_history_text(self) -> str:
        """Format conversation history as text for prompt injection."""
        if not self._messages:
            return ""
        parts = []
        for msg in self._messages:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        return "\n".join(parts)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        """Keep only the last max_turns * 2 messages (user + assistant pairs)."""
        max_messages = self._max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def __len__(self) -> int:
        return len(self._messages)


class WorkingMemory:
    """Scratchpad for facts discovered during a single reasoning loop.

    The agent can store intermediate findings here as key-value pairs.
    This is cleared at the start of each new query.
    """

    def __init__(self) -> None:
        self._facts: dict[str, str] = {}

    def store(self, key: str, value: str) -> None:
        self._facts[key] = value

    def recall(self, key: str) -> str | None:
        return self._facts.get(key)

    def get_all(self) -> dict[str, str]:
        return dict(self._facts)

    def as_text(self) -> str:
        """Format all facts for prompt injection."""
        if not self._facts:
            return ""
        parts = [f"- {k}: {v}" for k, v in self._facts.items()]
        return "Known facts:\n" + "\n".join(parts)

    def clear(self) -> None:
        self._facts.clear()


class LongTermMemory:
    """Persistent memory stored in a separate Qdrant collection.

    After each conversation, the Q&A pair is embedded and stored.
    Before answering new queries, past interactions are searched for
    relevant context.
    """

    COLLECTION_NAME = "agent_memory"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._initialized = False
        self._client = None

    def _ensure_init(self) -> None:
        """Lazy initialization — only connect to Qdrant when first needed."""
        if self._initialized:
            return

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance

            if self._settings.qdrant_url:
                self._client = QdrantClient(
                    url=self._settings.qdrant_url,
                    api_key=self._settings.qdrant_api_key,
                    timeout=self._settings.request_timeout_seconds,
                    check_compatibility=False,
                )
            else:
                import os
                persist_dir = str(self._settings.artifacts_dir / "vector_index" / "qdrant_memory")
                os.makedirs(persist_dir, exist_ok=True)
                self._client = QdrantClient(path=persist_dir)

            # Create collection if it doesn't exist
            collections = [c.name for c in self._client.get_collections().collections]
            if self.COLLECTION_NAME not in collections:
                self._client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
                logger.info("Created long-term memory collection")

            self._initialized = True
        except Exception as e:
            logger.warning("Long-term memory init failed: %s", e)
            self._client = None

    def store_interaction(self, question: str, answer: str) -> None:
        """Store a Q&A pair in long-term memory.

        Args:
            question: The user's question.
            answer: The agent's answer.
        """
        self._ensure_init()
        if self._client is None:
            return

        try:
            from src.embedding.ollama_embedder import OllamaEmbedder
            from qdrant_client.models import PointStruct
            import uuid

            embedder = OllamaEmbedder(self._settings)
            # Embed the question (we search by question similarity)
            vector = embedder.embed_single(question)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            self._client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point],
            )
            logger.info("Stored interaction in long-term memory")
        except Exception as e:
            logger.warning("Failed to store interaction: %s", e)

    def search_past(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search for relevant past interactions.

        Args:
            query: The current query to find similar past interactions.
            top_k: Number of past interactions to retrieve.

        Returns:
            List of dicts with 'question', 'answer', 'score' keys.
        """
        self._ensure_init()
        if self._client is None:
            return []

        try:
            from src.embedding.ollama_embedder import OllamaEmbedder

            embedder = OllamaEmbedder(self._settings)
            vector = embedder.embed_single(query)

            results = self._client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )

            past = []
            for r in results:
                if r.score > 0.7:  # Only include genuinely similar past interactions
                    payload = r.payload or {}
                    past.append({
                        "question": payload.get("question", ""),
                        "answer": payload.get("answer", ""),
                        "score": r.score,
                    })

            if past:
                logger.info("Found %d relevant past interactions", len(past))
            return past
        except Exception as e:
            logger.warning("Past interaction search failed: %s", e)
            return []

    def format_past_context(self, past: list[dict]) -> str:
        """Format past interactions for prompt injection."""
        if not past:
            return ""
        parts = ["## Relevant Past Interactions"]
        for item in past:
            parts.append(f"Q: {item['question']}\nA: {item['answer']}")
        return "\n\n".join(parts)
