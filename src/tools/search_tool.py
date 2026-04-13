"""Knowledge base search tool — wraps the existing retriever as an agent tool.

This is the primary tool the agent uses to find information from ingested
documents.  It takes a natural language query, searches the vector store,
and returns formatted text excerpts with source citations.
"""
from __future__ import annotations

from src.core.base_tool import BaseTool
from src.config.settings import Settings
from src.retrieval.retriver import retrieve
from src.core.logger import get_logger

logger = get_logger("SearchTool")


class SearchTool(BaseTool):
    """Search the knowledge base for relevant information."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "search_knowledge_base"

    @property
    def description(self) -> str:
        return (
            "Search the knowledge base for information from ingested documents. "
            "Input should be a specific natural language question or search query. "
            "Returns relevant text excerpts with source citations. "
            "Use this when you need factual information about topics in the documents."
        )

    def execute(self, input_text: str) -> str:
        """Search the knowledge base and return formatted results.

        Args:
            input_text: Natural language search query.

        Returns:
            Formatted text with search results or a 'no results' message.
        """
        logger.info("Searching knowledge base: '%s'", input_text[:80])

        try:
            hits = retrieve(input_text, self._settings)
        except Exception as e:
            logger.error("Search failed: %s", e)
            return f"Search failed with error: {e}"

        if not hits:
            return "No relevant information found in the knowledge base for this query."

        # Format results for the LLM to read
        parts = []
        for i, hit in enumerate(hits, 1):
            source = hit.metadata.get("filename", "unknown")
            score = f"{hit.score:.3f}"
            parts.append(
                f"[Result {i}] (source: {source}, relevance: {score})\n{hit.text}"
            )

        return "\n\n---\n\n".join(parts)
