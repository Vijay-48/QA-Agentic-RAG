from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.schemas import AgentResult


class BaseAgent(ABC):
    """Abstract base class for all agent implementations."""

    @abstractmethod
    def run(self, query: str) -> AgentResult:
        """Process a single query through the agent's reasoning loop.

        Args:
            query: The user's natural language question.

        Returns:
            AgentResult with the final answer and full reasoning trace.
        """

    @abstractmethod
    def chat(self, query: str) -> AgentResult:
        """Process a query with conversation memory preserved.

        Args:
            query: The user's follow-up or new question.

        Returns:
            AgentResult with answer, reasoning, and memory context.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all short-term / conversation memory."""
