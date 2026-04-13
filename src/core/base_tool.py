from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Abstract base class for all agent tools.

    Every tool has a name and description that the LLM reads to decide
    when to use it.  The execute() method takes a text input (from the
    LLM's "Action Input") and returns a text observation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used by the agent to call this tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description the LLM uses to decide when to use this tool."""

    @abstractmethod
    def execute(self, input_text: str) -> str:
        """Run the tool and return a text observation.

        Args:
            input_text: Natural language input from the agent.

        Returns:
            A text string the agent can read as an observation.
        """
