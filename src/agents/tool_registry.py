"""Tool registry — central place to register, discover, and manage agent tools.

The registry serves two purposes:
1. The agent queries it to know what tools are available
2. It formats tool descriptions for inclusion in the system prompt
"""
from __future__ import annotations

from src.core.base_tool import BaseTool
from src.core.logger import get_logger

logger = get_logger("ToolRegistry")


class ToolRegistry:
    """Manages available tools for the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool.  Overwrites if name already exists."""
        self._tools[tool.name] = tool
        logger.info("Registered tool: '%s'", tool.name)

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name.  Returns None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        """Return all tools as a list of {name, description} dicts."""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    def tool_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def format_for_prompt(self) -> str:
        """Format all tools into a text block for the LLM system prompt.

        Output format:
            1. tool_name: description
            2. tool_name: description
        """
        lines = []
        for i, tool in enumerate(self._tools.values(), 1):
            lines.append(f"{i}. {tool.name}: {tool.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
