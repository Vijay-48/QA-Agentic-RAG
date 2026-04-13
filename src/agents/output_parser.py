"""Parse the LLM's agent output into structured Thought/Action/Final Answer.

The agent LLM outputs text in the ReAct format:
    Thought: I need to search for X
    Action: search_knowledge_base
    Action Input: what is X

Or:
    Thought: I have enough info now.
    Final Answer: X is ...

This module extracts those parts with robust fallback handling for
malformed output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.logger import get_logger

logger = get_logger("OutputParser")


@dataclass
class ParsedOutput:
    """Result of parsing agent LLM output."""
    thought: str = ""
    action: str = ""
    action_input: str = ""
    final_answer: str = ""

    @property
    def has_action(self) -> bool:
        return bool(self.action)

    @property
    def has_final_answer(self) -> bool:
        return bool(self.final_answer)


def parse_agent_output(text: str) -> ParsedOutput:
    """Parse the LLM's raw text output into structured components.

    Handles common edge cases:
    - Missing fields
    - Extra whitespace
    - Tool name typos (fuzzy matching would go here)
    - Answer given without "Final Answer:" prefix

    Args:
        text: Raw text output from the LLM.

    Returns:
        ParsedOutput with extracted fields.
    """
    result = ParsedOutput()

    # Clean up the text
    text = text.strip()

    # Extract "Final Answer:" — this takes priority over everything
    final_match = re.search(
        r"Final\s*Answer\s*:\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if final_match:
        result.final_answer = final_match.group(1).strip()
        # Also extract thought if present before the final answer
        thought_match = re.search(
            r"Thought\s*:\s*(.*?)(?=Final\s*Answer)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if thought_match:
            result.thought = thought_match.group(1).strip()
        return result

    # Extract "Thought:"
    thought_match = re.search(
        r"Thought\s*:\s*(.*?)(?=Action\s*:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if thought_match:
        result.thought = thought_match.group(1).strip()

    # Extract "Action:"
    action_match = re.search(
        r"Action\s*:\s*(.*?)(?=Action\s*Input\s*:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if action_match:
        result.action = action_match.group(1).strip().lower()
        # Clean common formatting issues
        result.action = result.action.rstrip(":")

    # Extract "Action Input:"
    input_match = re.search(
        r"Action\s*Input\s*:\s*(.*?)(?=Observation\s*:|Thought\s*:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if input_match:
        result.action_input = input_match.group(1).strip()

    # Fallback: if no structured output found, treat the whole thing
    # as a final answer (the LLM just responded directly)
    if not result.has_action and not result.has_final_answer and not result.thought:
        logger.warning("No structured output found, treating as final answer")
        result.final_answer = text

    return result
