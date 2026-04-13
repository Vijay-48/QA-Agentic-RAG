"""Summarization tool — condense long text into a concise summary.

Useful when the agent retrieves too much context and needs to distill
key points before forming a final answer.
"""
from __future__ import annotations

from src.core.base_tool import BaseTool
from src.config.settings import Settings
from src.generation.ollama_generator import generate_answer
from src.generation.answer_postprocess import postprocess_answer
from src.core.logger import get_logger

logger = get_logger("SummarizeTool")

_SUMMARIZE_PROMPT = """Summarize the following text concisely, keeping all key facts and details.
Focus on the most important information. Write 2-4 sentences maximum.

Text to summarize:
{text}

Concise summary:"""


class SummarizeTool(BaseTool):
    """Summarize long text into a concise form."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "summarize"

    @property
    def description(self) -> str:
        return (
            "Summarize a long piece of text into a concise form. "
            "Input should be the text you want to summarize. "
            "Returns a 2-4 sentence summary with key facts preserved. "
            "Use this when you have too much information and need to condense it."
        )

    def execute(self, input_text: str) -> str:
        """Summarize the input text.

        Args:
            input_text: The text to summarize.

        Returns:
            A concise summary.
        """
        logger.info("Summarizing %d characters of text", len(input_text))

        # If text is already short, no need to summarize
        if len(input_text.split()) < 50:
            return input_text

        prompt = _SUMMARIZE_PROMPT.format(text=input_text[:3000])
        try:
            raw = generate_answer(prompt, self._settings)
            summary = postprocess_answer(raw)
            logger.info("Summary: %d chars → %d chars", len(input_text), len(summary))
            return summary
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            return f"Summarization failed: {e}"
