"""Agentic RAG pipeline — the top-level orchestrator for agent-based Q&A.

This wraps the RAGAgent and provides a clean interface matching
the existing pipeline pattern (like qa_pipeline.py).
"""
from __future__ import annotations

from src.config.settings import Settings, load_settings
from src.core.schemas import AgentResult
from src.agents.rag_agent import RAGAgent
from src.core.logger import get_logger

logger = get_logger("AgenticPipeline")

# Module-level agent instance for reuse across calls in chat mode
_agent_instance: RAGAgent | None = None


def _get_agent(settings: Settings, enable_memory: bool = True) -> RAGAgent:
    """Get or create the agent singleton."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RAGAgent(
            settings=settings,
            enable_memory=enable_memory,
            enable_self_eval=False,
            max_iterations=7,
        )
    return _agent_instance


def agent_ask(
    question: str,
    settings: Settings | None = None,
) -> AgentResult:
    """Process a single question through the agentic pipeline (stateless).

    Args:
        question: The user's question.
        settings: Application settings.  Loaded if None.

    Returns:
        AgentResult with answer, reasoning trace, and tool usage.
    """
    if settings is None:
        settings = load_settings()

    agent = RAGAgent(
        settings=settings,
        enable_memory=False,
        enable_self_eval=False,
    )

    logger.info("Agentic ask: '%s'", question[:80])
    result = agent.run(question)
    logger.info(
        "Agentic result: %d steps, %d tool calls, %d LLM calls",
        len(result.reasoning_steps),
        len(result.tools_used),
        result.total_llm_calls,
    )
    return result


def agent_chat(
    question: str,
    settings: Settings | None = None,
) -> AgentResult:
    """Process a question with conversation memory (stateful).

    The agent remembers previous turns within the same session.
    Call agent_reset() to clear memory.

    Args:
        question: The user's question or follow-up.
        settings: Application settings.  Loaded if None.

    Returns:
        AgentResult with answer, reasoning trace, and tool usage.
    """
    if settings is None:
        settings = load_settings()

    agent = _get_agent(settings)
    logger.info("Agentic chat: '%s'", question[:80])
    result = agent.chat(question)
    return result


def agent_reset() -> None:
    """Reset the agent's conversation memory."""
    global _agent_instance
    if _agent_instance is not None:
        _agent_instance.reset()
    logger.info("Agent conversation reset")
