"""Main RAG Agent — orchestrates tools, memory, and the reasoning loop.

This is the top-level agent class that ties everything together:
- Initializes the tool registry with all available tools
- Manages conversation and long-term memory
- Delegates to the reasoning loop for actual processing
- Optionally runs self-evaluation on answers
"""
from __future__ import annotations

from src.core.base_agent import BaseAgent
from src.core.schemas import AgentResult
from src.config.settings import Settings, load_settings
from src.agents.tool_registry import ToolRegistry
from src.agents.reasoning_loop import run_reasoning_loop
from src.agents.memory import ConversationMemory, LongTermMemory
from src.agents.self_evaluator import self_evaluate, should_retry
from src.tools.search_tool import SearchTool
from src.tools.calculator_tool import CalculatorTool
from src.tools.summarize_tool import SummarizeTool
from src.core.logger import get_logger

logger = get_logger("RAGAgent")


class RAGAgent(BaseAgent):
    """Production RAG agent with tools, memory, and self-evaluation."""

    def __init__(
        self,
        settings: Settings | None = None,
        enable_memory: bool = True,
        enable_self_eval: bool = False,
        max_iterations: int = 7,
    ) -> None:
        """Initialize the agent with all components.

        Args:
            settings: Application settings.  Loaded if None.
            enable_memory: Whether to enable long-term memory.
            enable_self_eval: Whether to self-evaluate answers.
            max_iterations: Max reasoning loop iterations.
        """
        self._settings = settings or load_settings()
        self._max_iterations = max_iterations
        self._enable_self_eval = enable_self_eval

        # Initialize tool registry
        self._registry = ToolRegistry()
        self._registry.register(SearchTool(self._settings))
        self._registry.register(CalculatorTool())
        self._registry.register(SummarizeTool(self._settings))

        # Initialize memory
        self._conversation = ConversationMemory()
        self._long_term = LongTermMemory(self._settings) if enable_memory else None

        logger.info(
            "RAG Agent initialized: %d tools, memory=%s, self_eval=%s",
            len(self._registry),
            enable_memory,
            enable_self_eval,
        )

    def run(self, query: str) -> AgentResult:
        """Process a single query (stateless — no conversation memory used).

        Args:
            query: The user's question.

        Returns:
            AgentResult with answer and reasoning trace.
        """
        logger.info("Agent.run: '%s'", query[:80])

        result = run_reasoning_loop(
            query=query,
            tool_registry=self._registry,
            settings=self._settings,
            long_term_memory=self._long_term,
            max_iterations=self._max_iterations,
        )

        # Optional self-evaluation
        if self._enable_self_eval:
            result = self._maybe_retry(result)

        return result

    def chat(self, query: str) -> AgentResult:
        """Process a query with conversation context preserved.

        Unlike run(), this maintains conversation history between calls
        so the agent can understand follow-up questions.

        Args:
            query: The user's question or follow-up.

        Returns:
            AgentResult with answer and reasoning trace.
        """
        logger.info("Agent.chat: '%s'", query[:80])

        # Add user message to conversation history
        self._conversation.add_user_message(query)

        result = run_reasoning_loop(
            query=query,
            tool_registry=self._registry,
            settings=self._settings,
            conversation_memory=self._conversation,
            long_term_memory=self._long_term,
            max_iterations=self._max_iterations,
        )

        # Optional self-evaluation
        if self._enable_self_eval:
            result = self._maybe_retry(result)

        # Store the response in conversation memory
        self._conversation.add_assistant_message(result.final_answer)

        # Store in long-term memory
        if self._long_term:
            self._long_term.store_interaction(query, result.final_answer)

        return result

    def reset(self) -> None:
        """Clear conversation memory (start a new session)."""
        self._conversation.clear()
        logger.info("Conversation memory cleared")

    def _maybe_retry(self, result: AgentResult) -> AgentResult:
        """Run self-evaluation and retry if quality is too low.

        Args:
            result: The initial AgentResult.

        Returns:
            Original or improved AgentResult.
        """
        # Build context from reasoning trace
        context_parts = []
        for step in result.reasoning_steps:
            if step.observation:
                context_parts.append(step.observation)
        context = "\n".join(context_parts)

        scores = self_evaluate(
            question=result.question,
            answer=result.final_answer,
            context=context,
            settings=self._settings,
        )

        result.citations.append({"self_eval": scores})

        if should_retry(scores):
            logger.info("Self-eval score %.2f below threshold, retrying...", scores["overall"])
            # Retry with increased iterations
            retry_result = run_reasoning_loop(
                query=result.question + " (Please provide a more detailed and accurate answer)",
                tool_registry=self._registry,
                settings=self._settings,
                max_iterations=self._max_iterations + 2,
            )
            retry_result.total_llm_calls += result.total_llm_calls
            retry_result.citations.append({"self_eval_retry": scores})
            return retry_result

        return result

    def get_tool_names(self) -> list[str]:
        """Return names of all registered tools."""
        return self._registry.tool_names()

    def register_tool(self, tool) -> None:
        """Register an additional tool at runtime."""
        self._registry.register(tool)
