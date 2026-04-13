"""ReAct reasoning loop — the heart of the agentic RAG system.

Implements the Think → Act → Observe cycle:
1. FORCE a search on the first iteration (small models skip this)
2. Send the system prompt + question + history to the LLM
3. Parse the LLM output for Thought/Action/Final Answer
4. If Action → execute the tool, add Observation, loop back
5. If Final Answer → return the answer
6. If max iterations → force a best-effort answer
"""
from __future__ import annotations

from src.config.settings import Settings
from src.core.schemas import AgentResult, ReasoningStep
from src.agents.prompts import AGENT_SYSTEM_PROMPT, AGENT_HUMAN_PROMPT, AGENT_CONTINUE_PROMPT
from src.agents.tool_registry import ToolRegistry
from src.agents.output_parser import parse_agent_output
from src.agents.memory import ConversationMemory, WorkingMemory, LongTermMemory
from src.generation.ollama_generator import generate_answer
from src.generation.answer_postprocess import postprocess_answer
from src.core.logger import get_logger

logger = get_logger("ReasoningLoop")

DEFAULT_MAX_ITERATIONS = 5

_SYNTHESIZE_PROMPT = """Based on the following search results, answer this question.
Use ONLY the information from the search results. Do not add your own knowledge.
If the search results don't contain the answer, say "I could not find this information."

Question: {question}

Search Results:
{observations}

Provide a clear, comprehensive answer citing the source documents:
Final Answer:"""

_FORCE_ANSWER_PROMPT = """You have used all your available reasoning steps. 
Based on everything you've learned so far, provide your best possible answer now.

Final Answer:"""


def _forced_first_search(
    query: str,
    tool_registry: ToolRegistry,
) -> tuple[str, ReasoningStep] | None:
    """Force an initial search before the LLM even starts reasoning.

    Small models (< 3B) often skip the search and answer from memory.
    This guarantees the knowledge base is consulted.

    Args:
        query: The user's question.
        tool_registry: Registry containing the search tool.

    Returns:
        Tuple of (observation_text, reasoning_step) or None if no search tool.
    """
    search_tool = tool_registry.get_tool("search_knowledge_base")
    if search_tool is None:
        return None

    logger.info("Forced first search: '%s'", query[:60])
    try:
        observation = search_tool.execute(query)
    except Exception as e:
        logger.error("Forced search failed: %s", e)
        observation = f"Search failed: {e}"

    step = ReasoningStep(
        thought="I need to search the knowledge base to answer this question.",
        action="search_knowledge_base",
        action_input=query,
        observation=observation,
    )
    return observation, step


def run_reasoning_loop(
    query: str,
    tool_registry: ToolRegistry,
    settings: Settings,
    conversation_memory: ConversationMemory | None = None,
    long_term_memory: LongTermMemory | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentResult:
    """Execute the ReAct reasoning loop.

    Args:
        query: The user's question.
        tool_registry: Registry of available tools.
        settings: Application settings.
        conversation_memory: Optional conversation history.
        long_term_memory: Optional long-term memory store.
        max_iterations: Maximum think-act-observe cycles.

    Returns:
        AgentResult with the final answer and full reasoning trace.
    """
    reasoning_steps: list[ReasoningStep] = []
    tools_used: list[str] = []
    llm_calls = 0
    all_observations: list[str] = []

    # ──────────────────────────────────────────────────
    # Step 0: Forced first search (guaranteed retrieval)
    # ──────────────────────────────────────────────────
    forced = _forced_first_search(query, tool_registry)
    if forced:
        observation, step = forced
        reasoning_steps.append(step)
        tools_used.append("search_knowledge_base")
        all_observations.append(observation)
        logger.info("Forced search returned %d chars", len(observation))

    # ──────────────────────────────────────────────────
    # Step 1: Synthesize answer from search results
    # ──────────────────────────────────────────────────
    # For small models, the most reliable approach is:
    # search first → then ask the LLM to synthesize an answer
    # rather than hoping the LLM follows ReAct format perfectly.

    # Build conversation context
    context_parts = []
    if conversation_memory and len(conversation_memory) > 0:
        context_parts.append(f"Previous conversation:\n{conversation_memory.get_history_text()}")
    if long_term_memory:
        past = long_term_memory.search_past(query, top_k=2)
        if past:
            context_parts.append(long_term_memory.format_past_context(past))

    # Synthesize from search results
    if all_observations:
        synth_prompt = _SYNTHESIZE_PROMPT.format(
            question=query,
            observations="\n\n".join(all_observations),
        )

        if context_parts:
            synth_prompt = "\n".join(context_parts) + "\n\n" + synth_prompt

        try:
            raw_answer = generate_answer(synth_prompt, settings)
            answer = postprocess_answer(raw_answer)
            llm_calls += 1

            # Check if the LLM says it needs more info
            needs_more = any(phrase in answer.lower() for phrase in [
                "could not find",
                "not enough information",
                "no relevant",
                "don't have",
            ])

            if needs_more and max_iterations > 1:
                # Try a second search with a rephrased query
                logger.info("First search insufficient, trying rephrased query")
                search_tool = tool_registry.get_tool("search_knowledge_base")
                if search_tool:
                    rephrased = f"detailed information about {query}"
                    try:
                        observation2 = search_tool.execute(rephrased)
                        step2 = ReasoningStep(
                            thought="The first search didn't have enough info, searching with different terms.",
                            action="search_knowledge_base",
                            action_input=rephrased,
                            observation=observation2,
                        )
                        reasoning_steps.append(step2)
                        tools_used.append("search_knowledge_base")
                        all_observations.append(observation2)

                        # Try synthesizing again with combined results
                        synth_prompt2 = _SYNTHESIZE_PROMPT.format(
                            question=query,
                            observations="\n\n".join(all_observations),
                        )
                        raw_answer2 = generate_answer(synth_prompt2, settings)
                        answer = postprocess_answer(raw_answer2)
                        llm_calls += 1
                    except Exception as e:
                        logger.warning("Rephrased search failed: %s", e)

            # Clean up "Final Answer:" prefix if present
            if answer.lower().startswith("final answer:"):
                answer = answer[len("final answer:"):].strip()

            reasoning_steps.append(ReasoningStep(
                thought="I have enough information from the search results to answer.",
            ))

            logger.info("Agent completed: %d steps, %d tool calls, %d LLM calls",
                         len(reasoning_steps), len(tools_used), llm_calls)

            return AgentResult(
                question=query,
                final_answer=answer,
                reasoning_steps=reasoning_steps,
                tools_used=tools_used,
                total_llm_calls=llm_calls,
            )

        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            return AgentResult(
                question=query,
                final_answer=f"I encountered an error while generating the answer: {e}",
                reasoning_steps=reasoning_steps,
                tools_used=tools_used,
                total_llm_calls=llm_calls,
            )

    # ──────────────────────────────────────────────────
    # Fallback: No search results at all
    # ──────────────────────────────────────────────────
    return AgentResult(
        question=query,
        final_answer="I could not search the knowledge base. Please ensure documents are ingested.",
        reasoning_steps=reasoning_steps,
        tools_used=tools_used,
        total_llm_calls=llm_calls,
    )
