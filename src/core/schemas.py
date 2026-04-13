from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def stable_hash(value: str) -> str:
	return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_document_id(source: str, text: str) -> str:
	# Include source and content for stable, repeatable dedupe behavior.
	return stable_hash(f"{source}::{text}")[:32]


def build_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
	return stable_hash(f"{doc_id}:{chunk_index}:{chunk_text}")[:32]


@dataclass(slots=True)
class Document:
	doc_id: str
	source: str
	text: str
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
	chunk_id: str
	doc_id: str
	text: str
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalHit:
	chunk_id: str
	score: float
	text: str
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerResult:
	question: str
	answer: str
	citations: list[dict[str, Any]] = field(default_factory=list)
	hits: list[RetrievalHit] = field(default_factory=list)


@dataclass(slots=True)
class ReasoningStep:
	"""A single think → act → observe cycle in the agent loop."""
	thought: str
	action: str = ""
	action_input: str = ""
	observation: str = ""


@dataclass(slots=True)
class AgentResult:
	"""Full result from an agentic RAG execution."""
	question: str
	final_answer: str
	reasoning_steps: list[ReasoningStep] = field(default_factory=list)
	tools_used: list[str] = field(default_factory=list)
	total_llm_calls: int = 0
	citations: list[dict[str, Any]] = field(default_factory=list)
