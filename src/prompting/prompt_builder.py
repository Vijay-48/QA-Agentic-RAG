from __future__ import annotations
from src.core.schemas import RetrievalHit
from src.prompting.templates import QA_TEMPLATE


def format_context(hits:list[RetrievalHit]) -> str:
    blocks = []
    for hit in hits:
        source = hit.metadata.get("filename", "unknown")
        block = f"[Source: {source}]\n{hit.text}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)

def build_qa_prompt(question: str, hits: list[RetrievalHit]) -> str:
    context = format_context(hits)
    return QA_TEMPLATE.format(context=context, question=question)