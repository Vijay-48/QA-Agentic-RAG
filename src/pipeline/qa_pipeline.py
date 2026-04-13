from __future__ import annotations
from src.config.settings import Settings, load_settings
from src.core.schemas import AnswerResult
from src.retrieval.retriver import retrieve
from src.prompting.prompt_builder import build_qa_prompt
from src.generation.ollama_generator import generate_answer
from src.generation.answer_postprocess import postprocess_answer
from src.core.logger import get_logger

logger = get_logger("QAPipeline")

def ask(question: str, settings: Settings | None = None) -> AnswerResult:
    if settings is None:
        settings = load_settings()
    
    #step-1 = retrive relevent chunks

    logger.info("Question: %s", question)
    hits = retrieve(question, settings)

    if not hits:
        logger.warning("No relevant chinks found")
        return AnswerResult(
            question=question,
            answer= "No relevent information found in the knowldege base.",
            citations=[],
            hits=[],
        )

    #step-2 = build prompt with context
    prompt = build_qa_prompt(question, hits)

    #step-3 = Generate answer
    raw_answer = generate_answer(prompt, settings)

    #step-4 = Clean up the answer
    answer = postprocess_answer(raw_answer)

    #step-5 = Build citations for hits
    citations = [
        {"source": hit.metadata.get("filename", "unknown"), "score": hit.score}
        for hit in hits
    ]
    logger.info("Answer Generated (%d chars, %d citations)", len(answer), len(citations))
    return AnswerResult(
        question=question,
        answer=answer,
        citations=citations,
        hits=hits,
    )