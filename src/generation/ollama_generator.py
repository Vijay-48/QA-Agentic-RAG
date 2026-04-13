from __future__ import annotations
from src.config.settings import Settings
from src.core.exceptions import GenerationError
from src.core.logger import get_logger
import requests

logger = get_logger("Generation")

def generate_answer(prompt:str, settings: Settings) -> str:
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_chat_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout= settings.request_timeout_seconds)
        response.raise_for_status()
        data = response.json()
        answer = data["message"]["content"]
        logger.info("Generated answer (%d chars)", len(answer))
        return answer

    except requests.RequestException as e:
        raise GenerationError(f"Generation failed: {e}") from e
