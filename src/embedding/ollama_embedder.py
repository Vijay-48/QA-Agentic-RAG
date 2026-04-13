from __future__ import annotations
import requests
from src.config.settings import Settings
from src.core.exceptions import EmbeddingError
from src.core.logger import get_logger

logger = get_logger("embedding")

class OllamaEmbedder:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_embedding_model
        self.timeout = settings.request_timeout_seconds

    
    def embed_single(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except requests.RequestException as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_single(text))
        logger.info("Embedded batch of %d texts", len(texts))
        return embeddings