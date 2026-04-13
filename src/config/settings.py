from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _int_env(name: str, default: int) -> int:
	raw = os.getenv(name)
	if raw is None:
		return default
	try:
		return int(raw)
	except ValueError:
		return default


@dataclass(frozen=True)
class Settings:
	project_root: Path
	data_raw_dir: Path
	data_processed_dir: Path
	data_eval_dir: Path
	artifacts_dir: Path
	artifacts_logs_dir: Path
	artifacts_cache_dir: Path
	chroma_persist_dir: Path

	ollama_base_url: str
	ollama_embedding_model: str
	ollama_chat_model: str
	request_timeout_seconds: int

	chunk_strategy: str
	chunk_size_tokens: int
	chunk_overlap_tokens: int
	chunk_min_tokens: int
	chunk_hard_max_tokens: int

	embedding_batch_size: int
	retrieval_top_k: int
	chroma_collection_name: str
	vector_store: str
	qdrant_url: str
	qdrant_api_key: str


def load_settings() -> Settings:
	load_dotenv()

	project_root = Path(__file__).resolve().parents[2]
	data_dir = project_root / "data"
	artifacts_dir = project_root / "artifacts"

	return Settings(
		project_root=project_root,
		data_raw_dir=data_dir / "raw",
		data_processed_dir=data_dir / "processed",
		data_eval_dir=data_dir / "eval",
		artifacts_dir=artifacts_dir,
		artifacts_logs_dir=artifacts_dir / "logs",
		artifacts_cache_dir=artifacts_dir / "cache",
		chroma_persist_dir=artifacts_dir / "vector_index" / "chroma",
		ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
		ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
		ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:0.6b"),
		request_timeout_seconds=_int_env("REQUEST_TIMEOUT_SECONDS", 120),
		chunk_strategy=os.getenv("CHUNK_STRATEGY", "semantic"),
		chunk_size_tokens=_int_env("CHUNK_SIZE_TOKENS", 220),
		chunk_overlap_tokens=_int_env("CHUNK_OVERLAP_TOKENS", 40),
		chunk_min_tokens=_int_env("CHUNK_MIN_TOKENS", 40),
		chunk_hard_max_tokens=_int_env("CHUNK_HARD_MAX_TOKENS", 320),
		embedding_batch_size=_int_env("EMBEDDING_BATCH_SIZE", 16),
		retrieval_top_k=_int_env("RETRIEVAL_TOP_K", 5),
		chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_chunks"),
		vector_store=os.getenv("VECTOR_STORE", "qdrant"),
		qdrant_url=os.getenv("QDRANT_URL", ""),
		qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
	)


def ensure_runtime_dirs(settings: Settings) -> None:
	directories = [
		settings.data_raw_dir,
		settings.data_processed_dir,
		settings.data_eval_dir,
		settings.artifacts_dir,
		settings.artifacts_logs_dir,
		settings.artifacts_cache_dir,
		settings.chroma_persist_dir,
	]

	for directory in directories:
		directory.mkdir(parents=True, exist_ok=True)
