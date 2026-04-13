from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.schemas import Document, stable_hash


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def build_document_metadata(path: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
	metadata = {
		"source": str(path),
		"filename": path.name,
		"title": path.stem,
		"extension": path.suffix.lower(),
		"loaded_at": utc_now_iso(),
	}
	if extra:
		metadata.update(extra)
	return metadata


def build_chunk_metadata(
	document: Document,
	chunk_index: int,
	total_chunks: int,
	chunk_text: str,
) -> dict[str, Any]:
	metadata = dict(document.metadata)
	metadata.update(
		{
			"doc_id": document.doc_id,
			"chunk_index": chunk_index,
			"total_chunks": total_chunks,
			"chunk_content_hash": stable_hash(chunk_text),
		}
	)
	return metadata
