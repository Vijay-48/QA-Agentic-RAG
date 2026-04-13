from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from src.core.schemas import Document, build_document_id
from src.ingestion.cleaners import clean_text
from src.ingestion.metadata import build_document_metadata

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _read_pdf(path: Path) -> str:
	reader = PdfReader(str(path))
	page_texts = [(page.extract_text() or "") for page in reader.pages]
	return "\n".join(page_texts)


def _read_text(path: Path) -> str:
	return path.read_text(encoding="utf-8", errors="ignore")


def load_file_text(path: Path) -> str:
	extension = path.suffix.lower()
	if extension == ".pdf":
		return _read_pdf(path)
	if extension in {".txt", ".md"}:
		return _read_text(path)
	raise ValueError(f"Unsupported extension: {extension}")


def discover_input_files(paths: Iterable[str]) -> list[Path]:
	discovered: list[Path] = []

	for item in paths:
		path = Path(item)
		if not path.exists():
			continue

		if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
			discovered.append(path)
			continue

		if path.is_dir():
			for extension in SUPPORTED_EXTENSIONS:
				discovered.extend(path.rglob(f"*{extension}"))

	# Remove duplicates while preserving order.
	unique: dict[str, Path] = {}
	for file_path in discovered:
		unique[str(file_path.resolve())] = file_path

	return list(unique.values())


def load_documents(paths: Iterable[str]) -> list[Document]:
	documents: list[Document] = []

	for path in discover_input_files(paths):
		raw_text = load_file_text(path)
		cleaned_text = clean_text(raw_text)
		if not cleaned_text:
			continue

		source = str(path.resolve())
		doc_id = build_document_id(source, cleaned_text)
		metadata = build_document_metadata(path)
		documents.append(
			Document(
				doc_id=doc_id,
				source=source,
				text=cleaned_text,
				metadata=metadata,
			)
		)

	return documents
