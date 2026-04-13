from __future__ import annotations

from src.core.schemas import Document
from src.ingestion.cleaners import clean_text


def preprocess_document(document: Document) -> Document:
	cleaned_text = clean_text(document.text)
	document.text = cleaned_text
	return document


def preprocess_documents(documents: list[Document]) -> list[Document]:
	return [preprocess_document(document) for document in documents if document.text.strip()]
