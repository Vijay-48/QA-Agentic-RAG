from __future__ import annotations

import re


def normalize_newlines(text: str) -> str:
	return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_pdf_hyphen_breaks(text: str) -> str:
	# Converts words split by line wraps like "retrie-\nval" into "retrieval".
	return re.sub(r"-\n(?=\w)", "", text)


def collapse_whitespace(text: str) -> str:
	text = re.sub(r"[ \t]+", " ", text)
	text = re.sub(r"\n{3,}", "\n\n", text)
	return text


def strip_non_printable(text: str) -> str:
	# Keep newline and tab while removing other control characters.
	return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def clean_text(text: str) -> str:
	cleaned = normalize_newlines(text)
	cleaned = strip_non_printable(cleaned)
	cleaned = remove_pdf_hyphen_breaks(cleaned)
	cleaned = collapse_whitespace(cleaned)
	return cleaned.strip()
