from __future__ import annotations

import re


def preprocess_query(query: str) -> str:
	query = query.strip()
	query = re.sub(r"\s+", " ", query)
	return query
