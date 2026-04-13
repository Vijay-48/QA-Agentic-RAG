"""Qdrant payload filters for metadata-based search narrowing.

These filters are applied BEFORE vector similarity search, so they reduce
the search space rather than post-filtering results.
"""
from __future__ import annotations

from typing import Any

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range


def build_filters(
    filename: str | None = None,
    doc_id: str | None = None,
    min_chunk_index: int | None = None,
    max_chunk_index: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Filter | None:
    """Build a Qdrant Filter from optional metadata constraints.

    Args:
        filename: Filter to chunks from a specific file.
        doc_id: Filter to chunks from a specific document.
        min_chunk_index: Only chunks at or after this index.
        max_chunk_index: Only chunks at or before this index.
        extra: Additional key=value payload filters.

    Returns:
        A Qdrant Filter object, or None if no constraints are provided.
    """
    conditions: list[FieldCondition] = []

    if filename:
        conditions.append(
            FieldCondition(key="filename", match=MatchValue(value=filename))
        )

    if doc_id:
        conditions.append(
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
        )

    if min_chunk_index is not None or max_chunk_index is not None:
        range_params: dict[str, int] = {}
        if min_chunk_index is not None:
            range_params["gte"] = min_chunk_index
        if max_chunk_index is not None:
            range_params["lte"] = max_chunk_index
        conditions.append(
            FieldCondition(key="chunk_index", range=Range(**range_params))
        )

    if extra:
        for key, value in extra.items():
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

    if not conditions:
        return None

    return Filter(must=conditions)
