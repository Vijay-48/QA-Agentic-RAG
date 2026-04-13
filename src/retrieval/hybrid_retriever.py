"""Hybrid retriever combining BM25 (keyword) and dense (embedding) search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.
BM25 catches exact keyword matches; dense catches semantic similarity.
Together they cover both failure modes.
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.config.settings import Settings
from src.core.schemas import RetrievalHit
from src.embedding.ollama_embedder import OllamaEmbedder
from src.storage.store_factory import get_vector_store
from src.preprocessing.query_preprocess import preprocess_query
from src.core.logger import get_logger

logger = get_logger("HybridRetriever")


def _reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalHit]],
    k: int = 60,
) -> list[RetrievalHit]:
    """Merge multiple ranked lists using RRF.

    RRF_score(doc) = sum( 1 / (k + rank_in_list_i) ) for each list.
    k=60 is a constant that controls how much lower ranks are dampened.

    Args:
        ranked_lists: Multiple result lists, each sorted by their own scores.
        k: RRF constant (default 60 from the original paper).

    Returns:
        Merged list sorted by combined RRF score.
    """
    # Map chunk_id → (accumulated_rrf_score, best_hit_object)
    scores: dict[str, float] = {}
    best_hit: dict[str, RetrievalHit] = {}

    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list):
            rrf = 1.0 / (k + rank + 1)  # rank is 0-indexed, formula uses 1-indexed
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + rrf
            # Keep the hit with the highest original score for metadata
            if hit.chunk_id not in best_hit or hit.score > best_hit[hit.chunk_id].score:
                best_hit[hit.chunk_id] = hit

    # Build merged hits with RRF scores
    merged = []
    for chunk_id, rrf_score in scores.items():
        original = best_hit[chunk_id]
        merged.append(
            RetrievalHit(
                chunk_id=chunk_id,
                score=rrf_score,
                text=original.text,
                metadata=original.metadata,
            )
        )

    merged.sort(key=lambda h: h.score, reverse=True)
    return merged


def _bm25_search(
    query: str,
    all_chunks: list[dict],
    top_k: int,
) -> list[RetrievalHit]:
    """Run BM25 keyword search over stored chunks.

    Args:
        query: The search query.
        all_chunks: List of dicts with 'text', 'chunk_id', 'metadata' keys.
        top_k: Number of results to return.

    Returns:
        List of RetrievalHit sorted by BM25 score.
    """
    if not all_chunks:
        return []

    # Tokenize all document texts
    corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(corpus)

    # Score the query
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Pair scores with chunks and sort
    scored = list(zip(scores, all_chunks))
    scored.sort(key=lambda x: x[0], reverse=True)

    hits = []
    for score, chunk in scored[:top_k]:
        if score > 0:
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.get("chunk_id", ""),
                    score=float(score),
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                )
            )

    return hits


def _dense_search(
    query: str,
    settings: Settings,
    top_k: int,
) -> list[RetrievalHit]:
    """Run dense vector search (same as simple retriever)."""
    embedder = OllamaEmbedder(settings)
    query_vector = embedder.embed_single(query)
    store = get_vector_store(settings)
    raw_results = store.query(query_vector, top_k)

    return [
        RetrievalHit(
            chunk_id=r["chunk_id"],
            score=r["score"],
            text=r["text"],
            metadata=r["metadata"],
        )
        for r in raw_results
    ]


def hybrid_retrieve(
    query: str,
    settings: Settings,
    top_k: int | None = None,
    dense_weight: int = 1,
    bm25_weight: int = 1,
) -> list[RetrievalHit]:
    """Retrieve using both BM25 and dense search, merged with RRF.

    Args:
        query: The user's query.
        settings: Application settings.
        top_k: Final number of results to return.
        dense_weight: How many copies of dense results in fusion (default 1).
        bm25_weight: How many copies of BM25 results in fusion (default 1).

    Returns:
        Merged, re-ranked list of RetrievalHit.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    query = preprocess_query(query)

    # Fetch more candidates than needed for better fusion
    candidate_k = top_k * 4

    # 1. Dense search
    logger.info("Running dense search for: '%s'", query[:50])
    dense_hits = _dense_search(query, settings, candidate_k)

    # 2. BM25 search — need all chunks for the BM25 index
    logger.info("Running BM25 search for: '%s'", query[:50])
    store = get_vector_store(settings)

    # Scroll all chunks from the vector store for BM25 indexing
    # For large datasets, this should be cached
    try:
        from qdrant_client import QdrantClient

        if hasattr(store, "client") and hasattr(store, "collection_name"):
            all_points, _ = store.client.scroll(
                collection_name=store.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            all_chunks = []
            for point in all_points:
                payload = dict(point.payload) if point.payload else {}
                text = payload.pop("text", "")
                chunk_id = payload.pop("chunk_id", str(point.id))
                all_chunks.append({
                    "text": text,
                    "chunk_id": chunk_id,
                    "metadata": payload,
                })
        else:
            all_chunks = []
    except Exception as e:
        logger.warning("BM25 corpus loading failed: %s. Using dense only.", e)
        all_chunks = []

    bm25_hits = _bm25_search(query, all_chunks, candidate_k) if all_chunks else []

    # 3. Fuse results
    if bm25_hits:
        ranked_lists = (
            [dense_hits] * dense_weight
            + [bm25_hits] * bm25_weight
        )
        merged = _reciprocal_rank_fusion(ranked_lists)
    else:
        # Fallback to dense only if BM25 failed
        merged = dense_hits

    result = merged[:top_k]
    logger.info(
        "Hybrid retrieval: %d dense + %d BM25 → %d merged → top %d",
        len(dense_hits), len(bm25_hits), len(merged), len(result),
    )
    return result
