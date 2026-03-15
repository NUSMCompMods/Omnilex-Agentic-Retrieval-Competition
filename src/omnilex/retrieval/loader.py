"""Index loading helpers for retrieval backends."""

from __future__ import annotations

import pickle
from pathlib import Path

from .bm25_index import BM25Index
from .semantic_index import (
    EmbeddingBackend,
    RerankerBackend,
    SemanticIndex,
    _UNSET,
)


def load_index(
    path: Path | str,
    *,
    batch_size: int = 32,
    device: str | None = None,
    embedder: EmbeddingBackend | None = None,
    reranker: RerankerBackend | None = None,
    candidate_k: int | None = None,
    reranker_model_name: str | None | object = _UNSET,
) -> BM25Index | SemanticIndex:
    """Load either a BM25 or semantic index from disk."""
    path = Path(path)
    with open(path, "rb") as file_obj:
        data = pickle.load(file_obj)

    index_type = data.get("index_type")
    if index_type == "semantic" or "embeddings" in data:
        return SemanticIndex.from_serialized_data(
            data,
            batch_size=batch_size,
            device=device,
            embedder=embedder,
            reranker=reranker,
            candidate_k=candidate_k,
            reranker_model_name=reranker_model_name,
        )

    return BM25Index.from_serialized_data(data)
