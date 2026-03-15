"""Retrieval tools and indexing for Swiss legal documents."""

from .bm25_index import BM25Index, build_index, load_jsonl_corpus, search
from .loader import load_index
from .semantic_index import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANK_MODEL,
    CrossEncoderReranker,
    SemanticIndex,
    SentenceTransformerEmbedder,
)
from .tools import CombinedSearchTool, CourtSearchTool, LawSearchTool

__all__ = [
    "BM25Index",
    "CombinedSearchTool",
    "CrossEncoderReranker",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RERANK_MODEL",
    "SemanticIndex",
    "SentenceTransformerEmbedder",
    "build_index",
    "load_jsonl_corpus",
    "load_index",
    "search",
    "LawSearchTool",
    "CourtSearchTool",
]
