"""Dense semantic search with optional reranking for legal corpora."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import ClassVar, Protocol, Sequence

import numpy as np

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
_UNSET = object()


class EmbeddingBackend(Protocol):
    """Protocol for text embedding backends."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts into a dense matrix."""


class RerankerBackend(Protocol):
    """Protocol for document rerankers."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one reranking score per document."""


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings while avoiding division by zero."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.size == 0:
        return embeddings

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers embedder with in-process model caching."""

    _MODEL_CACHE: ClassVar[dict[tuple[str, str | None], object]] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

    def _get_model(self):
        key = (self.model_name, self.device)
        model = self._MODEL_CACHE.get(key)
        if model is not None:
            return model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Semantic retrieval requires sentence-transformers. "
                "Install the project requirements before building/searching semantic indices."
            ) from exc

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device

        model = SentenceTransformer(self.model_name, **kwargs)
        self._MODEL_CACHE[key] = model
        return model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self._get_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class CrossEncoderReranker:
    """Lazy cross-encoder reranker with in-process model caching."""

    _MODEL_CACHE: ClassVar[dict[tuple[str, str | None, int], object]] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        batch_size: int = 8,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length

    def _get_model(self):
        key = (self.model_name, self.device, self.max_length)
        model = self._MODEL_CACHE.get(key)
        if model is not None:
            return model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Semantic reranking requires sentence-transformers. "
                "Install the project requirements before reranking results."
            ) from exc

        kwargs = {"max_length": self.max_length}
        if self.device:
            kwargs["device"] = self.device

        model = CrossEncoder(self.model_name, **kwargs)
        self._MODEL_CACHE[key] = model
        return model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []

        model = self._get_model()
        pairs = [(query, document) for document in documents]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32).tolist()


class SemanticIndex:
    """Semantic index backed by dense embeddings and optional reranking."""

    def __init__(
        self,
        documents: list[dict] | None = None,
        *,
        text_field: str = "text",
        citation_field: str = "citation",
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model_name: str | None = DEFAULT_RERANK_MODEL,
        candidate_k: int = 30,
        batch_size: int = 32,
        device: str | None = None,
        embedder: EmbeddingBackend | None = None,
        reranker: RerankerBackend | None = None,
    ) -> None:
        self.text_field = text_field
        self.citation_field = citation_field
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        self.device = device

        self.documents: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._document_payloads: list[str] = []

        self.embedder = embedder or SentenceTransformerEmbedder(
            embedding_model_name,
            batch_size=batch_size,
            device=device,
        )
        if reranker is not None:
            self.reranker = reranker
        elif reranker_model_name:
            self.reranker = CrossEncoderReranker(
                reranker_model_name,
                device=device,
            )
        else:
            self.reranker = None

        if documents:
            self.build(documents)

    @staticmethod
    def compose_document_payload(
        document: dict,
        text_field: str = "text",
        citation_field: str = "citation",
    ) -> str:
        """Compose a metadata-aware text payload for indexing and reranking."""
        parts: list[str] = []

        citation = document.get(citation_field, document.get("citation"))
        if citation:
            parts.append(f"citation: {citation}")

        title = document.get("title")
        if title:
            parts.append(f"title: {title}")

        regeste = document.get("regeste")
        if regeste:
            parts.append(f"regeste: {regeste}")

        text = document.get(text_field, "")
        if text:
            parts.append(f"text: {text}")

        return "\n".join(parts).strip()

    def _embedding_payload(self, document: dict) -> str:
        return self.compose_document_payload(
            document,
            text_field=self.text_field,
            citation_field=self.citation_field,
        )

    def _ensure_embedder(self) -> EmbeddingBackend:
        if self.embedder is None:
            raise ValueError("No embedder configured for semantic search.")
        return self.embedder

    def build(self, documents: list[dict]) -> None:
        """Build dense embeddings for the provided documents."""
        self.documents = list(documents)
        self._document_payloads = [self._embedding_payload(document) for document in self.documents]

        if not self.documents:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return

        embedder = self._ensure_embedder()
        embeddings = embedder.encode([f"passage: {payload}" for payload in self._document_payloads])
        if len(embeddings) != len(self.documents):
            raise ValueError("Embedding backend returned a mismatched number of vectors.")

        self.embeddings = _normalize_embeddings(np.asarray(embeddings, dtype=np.float32))

    def _search_dense(self, query: str, candidate_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.embeddings is None:
            raise ValueError("Index not built. Call build() first.")

        if self.embeddings.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)

        embedder = self._ensure_embedder()
        query_embedding = embedder.encode([f"query: {query.strip()}"])
        query_vector = _normalize_embeddings(query_embedding)[0]

        dense_scores = self.embeddings @ query_vector
        candidate_k = min(len(self.documents), max(1, candidate_k))
        top_indices = np.argpartition(dense_scores, -candidate_k)[-candidate_k:]
        ordered_indices = top_indices[np.argsort(dense_scores[top_indices])[::-1]]
        return ordered_indices.astype(int), dense_scores.astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
        *,
        candidate_k: int | None = None,
        rerank: bool = True,
    ) -> list[dict]:
        """Search the semantic index and optionally rerank the dense candidates."""
        query = query.strip()
        if not query:
            return []

        candidate_total = max(top_k, candidate_k or self.candidate_k)
        candidate_indices, dense_scores = self._search_dense(query, candidate_total)
        if len(candidate_indices) == 0:
            return []

        rerank_scores: np.ndarray | None = None
        rerank_order = np.arange(len(candidate_indices))
        if rerank and self.reranker is not None:
            candidate_payloads = [self._document_payloads[index] for index in candidate_indices]
            rerank_scores = np.asarray(
                self.reranker.score(query, candidate_payloads),
                dtype=np.float32,
            )
            if len(rerank_scores) != len(candidate_indices):
                raise ValueError("Reranker returned a mismatched number of scores.")
            rerank_order = np.argsort(rerank_scores)[::-1]

        results: list[dict] = []
        for position in rerank_order[: min(top_k, len(candidate_indices))]:
            index = int(candidate_indices[position])
            document = self.documents[index].copy()

            if return_scores:
                dense_score = float(dense_scores[index])
                rerank_score = None if rerank_scores is None else float(rerank_scores[position])
                document["_dense_score"] = dense_score
                document["_rerank_score"] = rerank_score
                document["_score"] = rerank_score if rerank_score is not None else dense_score

            results.append(document)

        return results

    def save(self, path: Path | str) -> None:
        """Save the semantic index to disk."""
        if self.embeddings is None:
            raise ValueError("Index not built. Call build() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "index_type": "semantic",
            "documents": self.documents,
            "document_payloads": self._document_payloads,
            "embeddings": np.asarray(self.embeddings, dtype=np.float32),
            "text_field": self.text_field,
            "citation_field": self.citation_field,
            "embedding_model_name": self.embedding_model_name,
            "reranker_model_name": self.reranker_model_name,
            "candidate_k": self.candidate_k,
        }

        with open(path, "wb") as file_obj:
            pickle.dump(data, file_obj)

    @classmethod
    def from_serialized_data(
        cls,
        data: dict,
        *,
        batch_size: int = 32,
        device: str | None = None,
        embedder: EmbeddingBackend | None = None,
        reranker: RerankerBackend | None = None,
        candidate_k: int | None = None,
        reranker_model_name: str | None | object = _UNSET,
    ) -> "SemanticIndex":
        """Construct an index from serialized semantic index data."""
        if data.get("index_type") != "semantic":
            raise ValueError("Serialized data is not a semantic index.")

        resolved_reranker_model = data.get("reranker_model_name", DEFAULT_RERANK_MODEL)
        if reranker_model_name is not _UNSET:
            resolved_reranker_model = reranker_model_name

        instance = cls(
            text_field=data.get("text_field", "text"),
            citation_field=data.get("citation_field", "citation"),
            embedding_model_name=data.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL),
            reranker_model_name=resolved_reranker_model,
            candidate_k=candidate_k or data.get("candidate_k", 30),
            batch_size=batch_size,
            device=device,
            embedder=embedder,
            reranker=reranker,
        )
        instance.documents = data.get("documents", [])
        instance._document_payloads = data.get("document_payloads") or [
            instance._embedding_payload(document) for document in instance.documents
        ]
        instance.embeddings = np.asarray(data.get("embeddings"), dtype=np.float32)
        return instance

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        batch_size: int = 32,
        device: str | None = None,
        embedder: EmbeddingBackend | None = None,
        reranker: RerankerBackend | None = None,
        candidate_k: int | None = None,
        reranker_model_name: str | None | object = _UNSET,
    ) -> "SemanticIndex":
        """Load a semantic index from disk."""
        path = Path(path)
        with open(path, "rb") as file_obj:
            data = pickle.load(file_obj)

        return cls.from_serialized_data(
            data,
            batch_size=batch_size,
            device=device,
            embedder=embedder,
            reranker=reranker,
            candidate_k=candidate_k,
            reranker_model_name=reranker_model_name,
        )
