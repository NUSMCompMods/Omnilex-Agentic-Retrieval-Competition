"""Tests for semantic retrieval and reranking."""

from __future__ import annotations

import pickle

import numpy as np

from omnilex.retrieval import BM25Index, LawSearchTool, SemanticIndex, load_index


class KeywordEmbedder:
    """Deterministic embedding backend for tests."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vector = np.array([0.05, 0.05, 0.05], dtype=np.float32)
            if "vertrag" in lower or "contract" in lower:
                vector += np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if "fahrl" in lower or "homicide" in lower:
                vector += np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if "meinungsfreiheit" in lower or "speech" in lower:
                vector += np.array([0.0, 0.0, 1.0], dtype=np.float32)
            vectors.append(vector)
        return np.vstack(vectors)


class OrderedEmbedder:
    """Embedder that yields a stable dense ranking for rerank tests."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            if text.startswith("query:"):
                vectors.append(np.array([1.0, 0.0], dtype=np.float32))
            elif "Art. 1 OR" in text:
                vectors.append(np.array([0.95, 0.05], dtype=np.float32))
            elif "BGE 119 II 449 E. 3.4" in text:
                vectors.append(np.array([0.85, 0.15], dtype=np.float32))
            else:
                vectors.append(np.array([0.1, 0.9], dtype=np.float32))
        return np.vstack(vectors)


class CitationReranker:
    """Reranker that prefers case law over statutes for the same query."""

    def score(self, query, documents):
        del query
        scores = []
        for document in documents:
            if "BGE 119 II 449 E. 3.4" in document:
                scores.append(0.95)
            elif "Art. 1 OR" in document:
                scores.append(0.70)
            else:
                scores.append(0.10)
        return scores


def test_semantic_index_build_search_save_load_roundtrip(sample_laws_corpus, tmp_path):
    index = SemanticIndex(
        documents=sample_laws_corpus,
        embedder=KeywordEmbedder(),
        reranker_model_name=None,
        candidate_k=5,
    )

    results = index.search(
        "What are the contract formation requirements?",
        top_k=2,
        return_scores=True,
    )

    assert len(results) == 2
    assert results[0]["citation"] == "Art. 1 OR"
    assert results[0]["_dense_score"] >= results[1]["_dense_score"]
    assert results[0]["_rerank_score"] is None
    assert index.embeddings is not None
    assert index.embeddings.dtype == np.float32

    path = tmp_path / "semantic_index.pkl"
    index.save(path)

    loaded = SemanticIndex.load(
        path,
        embedder=KeywordEmbedder(),
        reranker_model_name=None,
    )
    reloaded_results = loaded.search(
        "What are the contract formation requirements?",
        top_k=2,
        return_scores=True,
    )

    assert len(reloaded_results) == 2
    assert reloaded_results[0]["citation"] == "Art. 1 OR"
    assert loaded.embeddings is not None
    assert loaded.embeddings.dtype == np.float32


def test_reranker_changes_order_after_dense_retrieval():
    documents = [
        {
            "citation": "Art. 1 OR",
            "title": "Obligationenrecht - Abschluss des Vertrages",
            "text": "Zum Abschlusse eines Vertrages ist die gegenseitige Willensäusserung nötig.",
        },
        {
            "citation": "BGE 119 II 449 E. 3.4",
            "regeste": "Vertragsabschluss durch konkludentes Verhalten",
            "text": "Der Vertrag kommt durch übereinstimmende Willenserklärungen zustande.",
        },
    ]

    index = SemanticIndex(
        documents=documents,
        embedder=OrderedEmbedder(),
        reranker=CitationReranker(),
        candidate_k=2,
    )

    results = index.search(
        "contract formation requirements",
        top_k=2,
        return_scores=True,
    )

    assert [result["citation"] for result in results] == [
        "BGE 119 II 449 E. 3.4",
        "Art. 1 OR",
    ]
    assert results[0]["_dense_score"] < results[1]["_dense_score"]
    assert results[0]["_rerank_score"] > results[1]["_rerank_score"]


def test_load_index_supports_semantic_and_legacy_bm25(sample_laws_corpus, tmp_path):
    semantic_path = tmp_path / "semantic.pkl"
    semantic_index = SemanticIndex(
        documents=sample_laws_corpus,
        embedder=KeywordEmbedder(),
        reranker_model_name=None,
    )
    semantic_index.save(semantic_path)

    loaded_semantic = load_index(
        semantic_path,
        embedder=KeywordEmbedder(),
        reranker_model_name=None,
    )
    assert isinstance(loaded_semantic, SemanticIndex)

    legacy_bm25 = BM25Index(sample_laws_corpus)
    legacy_path = tmp_path / "legacy_bm25.pkl"
    with open(legacy_path, "wb") as file_obj:
        pickle.dump(
            {
                "documents": legacy_bm25.documents,
                "tokenized_corpus": legacy_bm25._tokenized_corpus,
                "text_field": legacy_bm25.text_field,
                "citation_field": legacy_bm25.citation_field,
            },
            file_obj,
        )

    loaded_bm25 = load_index(legacy_path)
    assert isinstance(loaded_bm25, BM25Index)


def test_law_search_tool_returns_metadata_scores(sample_laws_corpus):
    index = SemanticIndex(
        documents=sample_laws_corpus,
        embedder=KeywordEmbedder(),
        reranker=CitationReranker(),
        candidate_k=3,
    )
    tool = LawSearchTool(index=index, top_k=2, max_excerpt_length=120)

    metadata_results = tool.search_with_metadata("contract formation requirements")
    formatted = tool("contract formation requirements")

    assert metadata_results
    assert {"_dense_score", "_rerank_score", "_score"} <= metadata_results[0].keys()
    assert formatted.startswith("- ")
    assert "Art. 1 OR" in formatted
