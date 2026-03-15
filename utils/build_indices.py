#!/usr/bin/env python3
"""Build semantic indices for legal document corpora."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    SemanticIndex,
    load_jsonl_corpus,
)


def resolve_corpus_path(possible_paths: list[Path]) -> Path | None:
    """Return the first existing corpus file path."""
    for path in possible_paths:
        if path.exists():
            return path
    return None


def build_semantic_index(
    *,
    corpus_name: str,
    possible_paths: list[Path],
    output_path: Path,
    test_query: str,
    embedding_model: str,
    candidate_k: int,
    batch_size: int,
    device: str | None,
) -> None:
    """Build a semantic index for one corpus."""
    print(f"Building semantic index for {corpus_name}...")

    corpus_path = resolve_corpus_path(possible_paths)
    if corpus_path is None:
        print(f"  Warning: No {corpus_name} corpus found. Skipping index build.")
        print(f"  Expected one of: {[str(path) for path in possible_paths]}")
        return

    print(f"  Loading corpus from {corpus_path}")
    documents = load_jsonl_corpus(corpus_path)
    print(f"  Loaded {len(documents)} documents")

    if not documents:
        print("  Warning: Empty corpus. Skipping index build.")
        return

    index = SemanticIndex(
        documents=documents,
        text_field="text",
        citation_field="citation",
        embedding_model_name=embedding_model,
        candidate_k=candidate_k,
        batch_size=batch_size,
        device=device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(output_path)
    print(f"  Index saved to {output_path}")

    # Smoke-test dense retrieval without triggering the reranker download here.
    results = index.search(test_query, top_k=3, return_scores=True, rerank=False)
    print(f"  Dense search smoke test for '{test_query}': {len(results)} results")


def build_laws_index(
    input_dir: Path,
    output_dir: Path,
    *,
    embedding_model: str,
    candidate_k: int,
    batch_size: int,
    device: str | None,
) -> None:
    """Build semantic index for federal laws corpus."""
    build_semantic_index(
        corpus_name="federal laws",
        possible_paths=[
            input_dir / "samples" / "federal_laws.jsonl",
            input_dir / "federal_laws.jsonl",
            input_dir / "laws" / "federal_laws.jsonl",
            input_dir / "corpus" / "laws.jsonl",
        ],
        output_path=output_dir / "laws_index.pkl",
        test_query="Vertrag",
        embedding_model=embedding_model,
        candidate_k=candidate_k,
        batch_size=batch_size,
        device=device,
    )


def build_courts_index(
    input_dir: Path,
    output_dir: Path,
    *,
    embedding_model: str,
    candidate_k: int,
    batch_size: int,
    device: str | None,
) -> None:
    """Build semantic index for court decisions corpus."""
    build_semantic_index(
        corpus_name="court decisions",
        possible_paths=[
            input_dir / "samples" / "court_decisions.jsonl",
            input_dir / "court_decisions.jsonl",
            input_dir / "courts" / "bge.jsonl",
            input_dir / "corpus" / "courts.jsonl",
        ],
        output_path=output_dir / "courts_index.pkl",
        test_query="Meinungsfreiheit",
        embedding_model=embedding_model,
        candidate_k=candidate_k,
        batch_size=batch_size,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic indices for legal corpora")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Input directory with corpus files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for indices",
    )
    parser.add_argument(
        "--laws-only",
        action="store_true",
        help="Only build laws index",
    )
    parser.add_argument(
        "--courts-only",
        action="store_true",
        help="Only build courts index",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model to use for dense indexing",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
        help="Number of dense candidates to keep for reranking at query time",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used while embedding the corpus",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional device override for sentence-transformers "
            "(for example 'cpu', 'cuda', or 'mps')"
        ),
    )

    args = parser.parse_args()

    if not args.courts_only:
        build_laws_index(
            args.input_dir,
            args.output_dir,
            embedding_model=args.embedding_model,
            candidate_k=args.candidate_k,
            batch_size=args.batch_size,
            device=args.device,
        )

    if not args.laws_only:
        build_courts_index(
            args.input_dir,
            args.output_dir,
            embedding_model=args.embedding_model,
            candidate_k=args.candidate_k,
            batch_size=args.batch_size,
            device=args.device,
        )

    print("\nIndex building complete!")


if __name__ == "__main__":
    main()
