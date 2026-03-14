#!/usr/bin/env python3
"""Normalize a submission file to canonical citation IDs.

Usage:
    python utils/normalize_submission.py output/submission.csv
    python utils/normalize_submission.py output/submission.csv --output output/submission_normalized.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.citations.normalizer import CitationNormalizer


def normalize_submission(
    input_path: Path,
    output_path: Path,
    citation_separator: str = ";",
) -> tuple[int, int]:
    """Normalize predicted citations and write the cleaned CSV."""

    df = pd.read_csv(input_path)

    required_cols = {"query_id", "predicted_citations"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    normalizer = CitationNormalizer()
    raw_total = 0
    normalized_total = 0

    normalized_rows = []
    for value in df["predicted_citations"].fillna(""):
        raw_citations = [c.strip() for c in str(value).split(citation_separator) if c.strip()]
        canonical_citations = normalizer.canonicalize_list(raw_citations)
        raw_total += len(raw_citations)
        normalized_total += len(canonical_citations)
        normalized_rows.append(citation_separator.join(canonical_citations))

    df["predicted_citations"] = normalized_rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return raw_total, normalized_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize submission citations")
    parser.add_argument("submission", type=Path, help="Path to submission CSV")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path. Defaults to overwriting the input file.",
    )
    args = parser.parse_args()

    output_path = args.output or args.submission
    raw_total, normalized_total = normalize_submission(args.submission, output_path)

    print(f"Normalized submission saved to: {output_path}")
    print(f"Raw citations: {raw_total}")
    print(f"Canonical citations kept: {normalized_total}")


if __name__ == "__main__":
    main()
