"""Helpers for competition submission CSV I/O."""

from pathlib import Path

import pandas as pd


def read_csv_preserve_empty_strings(path: Path | str) -> pd.DataFrame:
    """Load a CSV while preserving quoted empty strings as empty strings."""

    return pd.read_csv(path, keep_default_na=False)


def _escape_csv_field(value: object, *, always_quote: bool = False) -> str:
    """Escape a single CSV field."""

    text = "" if value is None else str(value)
    escaped = text.replace('"', '""')
    needs_quotes = always_quote or any(ch in text for ch in [",", '"', "\n", "\r"])
    return f'"{escaped}"' if needs_quotes else escaped


def write_submission_csv(submission_df: pd.DataFrame, output_path: Path | str) -> Path:
    """Write a submission CSV with explicit quoted citation strings.

    The competition backend accepts empty predictions as `""`, but some generic
    CSV writers serialize them as blank trailing fields. This helper always
    quotes `predicted_citations` so empty predictions survive round-tripping.
    """

    required_cols = {"query_id", "predicted_citations"}
    missing = required_cols - set(submission_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    submission_df = submission_df.loc[:, ["query_id", "predicted_citations"]].copy()
    if submission_df["query_id"].isna().any():
        raise ValueError("Submission contains null query_id values")

    submission_df["predicted_citations"] = submission_df["predicted_citations"].fillna("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("query_id,predicted_citations\n")
        for row in submission_df.itertuples(index=False):
            query_id = _escape_csv_field(row.query_id)
            predicted_citations = _escape_csv_field(row.predicted_citations, always_quote=True)
            handle.write(f"{query_id},{predicted_citations}\n")

    return output_path
