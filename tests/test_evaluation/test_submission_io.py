"""Tests for submission CSV I/O helpers."""

from pathlib import Path

import pandas as pd

from omnilex.evaluation import write_submission_csv
from omnilex.evaluation.submission_io import read_csv_preserve_empty_strings
from omnilex.evaluation.scorer import Scorer


def test_write_submission_csv_quotes_empty_predictions(tmp_path: Path):
    submission_df = pd.DataFrame(
        [
            {
                "query_id": "test_0001",
                "predicted_citations": "Art. 11 Abs. 2 OR;BGE 139 I 2 E. 3.1",
            },
            {"query_id": "test_0002", "predicted_citations": ""},
        ]
    )

    output_path = tmp_path / "submission.csv"
    write_submission_csv(submission_df, output_path)

    assert output_path.read_text(encoding="utf-8") == (
        "query_id,predicted_citations\n"
        'test_0001,"Art. 11 Abs. 2 OR;BGE 139 I 2 E. 3.1"\n'
        'test_0002,""\n'
    )


def test_read_csv_preserve_empty_strings_keeps_empty_predictions(tmp_path: Path):
    input_path = tmp_path / "submission.csv"
    input_path.write_text(
        "query_id,predicted_citations\n"
        'test_0001,"Art. 11 Abs. 2 OR;BGE 139 I 2 E. 3.1"\n'
        'test_0002,""\n',
        encoding="utf-8",
    )

    df = read_csv_preserve_empty_strings(input_path)

    assert df["predicted_citations"].tolist() == [
        "Art. 11 Abs. 2 OR;BGE 139 I 2 E. 3.1",
        "",
    ]


def test_scorer_load_submission_preserves_empty_predictions(tmp_path: Path):
    input_path = tmp_path / "submission.csv"
    input_path.write_text(
        "query_id,predicted_citations\n"
        'test_0001,"Art. 11 Abs. 2 OR"\n'
        'test_0002,""\n',
        encoding="utf-8",
    )

    loaded = Scorer().load_submission(input_path)

    assert loaded["predicted_citations"].tolist() == ["Art. 11 Abs. 2 OR", ""]
