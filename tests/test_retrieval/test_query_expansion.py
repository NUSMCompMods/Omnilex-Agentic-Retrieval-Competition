"""Tests for multilingual query expansion and merged retrieval."""

import pytest

from omnilex.llm.prompts import parse_query_expansion_output
from omnilex.retrieval import BM25Index, ExpandedQuery, LawSearchTool, MultilingualQueryExpander


class FakeLlm:
    """Small llama-compatible stub used for query expansion tests."""

    def __init__(self, text: str):
        self.text = text

    def __call__(self, prompt, **kwargs):
        return {"choices": [{"text": self.text}]}


class StubExpander:
    """Fixed expander used for search tool tests."""

    def __init__(self, expanded_queries: list[ExpandedQuery]):
        self.expanded_queries = expanded_queries

    def expand(self, query: str) -> list[ExpandedQuery]:
        return list(self.expanded_queries)


def test_parse_query_expansion_output_normalizes_whitespace_and_deduplicates():
    output = """```json
    {
      "queries": [
        {"language": "en", "query": "contract   formation   requirements"},
        {"language": "en", "query": "contract formation requirements"},
        {"language": "de", "query": "  Vertrag  Abschluss  Voraussetzungen  "},
        {"language": "fr", "query": "contrat formation"}
      ]
    }
    ```"""

    parsed = parse_query_expansion_output(output, allowed_languages={"en", "de"})

    assert parsed == [
        {"language": "en", "query": "contract formation requirements"},
        {"language": "de", "query": "Vertrag Abschluss Voraussetzungen"},
    ]


def test_parse_query_expansion_output_accepts_json_array():
    output = """[
      {"language": "en", "query": "contract formation"},
      {"language": "de", "query": "Vertrag Abschluss"}
    ]"""

    parsed = parse_query_expansion_output(output, allowed_languages={"en", "de"})

    assert parsed == [
        {"language": "en", "query": "contract formation"},
        {"language": "de", "query": "Vertrag Abschluss"},
    ]


def test_multilingual_query_expander_falls_back_to_original_on_malformed_json():
    expander = MultilingualQueryExpander(
        llm=FakeLlm("not valid json"),
        target_languages=["en", "de"],
        max_queries=4,
    )

    expanded = expander.expand("  valid   contract  ")

    assert expanded == [ExpandedQuery(language="en", query="valid contract")]


def test_multilingual_query_expander_respects_budget_and_language_order():
    response = """{
      "queries": [
        {"language": "en", "query": "contract formation requirements"},
        {"language": "de", "query": "Vertrag Abschluss Voraussetzungen"},
        {"language": "de", "query": "Vertragsabschluss OR"},
        {"language": "de", "query": "Zusatzquery"},
        {"language": "en", "query": "What are the requirements for a valid contract under Swiss law?"}
      ]
    }"""
    expander = MultilingualQueryExpander(
        llm=FakeLlm(response),
        target_languages=["en", "de"],
        max_queries=4,
    )

    expanded = expander.expand("What are the requirements for a valid contract under Swiss law?")

    assert expanded == [
        ExpandedQuery(
            language="en",
            query="What are the requirements for a valid contract under Swiss law?",
        ),
        ExpandedQuery(language="en", query="contract formation requirements"),
        ExpandedQuery(language="de", query="Vertrag Abschluss Voraussetzungen"),
        ExpandedQuery(language="de", query="Vertragsabschluss OR"),
    ]


def test_law_search_tool_merges_results_from_expanded_queries():
    documents = [
        {
            "id": "law_contract",
            "citation": "Art. 1 OR",
            "text": "contract formation requirements Vertrag Abschluss Voraussetzungen OR",
        },
        {
            "id": "law_lease",
            "citation": "Art. 266a OR",
            "text": "lease termination tenancy Kündigung Vermieter Miete OR",
        },
    ]
    index = BM25Index(documents=documents)
    expander = StubExpander(
        [
            ExpandedQuery(language="en", query="contract formation requirements"),
            ExpandedQuery(language="de", query="Vertrag Abschluss Voraussetzungen"),
            ExpandedQuery(language="de", query="Vertrag"),
        ]
    )
    tool = LawSearchTool(index=index, top_k=5, query_expander=expander)

    results = tool.search_with_metadata("What are the requirements for a valid contract?")

    assert results[0]["citation"] == "Art. 1 OR"
    assert results[0]["matched_queries"] == [
        "contract formation requirements",
        "Vertrag Abschluss Voraussetzungen",
        "Vertrag",
    ]
    assert results[0]["matched_languages"] == ["en", "de"]

    expected_best_score = max(
        index.search("contract formation requirements", top_k=5, return_scores=True)[0]["_score"],
        index.search("Vertrag Abschluss Voraussetzungen", top_k=5, return_scores=True)[0]["_score"],
        index.search("Vertrag", top_k=5, return_scores=True)[0]["_score"],
    )
    assert results[0]["_score"] == pytest.approx(expected_best_score)


def test_law_search_tool_without_expander_matches_single_query_behavior(sample_laws_corpus):
    index = BM25Index(documents=sample_laws_corpus)
    tool = LawSearchTool(index=index, top_k=2, max_excerpt_length=300)

    actual = tool.run("Vertrag")
    expected_results = index.search("Vertrag", top_k=2)
    expected = "\n".join(
        f"- {doc['citation']}: {doc['text'][:300] + '...' if len(doc['text']) > 300 else doc['text']}"
        for doc in expected_results
    )

    assert actual == expected
