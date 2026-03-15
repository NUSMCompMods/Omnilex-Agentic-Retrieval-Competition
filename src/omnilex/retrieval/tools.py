"""LLM-compatible search tools for agentic retrieval."""

from __future__ import annotations

from .bm25_index import BM25Index
from .query_expansion import ExpandedQuery, MultilingualQueryExpander


class _BaseSearchTool:
    """Shared search behavior for agent-facing retrieval tools."""

    name: str = ""
    description: str = ""
    no_results_template: str = "No relevant results found for: '{query}'"

    def __init__(
        self,
        index: BM25Index,
        top_k: int = 5,
        max_excerpt_length: int = 300,
        query_expander: MultilingualQueryExpander | None = None,
    ):
        self.index = index
        self.top_k = top_k
        self.max_excerpt_length = max_excerpt_length
        self.query_expander = query_expander
        self._last_results: list[dict] = []

    def __call__(self, query: str) -> str:
        return self.run(query)

    def run(self, query: str) -> str:
        """Execute search and return formatted results."""
        if not query or not query.strip():
            self._last_results = []
            return "Error: Empty query. Please provide search terms."

        results = self._search(query, include_metadata=False)
        self._last_results = results

        if not results:
            return self.no_results_template.format(query=query)

        formatted = []
        for doc in results:
            citation = doc.get("citation", "Unknown")
            text = doc.get("text", "")
            if len(text) > self.max_excerpt_length:
                text = text[: self.max_excerpt_length] + "..."
            formatted.append(f"- {citation}: {text}")

        return "\n".join(formatted)

    def get_last_citations(self) -> list[str]:
        """Return citations from the last search."""
        return [doc.get("citation", "") for doc in self._last_results if doc.get("citation")]

    def search_with_metadata(self, query: str) -> list[dict]:
        """Execute search and return ranked result objects with metadata."""
        if not query or not query.strip():
            return []
        return self._search(query, include_metadata=True)

    def _search(self, query: str, include_metadata: bool) -> list[dict]:
        if self.query_expander is None:
            results = self.index.search(query, top_k=self.top_k, return_scores=include_metadata)
            if include_metadata:
                for result in results:
                    result.setdefault("matched_queries", [query])
                    result.setdefault("matched_languages", [])
            return results

        expanded_queries = self.query_expander.expand(query)
        if not expanded_queries:
            return []

        merged_results: dict[str, dict] = {}
        for expanded_query in expanded_queries:
            matches = self.index.search(expanded_query.query, top_k=self.top_k, return_scores=True)
            for match in matches:
                self._merge_result(
                    merged_results,
                    match,
                    expanded_query,
                )

        ranked_results = sorted(
            merged_results.values(),
            key=lambda item: (item["_score"], item.get("citation", ""), item.get("id", "")),
            reverse=True,
        )[: self.top_k]

        if not include_metadata:
            for result in ranked_results:
                result.pop("_score", None)
                result.pop("matched_queries", None)
                result.pop("matched_languages", None)

        return ranked_results

    def _merge_result(
        self,
        merged_results: dict[str, dict],
        match: dict,
        expanded_query: ExpandedQuery,
    ) -> None:
        key = self._result_key(match)
        score = float(match.get("_score", 0.0))
        match_without_score = match.copy()
        match_without_score.pop("_score", None)

        existing = merged_results.get(key)
        if existing is None:
            merged_results[key] = {
                **match_without_score,
                "_score": score,
                "matched_queries": [expanded_query.query],
                "matched_languages": [expanded_query.language],
            }
            return

        if score > existing["_score"]:
            preserved_queries = existing["matched_queries"]
            preserved_languages = existing["matched_languages"]
            merged_results[key] = {
                **match_without_score,
                "_score": score,
                "matched_queries": preserved_queries,
                "matched_languages": preserved_languages,
            }
            existing = merged_results[key]

        if expanded_query.query not in existing["matched_queries"]:
            existing["matched_queries"].append(expanded_query.query)
        if expanded_query.language not in existing["matched_languages"]:
            existing["matched_languages"].append(expanded_query.language)

    def _result_key(self, match: dict) -> str:
        return str(match.get("citation") or match.get("id") or match.get("text") or repr(match))


class LawSearchTool(_BaseSearchTool):
    """Tool for searching Swiss federal laws corpus."""

    name: str = "search_laws"
    description: str = """Search Swiss federal laws (SR/Systematische Rechtssammlung) by keywords.
Input: Search query string (can be in German, French, Italian, or English)
Output: List of relevant law citations with text excerpts

Use this tool to find relevant federal law provisions for a legal question.
Example queries: "contract formation requirements", "Vertragsabschluss", "divorce grounds"
"""
    no_results_template: str = "No relevant federal laws found for: '{query}'"


class CourtSearchTool(_BaseSearchTool):
    """Tool for searching Swiss Federal Court decisions corpus."""

    name: str = "search_courts"
    description: str = """Search Swiss Federal Court decisions by keywords.
Input: Search query string (German, French, Italian, or English)
Output: List of relevant court decision citations with excerpts

Use this tool to find relevant case law and judicial interpretations.
Example queries: "negligence standard of care", "Sorgfaltspflicht", "contract interpretation"
"""
    no_results_template: str = "No relevant court decisions found for: '{query}'"


class CombinedSearchTool:
    """Tool that searches both laws and court decisions."""

    name: str = "search_all"
    description: str = """Search both Swiss federal laws (SR) and Federal Court decisions (BGE).
Input: Search query string
Output: Combined list of relevant citations from both corpora

Use this for comprehensive research when you need both statutory law and case law.
"""

    def __init__(
        self,
        law_index: BM25Index,
        court_index: BM25Index,
        top_k_each: int = 3,
        max_excerpt_length: int = 250,
        query_expander: MultilingualQueryExpander | None = None,
    ):
        self.law_tool = LawSearchTool(
            law_index,
            top_k=top_k_each,
            max_excerpt_length=max_excerpt_length,
            query_expander=query_expander,
        )
        self.court_tool = CourtSearchTool(
            court_index,
            top_k=top_k_each,
            max_excerpt_length=max_excerpt_length,
            query_expander=query_expander,
        )

    def __call__(self, query: str) -> str:
        return self.run(query)

    def run(self, query: str) -> str:
        law_results = self.law_tool.run(query)
        court_results = self.court_tool.run(query)

        output = [
            "=== Federal Laws (SR) ===",
            law_results,
            "",
            "=== Court Decisions ===",
            court_results,
        ]

        return "\n".join(output)


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all available tools."""
    tools = [
        ("search_laws", LawSearchTool.description),
        ("search_courts", CourtSearchTool.description),
    ]

    lines = []
    for i, (name, desc) in enumerate(tools, 1):
        lines.append(f"{i}. {name}: {desc.strip()}")

    return "\n\n".join(lines)
