"""LLM-compatible search tools for agentic retrieval."""

from __future__ import annotations

from typing import Protocol


class SearchIndex(Protocol):
    """Protocol for searchable retrieval indices."""

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict]:
        """Return the most relevant documents for a query."""


class _BaseSearchTool:
    """Shared formatting logic for tool wrappers around retrieval indices."""

    empty_query_error = "Error: Empty query. Please provide search terms."
    no_results_template = "No relevant results found for: '{query}'"

    def __init__(
        self,
        index: SearchIndex,
        *,
        top_k: int = 5,
        max_excerpt_length: int = 300,
    ) -> None:
        self.index = index
        self.top_k = top_k
        self.max_excerpt_length = max_excerpt_length
        self._last_results: list[dict] = []

    def __call__(self, query: str) -> str:
        return self.run(query)

    def run(self, query: str) -> str:
        """Execute search and return formatted results."""
        if not query or not query.strip():
            self._last_results = []
            return self.empty_query_error

        results = self.index.search(query, top_k=self.top_k)
        self._last_results = results

        if not results:
            return self.no_results_template.format(query=query)

        formatted = []
        for document in results:
            citation = document.get("citation", "Unknown")
            text = document.get("text", "")
            if len(text) > self.max_excerpt_length:
                text = text[: self.max_excerpt_length] + "..."
            formatted.append(f"- {citation}: {text}")

        return "\n".join(formatted)

    def get_last_citations(self) -> list[str]:
        """Return citations from the most recent search."""
        return [
            document.get("citation", "")
            for document in self._last_results
            if document.get("citation")
        ]

    def search_with_metadata(self, query: str) -> list[dict]:
        """Execute search and return full result objects."""
        return self.index.search(query, top_k=self.top_k, return_scores=True)


class LawSearchTool(_BaseSearchTool):
    """Tool for searching Swiss federal laws corpus."""

    name: str = "search_laws"
    description: str = """Search Swiss federal laws (SR/Systematische Rechtssammlung) semantically.
Input: Search query string (can be in German, French, Italian, or English)
Output: List of relevant law citations with text excerpts

Use this tool to find relevant federal law provisions for a legal question.
Example queries: "contract formation requirements", "Vertragsabschluss", "divorce grounds"
"""
    no_results_template = "No relevant federal laws found for: '{query}'"


class CourtSearchTool(_BaseSearchTool):
    """Tool for searching Swiss Federal Court decisions corpus."""

    name: str = "search_courts"
    description: str = """Search Swiss Federal Court decisions semantically.
Input: Search query string (German, French, Italian, or English)
Output: List of relevant court decision citations with excerpts

Use this tool to find relevant case law and judicial interpretations.
Example queries: "negligence standard of care", "Sorgfaltspflicht", "contract interpretation"
"""
    no_results_template = "No relevant court decisions found for: '{query}'"


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
        law_index: SearchIndex,
        court_index: SearchIndex,
        top_k_each: int = 3,
        max_excerpt_length: int = 250,
    ) -> None:
        self.law_tool = LawSearchTool(
            law_index,
            top_k=top_k_each,
            max_excerpt_length=max_excerpt_length,
        )
        self.court_tool = CourtSearchTool(
            court_index,
            top_k=top_k_each,
            max_excerpt_length=max_excerpt_length,
        )

    def __call__(self, query: str) -> str:
        return self.run(query)

    def run(self, query: str) -> str:
        """Execute search on both corpora."""
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
    for index, (name, description) in enumerate(tools, 1):
        lines.append(f"{index}. {name}: {description.strip()}")

    return "\n\n".join(lines)
