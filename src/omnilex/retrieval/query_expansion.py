"""Multilingual query expansion helpers for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omnilex.llm.loader import generate
from omnilex.llm.prompts import format_query_expansion_prompt, parse_query_expansion_output


@dataclass(frozen=True)
class ExpandedQuery:
    """Expanded query paired with a language code."""

    language: str
    query: str


class MultilingualQueryExpander:
    """Generate a small multilingual set of BM25-friendly query rewrites."""

    def __init__(
        self,
        llm: Any,
        target_languages: list[str] | tuple[str, ...] = ("en", "de"),
        max_queries: int = 4,
        temperature: float = 0.1,
        max_tokens: int = 256,
        source_language: str = "en",
    ):
        """Initialize the query expander.

        Args:
            llm: LLM instance compatible with omnilex.llm.loader.generate
            target_languages: Languages to generate rewrites for
            max_queries: Maximum number of total queries, including the original
            temperature: Sampling temperature for rewrite generation
            max_tokens: Maximum completion tokens for rewrite generation
            source_language: Default source language for incoming queries
        """
        self.llm = llm
        self.source_language = source_language.strip().lower() or "en"
        self.target_languages = self._normalize_languages(target_languages)
        self.max_queries = max(1, max_queries)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._cache: dict[str, list[ExpandedQuery]] = {}

    def expand(self, query: str) -> list[ExpandedQuery]:
        """Expand a query into a small multilingual variant set."""
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        cached = self._cache.get(normalized_query)
        if cached is not None:
            return list(cached)

        language_limits = self._build_language_limits()
        expanded: list[ExpandedQuery] = []
        seen_queries: set[str] = set()

        seed_language = self.source_language if self.source_language in language_limits else None
        if seed_language is not None and language_limits[seed_language] > 0:
            expanded.append(ExpandedQuery(language=seed_language, query=normalized_query))
            seen_queries.add(normalized_query.casefold())
            language_limits[seed_language] -= 1

        remaining_slots = self.max_queries - len(expanded)
        if remaining_slots <= 0 or not any(limit > 0 for limit in language_limits.values()):
            self._cache[normalized_query] = list(expanded)
            return list(expanded)

        prompt = format_query_expansion_prompt(
            query=normalized_query,
            language_counts=language_limits,
        )

        try:
            response = generate(
                self.llm,
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["```"],
            )
            parsed = parse_query_expansion_output(
                response,
                allowed_languages=set(self.target_languages),
            )
        except Exception:
            self._cache[normalized_query] = list(expanded)
            return list(expanded)

        for item in parsed:
            if len(expanded) >= self.max_queries:
                break

            language = item["language"]
            if language_limits.get(language, 0) <= 0:
                continue

            candidate_query = self._normalize_query(item["query"])
            if not candidate_query:
                continue

            dedupe_key = candidate_query.casefold()
            if dedupe_key in seen_queries:
                continue

            expanded.append(ExpandedQuery(language=language, query=candidate_query))
            seen_queries.add(dedupe_key)
            language_limits[language] -= 1

        if not expanded:
            expanded = [ExpandedQuery(language=self.source_language, query=normalized_query)]

        self._cache[normalized_query] = list(expanded)
        return list(expanded)

    def _build_language_limits(self) -> dict[str, int]:
        """Allocate the fixed query budget across languages."""
        language_count = len(self.target_languages)
        if language_count == 0:
            return {self.source_language: self.max_queries}

        base = self.max_queries // language_count
        remainder = self.max_queries % language_count

        prioritized_languages = list(self.target_languages)
        if self.source_language in prioritized_languages:
            prioritized_languages.remove(self.source_language)
            prioritized_languages.insert(0, self.source_language)

        limits = {language: base for language in self.target_languages}
        for language in prioritized_languages[:remainder]:
            limits[language] += 1

        return limits

    def _normalize_languages(
        self,
        target_languages: list[str] | tuple[str, ...],
    ) -> list[str]:
        """Normalize and deduplicate the target language list."""
        normalized_languages: list[str] = []
        seen_languages: set[str] = set()

        for language in [self.source_language, *target_languages]:
            normalized = language.strip().lower()
            if not normalized or normalized in seen_languages:
                continue
            normalized_languages.append(normalized)
            seen_languages.add(normalized)

        return normalized_languages or [self.source_language]

    def _normalize_query(self, query: str) -> str:
        """Normalize whitespace in a query string."""
        return " ".join(query.split())
