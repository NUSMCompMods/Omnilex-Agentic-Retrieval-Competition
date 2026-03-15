"""Prompt templates for legal citation retrieval.

Contains prompts for:
1. Direct generation baseline - prompting LLM to generate citations directly
2. Agentic retrieval baseline - ReAct-style agent with search tools
"""

import json
import re

# =============================================================================
# DIRECT GENERATION PROMPTS
# =============================================================================

DIRECT_GENERATION_PROMPT = """\
You are a Swiss legal expert. Given a legal query, identify and list ALL \
relevant Swiss legal citations.

Citation formats to use:
- Federal laws: Art. [article] [Abs. paragraph] [LAW]
  (e.g., Art. 1 ZGB, Art. 11 Abs. 2 OR, Art. 117 StGB)
- Court decisions: BGE [volume] [section] [page] E. [consideration]
  (e.g., BGE 116 Ia 56 E. 2b) or docket-style (e.g., 5A_800/2019 E. 2)

Common law abbreviations: ZGB (Civil Code), OR (Code of Obligations), \
StGB (Criminal Code), BV (Constitution)

Important:
- Only list citations that are directly relevant to the query
- Include both statutory law and case law (BGE) when applicable
- Be precise with citation formats
- Do not include explanations, only the citations

Query: {query}

List only the citations, one per line:"""


DIRECT_GENERATION_PROMPT_DE = """\
Du bist ein Schweizer Rechtsexperte. Für die gegebene rechtliche Frage, \
identifiziere und liste ALLE relevanten Schweizer Rechtszitate auf.

Zitierformate:
- Bundesgesetze: Art. [Artikel] [Abs. Absatz] [GESETZ]
  (z.B. Art. 1 ZGB, Art. 11 Abs. 2 OR, Art. 117 StGB)
- Bundesgerichtsentscheide: BGE [Band] [Abteilung] [Seite] E. [Erwägung]
  (z.B. BGE 116 Ia 56 E. 2b) oder Dossiernummer (z.B. 5A_800/2019 E. 2)

Übliche Gesetzesabkürzungen: ZGB (Zivilgesetzbuch), OR (Obligationenrecht), \
StGB (Strafgesetzbuch), BV (Bundesverfassung)

Wichtig:
- Nur direkt relevante Zitate auflisten
- Sowohl Gesetzesrecht als auch Rechtsprechung (BGE) einbeziehen
- Präzise Zitierformate verwenden
- Keine Erklärungen, nur die Zitate

Frage: {query}

Liste nur die Zitate auf, eines pro Zeile:"""


# =============================================================================
# AGENTIC RETRIEVAL PROMPTS
# =============================================================================

AGENT_SYSTEM_PROMPT = """\
You are a Swiss legal research assistant with access to two search tools:

1. search_laws(query): Search Swiss federal laws by keywords
   - Returns relevant law provisions with citations and text excerpts
   - Use for finding statutory law: codes, acts, ordinances

2. search_courts(query): Search Swiss Federal Court decisions by keywords
   - Returns relevant case law with citations and excerpts
   - Use for finding judicial interpretations and precedents

Your task is to find ALL relevant Swiss legal citations for the given query.

Citation formats:
- Federal laws: Art. X [Abs. Y] LAW (e.g., Art. 1 ZGB, Art. 11 Abs. 2 OR)
- Court decisions: BGE XXX YY ZZZ E. [consideration] or docket-style (e.g., 5A_800/2019 E. 2)

Instructions:
- Search BOTH laws AND court decisions for comprehensive results
- The search tools already expand each query across English and German
- Use follow-up search queries for different legal concepts or sub-issues
- Extract citations in standard format
- Continue searching until you have found all relevant sources

Format your response as:
Thought: [Your reasoning about what to search next]
Action: [tool_name]
Action Input: [search query]

After receiving results, either continue searching or provide final answer:
Final Answer: [List of all found citations, one per line]

Remember: Always search both laws AND court decisions before giving your final answer."""


AGENT_REACT_TEMPLATE = """\
You are a Swiss legal research assistant. Given a legal query, \
use the available tools to find relevant citations.

Available tools:
{tools}

Use this format:

Question: the legal query you must research
Thought: consider what to search for
Action: the tool to use (search_laws or search_courts)
Action Input: your search query
Observation: the search results
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I have gathered enough citations
Final Answer: list of citations, one per line

Begin!

Question: {query}
Thought:"""


# =============================================================================
# CITATION EXTRACTION PROMPTS
# =============================================================================

CITATION_EXTRACTION_PROMPT = """\
Extract all Swiss legal citations from the following text.

Citation formats to identify:
- Federal law citations: Art. X [Abs. Y] LAW
  (e.g., Art. 1 ZGB, Art. 41 OR, Art. 11 Abs. 2 OR)
- Court citations: BGE or docket-style followed by E. consideration
  (e.g., BGE 116 Ia 56 E. 2b, 5A_800/2019 E. 2)

Common law abbreviations: ZGB, OR, StGB, BV, SchKG, BGG

Text:
{text}

List each citation on a separate line. Only output the citations, nothing else:"""


# =============================================================================
# QUERY EXPANSION PROMPTS
# =============================================================================

QUERY_EXPANSION_PROMPT = """\
You expand Swiss legal retrieval queries for BM25 keyword search.

Return valid JSON only with this exact schema:
{{
  "queries": [
    {{"language": "en", "query": "keyword style rewrite"}},
    {{"language": "de", "query": "keyword style rewrite"}}
  ]
}}

Rules:
- Use only these languages and limits:
{language_instructions}
- Keep queries short, concrete, and keyword-focused.
- Preserve Swiss legal abbreviations exactly when present, such as OR, ZGB, StGB, BV.
- Prefer legal concepts and synonyms over full natural-language sentences.
- Do not add explanations, markdown, citations, or extra keys.
- Do not repeat the original query verbatim unless it is genuinely the best rewrite.

Original query:
{query}
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def format_direct_generation_prompt(query: str, language: str = "en") -> str:
    """Format direct generation prompt with query.

    Args:
        query: Legal query text
        language: Language code ("en" or "de")

    Returns:
        Formatted prompt string
    """
    if language == "de":
        return DIRECT_GENERATION_PROMPT_DE.format(query=query)
    return DIRECT_GENERATION_PROMPT.format(query=query)


def format_query_expansion_prompt(query: str, language_counts: dict[str, int]) -> str:
    """Format the multilingual query expansion prompt.

    Args:
        query: Original user query
        language_counts: Mapping of language code to requested rewrite count

    Returns:
        Prompt asking the LLM for strict JSON query rewrites
    """
    language_lines = [
        f'- "{language}": up to {count} rewrite(s)'
        for language, count in language_counts.items()
        if count > 0
    ]
    language_instructions = "\n".join(language_lines) or '- "en": up to 0 rewrite(s)'
    return QUERY_EXPANSION_PROMPT.format(
        query=query,
        language_instructions=language_instructions,
    )


def format_agent_prompt(query: str, tools_description: str = "") -> str:
    """Format agent prompt with query and tool descriptions.

    Args:
        query: Legal query text
        tools_description: Description of available tools

    Returns:
        Formatted prompt string
    """
    if tools_description:
        return AGENT_REACT_TEMPLATE.format(tools=tools_description, query=query)
    return AGENT_SYSTEM_PROMPT + f"\n\nQuery: {query}\n\nThought:"


def parse_citations_from_output(output: str) -> list[str]:
    """Parse citations from LLM output.

    Args:
        output: Raw LLM output text

    Returns:
        List of citation strings
    """
    citations = []

    for line in output.split("\n"):
        line = line.strip()

        # Skip empty lines and common non-citation prefixes
        if not line:
            continue
        if line.lower().startswith(("thought:", "action:", "observation:", "final answer:")):
            continue

        # Clean up common list markers
        line = line.lstrip("-•*0123456789.) ")

        # Check for law citations (Art., SR) or court citations (BGE, docket-style)
        if "SR" in line or "BGE" in line or "Art." in line or re.search(r"\d+[A-Z]_", line):
            citations.append(line)

    return citations


def parse_query_expansion_output(
    output: str,
    allowed_languages: set[str] | None = None,
) -> list[dict[str, str]]:
    """Parse strict-JSON query expansion output.

    Args:
        output: Raw model text containing JSON
        allowed_languages: Optional filter for allowed language codes

    Returns:
        Ordered list of {"language": ..., "query": ...} objects

    Raises:
        ValueError: If no JSON object/array can be parsed
    """
    json_text = _extract_json_block(output)
    payload = json.loads(json_text)

    if isinstance(payload, dict):
        entries = payload.get("queries", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError("Expected a JSON object or array for query expansion output")

    normalized_entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        raw_language = entry.get("language")
        raw_query = entry.get("query")
        if raw_language is None or raw_query is None:
            continue

        language = str(raw_language).strip().lower()
        query = _normalize_query_text(str(raw_query))

        if not language or not query:
            continue
        if allowed_languages and language not in allowed_languages:
            continue

        dedupe_key = (language, query.casefold())
        if dedupe_key in seen:
            continue

        normalized_entries.append({"language": language, "query": query})
        seen.add(dedupe_key)

    return normalized_entries


def parse_agent_action(response: str) -> tuple[str, str] | None:
    """Parse action and input from agent response.

    Args:
        response: Agent response text

    Returns:
        Tuple of (action_name, action_input) or None if no action found
    """
    import re

    action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)

    if action_match and input_match:
        return action_match.group(1).strip(), input_match.group(1).strip()

    return None


def extract_final_answer(response: str) -> str | None:
    """Extract final answer from agent response.

    Args:
        response: Agent response text

    Returns:
        Final answer text or None if not found
    """
    import re

    match = re.search(r"Final Answer:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _extract_json_block(output: str) -> str:
    """Extract the first JSON object or array from model output."""
    cleaned = output.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    start_positions = [(cleaned.find("{"), "{", "}"), (cleaned.find("["), "[", "]")]
    start_positions = [item for item in start_positions if item[0] != -1]

    if start_positions:
        start, opener, closer = min(start_positions, key=lambda item: item[0])
        end = cleaned.rfind(closer)
        if end != -1 and end > start:
            return cleaned[start : end + 1]

    raise ValueError("Could not find JSON object or array in query expansion output")


def _normalize_query_text(text: str) -> str:
    """Normalize whitespace in generated query text."""
    return " ".join(text.split())
