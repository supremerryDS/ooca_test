"""Knowledge-base search tool: FAQ articles relevant to the customer's issue."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from triage_agent.search import DEFAULT_TOP_K, KBCategory, KnowledgeBase
from triage_agent.tools.base import Tool

MAX_TOP_K = 5


class SearchKnowledgeBaseArgs(BaseModel):
    """Arguments for search_knowledge_base."""

    query: str = Field(
        description="Search query in ENGLISH (translate non-English tickets first). Use specific "
        "keywords, e.g. 'duplicate pending charges refund' or 'system default theme macOS'."
    )
    category: KBCategory | None = Field(description="Optional category filter. Use null to search everything.")
    top_k: int | None = Field(description=f"Max results, 1-{MAX_TOP_K}. Use null for the default ({DEFAULT_TOP_K}).")


class SearchKnowledgeBaseTool(Tool):
    """FAQ search for the model. Returns full article content so replies can quote it.

    Args:
        knowledge_base: The articles to search.
    """

    name = "search_knowledge_base"
    description = (
        "Search the help-center / internal FAQ for articles relevant to the customer's issue. "
        "Returns article IDs, titles and full content. Call once per distinct issue in the ticket."
    )
    args_model = SearchKnowledgeBaseArgs

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._kb = knowledge_base

    def run(self, args: SearchKnowledgeBaseArgs) -> dict[str, Any]:
        top_k = min(max(args.top_k or DEFAULT_TOP_K, 1), MAX_TOP_K)
        hits = self._kb.search(args.query, category=args.category, top_k=top_k)
        return {
            "query": args.query,
            "results": [
                {
                    "id": hit.article.id,
                    "title": hit.article.title,
                    "category": hit.article.category,
                    "relevance_score": hit.score,
                    "content": hit.article.content,
                }
                for hit in hits
            ],
            "note": None if hits else "No relevant articles found. Do not invent an answer.",
        }
