"""Knowledge-base search tool.

The mock uses BM25 keyword ranking over a JSON file of FAQ articles: no extra
dependencies, deterministic, easy to test. In production `KnowledgeBase` would be
backed by a vector or hybrid search over the real help center; the tool and the
agent would not change.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from triage_agent.datasources import load_json
from triage_agent.tools.base import Tool

KBCategory = Literal["billing", "plans", "technical", "enterprise", "product", "account"]

DEFAULT_TOP_K = 3
MAX_TOP_K = 5


class Article(BaseModel):
    """One help-center or internal FAQ article."""

    id: str
    title: str
    category: KBCategory
    tags: list[str] = Field(default_factory=list)
    content: str


class SearchHit(BaseModel):
    """An article matched by a search, with its relevance score (higher is better)."""

    article: Article
    score: float


class KnowledgeBase(Protocol):
    """Anything that can search articles. `BM25KnowledgeBase` today; vector search in production."""

    def search(self, query: str, *, category: str | None = None, top_k: int = DEFAULT_TOP_K) -> list[SearchHit]:
        """Find the articles most relevant to a query.

        Args:
            query: Free-text query, in English.
            category: Only search this category, or every category if None.
            top_k: Maximum number of results.

        Returns:
            Relevant articles, best first. Empty if nothing matches.
        """
        ...


# --------------------------------------------------------------------------- #
# BM25 implementation                                                          #
# --------------------------------------------------------------------------- #
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it my of on or "
    "so that the this to was what when why with you your".split()
)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words and drop common English stopwords.

    Thai has no spaces between words, so a Thai sentence becomes one token; the model is
    told to search in English for that reason.

    Args:
        text: Text to tokenize.

    Returns:
        The remaining words, in order.
    """
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


class BM25KnowledgeBase:
    """Keyword search with Okapi BM25 over title, tags and content.

    Title and tags count twice, because they are written to describe the article.
    The index is built once in the constructor; searches only score.

    Args:
        articles: Articles to index.
        k1: How quickly repeated terms stop adding score.
        b: How much long articles are penalized (0 = not at all, 1 = fully).
    """

    def __init__(self, articles: list[Article], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._articles = articles
        self._k1, self._b = k1, b
        self._doc_terms = [Counter(self._index_tokens(a)) for a in articles]
        self._doc_lens = [sum(terms.values()) for terms in self._doc_terms]
        self._avg_len = (sum(self._doc_lens) / len(self._doc_lens)) if articles else 0.0
        doc_freq = Counter(term for terms in self._doc_terms for term in terms)
        n = len(articles)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    @classmethod
    def from_json(cls, path: Path) -> BM25KnowledgeBase:
        """Build the index from a JSON list of articles.

        Args:
            path: The JSON file.

        Returns:
            A ready-to-search knowledge base.
        """
        return cls([Article.model_validate(item) for item in load_json(path)])

    def search(self, query: str, *, category: str | None = None, top_k: int = DEFAULT_TOP_K) -> list[SearchHit]:
        """Return up to `top_k` articles with a score above zero, best first. See `KnowledgeBase.search`."""
        query_terms = tokenize(query)
        hits = [
            SearchHit(article=article, score=round(self._score(query_terms, i), 3))
            for i, article in enumerate(self._articles)
            if category is None or article.category == category
        ]
        relevant = [hit for hit in hits if hit.score > 0]
        return sorted(relevant, key=lambda h: h.score, reverse=True)[:top_k]

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        """BM25 score of one article for the query terms (0 when no term matches)."""
        terms, doc_len = self._doc_terms[doc_index], self._doc_lens[doc_index]
        score = 0.0
        for term in query_terms:
            tf = terms.get(term, 0)
            if tf == 0:
                continue
            norm = tf + self._k1 * (1 - self._b + self._b * doc_len / self._avg_len)
            score += self._idf[term] * tf * (self._k1 + 1) / norm
        return score

    @staticmethod
    def _index_tokens(article: Article) -> list[str]:
        boosted = f"{article.title} {' '.join(article.tags)}"
        return tokenize(boosted) * 2 + tokenize(article.content)


# --------------------------------------------------------------------------- #
# Tool                                                                         #
# --------------------------------------------------------------------------- #
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
