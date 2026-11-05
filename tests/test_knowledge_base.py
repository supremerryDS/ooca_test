from __future__ import annotations

import pytest
from fakes import DATA_DIR

from triage_agent.search import BM25KnowledgeBase


@pytest.fixture(scope="module")
def kb() -> BM25KnowledgeBase:
    return BM25KnowledgeBase.from_json(DATA_DIR / "knowledge_base.json")


@pytest.mark.parametrize(
    ("query", "expected_top"),
    [
        ("duplicate pending charges refund", "KB-BILL-002"),
        ("plan still free after payment pro access", "KB-BILL-003"),
        ("error 500 cannot access multiple browsers", "KB-TECH-001"),
        ("status page says all systems operational region asia", "KB-TECH-002"),
        ("system default theme ignores macos dark appearance", "KB-APP-002"),
        ("schedule dark mode", "KB-APP-001"),
    ],
)
def test_search_ranks_the_right_article_first(kb, query, expected_top):
    hits = kb.search(query)
    assert hits, f"no results for {query!r}"
    assert hits[0].article.id == expected_top


def test_category_filter(kb):
    hits = kb.search("refund", category="plans")
    assert all(hit.article.category == "plans" for hit in hits)


def test_irrelevant_query_returns_nothing(kb):
    assert kb.search("zzzz qqqq") == []


def test_top_k_is_respected(kb):
    assert len(kb.search("charge refund payment plan", top_k=2)) == 2
