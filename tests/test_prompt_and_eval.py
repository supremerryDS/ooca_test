from __future__ import annotations

from fakes import DATA_DIR, decision_payload

from triage_agent.evaluation import load_labels, score_result, summarize
from triage_agent.prompts import load_system_prompt, render_ticket
from triage_agent.schemas import ToolCallRecord, TriageDecision, TriageResult


def test_system_prompt_mentions_every_tool(registry):
    prompt = load_system_prompt()
    for name in [*registry.names, "submit_triage"]:
        assert name in prompt


def test_rendered_ticket_fences_customer_text_and_shows_timing(tickets):
    text = render_ticket(tickets["T-1001"])
    assert "<ticket>" in text and "</ticket>" in text
    assert "customer_id: cus_1001" in text
    assert "(3h before latest message)" in text
    assert "THREE charges of $29.99" in text


def test_thai_ticket_is_rendered_verbatim(tickets):
    assert "ขึ้น error 500" in render_ticket(tickets["T-1002"])


def _result(ticket_id: str, tools: list[str], **decision) -> TriageResult:
    return TriageResult(
        ticket_id=ticket_id,
        status="completed",
        decision=TriageDecision.model_validate(decision_payload(**decision)),
        tool_calls=[ToolCallRecord(tool_name=t, arguments={}, ok=True) for t in tools],
        model="fake",
    )


def test_scoring_accepts_any_acceptable_label():
    labels = load_labels(DATA_DIR.parent / "evals" / "golden_labels.json")
    good = _result(
        "T-1001",
        ["get_customer_history", "get_billing_transactions", "search_knowledge_base"],
        urgency="critical",
        action="escalate_to_human",
        route_to="billing",
    )
    bad = _result(
        "T-1003", ["get_customer_history"], urgency="high", action="escalate_to_human", route_to="engineering"
    )

    scores = [score_result(good, labels["T-1001"]), score_result(bad, labels["T-1003"])]
    assert scores[0].all_ok
    assert not scores[1].urgency_ok and not scores[1].action_ok
    assert scores[1].team_ok  # T-1003 accepts any team
    assert scores[1].missing_tools == ("search_knowledge_base",)
    assert summarize(scores)["fully_correct"] == 0.5
