from __future__ import annotations

from fakes import decision_payload

from triage_agent.policy import PolicyContext, PolicyEngine, default_rules
from triage_agent.schemas import Action, Team, ToolCallRecord, TriageDecision, Urgency

ENGINE = PolicyEngine(default_rules(confidence_threshold=0.6))
KB_SEEN = PolicyContext(retrieved_kb_ids=frozenset({"KB-APP-001", "KB-APP-002"}))
ENTERPRISE = PolicyContext(retrieved_kb_ids=KB_SEEN.retrieved_kb_ids, customer_plan="enterprise")


def decide(**overrides) -> TriageDecision:
    return TriageDecision.model_validate(decision_payload(**overrides))


def with_flags(*flags: str) -> dict:
    return {**decision_payload()["extracted"], "risk_flags": list(flags)}


def test_clean_auto_response_is_untouched():
    final, overrides = ENGINE.apply(decide(), KB_SEEN)
    assert overrides == []
    assert final.action is Action.AUTO_RESPOND


def test_critical_is_never_auto_answered():
    final, overrides = ENGINE.apply(decide(urgency="critical"), KB_SEEN)
    assert final.action is Action.ESCALATE_TO_HUMAN
    assert final.route_to is Team.GENERAL_SUPPORT
    assert overrides[0].rule == "critical_requires_human"


def test_chargeback_threat_escalates_and_raises_urgency_floor():
    decision = decide(
        urgency="medium", action="route_to_specialist", route_to="billing", extracted=with_flags("chargeback_threat")
    )
    final, overrides = ENGINE.apply(decision, KB_SEEN)
    assert final.action is Action.ESCALATE_TO_HUMAN
    assert final.urgency is Urgency.HIGH
    assert final.route_to is Team.BILLING  # keeps the team the model chose
    assert overrides[0].changes["urgency"] == ["medium", "high"]


def test_safety_concern_goes_to_care_team():
    final, _ = ENGINE.apply(decide(extracted=with_flags("safety_concern")), KB_SEEN)
    assert (final.urgency, final.action, final.route_to) == (Urgency.CRITICAL, Action.ESCALATE_TO_HUMAN, Team.CARE_TEAM)


def test_hallucinated_citations_are_dropped_and_auto_reply_blocked():
    final, overrides = ENGINE.apply(decide(kb_article_ids=["KB-MADE-UP"]), KB_SEEN)
    assert final.kb_article_ids == []
    assert final.action is Action.ROUTE_TO_SPECIALIST
    assert [o.rule for o in overrides] == ["drop_ungrounded_citations", "auto_response_must_be_grounded"]


def test_low_confidence_auto_reply_needs_review():
    final, overrides = ENGINE.apply(decide(confidence=0.4), KB_SEEN)
    assert final.action is Action.ROUTE_TO_SPECIALIST
    assert overrides[-1].rule == "low_confidence_requires_review"


def extracted(issue_type: str, *flags: str) -> dict:
    return {**decision_payload()["extracted"], "issue_type": issue_type, "risk_flags": list(flags)}


def test_enterprise_outage_is_escalated_with_sla_floor():
    final, overrides = ENGINE.apply(decide(urgency="low", extracted=extracted("outage")), ENTERPRISE)
    assert (final.urgency, final.action, final.route_to) == (
        Urgency.HIGH,
        Action.ESCALATE_TO_HUMAN,
        Team.INCIDENT_RESPONSE,
    )
    assert overrides[0].rule == "enterprise_sla_requires_human"


def test_enterprise_multi_user_issue_keeps_the_team_the_model_chose():
    decision = decide(
        urgency="medium",
        action="route_to_specialist",
        route_to="technical_support",
        extracted=extracted("bug", "multiple_users_affected"),
    )
    final, _ = ENGINE.apply(decision, ENTERPRISE)
    assert (final.urgency, final.action, final.route_to) == (
        Urgency.HIGH,
        Action.ESCALATE_TO_HUMAN,
        Team.TECHNICAL_SUPPORT,
    )


def test_enterprise_how_to_question_is_untouched():
    final, overrides = ENGINE.apply(decide(), ENTERPRISE)
    assert overrides == []
    assert final.action is Action.AUTO_RESPOND


def test_same_outage_from_a_pro_customer_is_not_forced():
    pro = PolicyContext(retrieved_kb_ids=KB_SEEN.retrieved_kb_ids, customer_plan="pro")
    decision = decide(
        urgency="medium", action="route_to_specialist", route_to="technical_support", extracted=extracted("outage")
    )
    final, overrides = ENGINE.apply(decision, pro)
    assert overrides == []
    assert final.action is Action.ROUTE_TO_SPECIALIST


def test_context_takes_plan_from_customer_history_tool():
    records = [
        ToolCallRecord(tool_name="get_customer_history", arguments={}, ok=True, output={"plan": "enterprise"}),
        ToolCallRecord(
            tool_name="search_knowledge_base", arguments={}, ok=True, output={"results": [{"id": "KB-ENT-001"}]}
        ),
        ToolCallRecord(tool_name="get_customer_history", arguments={}, ok=False, error="not found"),
    ]
    ctx = PolicyContext.from_tool_calls(records)
    assert ctx.customer_plan == "enterprise"
    assert ctx.retrieved_kb_ids == {"KB-ENT-001"}
    assert PolicyContext.from_tool_calls([]).customer_plan is None
