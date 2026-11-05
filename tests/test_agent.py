from __future__ import annotations

import json

from fakes import ScriptedLLM, decision_payload, respond, tool_call

from triage_agent.llm.base import LLMError
from triage_agent.schemas import Action, Team, Urgency


def test_happy_path_gathers_evidence_then_submits(make_agent, tickets):
    llm = ScriptedLLM(
        [
            respond(
                tool_call("get_customer_history", {"customer_id": "cus_3003"}),
                tool_call(
                    "search_knowledge_base", {"query": "dark mode system default", "category": None, "top_k": None}
                ),
            ),
            respond(tool_call("submit_triage", decision_payload())),
        ]
    )
    result = make_agent(llm).triage(tickets["T-1003"])

    assert result.status == "completed"
    assert result.decision.action is Action.AUTO_RESPOND
    assert result.policy_overrides == []
    assert [r.tool_name for r in result.tool_calls] == ["get_customer_history", "search_knowledge_base"]
    assert all(r.ok for r in result.tool_calls)
    assert result.steps == 2
    assert result.usage.total_tokens == 240


def test_tool_results_are_sent_back_to_the_model(make_agent, tickets):
    lookup = tool_call("get_customer_history", {"customer_id": "cus_1001"})
    llm = ScriptedLLM([respond(lookup), respond(tool_call("submit_triage", decision_payload()))])
    make_agent(llm).triage(tickets["T-1001"])

    second_turn = llm.calls[1].messages
    tool_message = next(m for m in second_turn if m.role == "tool")
    assert tool_message.tool_call_id == lookup.id
    assert json.loads(tool_message.content)["plan"] == "free"


def test_invalid_decision_is_returned_to_the_model_for_correction(make_agent, tickets):
    invalid = decision_payload(customer_reply=None)  # auto_respond without a reply
    llm = ScriptedLLM(
        [
            respond(tool_call("search_knowledge_base", {"query": "dark mode", "category": None, "top_k": None})),
            respond(tool_call("submit_triage", invalid)),
            respond(tool_call("submit_triage", decision_payload())),
        ]
    )
    result = make_agent(llm).triage(tickets["T-1003"])

    assert result.status == "completed"
    feedback = llm.calls[2].messages[-1]
    assert feedback.role == "tool" and "customer_reply is required" in feedback.content


def test_submit_in_same_turn_as_lookups_is_rejected(make_agent, tickets):
    llm = ScriptedLLM(
        [
            respond(
                tool_call("get_customer_history", {"customer_id": "cus_3003"}),
                tool_call("submit_triage", decision_payload()),
            ),
            respond(tool_call("search_knowledge_base", {"query": "dark mode", "category": None, "top_k": None})),
            respond(tool_call("submit_triage", decision_payload())),
        ]
    )
    result = make_agent(llm).triage(tickets["T-1003"])

    assert result.steps == 3
    rejected = [m for m in llm.calls[1].messages if m.role == "tool"][-1]
    assert "on its own" in rejected.content


def test_last_step_forces_submit(make_agent, tickets):
    llm = ScriptedLLM(
        [
            respond(tool_call("get_customer_history", {"customer_id": "cus_3003"})),
            respond(tool_call("submit_triage", decision_payload(action="route_to_specialist", route_to="engineering"))),
        ]
    )
    make_agent(llm, max_steps=2).triage(tickets["T-1003"])

    assert llm.calls[0].force_tool is None
    assert llm.calls[1].force_tool == "submit_triage"


def test_llm_outage_falls_back_to_human_escalation(make_agent, tickets):
    llm = ScriptedLLM([LLMError("503 from provider")])
    result = make_agent(llm).triage(tickets["T-1002"])

    assert result.status == "fallback"
    assert result.decision.action is Action.ESCALATE_TO_HUMAN
    assert result.decision.route_to is Team.GENERAL_SUPPORT
    assert result.decision.urgency is Urgency.HIGH
    assert "503" in (result.error or "")


def test_unexpected_error_still_falls_back_to_a_human(make_agent, tickets):
    llm = ScriptedLLM([KeyError("bug somewhere")])
    result = make_agent(llm).triage(tickets["T-1003"])

    assert result.status == "fallback"
    assert result.decision.action is Action.ESCALATE_TO_HUMAN
    assert "KeyError" in (result.error or "")


def test_no_valid_decision_within_step_budget_falls_back(make_agent, tickets):
    lookup = respond(tool_call("get_customer_history", {"customer_id": "cus_3003"}))
    bad_submit = respond(tool_call("submit_triage", "{not json"))
    llm = ScriptedLLM([lookup, bad_submit])
    result = make_agent(llm, max_steps=2).triage(tickets["T-1003"])

    assert result.status == "fallback"
    assert result.llm_decision is None


def test_policy_overrides_are_applied_to_llm_decision(make_agent, tickets):
    risky = decision_payload(
        urgency="medium",
        action="route_to_specialist",
        route_to="billing",
        kb_article_ids=[],
        extracted={**decision_payload()["extracted"], "risk_flags": ["chargeback_threat"]},
    )
    llm = ScriptedLLM(
        [
            respond(tool_call("get_billing_transactions", {"customer_id": "cus_1001"})),
            respond(tool_call("submit_triage", risky)),
        ]
    )
    result = make_agent(llm).triage(tickets["T-1001"])

    assert result.llm_decision.action is Action.ROUTE_TO_SPECIALIST
    assert result.decision.action is Action.ESCALATE_TO_HUMAN
    assert result.decision.urgency is Urgency.HIGH
    assert result.decision.route_to is Team.BILLING
    assert [o.rule for o in result.policy_overrides] == ["high_risk_requires_human"]


def test_enterprise_sla_rule_uses_the_plan_returned_by_customer_history(make_agent, tickets):
    underrated = decision_payload(
        urgency="medium",
        action="route_to_specialist",
        route_to="technical_support",
        kb_article_ids=[],
        extracted={**decision_payload()["extracted"], "issue_type": "outage", "risk_flags": []},
    )
    llm = ScriptedLLM(
        [
            respond(tool_call("get_customer_history", {"customer_id": "cus_2002"})),
            respond(tool_call("submit_triage", underrated)),
        ]
    )
    result = make_agent(llm).triage(tickets["T-1002"])

    assert result.decision.action is Action.ESCALATE_TO_HUMAN
    assert result.decision.urgency is Urgency.HIGH
    assert [o.rule for o in result.policy_overrides] == ["enterprise_sla_requires_human"]


def test_monitoring_evidence_and_enterprise_sla_combine_for_ticket_2(make_agent, tickets):
    underrated = decision_payload(
        urgency="medium",
        action="route_to_specialist",
        route_to="technical_support",
        kb_article_ids=[],
        extracted={**decision_payload()["extracted"], "issue_type": "outage", "risk_flags": []},
    )
    llm = ScriptedLLM(
        [
            respond(
                tool_call("get_customer_history", {"customer_id": "cus_2002"}),
                tool_call("check_service_status", {"region": "asia-southeast"}),
            ),
            respond(tool_call("submit_triage", underrated)),
        ]
    )
    result = make_agent(llm).triage(tickets["T-1002"])

    decision = result.decision
    assert (decision.urgency, decision.action, decision.route_to) == (
        Urgency.HIGH,
        Action.ESCALATE_TO_HUMAN,
        Team.INCIDENT_RESPONSE,
    )
    assert [o.rule for o in result.policy_overrides] == [
        "active_incident_in_customer_region",
        "enterprise_sla_requires_human",
    ]
