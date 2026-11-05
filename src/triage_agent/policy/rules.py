"""The business rules, and the order they run in.

Each rule looks at the decision and the facts from tool results, and either leaves
the decision alone or says which fields to change and why.
"""

from __future__ import annotations

from typing import ClassVar

from triage_agent.policy.context import PolicyContext
from triage_agent.policy.engine import PolicyRule, RuleOutcome
from triage_agent.schemas import Action, IssueType, RiskFlag, Team, TriageDecision, Urgency


class SafetyConcernRule(PolicyRule):
    """A user in distress is never handled by automation (critical for a wellbeing platform)."""

    name = "safety_concern_to_care_team"

    def check(self, decision: TriageDecision, ctx: PolicyContext) -> RuleOutcome | None:
        if RiskFlag.SAFETY_CONCERN not in decision.extracted.risk_flags:
            return None
        return RuleOutcome(
            reason="Customer wellbeing/safety concern: immediate human ownership by the care team.",
            updates={"urgency": Urgency.CRITICAL, "action": Action.ESCALATE_TO_HUMAN, "route_to": Team.CARE_TEAM},
        )


class CriticalNeedsHumanRule(PolicyRule):
    """A critical ticket always gets immediate human ownership."""

    name = "critical_requires_human"

    def check(self, decision: TriageDecision, ctx: PolicyContext) -> RuleOutcome | None:
        if decision.urgency is not Urgency.CRITICAL or decision.action is Action.ESCALATE_TO_HUMAN:
            return None
        return RuleOutcome(
            reason="Critical tickets always get immediate human ownership.",
            updates={"action": Action.ESCALATE_TO_HUMAN, "route_to": decision.route_to or Team.GENERAL_SUPPORT},
        )


class HighRiskNeedsHumanRule(PolicyRule):
    """Chargebacks, legal threats and security/data issues need a human with authority."""

    name = "high_risk_requires_human"
    HIGH_RISK_FLAGS: ClassVar[frozenset[RiskFlag]] = frozenset(
        {RiskFlag.CHARGEBACK_THREAT, RiskFlag.LEGAL_THREAT, RiskFlag.SECURITY_OR_DATA_LOSS}
    )

    def check(self, decision: TriageDecision, ctx: PolicyContext) -> RuleOutcome | None:
        flags = self.HIGH_RISK_FLAGS.intersection(decision.extracted.risk_flags)
        if not flags:
            return None
        return RuleOutcome(
            reason=f"High-risk flags present ({', '.join(sorted(f.value for f in flags))}).",
            updates={
                "urgency": _at_least(decision.urgency, Urgency.HIGH),
                "action": Action.ESCALATE_TO_HUMAN,
                "route_to": decision.route_to or Team.GENERAL_SUPPORT,
            },
        )


class ActiveIncidentRule(PolicyRule):
    """Monitoring confirms a problem in the customer's region: the ticket belongs to incident response.

    Applies to every plan, because an outage is urgent whoever reports it. The evidence comes
    from `check_service_status`, not from the model's judgement, so it also separates a real
    outage from a problem on the customer's side.

    The ticket is routed to the incident team but not escalated to a person on its own: during
    an outage hundreds of tickets describe the same problem, so they are linked to the incident
    and answered together. Other rules (critical, enterprise SLA, chargeback) can still escalate.
    """

    name = "active_incident_in_customer_region"
    RELATED_ISSUES: ClassVar[frozenset[IssueType]] = frozenset({IssueType.OUTAGE, IssueType.ACCOUNT_ACCESS})

    def check(self, decision: TriageDecision, ctx: PolicyContext) -> RuleOutcome | None:
        incidents = ctx.incidents_affecting_customer()
        issues = {decision.extracted.issue_type, *decision.extracted.secondary_issue_types}
        if not incidents or not self.RELATED_ISSUES & issues:
            return None
        keep_escalation = decision.action is Action.ESCALATE_TO_HUMAN
        found = "; ".join(
            f"{region}: {', '.join(ids) or 'degraded, no incident yet'}" for region, ids in incidents.items()
        )
        return RuleOutcome(
            reason=f"Monitoring confirms a problem in the customer's region ({found}).",
            updates={
                "urgency": _at_least(decision.urgency, Urgency.HIGH),
                "action": Action.ESCALATE_TO_HUMAN if keep_escalation else Action.ROUTE_TO_SPECIALIST,
                "route_to": Team.INCIDENT_RESPONSE,
            },
        )


class EnterpriseSlaRule(PolicyRule):
    """Enterprise customers have a contractual SLA: outages or multi-user impact get a human fast.

    The plan comes from `get_customer_history` (our CRM), not from the ticket text.
    """

    name = "enterprise_sla_requires_human"

    def check(self, decision: TriageDecision, ctx: PolicyContext) -> RuleOutcome | None:
        if ctx.customer_plan != "enterprise":
            return None
        extracted = decision.extracted
        is_widespread = (
            extracted.issue_type is IssueType.OUTAGE or RiskFlag.MULTIPLE_USERS_AFFECTED in extracted.risk_flags
        )
        if not is_widespread:
            return None
        return RuleOutcome(
            reason="Enterprise customer with an outage or multi-user impact (contractual SLA).",
            updates={
                "urgency": _at_least(decision.urgency, Urgency.HIGH),
                "action": Action.ESCALATE_TO_HUMAN,
                "route_to": decision.route_to or Team.INCIDENT_RESPONSE,
            },
        )


class DropUngroundedCitationsRule(PolicyRule):
    """Remove KB article IDs the model cited but never actually retrieved (hallucinations)."""

    name = "drop_ungrounded_citations"

    def check(self, decision: TriageDecision, ctx: PolicyContext) -> RuleOutcome | None:
        grounded = [kb_id for kb_id in decision.kb_article_ids if kb_id in ctx.retrieved_kb_ids]
        if grounded == decision.kb_article_ids:
            return None
        return RuleOutcome(
            reason="Cited KB articles that were not returned by search_knowledge_base.",
            updates={"kb_article_ids": grounded},
        )


class GroundedAutoResponseRule(PolicyRule):
    """Only send an automatic reply if it is backed by at least one retrieved KB article."""

    name = "auto_response_must_be_grounded"

    def check(self, decision: TriageDecision, ctx: PolicyContext) -> RuleOutcome | None:
        if decision.action is not Action.AUTO_RESPOND or decision.kb_article_ids:
            return None
        return RuleOutcome(
            reason="Auto-reply is not backed by any knowledge-base article; a human must review it.",
            updates={"action": Action.ROUTE_TO_SPECIALIST, "route_to": Team.GENERAL_SUPPORT},
        )


class LowConfidenceReviewRule(PolicyRule):
    """An automatic reply the model is not confident about goes to a human for review first.

    Args:
        threshold: Minimum confidence (0 to 1) for an automatic reply.
    """

    name = "low_confidence_requires_review"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def check(self, decision: TriageDecision, ctx: PolicyContext) -> RuleOutcome | None:
        if decision.action is not Action.AUTO_RESPOND or decision.confidence >= self.threshold:
            return None
        return RuleOutcome(
            reason=f"Confidence {decision.confidence:.2f} is below {self.threshold:.2f} for an automatic reply.",
            updates={"action": Action.ROUTE_TO_SPECIALIST, "route_to": Team.GENERAL_SUPPORT},
        )


def default_rules(confidence_threshold: float) -> list[PolicyRule]:
    """Return the production rule set, in the order it must run.

    Order matters: safety first, then escalation rules, then the checks on automatic replies.

    Args:
        confidence_threshold: Minimum confidence for an automatic reply.

    Returns:
        The rules, ready for `PolicyEngine`.
    """
    return [
        SafetyConcernRule(),
        CriticalNeedsHumanRule(),
        HighRiskNeedsHumanRule(),
        ActiveIncidentRule(),
        EnterpriseSlaRule(),
        DropUngroundedCitationsRule(),
        GroundedAutoResponseRule(),
        LowConfidenceReviewRule(confidence_threshold),
    ]


def _at_least(current: Urgency, floor: Urgency) -> Urgency:
    """Raise urgency to `floor` if it is lower; never lower it."""
    return current if current.rank >= floor.rank else floor
