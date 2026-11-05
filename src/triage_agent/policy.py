"""Deterministic guardrails applied *after* the LLM decides.

The LLM is good at reading messy conversations; it is not a reliable place to
enforce business rules. So the LLM extracts signals (urgency, risk flags,
confidence, cited articles) and this module enforces the non-negotiable rules
in plain code. Every change is recorded as a `PolicyOverride`, which makes the
system auditable and gives us a metric ("how often do guardrails fire?").

Adding a rule = subclass `PolicyRule` and add it to `default_rules`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from triage_agent.schemas import (
    Action,
    IssueType,
    PolicyOverride,
    RiskFlag,
    Team,
    ToolCallRecord,
    TriageDecision,
    Urgency,
)
from triage_agent.tools.customer_history import GetCustomerHistoryTool
from triage_agent.tools.knowledge_base import SearchKnowledgeBaseTool
from triage_agent.tools.service_status import CheckServiceStatusTool


@dataclass(frozen=True)
class PolicyContext:
    """Facts about the run that rules may need beyond the decision itself.

    Everything here comes from tool results (our own systems), never from the
    ticket text, so a customer cannot change it by what they write.

    Attributes:
        retrieved_kb_ids: IDs of knowledge-base articles that search actually returned.
        customer_plan: Plan from `get_customer_history`, or None if it was not looked up.
        customer_region: Region from `get_customer_history`, or None if it was not looked up.
        degraded_regions: Regions where monitoring showed a problem, mapped to their
            active incident IDs (the tuple may be empty).
    """

    retrieved_kb_ids: frozenset[str] = field(default_factory=frozenset)
    customer_plan: str | None = None
    customer_region: str | None = None
    degraded_regions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_tool_calls(cls, tool_calls: Iterable[ToolCallRecord]) -> PolicyContext:
        """Collect the facts from the successful tool calls of one run.

        Args:
            tool_calls: Every tool call the agent made, successful or not.

        Returns:
            The context the rules check against.
        """
        records = [record for record in tool_calls if record.ok]
        ids = {
            result["id"]
            for record in records
            if record.tool_name == SearchKnowledgeBaseTool.name
            for result in record.output.get("results", [])
        }
        customers = [record.output for record in records if record.tool_name == GetCustomerHistoryTool.name]
        customer = customers[-1] if customers else {}
        degraded = {}
        for record in records:
            if record.tool_name != CheckServiceStatusTool.name:
                continue
            monitoring = record.output.get("internal_monitoring", {})
            incidents = tuple(i.get("incident_id", "unknown") for i in monitoring.get("active_incidents", []))
            if monitoring.get("status") != "operational" or incidents:
                degraded[record.output["region"]] = incidents
        return cls(
            retrieved_kb_ids=frozenset(ids),
            customer_plan=customer.get("plan"),
            customer_region=customer.get("region"),
            degraded_regions=degraded,
        )

    def incidents_affecting_customer(self) -> dict[str, tuple[str, ...]]:
        """Degraded regions that matter for this customer (all of them if the region is unknown)."""
        if self.customer_region is None:
            return dict(self.degraded_regions)
        return {r: ids for r, ids in self.degraded_regions.items() if r == self.customer_region}


@dataclass(frozen=True)
class RuleOutcome:
    """What a rule wants to change, and why.

    Attributes:
        reason: Human-readable explanation, stored in the policy override.
        updates: `TriageDecision` field names mapped to their new values.
    """

    reason: str
    updates: dict[str, Any]


class PolicyRule(ABC):
    """One business rule checked against every decision.

    Attributes:
        name: Stable identifier recorded in `PolicyOverride.rule`.
    """

    name: ClassVar[str]

    @abstractmethod
    def check(self, decision: TriageDecision, ctx: PolicyContext) -> RuleOutcome | None:
        """Decide whether the rule fires for this decision.

        Args:
            decision: The decision so far (earlier rules may already have changed it).
            ctx: Facts from the tool calls of this run.

        Returns:
            The field updates to apply, or None if the rule does not fire.
        """


# --------------------------------------------------------------------------- #
# Rules                                                                        #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Engine                                                                       #
# --------------------------------------------------------------------------- #
class PolicyEngine:
    """Applies rules to a decision in order and records every change they make.

    Args:
        rules: The rules to apply, in order.
    """

    def __init__(self, rules: Sequence[PolicyRule]) -> None:
        self._rules = list(rules)

    def apply(self, decision: TriageDecision, ctx: PolicyContext) -> tuple[TriageDecision, list[PolicyOverride]]:
        """Run every rule in order; each rule sees the changes made by the rules before it.

        Args:
            decision: The decision proposed by the model (or the fail-safe decision).
            ctx: Facts from the tool calls of this run.

        Returns:
            The final decision, and one override for each rule that changed something.
        """
        overrides: list[PolicyOverride] = []
        for rule in self._rules:
            outcome = rule.check(decision, ctx)
            if outcome is None:
                continue
            changes = {
                name: [_plain(getattr(decision, name)), _plain(value)]
                for name, value in outcome.updates.items()
                if getattr(decision, name) != value
            }
            if not changes:
                continue
            decision = decision.model_copy(update=outcome.updates)
            overrides.append(PolicyOverride(rule=rule.name, reason=outcome.reason, changes=changes))
        return decision, overrides


def _at_least(current: Urgency, floor: Urgency) -> Urgency:
    """Raise urgency to `floor` if it is lower; never lower it."""
    return current if current.rank >= floor.rank else floor


def _plain(value: Any) -> Any:
    """Turn enums (also inside lists) into their plain values so overrides serialize to JSON."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value
