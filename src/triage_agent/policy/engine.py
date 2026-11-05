"""The rule interface, and the engine that applies rules in order and records every change."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from triage_agent.policy.context import PolicyContext
from triage_agent.schemas import PolicyOverride, TriageDecision


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


def _plain(value: Any) -> Any:
    """Turn enums (also inside lists) into their plain values so overrides serialize to JSON."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value
