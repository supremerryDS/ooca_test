"""Offline evaluation against hand-labelled tickets.

Labels list *acceptable* answers (e.g. urgency high OR critical) because some
tickets are genuinely ambiguous; exact-match labels would punish reasonable calls.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from triage_agent.schemas import Action, Team, TriageResult, Urgency


class GoldenLabel(BaseModel):
    """Hand-written expectations for one ticket.

    Attributes:
        ticket_id: The ticket these expectations belong to.
        notes: Why these answers are expected (for humans only).
        acceptable_urgency: Every urgency that counts as correct.
        acceptable_actions: Every action that counts as correct.
        acceptable_teams: Every team that counts as correct, or None if any team (or none) is fine.
        expected_tools: Tools the agent should have called successfully.
    """

    ticket_id: str
    notes: str = ""
    acceptable_urgency: list[Urgency]
    acceptable_actions: list[Action]
    acceptable_teams: list[Team] | None = None
    expected_tools: list[str] = []


@dataclass(frozen=True)
class TicketScore:
    """Pass/fail for each scored field of one ticket.

    Attributes:
        ticket_id: The scored ticket.
        urgency_ok: Urgency is one of the acceptable values.
        action_ok: Action is one of the acceptable values.
        team_ok: Team is acceptable (always true when the label accepts any team).
        tools_ok: Every expected tool was called successfully.
        missing_tools: Expected tools that were not called.
    """

    ticket_id: str
    urgency_ok: bool
    action_ok: bool
    team_ok: bool
    tools_ok: bool
    missing_tools: tuple[str, ...]

    @property
    def all_ok(self) -> bool:
        """Whether every field passed."""
        return self.urgency_ok and self.action_ok and self.team_ok and self.tools_ok


def load_labels(path: Path) -> dict[str, GoldenLabel]:
    """Read golden labels from a JSON list.

    Args:
        path: The JSON file.

    Returns:
        Labels keyed by ticket ID.
    """
    with path.open(encoding="utf-8") as fh:
        labels = [GoldenLabel.model_validate(item) for item in json.load(fh)]
    return {label.ticket_id: label for label in labels}


def score_result(result: TriageResult, label: GoldenLabel) -> TicketScore:
    """Compare the final decision for one ticket with its label.

    Args:
        result: What the agent produced (the decision after guardrails is scored).
        label: The expected answers.

    Returns:
        Pass/fail per field, plus any expected tools that were not called.
    """
    decision = result.decision
    called = {record.tool_name for record in result.tool_calls if record.ok}
    missing = tuple(tool for tool in label.expected_tools if tool not in called)
    return TicketScore(
        ticket_id=result.ticket_id,
        urgency_ok=decision.urgency in label.acceptable_urgency,
        action_ok=decision.action in label.acceptable_actions,
        team_ok=label.acceptable_teams is None or decision.route_to in label.acceptable_teams,
        tools_ok=not missing,
        missing_tools=missing,
    )


def summarize(scores: Sequence[TicketScore]) -> dict[str, float]:
    """Turn per-ticket scores into accuracy figures between 0 and 1.

    Args:
        scores: One score per ticket.

    Returns:
        Accuracy for urgency, action, team and tool usage, plus the share of tickets that
        were fully correct. Empty if there are no scores.
    """
    if not scores:
        return {}
    n = len(scores)
    return {
        "urgency_accuracy": sum(s.urgency_ok for s in scores) / n,
        "action_accuracy": sum(s.action_ok for s in scores) / n,
        "team_accuracy": sum(s.team_ok for s in scores) / n,
        "tool_usage_accuracy": sum(s.tools_ok for s in scores) / n,
        "fully_correct": sum(s.all_ok for s in scores) / n,
    }
