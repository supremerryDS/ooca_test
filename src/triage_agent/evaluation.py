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
    ticket_id: str
    notes: str = ""
    acceptable_urgency: list[Urgency]
    acceptable_actions: list[Action]
    acceptable_teams: list[Team] | None = None  # None = any team (or none) is fine
    expected_tools: list[str] = []


@dataclass(frozen=True)
class TicketScore:
    ticket_id: str
    urgency_ok: bool
    action_ok: bool
    team_ok: bool
    tools_ok: bool
    missing_tools: tuple[str, ...]

    @property
    def all_ok(self) -> bool:
        return self.urgency_ok and self.action_ok and self.team_ok and self.tools_ok


def load_labels(path: Path) -> dict[str, GoldenLabel]:
    with path.open(encoding="utf-8") as fh:
        labels = [GoldenLabel.model_validate(item) for item in json.load(fh)]
    return {label.ticket_id: label for label in labels}


def score_result(result: TriageResult, label: GoldenLabel) -> TicketScore:
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
