"""Terminal rendering (rich). Pure presentation: no business logic here."""

from __future__ import annotations

import json
from collections.abc import Sequence

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from triage_agent.evaluation import TicketScore
from triage_agent.schemas import TriageResult, Urgency

URGENCY_STYLE = {
    Urgency.CRITICAL: "bold white on red",
    Urgency.HIGH: "bold dark_orange",
    Urgency.MEDIUM: "bold yellow",
    Urgency.LOW: "bold green",
}


def render_result(console: Console, result: TriageResult) -> None:
    d = result.decision
    x = d.extracted

    title = Text.assemble(
        (f" {result.ticket_id} ", "bold"),
        (f" {d.urgency.value.upper()} ", URGENCY_STYLE[d.urgency]),
        f"  {d.action.value}",
        (f" → {d.route_to.value}" if d.route_to else "", "bold cyan"),
        (" [FALLBACK]" if result.status == "fallback" else "", "bold red"),
    )

    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="dim", no_wrap=True)
    facts.add_column()
    facts.add_row("Summary", x.summary)
    facts.add_row("Product area", x.product_area)
    issue = x.issue_type.value + (
        f" (+ {', '.join(i.value for i in x.secondary_issue_types)})" if x.secondary_issue_types else ""
    )
    facts.add_row("Issue type", issue)
    facts.add_row("Sentiment", f"{x.sentiment.value} ({x.sentiment_trend.value})")
    facts.add_row("Language", x.customer_language)
    facts.add_row("Risk flags", ", ".join(f.value for f in x.risk_flags) or "-")
    facts.add_row("Key facts", "\n".join(f"• {fact}" for fact in x.key_facts) or "-")
    facts.add_row("Why urgency", d.urgency_rationale)
    facts.add_row("Why action", d.action_rationale)
    facts.add_row("KB articles", ", ".join(d.kb_article_ids) or "-")
    facts.add_row("Confidence", f"{d.confidence:.2f}")

    parts: list = [facts]
    if d.customer_reply:
        parts.append(Panel(d.customer_reply, title="Reply to customer", border_style="green"))
    parts.append(Panel(d.internal_note, title="Internal note", border_style="blue"))
    if result.policy_overrides:
        parts.append(_overrides_table(result))
    parts.append(_tools_table(result))
    parts.append(
        Text(
            f"model={result.model}  steps={result.steps}  tokens={result.usage.total_tokens}  "
            f"latency={result.latency_ms / 1000:.1f}s" + (f"  error={result.error}" if result.error else ""),
            style="dim",
        )
    )
    console.print(Panel(Group(*parts), title=title, title_align="left", border_style="white"))


def _overrides_table(result: TriageResult) -> Table:
    table = Table(title="Policy overrides", title_justify="left", expand=True)
    table.add_column("Rule")
    table.add_column("Change")
    table.add_column("Reason")
    for override in result.policy_overrides:
        change = "\n".join(f"{field}: {before} → {after}" for field, (before, after) in override.changes.items())
        table.add_row(override.rule, change, override.reason)
    return table


def _tools_table(result: TriageResult) -> Table:
    table = Table(title="Tool calls", title_justify="left", expand=True)
    table.add_column("#", justify="right")
    table.add_column("Tool")
    table.add_column("Arguments")
    table.add_column("OK")
    table.add_column("ms", justify="right")
    for i, record in enumerate(result.tool_calls, start=1):
        args = (
            record.arguments if isinstance(record.arguments, str) else json.dumps(record.arguments, ensure_ascii=False)
        )
        status = "✓" if record.ok else f"✗ {record.error}"
        table.add_row(str(i), record.tool_name, args, status, f"{record.duration_ms:.0f}")
    return table


def render_scores(console: Console, scores: Sequence[TicketScore], summary: dict[str, float]) -> None:
    table = Table(title="Evaluation", title_justify="left")
    for column in ("Ticket", "Urgency", "Action", "Team", "Tools"):
        table.add_column(column)

    def mark(ok: bool) -> str:
        return "[green]✓[/]" if ok else "[red]✗[/]"

    for s in scores:
        tools = mark(s.tools_ok) + (f" missing: {', '.join(s.missing_tools)}" if s.missing_tools else "")
        table.add_row(s.ticket_id, mark(s.urgency_ok), mark(s.action_ok), mark(s.team_ok), tools)
    console.print(table)
    for metric, value in summary.items():
        console.print(f"  {metric:<22} {value:.0%}")
