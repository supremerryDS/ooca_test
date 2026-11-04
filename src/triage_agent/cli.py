"""Terminal entry point.

triage-agent run                      # triage all sample tickets
triage-agent run --ticket-id T-1002   # just one
triage-agent run --output results.json
triage-agent eval                     # score against evals/golden_labels.json
triage-agent tools                    # print the tool schemas sent to OpenAI
triage-agent serve                    # HTTP API, interactive docs at /docs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from rich.console import Console

from triage_agent.bootstrap import build_agent, build_tools, load_tickets
from triage_agent.config import PROJECT_ROOT, Settings
from triage_agent.display import render_result, render_scores
from triage_agent.evaluation import load_labels, score_result, summarize
from triage_agent.schemas import Ticket, TriageResult

DEFAULT_TICKETS = PROJECT_ROOT / "data" / "sample_tickets.json"
DEFAULT_LABELS = PROJECT_ROOT / "evals" / "golden_labels.json"

console = Console()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    if args.model:
        settings = replace(settings, model=args.model)
    return args.handler(args, settings)


# --------------------------------------------------------------------------- #
# Commands                                                                     #
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    if not _has_api_key():
        return 2
    tickets = _select(load_tickets(args.tickets), args.ticket_id)
    results = _triage_all(tickets, settings)
    _maybe_write(results, args.output)
    return 0 if all(r.status == "completed" for r in results) else 1


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    if not _has_api_key():
        return 2
    labels = load_labels(args.labels)
    tickets = [t for t in load_tickets(args.tickets) if t.ticket_id in labels]
    results = _triage_all(tickets, settings)
    scores = [score_result(r, labels[r.ticket_id]) for r in results]
    render_scores(console, scores, summarize(scores))
    _maybe_write(results, args.output)
    return 0


def cmd_tools(args: argparse.Namespace, settings: Settings) -> int:
    from triage_agent.agent import SUBMIT_TOOL
    from triage_agent.llm.openai_client import to_openai_tool

    specs = [*build_tools(settings.data_dir).specs, SUBMIT_TOOL]
    console.print_json(json.dumps([to_openai_tool(spec) for spec in specs], ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from triage_agent.api import create_app

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[yellow]OPENAI_API_KEY is not set: /docs works, but triage requests will return 503.[/]")
    console.print(f"API docs: [bold]http://{args.host}:{args.port}/docs[/]  (Ctrl+C to stop)")
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _triage_all(tickets: list[Ticket], settings: Settings) -> list[TriageResult]:
    agent = build_agent(settings)
    results = []
    for ticket in tickets:
        with console.status(f"Triaging {ticket.ticket_id} with {settings.model}..."):
            result = agent.triage(ticket)
        render_result(console, result)
        results.append(result)
    return results


def _select(tickets: list[Ticket], ids: list[str] | None) -> list[Ticket]:
    if not ids:
        return tickets
    selected = [t for t in tickets if t.ticket_id in ids]
    unknown = set(ids) - {t.ticket_id for t in selected}
    if unknown:
        raise SystemExit(f"Unknown ticket id(s): {', '.join(sorted(unknown))}")
    return selected


def _maybe_write(results: list[TriageResult], output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([r.model_dump(mode="json") for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[dim]Saved {len(results)} result(s) to {output}[/]")


def _has_api_key() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    console.print("[red]OPENAI_API_KEY is not set.[/] Copy .env.example to .env and add your key.")
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage-agent", description="Support ticket triage agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="log tool calls and agent steps")
    parser.add_argument("--model", help="override OPENAI_MODEL for this run")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="triage tickets and print the decisions")
    run.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    run.add_argument("--ticket-id", action="append", help="only this ticket (repeatable)")
    run.add_argument("--output", type=Path, help="write results as JSON")
    run.set_defaults(handler=cmd_run)

    ev = sub.add_parser("eval", help="triage labelled tickets and report accuracy")
    ev.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    ev.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ev.add_argument("--output", type=Path, help="write results as JSON")
    ev.set_defaults(handler=cmd_eval)

    tools = sub.add_parser("tools", help="print tool definitions (JSON schema sent to the model)")
    tools.set_defaults(handler=cmd_tools)

    serve = sub.add_parser("serve", help="run the HTTP API (FastAPI) with interactive docs at /docs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=cmd_serve)
    return parser


if __name__ == "__main__":
    sys.exit(main())
