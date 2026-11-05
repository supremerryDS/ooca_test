"""The triage agent: a bounded tool-calling loop that ends with a structured decision.

Flow for one ticket
-------------------
1. Build the conversation: system prompt + rendered ticket.
2. Loop (max `max_steps`):
     - ask the LLM for the next step; it MUST answer with tool calls;
     - run lookup tools (customer history, KB, billing, status) and feed results back;
     - when it calls `submit_triage`, validate the payload with Pydantic.
       Invalid -> the validation error goes back to the model to fix.
       Valid   -> stop.
   On the last step the model is forced to call `submit_triage`.
3. Apply deterministic policy guardrails to the decision.
4. If anything fails (LLM down, no valid decision), return a fail-safe decision
   that escalates to a human. A ticket is never dropped.
"""

from __future__ import annotations

import json
import logging
import time

from pydantic import ValidationError

from triage_agent.llm.base import ChatMessage, LLMClient, LLMError, ToolCall, ToolSpec
from triage_agent.policy import PolicyContext, PolicyEngine
from triage_agent.prompts import render_ticket
from triage_agent.schemas import Ticket, TokenUsage, ToolCallRecord, TriageDecision, TriageResult
from triage_agent.tools.base import ToolRegistry, tool_output_for_llm

logger = logging.getLogger(__name__)

SUBMIT_TOOL = ToolSpec(
    name="submit_triage",
    description=(
        "Submit the final triage decision for this ticket. Call it exactly once, on its own, "
        "after you have gathered evidence with the other tools."
    ),
    parameters=TriageDecision,
)


class AgentError(RuntimeError):
    """The agent could not produce a valid decision."""


class _Run:
    """Mutable state of a single triage run (kept out of the agent so the agent is reusable)."""

    def __init__(self) -> None:
        self.conversation: list[ChatMessage] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.usage = TokenUsage()
        self.steps = 0


class TriageAgent:
    """Triages support tickets with an LLM, lookup tools and deterministic guardrails.

    One instance is reusable and safe to share: all per-ticket state lives in a
    private run object, so the CLI, the API and a queue worker can call `triage`
    on the same agent.

    Args:
        llm: Client used to talk to the model.
        tools: Lookup tools the model may call.
        policy: Guardrails applied to every decision after the model answers.
        system_prompt: Instructions for the model.
        max_steps: Maximum number of model turns per ticket. On the last turn the
            model is forced to submit a decision. Must be at least 2.

    Raises:
        ValueError: If `max_steps` is lower than 2.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        policy: PolicyEngine,
        system_prompt: str,
        *,
        max_steps: int = 8,
    ) -> None:
        if max_steps < 2:
            raise ValueError("max_steps must be at least 2 (gather evidence, then submit)")
        self._llm = llm
        self._tools = tools
        self._policy = policy
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._tool_specs = [*tools.specs, SUBMIT_TOOL]

    def triage(self, ticket: Ticket) -> TriageResult:
        """Triage one ticket end to end.

        Runs the tool-calling loop, then applies the policy guardrails. Never raises:
        if the model, a tool, a guardrail or anything else fails, the result holds a fail-safe
        decision that sends the ticket to a human.

        Args:
            ticket: The ticket to triage.

        Returns:
            The final decision after guardrails, the model's original decision,
            any policy overrides, every tool call, token usage and latency.
        """
        started = time.perf_counter()
        run = _Run()
        llm_decision: TriageDecision | None = None
        error: str | None = None

        try:
            run.conversation = [ChatMessage.system(self._system_prompt), ChatMessage.user(render_ticket(ticket))]
            llm_decision = self._run_loop(run)
            proposed = llm_decision
        except (AgentError, LLMError) as exc:
            logger.error("Triage failed for %s: %s", ticket.ticket_id, exc)
            error = str(exc)
            proposed = TriageDecision.fallback(reason=error)
        except Exception as exc:  # noqa: BLE001 - last resort: a ticket must never be dropped
            logger.exception("Unexpected error while triaging %s", ticket.ticket_id)
            error = f"Unexpected error: {type(exc).__name__}: {exc}"
            proposed = TriageDecision.fallback(reason=error)

        try:
            final, overrides = self._policy.apply(proposed, PolicyContext.from_tool_calls(run.tool_calls))
        except Exception as exc:  # noqa: BLE001 - a broken guardrail must not drop the ticket either
            logger.exception("Policy guardrails failed for %s", ticket.ticket_id)
            error = error or f"Policy error: {type(exc).__name__}: {exc}"
            final, overrides = TriageDecision.fallback(reason=error), []

        return TriageResult(
            ticket_id=ticket.ticket_id,
            status="completed" if error is None else "fallback",
            decision=final,
            llm_decision=llm_decision,
            policy_overrides=overrides,
            tool_calls=run.tool_calls,
            model=self._llm.model,
            steps=run.steps,
            usage=run.usage,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            error=error,
        )

    # ------------------------------------------------------------------ #
    # Loop                                                                 #
    # ------------------------------------------------------------------ #
    def _run_loop(self, run: _Run) -> TriageDecision:
        """Ask the model for tool calls until it submits a valid decision or runs out of steps."""
        for step in range(1, self._max_steps + 1):
            run.steps = step
            is_last_step = step == self._max_steps
            response = self._llm.chat(
                run.conversation,
                tools=self._tool_specs,
                force_tool=SUBMIT_TOOL.name if is_last_step else None,
            )
            run.usage.add(response.usage)
            run.conversation.append(ChatMessage.assistant(response.content, response.tool_calls))

            if not response.tool_calls:
                run.conversation.append(
                    ChatMessage.user("Reply only with tool calls. Call submit_triage when you are ready.")
                )
                continue

            decision = self._handle_tool_calls(response.tool_calls, run)
            if decision is not None:
                return decision

        raise AgentError(f"No valid triage decision after {self._max_steps} steps")

    def _handle_tool_calls(self, calls: list[ToolCall], run: _Run) -> TriageDecision | None:
        """Run lookups, answer every call, and return the decision if one was accepted."""
        lookups = [c for c in calls if c.name != SUBMIT_TOOL.name]
        submissions = [c for c in calls if c.name == SUBMIT_TOOL.name]

        # Every tool call must get a tool message back, in order, or the API rejects the next turn.
        for call in lookups:
            record = self._tools.execute(call.name, call.arguments)
            run.tool_calls.append(record)
            logger.info("tool %s ok=%s", call.name, record.ok)
            run.conversation.append(ChatMessage.tool_result(call.id, tool_output_for_llm(record)))

        decision: TriageDecision | None = None
        for call in submissions:
            if lookups:
                reply = _error("Call submit_triage on its own, after reading the results of your lookups.")
            elif decision is not None:
                reply = _error("A decision was already submitted in this turn.")
            else:
                decision, reply = self._parse_decision(call)
            run.conversation.append(ChatMessage.tool_result(call.id, reply))
        return decision

    @staticmethod
    def _parse_decision(call: ToolCall) -> tuple[TriageDecision | None, str]:
        """Validate a `submit_triage` call; return the decision (or None) and the reply for the model."""
        try:
            decision = TriageDecision.model_validate_json(call.arguments)
        except ValidationError as exc:
            logger.warning("Invalid submit_triage payload: %s", exc)
            return None, _error(f"Invalid decision, fix and resubmit: {exc.errors(include_url=False)}")
        return decision, '{"status": "accepted"}'


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False, default=str)
