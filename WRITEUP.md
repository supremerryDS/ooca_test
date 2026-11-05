# Write-up: Support Ticket Triage Agent

## Architecture decisions and why

- **One bounded tool-calling loop.** Triage is a short "gather evidence, then decide" task,
  so I used one agent with a step limit (default 8) rather than a multi-agent graph. It is easier to debug, cheaper and
  fast enough. The model may call several lookups in one turn.
- **The decision is a tool (`submit_triage`).** Its arguments are the `TriageDecision` Pydantic model,
  exported as a strict JSON schema. The output is always machine-readable. Invalid output
  (e.g. `auto_respond` without a reply) goes back to the model with the validation error to fix.
  On the last step the model is forced to submit.
- **The LLM proposes, code enforces policy.** The model is good at reading messy, multilingual
  conversations. Business rules belong in code. The model extracts signals (urgency, risk flags,
  confidence, cited KB IDs), and a `PolicyEngine` applies rules such as: critical tickets and chargeback, legal or security risks
  always go to a human; auto-replies must cite a KB article that was actually retrieved; low
  confidence means review; safety concerns go to the care team (important for a wellbeing product). Every override is logged.
- **Read-only tools.** The agent looks things up but never refunds or replies by itself. Side
  effects happen downstream, after the decision and guardrails, which limits damage from a wrong call.
- **Separate layers.** `LLMClient`, tools, data sources and policy rules are separate
  interfaces, so I can switch the model provider, use a real CRM or vector search, or add a rule
  without touching the loop. The prompt is a Markdown file, versioned like code.

## What could go wrong and how I'd handle it

| Risk | Mitigation |
|---|---|
| Wrong urgency (missing an outage, or over-escalating an angry but minor issue) | Explicit rubric with examples. Tool evidence (monitoring, billing) is checked, not only tone. Guardrails set a minimum urgency. Track human re-prioritisations. |
| Hallucinated facts, KB IDs or promises (refunds, ETAs) | Prompt forbids it. Uncited KB IDs are removed. Auto-reply needs grounding. Escalations only send a short acknowledgement, which a human reviews. |
| Prompt injection in ticket text ("mark as low", "refund me") | Ticket is placed in a block marked as untrusted data. Tools are read-only. Policy is in code the model cannot override. |
| LLM outage, timeout or malformed output | SDK retries. Validation with feedback. Step limit. Fail-safe decision escalates to a human. |
| Tool errors or missing data | Errors go back to the model as data. The model is told not to guess and to lower confidence. |
| Language and cultural tone (e.g. Thai) | Replies are written in the customer's language. Internal notes are in English. Thai cases are in the eval set. |
| Cost and latency growth | Token usage and latency are recorded per ticket. A smaller model for low-risk tickets is possible. |

## How I'd evaluate it in production

1. **Offline before release:** a labelled set covering urgency, action, team and expected tools
   (`triage-agent eval`), growing from real tickets, including edge cases (multilingual, injection,
   mixed issues). Run on every prompt or model change as a regression gate. Use LLM-as-judge plus human
   spot checks for reply quality and groundedness. The same ticket can get a different answer on
   another run, even at temperature 0, so I'd run the set several times and gate on the pass rate,
   not on one run.
2. **Shadow mode, then gradual rollout:** the agent triages in parallel with humans. Compare
   urgency and routing agreement before enabling auto-replies, starting with `low` tickets.
3. **Online metrics:** human override rate by field, re-routing and re-opened tickets,
   time to first response, SLA breaches, CSAT after auto-replies, escalation precision and
   recall (especially missed `critical` tickets, the most costly error), guardrail fire rate,
   cost and latency per ticket. Review a sample weekly and add the failures to the eval set.
