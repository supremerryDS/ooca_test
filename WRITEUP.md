# Write-up: Support Ticket Triage Agent

## Architecture decisions and why

- **One bounded tool-calling loop.** Triage is "gather evidence, then decide", so one agent with
  a step limit (default 8) is enough: easier to debug and cheaper than a multi-agent setup.
- **The decision is a tool.** `submit_triage` takes the `TriageDecision` Pydantic model as a strict
  JSON schema, so the output is always machine-readable. Invalid decisions go back to the model
  with the validation error; the last step forces a submit.
- **The LLM proposes, code enforces.** The model reads messy, multilingual conversations and
  extracts signals such as urgency and risk flags. A `PolicyEngine` enforces business rules in
  code: critical, chargeback, legal and security tickets go to a human; auto-replies must cite
  an article that search returned; low confidence means review; safety concerns go to the care
  team. Every override is recorded in the result.
- **Read-only tools, swappable layers.** Tools only look things up; refunds and replies happen
  downstream, after the guardrails. LLM client, tools, data sources and rules are separate
  interfaces: provider, CRM, vector search or a new rule can change without touching the loop.

## What could go wrong and how I'd handle it

- **Wrong urgency:** rubric with examples; tool evidence counts more than tone; guardrails set a
  floor; track how often humans re-prioritise.
- **Hallucinated facts or promises:** uncited KB IDs are removed; an auto-reply needs a retrieved
  article; every other reply is a draft for a human.
- **Prompt injection:** the ticket is fenced as untrusted data; tools are read-only; policy lives in code.
- **LLM outage or bad output:** SDK retries, validation feedback and a step limit, then a fail-safe
  escalation to a human.
- **Tool errors or a wrong KB category:** errors go back to the model as data; search widens to
  all categories if the guess was wrong.
- **Tone in Thai and other languages:** replies use the customer's language; a Thai ticket is in
  the eval set.

## How I'd evaluate it in production

1. **Offline before release:** a labelled set (`triage-agent eval`) for urgency, action, team and
   tool use, grown from real tickets and edge cases, run on every prompt or model change. Answers
   vary between runs even at temperature 0, so I'd run it several times and gate on the pass
   rate. LLM-as-judge and human spot checks cover reply quality.
2. **Shadow mode, then rollout:** the agent runs beside humans; once agreement is high, enable
   auto-replies for `low` tickets first.
3. **Online metrics:** override rate per field, re-opened tickets, time to first response, SLA
   breaches, CSAT after auto-replies, recall on `critical` (the costliest miss), guardrail fire
   rate, cost and latency. Each week, failures join the eval set.
