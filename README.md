# Support Ticket Triage Agent

An AI agent that triages incoming customer-support tickets with an OpenAI GPT model.
For every ticket it:

1. **Classifies urgency**: `critical` / `high` / `medium` / `low`
2. **Extracts key information**: product area, issue type(s), sentiment and its trend, language, key facts, risk flags
3. **Searches a knowledge base** for relevant FAQ/docs
4. **Decides the next action**: `auto_respond`, `route_to_specialist` (with team) or `escalate_to_human`
5. Uses **4 tools**: customer history, knowledge-base search, billing transactions, service status

It runs as a terminal console app **and** as an HTTP API (FastAPI) with interactive docs. There is no chat UI.

---

## Quick start

Requires **Python 3.10+**. Check with `python3 --version`. If your Python is older, see
*Only have Python 3.9 or older?* below.

**1. Install** (macOS / Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows, activate with `.venv\Scripts\activate`. Upgrade pip first: installing from
`pyproject.toml` in editable mode needs pip 21.3 or newer.

<details>
<summary><b>Only have Python 3.9 or older?</b></summary>

If you already created `.venv` with the old Python, remove it first:

```bash
deactivate
rm -rf .venv
```

Then install Python 3.12 using **one** of these options, and continue with
`python -m pip install --upgrade pip` and `pip install -e ".[dev]"` from step 1.

conda / Anaconda:

```bash
conda create -n triage python=3.12 -y
conda activate triage
```

Homebrew (macOS):

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
```

pyenv:

```bash
pyenv install 3.12
pyenv local 3.12
python -m venv .venv
source .venv/bin/activate
```

Installer from <https://www.python.org/downloads/>, then:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows use `py -3.12 -m venv .venv` and then `.venv\Scripts\activate`.

If `pip install -e` fails with *"File setup.py not found ... editable mode currently requires a
setup.py based build"*, your pip is too old. Run `python -m pip install --upgrade pip` inside the
environment and try again.
</details>

**2. Add your OpenAI key:**

```bash
cp .env.example .env
```

Then open `.env` and set `OPENAI_API_KEY`.

**3. Run:**

| Command | What it does |
|---|---|
| `triage-agent run` | Triage the 3 sample tickets |
| `triage-agent run --ticket-id T-1002` | Triage one ticket |
| `triage-agent run --output outputs/results.json` | Also save the full results as JSON |
| `triage-agent eval` | Score the agent against `evals/golden_labels.json` |
| `triage-agent tools` | Print the tool definitions (JSON schema sent to OpenAI). No key needed |
| `triage-agent -v run` | Verbose: log every tool call |
| `triage-agent serve` | Start the HTTP API on <http://127.0.0.1:8000> (see below) |

`python -m triage_agent ...` works too.

**4. Or use the HTTP API:**

```bash
triage-agent serve
```

Open <http://127.0.0.1:8000/docs> in a browser. It is an interactive Swagger UI: pick an endpoint,
click **Try it out**, then **Execute**. Stop the server with `Ctrl+C`. Use `--port` to change the port.

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness check, configured model, whether an API key is set |
| `GET /samples` | List the 3 sample tickets |
| `POST /samples/{ticket_id}/triage` | Triage a sample ticket by ID (`T-1001`, `T-1002`, `T-1003`) |
| `POST /triage` | Triage any ticket in the request body (the 3 samples are in the *Examples* dropdown) |

The response is the full `TriageResult`: the final decision, the LLM's original decision, policy
overrides, every tool call, token usage and latency. The server starts without an API key so you
can browse the docs. Triage requests then return `503` until `OPENAI_API_KEY` is set.

Example with curl:

```bash
curl -X POST http://127.0.0.1:8000/samples/T-1002/triage
```

Command blocks in this README contain no inline `# comments` on purpose: zsh, the default macOS
shell, passes them to the command as extra arguments.

Configuration is read from environment variables or `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | Your OpenAI key |
| `OPENAI_MODEL` | `gpt-5.4-mini` | Any GPT model with tool calling, e.g. `gpt-4.1` |
| `OPENAI_REASONING_EFFORT` | unset | `low`/`medium`/`high` for reasoning models |
| `OPENAI_TIMEOUT_S` | `60` | Timeout for one OpenAI request, in seconds |
| `TRIAGE_MAX_STEPS` | `8` | Max LLM turns per ticket before forcing a decision |
| `TRIAGE_CONFIDENCE_THRESHOLD` | `0.6` | Below this an auto-reply is sent to human review |
| `TRIAGE_DATA_DIR` | `./data` | Folder with mock data and the knowledge base |

Run the tests (no API key needed, the LLM is faked):

```bash
pytest
```

---

## How it works

```
             ┌─────────────────────────── TriageAgent (bounded loop, max N steps) ───────────────────────────┐
 Ticket ───▶ │ system prompt + <ticket>  ──▶  GPT (tool_choice=required)  ──▶  tool calls                     │
             │                                   ▲                              │                              │
             │                                   └──── tool results (JSON) ◀────┤ ToolRegistry                 │
             │                                                                  │  ├ get_customer_history      │
             │                                                                  │  ├ search_knowledge_base     │
             │                                                                  │  ├ get_billing_transactions  │
             │                                                                  │  └ check_service_status      │
             │                                                                  ▼                              │
             │                                        submit_triage(TriageDecision)  → Pydantic validation     │
             └───────────────────────────────────────────────────────────────────┬─────────────────────────────┘
                                                                                 ▼
                                            PolicyEngine (deterministic guardrails) ──▶ TriageResult
                                            (decision + overrides + tool trace + tokens + latency)
```

1. The ticket is rendered into a `<ticket>` block (marked as untrusted customer data) with
   relative timestamps, so the model sees how long the customer has been waiting. Tag-like
   text from the customer cannot close that block, and long threads keep their newest messages
   within a fixed length budget to bound cost. Clearly abnormal input (empty IDs, a message over
   10,000 characters, more than 100 messages) is rejected with HTTP 422, and timestamps without
   a timezone are read as UTC.
2. The model must answer with tool calls. It looks up the customer, then billing, service status
   and the knowledge base as needed.
3. It ends by calling **`submit_triage`**, whose arguments *are* the `TriageDecision` schema
   (strict JSON schema generated from Pydantic). Invalid decisions are sent back to the model with
   the validation error so it can fix them. On the last step the model is forced to submit.
4. **Guardrails in code** run on the decision, for example: critical tickets and chargeback threats
   always go to a human; a ticket about errors or access in a region where monitoring shows an
   incident goes to the incident team at `high` or above, whatever the customer's plan; an
   enterprise customer with an outage or multi-user impact always gets a human; auto-replies must
   cite a KB article that was actually retrieved; and low-confidence auto-replies go to review.
   Rules read plan, region and incidents from tool results, never from the ticket text. Every
   change is recorded as a `PolicyOverride`.
5. If the LLM fails or never produces a valid decision, a **fail-safe decision** escalates the
   ticket to a human. No ticket is dropped.

### Tools

| Tool | Input | What it returns (mocked) |
|---|---|---|
| `get_customer_history` | `customer_id` | Plan, tenure, seats, region, devices/app versions, SLA, previous tickets |
| `search_knowledge_base` | `query`, `category?`, `top_k?` | Top FAQ articles (BM25 ranking over `data/knowledge_base.json`) |
| `get_billing_transactions` | `customer_id` | Subscription status, payment attempts (gateway status vs. what the customer saw), refunds |
| `check_service_status` | `region` | Internal monitoring (error rate, incidents) vs. the public status page |
| `submit_triage` | `TriageDecision` | Terminal tool: the final structured answer |

Run `triage-agent tools` to see the exact JSON schemas.

### Output (`TriageDecision`)

`urgency`, `urgency_rationale`, `extracted` (`product_area`, `issue_type`, `secondary_issue_types`,
`summary`, `key_facts`, `customer_language`, `sentiment`, `sentiment_trend`, `risk_flags`),
`kb_article_ids`, `action`, `route_to`, `action_rationale`, `customer_reply` (in the customer's
language), `internal_note`, `confidence`.

### Sample tickets and expected outcome

| Ticket | Situation | Expected urgency | Expected action |
|---|---|---|---|
| T-1001 | Free user charged 3× $29.99, still no Pro, presentation in 2h, threatens chargeback | high (critical accepted) | escalate → billing |
| T-1002 | Enterprise (45 seats, Thailand) gets HTTP 500 on every browser; status page says OK | critical | escalate → incident_response |
| T-1003 | Pro user: dark mode question, "System Default" bug, scheduling feature request | low | auto_respond |

`triage-agent eval` checks these automatically (urgency, action, team and whether the right tools were called).

### Mock data assumptions

The customer facts come from the assignment. To make the tools meaningful, the mock data adds:

- **Billing (T-1001):** three card *authorizations* exist and plan provisioning timed out, so the customer saw "failed" even though money was held.
- **Monitoring (T-1002):** `asia-southeast` is degraded (37% HTTP 5xx) with an incident that is **not yet on the public status page**.
- **Customer (T-1003):** uses macOS desktop app 4.1.3. The KB has a known issue for 4.1.x and explains that the explicit Dark option arrives in 4.2.

---

## Project structure

```
src/triage_agent/
  agent.py              # the tool-calling loop (+ submit_triage, fail-safe)
  schemas.py            # Ticket in, TriageDecision / TriageResult out (Pydantic)
  policy/               # deterministic guardrails applied after the LLM
    context.py          # facts from tool results that rules check against
    rules.py            # the business rules, in the order they run
    engine.py           # applies the rules and records every change as an override
  prompts/
    system_prompt.md    # the system prompt (rubrics, rules)
    __init__.py         # prompt loading + ticket rendering
  tools/
    base.py             # Tool base class + ToolRegistry (validation, error containment)
    customer_history.py, billing.py, service_status.py, knowledge_base.py
  llm/
    base.py             # provider-agnostic LLMClient protocol + message types
    openai_client.py    # OpenAI Chat Completions adapter (strict function calling)
  datasources.py        # backend protocols + JSON mock backend
  search.py             # knowledge-base interface + BM25 search over the FAQ articles
  evaluation.py         # golden-label scoring
  bootstrap.py          # wiring (composition root)
  cli.py / display.py   # terminal app
  api.py                # HTTP API (FastAPI), same agent as the CLI
data/                   # sample tickets, mock CRM/billing/monitoring, knowledge base
evals/golden_labels.json
tests/                  # pytest, uses a scripted fake LLM
WRITEUP.md              # 1-page design write-up
```

## Extending

- **New tool:** subclass `Tool` (name, description, Pydantic `args_model`, `run`) and add it in
  `bootstrap.build_tools`. The schema is generated automatically. Mention it in the system prompt.
- **Real backends:** implement `CustomerDirectory` / `BillingSystem` / `StatusMonitor` /
  `KnowledgeBase` (e.g. vector search) and pass them in `bootstrap.py`. Tools don't change.
- **New guardrail:** subclass `PolicyRule` in `policy/rules.py` and add it to `default_rules`.
- **Another LLM provider:** implement `LLMClient.chat` in a new adapter.
- **New API endpoint:** add a route to the `router` in `api.py`. The agent is injected with
  `Depends(get_agent)`, and tests can pass a fake agent to `create_app(agent=...)`.
