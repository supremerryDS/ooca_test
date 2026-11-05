"""Deterministic guardrails applied *after* the LLM decides.

The LLM is good at reading messy conversations; it is not a reliable place to
enforce business rules. So the LLM extracts signals (urgency, risk flags,
confidence, cited articles) and this package enforces the non-negotiable rules
in plain code. Every change is recorded as a `PolicyOverride`, which makes the
system auditable and gives us a metric ("how often do guardrails fire?").

- `context.py`: the facts from tool results that rules check against.
- `rules.py`: the business rules and the order they run in.
- `engine.py`: the rule interface, and the engine that applies rules and records overrides.

Adding a rule = subclass `PolicyRule` in `rules.py` and add it to `default_rules`.
"""

from triage_agent.policy.context import PolicyContext
from triage_agent.policy.engine import PolicyEngine, PolicyRule, RuleOutcome
from triage_agent.policy.rules import default_rules

__all__ = ["PolicyContext", "PolicyEngine", "PolicyRule", "RuleOutcome", "default_rules"]
