"""Domain models: what goes into the agent (tickets) and what comes out (decisions).

The `TriageDecision` model doubles as the JSON schema of the `submit_triage` tool,
so the LLM is forced to answer in exactly this shape and every answer is validated
by Pydantic before the rest of the system trusts it.

Note: numeric bounds are enforced with validators instead of `Field(ge=..., le=...)`
so the generated JSON schema stays within what OpenAI strict mode accepts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Hard limits on incoming tickets. Anything beyond these is not a normal support ticket
# and is rejected (HTTP 422 from the API). Long but normal threads are accepted and
# trimmed to the newest messages when the prompt is built (see prompts.render_ticket).
MAX_ID_CHARS = 100
MAX_MESSAGE_CHARS = 10_000
MAX_MESSAGES = 100


# --------------------------------------------------------------------------- #
# Enums (shared vocabulary between the prompt, the policy layer and routing)   #
# --------------------------------------------------------------------------- #
class Urgency(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        """Higher number = more urgent. Useful for sorting queues."""
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]


class IssueType(str, Enum):
    BILLING = "billing"
    OUTAGE = "outage"
    BUG = "bug"
    HOW_TO = "how_to"
    FEATURE_REQUEST = "feature_request"
    ACCOUNT_ACCESS = "account_access"
    OTHER = "other"


class Sentiment(str, Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class SentimentTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"


class RiskFlag(str, Enum):
    CHARGEBACK_THREAT = "chargeback_threat"
    LEGAL_THREAT = "legal_threat"
    CHURN_RISK = "churn_risk"
    REVENUE_AT_RISK = "revenue_at_risk"
    HARD_DEADLINE = "hard_deadline"
    MULTIPLE_USERS_AFFECTED = "multiple_users_affected"
    SECURITY_OR_DATA_LOSS = "security_or_data_loss"
    SAFETY_CONCERN = "safety_concern"


class Action(str, Enum):
    AUTO_RESPOND = "auto_respond"
    ROUTE_TO_SPECIALIST = "route_to_specialist"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class Team(str, Enum):
    GENERAL_SUPPORT = "general_support"
    BILLING = "billing"
    TECHNICAL_SUPPORT = "technical_support"
    ENGINEERING = "engineering"
    INCIDENT_RESPONSE = "incident_response"
    CUSTOMER_SUCCESS = "customer_success"
    PRODUCT = "product"
    CARE_TEAM = "care_team"


# --------------------------------------------------------------------------- #
# Input                                                                        #
# --------------------------------------------------------------------------- #
class TicketMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    author: Literal["customer", "support"] = "customer"
    sent_at: datetime
    text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("sent_at")
    @classmethod
    def _assume_utc_when_timezone_missing(cls, value: datetime) -> datetime:
        # Helpdesks do not always send a timezone. Mixing timestamps with and without one
        # would make comparisons fail, so a missing timezone is treated as UTC.
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class Ticket(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    ticket_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    customer_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    channel: str = Field(default="unknown", min_length=1, max_length=MAX_ID_CHARS)
    messages: list[TicketMessage] = Field(min_length=1, max_length=MAX_MESSAGES)

    @property
    def latest_message_at(self) -> datetime:
        return max(m.sent_at for m in self.messages)


# --------------------------------------------------------------------------- #
# Output: the structured decision the LLM must submit                          #
# --------------------------------------------------------------------------- #
class ExtractedInfo(BaseModel):
    """Key facts pulled out of the conversation."""

    product_area: str = Field(
        description="Product or feature area involved, e.g. 'Billing / Pro upgrade', "
        "'Web app (asia-southeast)', 'macOS desktop app / Appearance settings'."
    )
    issue_type: IssueType = Field(description="Primary issue type.")
    secondary_issue_types: list[IssueType] = Field(
        description="Other issue types raised in the same ticket (empty list if none)."
    )
    summary: str = Field(description="One or two sentence English summary of the whole ticket.")
    key_facts: list[str] = Field(
        description="Concrete facts a human agent needs: amounts, counts, versions, deadlines, "
        "error codes, what the customer already tried."
    )
    customer_language: str = Field(description="ISO 639-1 code of the language the customer writes in.")
    sentiment: Sentiment = Field(description="Customer sentiment in the most recent messages.")
    sentiment_trend: SentimentTrend = Field(description="How sentiment changed across the messages.")
    risk_flags: list[RiskFlag] = Field(description="Business or safety risks present (empty list if none).")


class TriageDecision(BaseModel):
    """Final triage decision for one ticket. Submit exactly once, after gathering evidence."""

    urgency: Urgency = Field(description="Urgency according to the rubric in the instructions.")
    urgency_rationale: str = Field(description="Why this urgency, citing evidence from the ticket and tools.")
    extracted: ExtractedInfo
    kb_article_ids: list[str] = Field(
        description="IDs of knowledge-base articles returned by search_knowledge_base that support "
        "the reply or the routing. Never invent IDs."
    )
    action: Action = Field(description="Next action for this ticket.")
    route_to: Team | None = Field(
        description="Team that should own the ticket. Required for route_to_specialist and "
        "escalate_to_human; null when auto_respond fully resolves the ticket."
    )
    action_rationale: str = Field(description="Why this action and team.")
    customer_reply: str | None = Field(
        description="Message to send to the customer, written in the customer's language. "
        "For auto_respond this is the full answer; otherwise a short acknowledgement that sets "
        "expectations without promising refunds, causes or fix times."
    )
    internal_note: str = Field(
        description="Hand-off note for the human/specialist: what is known, what tools showed, "
        "what to do next. Always in English."
    )
    confidence: float = Field(description="Confidence in this decision, between 0.0 and 1.0.")

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return value

    @model_validator(mode="after")
    def _check_action_consistency(self) -> TriageDecision:
        if self.action is not Action.AUTO_RESPOND and self.route_to is None:
            raise ValueError(f"route_to is required when action is '{self.action.value}'")
        if self.action is Action.AUTO_RESPOND and not (self.customer_reply or "").strip():
            raise ValueError("customer_reply is required when action is 'auto_respond'")
        return self

    @classmethod
    def fallback(cls, reason: str) -> TriageDecision:
        """Safe default used when the agent cannot produce a valid decision: send to a human."""
        return cls(
            urgency=Urgency.HIGH,  # unknown != unimportant; err on the side of a fast human look
            urgency_rationale="Automatic triage failed; urgency could not be assessed.",
            extracted=ExtractedInfo(
                product_area="unknown",
                issue_type=IssueType.OTHER,
                secondary_issue_types=[],
                summary="Automatic triage failed; needs manual review.",
                key_facts=[],
                customer_language="unknown",
                sentiment=Sentiment.NEUTRAL,
                sentiment_trend=SentimentTrend.STABLE,
                risk_flags=[],
            ),
            kb_article_ids=[],
            action=Action.ESCALATE_TO_HUMAN,
            route_to=Team.GENERAL_SUPPORT,
            action_rationale="Fail-safe: the agent did not return a valid decision.",
            customer_reply=None,
            internal_note=f"Triage agent fallback. Reason: {reason}",
            confidence=0.0,
        )


# --------------------------------------------------------------------------- #
# Run result: decision + everything needed to audit / evaluate it              #
# --------------------------------------------------------------------------- #
class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, Any] | str
    ok: bool
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class PolicyOverride(BaseModel):
    rule: str
    reason: str
    changes: dict[str, list[Any]] = Field(description="field -> [before, after]")


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: TokenUsage | None) -> None:
        if other is not None:
            self.prompt_tokens += other.prompt_tokens
            self.completion_tokens += other.completion_tokens


class TriageResult(BaseModel):
    ticket_id: str
    status: Literal["completed", "fallback"]
    decision: TriageDecision = Field(description="Final decision, after policy guardrails.")
    llm_decision: TriageDecision | None = Field(
        default=None, description="Decision as proposed by the LLM, before guardrails."
    )
    policy_overrides: list[PolicyOverride] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    model: str
    steps: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    error: str | None = None
