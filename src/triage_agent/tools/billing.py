"""Billing tool: subscription state, payment attempts, refunds (read-only)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from triage_agent.datasources import BillingSystem
from triage_agent.tools.base import Tool, ToolError


class GetBillingTransactionsArgs(BaseModel):
    customer_id: str = Field(description="Customer ID from the ticket header.")


class GetBillingTransactionsTool(Tool):
    name = "get_billing_transactions"
    description = (
        "Read-only view of a customer's billing account: current plan, subscription status, "
        "recent payment attempts (gateway status vs. what the customer saw), refunds and disputes. "
        "Use for any ticket about payments, charges, refunds or plan upgrades."
    )
    args_model = GetBillingTransactionsArgs

    def __init__(self, billing: BillingSystem) -> None:
        self._billing = billing

    def run(self, args: GetBillingTransactionsArgs) -> dict[str, Any]:
        account = self._billing.get_billing_account(args.customer_id)
        if account is None:
            raise ToolError(f"No billing account found for customer '{args.customer_id}'.")
        return account
