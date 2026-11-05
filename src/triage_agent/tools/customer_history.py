"""Customer history tool: plan, tenure, region, devices and past tickets."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from triage_agent.datasources import CustomerDirectory
from triage_agent.tools.base import Tool, ToolError


class GetCustomerHistoryArgs(BaseModel):
    """Arguments for get_customer_history."""

    customer_id: str = Field(description="Customer ID from the ticket header, e.g. 'cus_1001'.")


class GetCustomerHistoryTool(Tool):
    """Customer lookup: plan, tenure, region, devices, SLA and previous tickets.

    Args:
        directory: Where customer profiles are read from.
    """

    name = "get_customer_history"
    description = (
        "Look up a customer's profile and support history: plan, tenure, seats, region, "
        "devices/app versions, account manager, contractual SLA and previous tickets. "
        "Call this first for every ticket."
    )
    args_model = GetCustomerHistoryArgs

    def __init__(self, directory: CustomerDirectory) -> None:
        self._directory = directory

    def run(self, args: GetCustomerHistoryArgs) -> dict[str, Any]:
        customer = self._directory.get_customer(args.customer_id)
        if customer is None:
            raise ToolError(f"No customer found with id '{args.customer_id}'.")
        return customer
