"""Service status tool: internal monitoring for a region vs. the public status page."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from triage_agent.datasources import StatusMonitor
from triage_agent.tools.base import Tool, ToolError

Region = Literal["us-east", "eu-west", "asia-southeast"]


class CheckServiceStatusArgs(BaseModel):
    region: Region = Field(
        description="Region to check. Use the customer's region from get_customer_history "
        "('asia-southeast' covers Thailand and the rest of South-East Asia)."
    )


class CheckServiceStatusTool(Tool):
    name = "check_service_status"
    description = (
        "Check INTERNAL monitoring for a region (error rates, latency, active incidents, including "
        "incidents not yet on the public status page). Use for outages, errors, slowness or "
        "'cannot access' tickets."
    )
    args_model = CheckServiceStatusArgs

    def __init__(self, monitor: StatusMonitor) -> None:
        self._monitor = monitor

    def run(self, args: CheckServiceStatusArgs) -> dict[str, Any]:
        region_status = self._monitor.get_region_status(args.region)
        if region_status is None:
            raise ToolError(f"No monitoring data for region '{args.region}'.")
        return {
            "region": args.region,
            "internal_monitoring": region_status,
            "public_status_page": self._monitor.get_public_status(),
        }
