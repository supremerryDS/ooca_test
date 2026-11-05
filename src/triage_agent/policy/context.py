"""The facts from tool results that guardrails check against."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from triage_agent.schemas import ToolCallRecord
from triage_agent.tools.customer_history import GetCustomerHistoryTool
from triage_agent.tools.knowledge_base import SearchKnowledgeBaseTool
from triage_agent.tools.service_status import CheckServiceStatusTool


@dataclass(frozen=True)
class PolicyContext:
    """Facts about the run that rules may need beyond the decision itself.

    Everything here comes from tool results (our own systems), never from the
    ticket text, so a customer cannot change it by what they write.

    Attributes:
        retrieved_kb_ids: IDs of knowledge-base articles that search actually returned.
        customer_plan: Plan from `get_customer_history`, or None if it was not looked up.
        customer_region: Region from `get_customer_history`, or None if it was not looked up.
        degraded_regions: Regions where monitoring showed a problem, mapped to their
            active incident IDs (the tuple may be empty).
    """

    retrieved_kb_ids: frozenset[str] = field(default_factory=frozenset)
    customer_plan: str | None = None
    customer_region: str | None = None
    degraded_regions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_tool_calls(cls, tool_calls: Iterable[ToolCallRecord]) -> PolicyContext:
        """Collect the facts from the successful tool calls of one run.

        Args:
            tool_calls: Every tool call the agent made, successful or not.

        Returns:
            The context the rules check against.
        """
        records = [record for record in tool_calls if record.ok]
        ids = {
            result["id"]
            for record in records
            if record.tool_name == SearchKnowledgeBaseTool.name
            for result in record.output.get("results", [])
        }
        customers = [record.output for record in records if record.tool_name == GetCustomerHistoryTool.name]
        customer = customers[-1] if customers else {}
        degraded = {}
        for record in records:
            if record.tool_name != CheckServiceStatusTool.name:
                continue
            monitoring = record.output.get("internal_monitoring", {})
            incidents = tuple(i.get("incident_id", "unknown") for i in monitoring.get("active_incidents", []))
            if monitoring.get("status") != "operational" or incidents:
                degraded[record.output["region"]] = incidents
        return cls(
            retrieved_kb_ids=frozenset(ids),
            customer_plan=customer.get("plan"),
            customer_region=customer.get("region"),
            degraded_regions=degraded,
        )

    def incidents_affecting_customer(self) -> dict[str, tuple[str, ...]]:
        """Degraded regions that matter for this customer (all of them if the region is unknown)."""
        if self.customer_region is None:
            return dict(self.degraded_regions)
        return {r: ids for r, ids in self.degraded_regions.items() if r == self.customer_region}
