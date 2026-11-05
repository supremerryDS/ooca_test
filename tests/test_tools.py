from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from triage_agent.agent import SUBMIT_TOOL
from triage_agent.llm.openai_client import to_openai_tool
from triage_agent.tools import Tool, ToolRegistry


def test_registry_exposes_all_tools(registry):
    assert set(registry.names) == {
        "get_customer_history",
        "search_knowledge_base",
        "get_billing_transactions",
        "check_service_status",
    }


def test_unknown_tool_returns_error_instead_of_raising(registry):
    record = registry.execute("delete_everything", "{}")
    assert not record.ok and "Unknown tool" in record.error


def test_invalid_arguments_return_validation_error(registry):
    record = registry.execute("check_service_status", json.dumps({"region": "mars"}))
    assert not record.ok and "Invalid arguments" in record.error


def test_unknown_customer_is_a_tool_error(registry):
    record = registry.execute("get_customer_history", json.dumps({"customer_id": "cus_404"}))
    assert not record.ok and "No customer found" in record.error


def test_billing_shows_three_authorizations_for_ticket_1(registry):
    record = registry.execute("get_billing_transactions", json.dumps({"customer_id": "cus_1001"}))
    attempts = record.output["payment_attempts"]
    assert record.ok
    assert len(attempts) == 3
    assert {a["gateway_status"] for a in attempts} == {"authorized"}
    assert record.output["current_plan"] == "free"


def test_service_status_reveals_unpublished_asia_incident(registry):
    record = registry.execute("check_service_status", json.dumps({"region": "asia-southeast"}))
    assert record.ok
    assert record.output["internal_monitoring"]["status"] == "degraded"
    assert record.output["public_status_page"]["overall"] == "all_systems_operational"


def test_kb_search_ignores_a_wrong_category_guess(registry):
    args = {"query": "system default theme ignores macos dark appearance", "category": "technical", "top_k": None}
    record = registry.execute("search_knowledge_base", json.dumps(args))
    assert record.ok
    assert record.output["results"][0]["id"] == "KB-APP-002"
    assert "all categories" in record.output["note"]


def test_kb_search_keeps_the_category_when_it_matches(registry):
    args = {"query": "system default theme ignores macos dark appearance", "category": "product", "top_k": None}
    record = registry.execute("search_knowledge_base", json.dumps(args))
    assert {r["category"] for r in record.output["results"]} == {"product"}
    assert record.output["note"] is None


def test_crashing_tool_is_contained():
    class Args(BaseModel):
        pass

    class Broken(Tool):
        name = "broken"
        description = "always fails"
        args_model = Args

        def run(self, args):
            raise RuntimeError("boom")

    record = ToolRegistry([Broken()]).execute("broken", "{}")
    assert not record.ok and "Internal tool error" in record.error


def test_duplicate_tool_names_are_rejected(registry):
    tool = next(iter(registry._tools.values()))
    with pytest.raises(ValueError):
        ToolRegistry([tool, tool])


@pytest.mark.parametrize("spec_index", range(5))
def test_tool_schemas_are_openai_strict_compatible(registry, spec_index):
    spec = [*registry.specs, SUBMIT_TOOL][spec_index]
    schema = to_openai_tool(spec)
    assert schema["function"]["strict"] is True

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
            # keywords we deliberately avoid for maximum model compatibility
            assert not {"default", "minimum", "maximum", "minItems"} & set(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema["function"]["parameters"])
