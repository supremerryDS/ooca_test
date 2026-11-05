"""Backends the tools read from.

Each backend is a small Protocol so a real implementation (CRM API, Stripe,
Datadog/Prometheus, ...) can replace the JSON mock without changing any tool.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol


class CustomerDirectory(Protocol):
    """Source of customer profiles (a CRM in production)."""

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        """Return the customer's profile and support history, or None if the ID is unknown."""
        ...


class BillingSystem(Protocol):
    """Source of billing data (a payment provider in production)."""

    def get_billing_account(self, customer_id: str) -> dict[str, Any] | None:
        """Return plan, subscription status and payment attempts, or None if the ID is unknown."""
        ...


class StatusMonitor(Protocol):
    """Source of service health (internal monitoring and the public status page)."""

    def get_public_status(self) -> dict[str, Any]:
        """Return what the public status page currently says."""
        ...

    def get_region_status(self, region: str) -> dict[str, Any] | None:
        """Return internal monitoring for a region, or None if the region is unknown."""
        ...


class MockBackend:
    """Reads mocked customer, billing and monitoring data from JSON files.

    Implements `CustomerDirectory`, `BillingSystem` and `StatusMonitor`. Each file is
    read on first use and then kept in memory.

    Args:
        data_dir: Folder with `customers.json`, `billing.json` and `service_status.json`.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    # CustomerDirectory
    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        return self._customers.get(customer_id)

    # BillingSystem
    def get_billing_account(self, customer_id: str) -> dict[str, Any] | None:
        return self._billing.get(customer_id)

    # StatusMonitor
    def get_public_status(self) -> dict[str, Any]:
        return self._status["public_status_page"]

    def get_region_status(self, region: str) -> dict[str, Any] | None:
        return self._status["regions"].get(region)

    @cached_property
    def _customers(self) -> dict[str, Any]:
        return load_json(self._data_dir / "customers.json")

    @cached_property
    def _billing(self) -> dict[str, Any]:
        return load_json(self._data_dir / "billing.json")

    @cached_property
    def _status(self) -> dict[str, Any]:
        return load_json(self._data_dir / "service_status.json")


def load_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
