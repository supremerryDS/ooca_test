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
    def get_customer(self, customer_id: str) -> dict[str, Any] | None: ...


class BillingSystem(Protocol):
    def get_billing_account(self, customer_id: str) -> dict[str, Any] | None: ...


class StatusMonitor(Protocol):
    def get_public_status(self) -> dict[str, Any]: ...

    def get_region_status(self, region: str) -> dict[str, Any] | None: ...


class MockBackend:
    """Reads mocked customer, billing and monitoring data from JSON files in `data_dir`.

    Implements CustomerDirectory, BillingSystem and StatusMonitor.
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
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
