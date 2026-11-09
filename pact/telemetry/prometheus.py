"""Prometheus HTTP API client for instant PromQL queries."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pact.config import PrometheusConfig

Fetch = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class PrometheusSample:
    metric: Mapping[str, str]
    timestamp_s: float
    value: float


class PrometheusClient:
    """Queries ``/api/v1/query``. Inject ``fetch`` in tests to avoid the network."""

    def __init__(
        self,
        config: PrometheusConfig,
        *,
        fetch: Fetch | None = None,
    ) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._timeout_s = config.timeout_s
        self._fetch = fetch or self._http_fetch

    def instant_query(self, query: str) -> list[PrometheusSample]:
        params = urlencode({"query": query})
        url = f"{self._base_url}/api/v1/query?{params}"
        payload = self._fetch(url)
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {payload!r}")
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise TypeError("Prometheus response missing data mapping")
        result = data.get("result", [])
        if not isinstance(result, Sequence):
            raise TypeError("Prometheus result must be a sequence")
        return [_parse_sample(item) for item in result]

    def scalar(self, query: str) -> float:
        """Sum sample values. Empty result is 0.0."""

        return sum(sample.value for sample in self.instant_query(query))

    def _http_fetch(self, url: str) -> Mapping[str, Any]:
        request = Request(url, method="GET")
        with urlopen(request, timeout=self._timeout_s) as response:
            raw: Any = json.loads(response.read().decode())
        if not isinstance(raw, dict):
            raise TypeError("Prometheus JSON root must be an object")
        return raw


def _parse_sample(item: object) -> PrometheusSample:
    if not isinstance(item, Mapping):
        raise TypeError("Prometheus sample must be a mapping")
    metric_raw = item.get("metric", {})
    if not isinstance(metric_raw, Mapping):
        raise TypeError("Prometheus sample metric must be a mapping")
    metric = {str(k): str(v) for k, v in metric_raw.items()}
    value_raw = item.get("value")
    if not isinstance(value_raw, Sequence) or len(value_raw) < 2:
        raise TypeError("Prometheus sample value must be [timestamp, number]")
    return PrometheusSample(
        metric=metric,
        timestamp_s=float(value_raw[0]),
        value=float(value_raw[1]),
    )
