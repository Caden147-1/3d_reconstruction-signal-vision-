"""Shared measurement validity rules for WiFi association and retrieval."""

from __future__ import annotations

import math
from typing import Any, Mapping


MAX_WIFI_AGE_MS = 5000
MIN_WIFI_APS = 3
MISSING_RSSI = -110.0
WIFI_QUALITY_FIELDS = [
    "wifi_observation_status",
    "wifi_measurement_min_elapsed_ts",
    "wifi_measurement_max_elapsed_ts",
    "wifi_fresh_ap_count",
    "wifi_max_age_ms",
]


def finite_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def visible_fingerprint(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        field: value
        for field, raw in row.items()
        if field.startswith("ap_") and MISSING_RSSI < (value := finite_number(raw)) < 0
    }


def valid_fingerprint(row: Mapping[str, Any]) -> bool:
    if row.get("wifi_observation_status") != "fresh":
        return False
    earliest = finite_number(row.get("wifi_measurement_min_elapsed_ts"))
    latest = finite_number(row.get("wifi_measurement_max_elapsed_ts"))
    age = finite_number(row.get("wifi_max_age_ms"))
    gap = row.get("wifi_gap_ms")
    if gap not in (None, "") and not 0 <= finite_number(gap) <= MAX_WIFI_AGE_MS:
        return False
    return (
        math.isfinite(earliest) and math.isfinite(latest)
        and earliest >= 0
        and 0 <= latest - earliest <= MAX_WIFI_AGE_MS
        and 0 <= age <= MAX_WIFI_AGE_MS
        and len(visible_fingerprint(row)) >= MIN_WIFI_APS
    )


def measurement_gap_ms(row: Mapping[str, Any], target_ms: int) -> float:
    """Bound the age of every AP in the fingerprint, not the log entry age."""
    if not valid_fingerprint(row):
        return math.inf
    return max(
        abs(target_ms - finite_number(row["wifi_measurement_min_elapsed_ts"])),
        abs(target_ms - finite_number(row["wifi_measurement_max_elapsed_ts"])),
    )
