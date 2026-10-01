"""Shared measurement validity rules for WiFi association and retrieval."""

from __future__ import annotations

import math
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
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
    "wifi_confidence",
    "wifi_unique_measurement_count",
    "wifi_measurement_ids",
    "wifi_ap_reliability",
]
WIFI_ASSOCIATION_FIELDS = [
    "wifi_association_basis",
    "wifi_stationary_start_elapsed_ts",
    "wifi_stationary_end_elapsed_ts",
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
    """Validate the RF measurement at scan time, independently of capture delay.

    Checkpoint association is established by image_mapping. wifi_gap_ms is
    diagnostic: elapsed time alone does not invalidate a same-location capture.
    """
    if row.get("wifi_observation_status") != "fresh":
        return False
    earliest = finite_number(row.get("wifi_measurement_min_elapsed_ts"))
    latest = finite_number(row.get("wifi_measurement_max_elapsed_ts"))
    age = finite_number(row.get("wifi_max_age_ms"))
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


def fingerprint_confidence(row: Mapping[str, Any]) -> float:
    """Confidence for retrieval/positioning; strict fresh validity remains separate."""
    if row.get("wifi_observation_status") == "aggregated_checkpoint":
        value = finite_number(row.get("wifi_confidence"))
        if len(visible_fingerprint(row)) >= 2 and math.isfinite(value):
            return max(0.0, min(1.0, value))
        return 0.0
    return 1.0 if valid_fingerprint(row) else 0.0


def weighted_median(samples: list[tuple[float, float]]) -> float:
    ordered = sorted(samples)
    halfway = sum(weight for _, weight in ordered) / 2
    accumulated = 0.0
    for value, weight in ordered:
        accumulated += weight
        if accumulated >= halfway:
            return value
    return ordered[-1][0]


def fingerprint_similarity(a: Mapping[str, Any], b: Mapping[str, Any], scale_db: float = 12.0) -> dict:
    """Continuous retrieval score shared by matching and its diagnostics."""
    va, vb = visible_fingerprint(a), visible_fingerprint(b)
    shared, union = va.keys() & vb.keys(), va.keys() | vb.keys()
    result = {"score": 0.0, "rssi_rms_db": None, "shared_aps": len(shared),
              "ap_overlap": len(shared) / len(union) if union else 0.0}
    if not a.get("session") or a.get("session") != b.get("session"):
        return {**result, "decision": "different_session"}
    if fingerprint_confidence(a) <= 0 or fingerprint_confidence(b) <= 0:
        return {**result, "decision": "no_usable_fingerprint"}
    ids_a = set(json.loads(a.get("wifi_measurement_ids") or "[]"))
    ids_b = set(json.loads(b.get("wifi_measurement_ids") or "[]"))
    if ids_a and ids_b:
        reused = bool(ids_a & ids_b)
    else:
        reused = max(finite_number(a.get("wifi_measurement_min_elapsed_ts")), finite_number(b.get("wifi_measurement_min_elapsed_ts"))) <= min(finite_number(a.get("wifi_measurement_max_elapsed_ts")), finite_number(b.get("wifi_measurement_max_elapsed_ts")))
    if reused:
        return {**result, "decision": "reused_measurement"}
    if len(shared) < 2:
        return {**result, "decision": "insufficient_shared_aps"}
    ra = json.loads(a.get("wifi_ap_reliability") or "{}")
    rb = json.loads(b.get("wifi_ap_reliability") or "{}")
    weights = {ap: math.sqrt(max(0.0, finite_number(ra.get(ap, 1.0)))
                             * max(0.0, finite_number(rb.get(ap, 1.0)))) for ap in shared}
    total = sum(weights.values())
    if total <= 0:
        return {**result, "decision": "no_reliable_shared_aps"}
    rms = math.sqrt(sum(weights[ap] * (va[ap] - vb[ap]) ** 2 for ap in shared) / total)
    confidence = math.sqrt(fingerprint_confidence(a) * fingerprint_confidence(b))
    score = confidence * math.sqrt(result["ap_overlap"]) * min(1.0, len(shared) / 4) * math.exp(-rms / max(1e-6, scale_db))
    return {**result, "score": score, "rssi_rms_db": rms, "decision": "ranked_candidate_requires_visual_geometry"}


def aggregate_checkpoint_fingerprints(path: Path, points: list[Any]) -> tuple[dict, dict]:
    """Deduplicate raw AP observations, then form confidence-weighted point summaries.

    A measurement is owned by one checkpoint only. Uncertain nearest-checkpoint
    assignment is explicitly weaker than a recorded stationary interval. Cached
    repeats cannot increase confidence, and missing AP values are never invented.
    """
    summary = {"unique_measurements": 0, "assigned_measurements": 0,
               "unassigned_measurements": 0, "aggregated_points": 0,
               "max_association_gap_ms": 120000,
               "policy": "unique_AP_timestamp_weighted_median_soft_checkpoint_assignment"}
    if not path.exists():
        return {}, {**summary, "status": "missing_raw_observations"}
    by_session = defaultdict(list)
    for point in points:
        by_session[point.session].append(point)
    unique = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = finite_number(row.get("ap_elapsed_ts"))
            logged = finite_number(row.get("scan_elapsed_ts"))
            rssi = finite_number(row.get("rssi"))
            if not row.get("mac") or not row.get("session") or not (
                math.isfinite(timestamp) and timestamp >= 0 and math.isfinite(logged)
                and logged >= timestamp and MISSING_RSSI < rssi < 0
            ):
                continue
            key = (row["session"], row["mac"], int(timestamp))
            age = logged - timestamp
            if key not in unique or age < unique[key][1]:
                unique[key] = (rssi, age)
    summary["unique_measurements"] = len(unique)
    groups = defaultdict(list)
    for (session, mac, timestamp), (rssi, age) in unique.items():
        candidates = by_session.get(session, [])
        if not candidates:
            continue
        stationary = [p for p in candidates if p.stationary_end_elapsed_ts is not None
                      and p.elapsed_ts <= timestamp < p.stationary_end_elapsed_ts]
        point = min(stationary or candidates, key=lambda p: abs(p.elapsed_ts - timestamp))
        gap = abs(point.elapsed_ts - timestamp)
        if not stationary and gap > summary["max_association_gap_ms"]:
            continue
        association = 1.0 if stationary else 0.9 if gap <= MAX_WIFI_AGE_MS else 0.65 * math.exp(-gap / 60000)
        weight = association * math.exp(-age / 60000)
        if weight < 1e-6:
            continue
        groups[point.global_point_index].append((mac, timestamp, rssi, age, weight, bool(stationary), gap))
    result = {}
    for point in points:
        samples = groups.get(point.global_point_index, [])
        if not samples:
            continue
        per_ap = defaultdict(list)
        for sample in samples:
            per_ap[sample[0]].append(sample)
        row = {}
        reliability = {}
        for mac, values in per_ap.items():
            rssi = weighted_median([(v[2], v[4]) for v in values])
            mad = median(abs(v[2] - rssi) for v in values)
            field = f"ap_{mac}"
            row[field] = round(rssi, 3)
            reliability[field] = round(sum(v[4] for v in values) / len(values) / (1 + mad / 6), 6)
        times = [v[1] for v in samples]
        confidence = sum(reliability.values()) / len(reliability) * min(1.0, len(per_ap) / 6)
        all_stationary = all(v[5] for v in samples)
        row.update({
            "wifi_observation_status": "aggregated_checkpoint",
            "wifi_confidence": round(confidence, 6),
            "wifi_measurement_min_elapsed_ts": min(times),
            "wifi_measurement_max_elapsed_ts": max(times),
            "wifi_unique_measurement_count": len(samples),
            "wifi_measurement_ids": json.dumps([f"{point.session}|{v[0]}|{v[1]}" for v in samples]),
            "wifi_ap_reliability": json.dumps(reliability, sort_keys=True),
            "wifi_fresh_ap_count": len({v[0] for v in samples if v[3] <= MAX_WIFI_AGE_MS}),
            "wifi_max_age_ms": max(v[3] for v in samples),
            "wifi_gap_ms": max(v[6] for v in samples),
            "wifi_association_basis": "aggregated_stationary" if all_stationary else "aggregated_soft_checkpoint",
            "wifi_stationary_start_elapsed_ts": point.elapsed_ts if all_stationary else "",
            "wifi_stationary_end_elapsed_ts": point.stationary_end_elapsed_ts if all_stationary else "",
            "elapsed_ts": "", "epoch_ts": "", "scan_index": "",
        })
        result[point.global_point_index] = row
        summary["assigned_measurements"] += len(samples)
    summary["unassigned_measurements"] = len(unique) - summary["assigned_measurements"]
    summary["aggregated_points"] = sum(fingerprint_confidence(row) > 0 for row in result.values())
    return result, {**summary, "status": "ok"}
