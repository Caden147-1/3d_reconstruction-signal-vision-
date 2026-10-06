"""Shared measurement validity rules for WiFi association and retrieval."""

from __future__ import annotations

import math
import json
from typing import Any, Iterable, Mapping


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

    Video association is checked by the video/sensor synchronization layer.
    Measurement freshness alone does not establish a camera position.
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
    return 1.0 if valid_fingerprint(row) else 0.0


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


def fingerprint_sequence_context(
    a: Mapping[str, Any], b: Mapping[str, Any],
    checkpoints: Iterable[Mapping[str, Any]], *, scale_db: float = 12.0,
) -> dict:
    """Bounded retrieval adjustment from independent neighboring RF observations.

    Only an already valid RF pair can receive context. One predecessor and one
    successor are considered in both traversal directions; missing stops are
    never bridged. Distinct AP IDs from the same scan are not independent stops.
    The multiplier is at most +/-15%, halved with only one usable flank, and
    attenuated by the weakest fingerprint confidence. It is not a pose weight,
    a physical distance, or evidence that a visual loop has been established.
    """
    neutral = dict(status="neutral", reason="insufficient_independent_context",
                   multiplier=1.0, direction=None, support=None,
                   usable_neighbor_pairs=0, independent_observation_count=0,
                   neighbor_pairs=[])
    try:
        base = fingerprint_similarity(a, b, scale_db)
    except (TypeError, ValueError):
        return {**neutral, "reason": "invalid_fingerprint_metadata"}
    if not math.isfinite(base["score"]) or base["score"] <= 0:
        return {**neutral, "reason": "no_valid_central_rf_pair"}

    session = a.get("session")

    def point_index(row):
        value = finite_number(row.get("global_point_index"))
        return int(value) if math.isfinite(value) and value > 0 and value.is_integer() else None

    def observation(row):
        if row is None or row.get("session") != session:
            return None
        timestamp = finite_number(row.get("point_elapsed_ts"))
        confidence = fingerprint_confidence(row)
        visible = visible_fingerprint(row)
        if not math.isfinite(timestamp) or timestamp < 0 or confidence <= 0:
            return None
        try:
            ids = json.loads(row.get("wifi_measurement_ids") or "[]")
            if not isinstance(ids, list) or not ids:
                return None
            measurements = set()
            for identity in ids:
                observed_session, mac, raw_time = identity.rsplit("|", 2)
                measured = int(raw_time)
                if observed_session != session or measured < 0 or "ap_" + mac not in visible:
                    return None
                measurements.add((mac, measured))
        except (AttributeError, TypeError, ValueError):
            return None
        if len({mac for mac, _ in measurements}) < MIN_WIFI_APS:
            return None
        start, end = min(t for _, t in measurements), max(t for _, t in measurements)
        if (start != finite_number(row.get("wifi_measurement_min_elapsed_ts"))
                or end != finite_number(row.get("wifi_measurement_max_elapsed_ts"))
                or max(abs(start - timestamp), abs(end - timestamp)) > 120000):
            return None
        return dict(start=start, end=end, confidence=confidence)

    def independent(observations):
        ordered = sorted(observations, key=lambda item: item["start"])
        # Conservative scan-level guard: changing AP names cannot turn reused
        # timestamps, overlapping windows or one scan burst into new evidence.
        return all(right["start"] - left["end"] > MAX_WIFI_AGE_MS
                   for left, right in zip(ordered, ordered[1:]))

    left_index, right_index = point_index(a), point_index(b)
    left_observation, right_observation = observation(a), observation(b)
    if (left_index is None or right_index is None or left_index == right_index
            or left_observation is None or right_observation is None
            or not independent([left_observation, right_observation])):
        return {**neutral, "reason": "central_observation_independence_unavailable"}
    by_index = {}
    for row in checkpoints:
        if row.get("session") == session and (index := point_index(row)) is not None:
            # Multiple representatives for a stop make its sequence ambiguous.
            by_index[index] = None if index in by_index else row
    if by_index.get(left_index) is None or by_index.get(right_index) is None:
        return {**neutral, "reason": "ambiguous_checkpoint_order"}

    def neighbor(center, index, offset):
        row = by_index.get(index + offset)
        if row is None:
            return None
        gap = offset * (finite_number(row.get("point_elapsed_ts"))
                        - finite_number(center.get("point_elapsed_ts")))
        return row if math.isfinite(gap) and 0 < gap <= 180000 else None

    def observed_in_order(center, other, offset):
        gap = other["start"] - center["end"] if offset > 0 else center["start"] - other["end"]
        return MAX_WIFI_AGE_MS < gap <= 180000

    candidates = []
    for direction in (1, -1):
        observations = [left_observation, right_observation]
        pairs = []
        for offset in (-1, 1):
            left = neighbor(a, left_index, offset)
            right = neighbor(b, right_index, offset * direction)
            lo, ro = observation(left), observation(right)
            if (lo is None or ro is None
                    or not observed_in_order(left_observation, lo, offset)
                    or not observed_in_order(right_observation, ro, offset * direction)
                    or not independent([left_observation, right_observation, lo, ro])):
                continue
            try:
                similarity = fingerprint_similarity(left, right, scale_db)
            except (TypeError, ValueError):
                continue
            if (similarity["decision"] != "ranked_candidate_requires_visual_geometry"
                    or similarity["shared_aps"] < MIN_WIFI_APS):
                continue
            shape = similarity["score"] / math.sqrt(lo["confidence"] * ro["confidence"])
            if not math.isfinite(shape):
                continue
            pairs.append(dict(left_checkpoint=point_index(left), right_checkpoint=point_index(right),
                              support=max(0.0, min(1.0, shape))))
            observations.extend((lo, ro))
        if not pairs or not independent(observations):
            continue
        support = sum(pair["support"] for pair in pairs) / len(pairs)
        confidence = min(item["confidence"] for item in observations)
        multiplier = 1.0 + .15 * (len(pairs) / 2) * confidence * (2 * support - 1)
        candidates.append(dict(status="applied", reason="independent_neighbor_rf_context",
                               multiplier=round(multiplier, 6),
                               direction="forward" if direction == 1 else "reverse",
                               support=round(support, 6), usable_neighbor_pairs=len(pairs),
                               independent_observation_count=len(observations), neighbor_pairs=pairs))
    # Prefer the better-supported traversal; deterministic forward tie-break.
    return max(candidates, key=lambda item: (item["support"], item["usable_neighbor_pairs"])) if candidates else neutral
