"""Bounded soft retrieval from authenticated, uncertain sensor time windows.

Window membership is a retrieval hypothesis. It never supplies a camera pose,
an exact sensor timestamp, or a union fingerprint to an image row.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from video_sensor_fusion import has_valid_video_sensor_sync
from wifi_observations import MAX_WIFI_AGE_MS, finite_number, fingerprint_similarity, measurement_gap_ms


_CONTINUOUS = {"observed_step_link", "stationary_checkpoint", "documented_checkpoint_pause"}
_BASIS = "fuzzy_time_window_requires_visual_geometry"


def _even_indices(count, limit):
    if count <= limit:
        return range(count)
    if limit <= 0:
        return ()
    if limit == 1:
        return (count // 2,)
    return sorted({round(i * (count - 1) / (limit - 1)) for i in range(limit)})


def measurement_key(row):
    """Use original measurement identity across explicit and fuzzy retrieval."""
    ids = json.loads(row.get("wifi_measurement_ids") or "[]")
    if not isinstance(ids, list) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Malformed original WiFi measurement IDs")
    identity = ("ids", tuple(sorted(set(ids)))) if ids else (
        "interval", finite_number(row.get("wifi_measurement_min_elapsed_ts")),
        finite_number(row.get("wifi_measurement_max_elapsed_ts")))
    return row["session"], identity


class WifiRetrievalBudget:
    """One representative/neighbor/frame quota shared by both clock modes."""

    def __init__(self, neighbors=3):
        self.max_scans, self.max_frames, self.max_neighbors = 256, 4, min(3, max(0, neighbors))
        self.scans = set()
        self.neighbors, self.frames = defaultdict(set), defaultdict(set)

    def representatives(self, scans):
        unique = {}
        for item in scans:
            try:
                key = measurement_key(item[0])
            except (ValueError, TypeError, AttributeError, KeyError):
                continue
            unique.setdefault(key, item)
        previous = [(key, item) for key, item in unique.items() if key in self.scans]
        new = [(key, item) for key, item in unique.items() if key not in self.scans]
        chosen = [*previous, *(new[index] for index in _even_indices(len(new), self.max_scans - len(self.scans)))]
        for key, _ in chosen:
            self.scans.add(key)
        return [(*item, key) for key, item in chosen]

    def accept(self, keys, names):
        if not keys or len(keys) != 2 or keys[0] == keys[1]:
            return False
        for key, other, name in ((keys[0], keys[1], names[0]), (keys[1], keys[0], names[1])):
            if (key not in self.scans
                    or (other not in self.neighbors[key] and len(self.neighbors[key]) >= self.max_neighbors)
                    or (name not in self.frames[key] and len(self.frames[key]) >= self.max_frames)):
                return False
        self.neighbors[keys[0]].add(keys[1])
        self.neighbors[keys[1]].add(keys[0])
        self.frames[keys[0]].add(names[0])
        self.frames[keys[1]].add(names[1])
        return True


def _context(path):
    if path is None:
        return None, "", "missing_context"
    try:
        data = Path(path).read_bytes()
        value = json.loads(data)
    except (OSError, UnicodeError, ValueError, TypeError):
        return None, "", "invalid_context"
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or value.get("method") != "fuzzy_time_window"
            or not isinstance(value.get("models"), dict)
            or not isinstance(value.get("route_samples"), list)
            or not isinstance(value.get("wifi_scans"), list)
            or not all(isinstance(row, dict)
                       and all(isinstance(cell, (str, int, float, bool, type(None))) for cell in row.values())
                       for key in ("route_samples", "wifi_scans") for row in value[key])):
        return None, "", "invalid_context_schema"
    uncertainty = finite_number(value.get("uncertainty_s"))
    if not math.isfinite(uncertainty) or uncertainty < 0:
        return None, "", "invalid_context_uncertainty"
    return value, hashlib.sha256(data).hexdigest(), "verified_context"


def _frame_window(row, context, digest):
    from video_sensor_fuzzy import has_fuzzy_video_sensor_window

    if not has_fuzzy_video_sensor_window(row) or row.get("video_sensor_fuzzy_source_sha256") != digest:
        return None
    model = context["models"].get(row.get("video_name"))
    if not isinstance(model, dict) or model.get("session") != row.get("session"):
        return None
    offset, scale, start, end = [finite_number(model.get(field)) for field in
                                  ("offset_ms", "scale", "sensor_start_ms", "sensor_end_ms")]
    video = finite_number(row.get("video_time_s"))
    uncertainty = finite_number(row.get("video_sensor_time_uncertainty_s"))
    low = finite_number(row.get("video_sensor_window_start_ms"))
    high = finite_number(row.get("video_sensor_window_end_ms"))
    if (not all(math.isfinite(v) for v in (offset, scale, start, end, video, uncertainty, low, high))
            or scale != 1 or start < 0 or end < start or video < 0
            or uncertainty < 0 or abs(uncertainty - float(context["uncertainty_s"])) > 1e-6):
        return None
    center = offset + video * 1000
    expected_low, expected_high = max(start, center - uncertainty * 1000), min(end, center + uncertainty * 1000)
    if expected_low > expected_high or abs(low - expected_low) > 1e-3 or abs(high - expected_high) > 1e-3:
        return None
    duration = finite_number(model.get("video_duration_s"))
    if math.isfinite(duration) and video > duration + 1e-6:
        return None
    return dict(row=row, low=low, high=high, center=center, uncertainty=uncertainty, fuzzy=True)


def _segments(samples):
    grouped = defaultdict(list)
    for row in samples:
        timestamp = finite_number(row.get("elapsed_ts"))
        if row.get("session") and math.isfinite(timestamp) and timestamp >= 0:
            grouped[row["session"]].append((timestamp, row))
    result = defaultdict(list)
    for session, records in grouped.items():
        records.sort(key=lambda item: item[0])
        current, previous = [], None
        for timestamp, row in records:
            distance = finite_number(row.get("trajectory_route_distance_m"))
            valid = (row.get("route_distance_status") == "estimated_steps"
                     and row.get("contour_id") and math.isfinite(distance) and distance >= 0)
            continuous = (valid and previous is not None and row["contour_id"] == previous[1]["contour_id"]
                          and row.get("link_status") in _CONTINUOUS and timestamp > previous[0]
                          and distance >= previous[2]
                          and (timestamp - previous[0] <= 10000 or row.get("link_status") == "documented_checkpoint_pause"))
            if not continuous and current:
                result[session].append(current)
                current = []
            if valid:
                current.append((timestamp, row, distance))
                previous = (timestamp, row, distance)
            else:
                previous = None
        if current:
            result[session].append(current)
    return result


def _route_interval(frame, segments):
    containing = [segment for segment in segments
                  if segment[0][0] <= frame["low"] <= frame["high"] <= segment[-1][0]]
    if len(containing) != 1:
        return None
    segment = containing[0]
    times = [item[0] for item in segment]

    def at(timestamp):
        index = bisect_left(times, timestamp)
        if index < len(times) and times[index] == timestamp:
            return segment[index][2]
        a, b = segment[index - 1], segment[index]
        return a[2] + (b[2] - a[2]) * (timestamp - a[0]) / (b[0] - a[0])

    return dict(frame, route_low=at(frame["low"]), route_high=at(frame["high"]),
                segment_key=(frame["row"]["session"], segment[0][1]["contour_id"], segment[0][0]))


def _evidence(a, b, digest):
    a, b = sorted((a, b), key=lambda frame: frame["row"]["image_file"])
    return dict(edge_weight_basis=_BASIS, video_sensor_fuzzy_source_sha256=digest,
                sensor_association_mode="fuzzy_time_window",
                sensor_time_uncertainty_s=max(a["uncertainty"], b["uncertainty"]),
                sensor_time_uncertainty_a_s=a["uncertainty"], sensor_time_uncertainty_b_s=b["uncertainty"],
                sensor_window_a_start_ms=a["low"], sensor_window_a_end_ms=a["high"],
                sensor_window_b_start_ms=b["low"], sensor_window_b_end_ms=b["high"],
                temporal_uncertainty_is_assumed=True)


def add_fuzzy_video_sensor_candidates(rows, context_path, add, *, enabled=True, wifi_enabled=True,
                                      route_distance_m=8., route_neighbors=4, wifi_neighbors=3,
                                      wifi_budget=None):
    """Use the caller's pair set and endpoint budgets through ``add``.

    Counts describe accepted edges after shared deduplication, not all retrieval
    hypotheses. Malformed optional context only disables this candidate source.
    """
    fuzzy_rows = [row for row in rows if row.get("source_type") == "video_base_frame"
                  and row.get("image_file") and row.get("video_sensor_sync_status") == "fuzzy_time_window"]
    summary = dict(fuzzy_status="disabled" if not enabled else "no_fuzzy_video_frames",
                   fuzzy_input_frame_count=len(fuzzy_rows), fuzzy_frame_count=0,
                   fuzzy_rejected_frame_count=0, fuzzy_route_frame_count=0, fuzzy_route_group_count=0,
                   fuzzy_route_candidate_count=0, fuzzy_wifi_candidate_count=0,
                   fuzzy_wifi_valid_scan_count=0, fuzzy_wifi_representative_scan_count=0,
                   fuzzy_wifi_omitted_scan_count=0, fuzzy_wifi_scan_pair_count=0,
                   fuzzy_wifi_pair_comparison_count=0, fuzzy_wifi_ambiguous_window_pair_count=0,
                   fuzzy_max_representative_scans=256, fuzzy_max_frames_per_scan=4,
                   fuzzy_wifi_neighbors_per_scan=min(3, max(0, wifi_neighbors)),
                   fuzzy_camera_pose_constraints=False)
    if not enabled or not fuzzy_rows:
        return summary
    context, digest, status = _context(context_path)
    summary["fuzzy_status"] = status
    if context is None:
        summary["fuzzy_rejected_frame_count"] = len(fuzzy_rows)
        return summary
    fuzzy = [frame for row in fuzzy_rows if (frame := _frame_window(row, context, digest)) is not None]
    summary.update(fuzzy_frame_count=len(fuzzy), fuzzy_rejected_frame_count=len(fuzzy_rows) - len(fuzzy),
                   fuzzy_context_sha256=digest)
    if not fuzzy:
        summary["fuzzy_status"] = "no_verified_fuzzy_frames"
        return summary
    summary["fuzzy_status"] = "evaluated"
    sessions = {frame["row"]["session"] for frame in fuzzy}
    frames = list(fuzzy)
    names = {frame["row"]["image_file"] for frame in frames}
    for row in rows:
        if (row.get("source_type") == "video_base_frame" and row.get("image_file") not in names
                and row.get("session") in sessions and has_valid_video_sensor_sync(row)):
            timestamp = float(row["sensor_elapsed_ts"])
            frames.append(dict(row=row, low=timestamp, high=timestamp, center=timestamp, uncertainty=0., fuzzy=False))
    segments = _segments(context["route_samples"])
    groups = defaultdict(list)
    for frame in frames:
        route = _route_interval(frame, segments.get(frame["row"]["session"], ()))
        if route is not None:
            groups[route["segment_key"]].append(route)
            summary["fuzzy_route_frame_count"] += int(frame["fuzzy"])
    summary["fuzzy_route_group_count"] = sum(any(frame["fuzzy"] for frame in group) for group in groups.values())
    if route_distance_m > 0 and route_neighbors > 0:
        for group in groups.values():
            group.sort(key=lambda frame: ((frame["route_low"] + frame["route_high"]) / 2, frame["row"]["image_file"]))
            centers = [(frame["route_low"] + frame["route_high"]) / 2 for frame in group]
            max_width = max((frame["route_high"] - frame["route_low"]) / 2 for frame in group)
            for index, frame in enumerate(group):
                width = (frame["route_high"] - frame["route_low"]) / 2
                low = bisect_left(centers, centers[index] - route_distance_m - width - max_width)
                high = bisect_right(centers, centers[index] + route_distance_m + width + max_width)
                candidates = {low + offset for offset in _even_indices(high - low, 32)}
                candidates.update(range(max(low, index - 16), min(high, index + 17)))
                ranked = []
                for other_index in candidates:
                    other = group[other_index]
                    if other_index == index or not (frame["fuzzy"] or other["fuzzy"]):
                        continue
                    gap = max(0., frame["route_low"] - other["route_high"], other["route_low"] - frame["route_high"])
                    if gap <= route_distance_m:
                        ranked.append((gap, abs(frame["center"] - other["center"]), other["row"]["image_file"], other))
                for gap, _, _, other in sorted(ranked, key=lambda item: item[:3]):
                    weight = .35 / (1 + gap) / (1 + (frame["uncertainty"] + other["uncertainty"]) / 10)
                    if add(frame["row"], other["row"], "video_imu_route", weight,
                           trajectory_distance_min_m=gap,
                           trajectory_distance_max_m=max(abs(frame["route_low"] - other["route_high"]),
                                                         abs(other["route_low"] - frame["route_high"])),
                           **_evidence(frame, other, digest)):
                        summary["fuzzy_route_candidate_count"] += 1
    if not wifi_enabled or wifi_neighbors <= 0:
        return summary
    frames_by_session = defaultdict(list)
    for frame in frames:
        frames_by_session[frame["row"]["session"]].append(frame)
    frame_lookup = {}
    for session, group in frames_by_session.items():
        group.sort(key=lambda frame: (frame["center"], frame["row"]["image_file"]))
        frame_lookup[session] = (group, [frame["center"] for frame in group],
                                 max(max(frame["center"] - frame["low"], frame["high"] - frame["center"])
                                     for frame in group))
    scans, seen = [], set()
    for scan in context["wifi_scans"]:
        timestamp = finite_number(scan.get("elapsed_ts"))
        if scan.get("session") not in sessions or not math.isfinite(timestamp) or timestamp < 0:
            continue
        try:
            if measurement_gap_ms(scan, timestamp) > MAX_WIFI_AGE_MS:
                continue
            key = measurement_key(scan)
        except (TypeError, ValueError, AttributeError):
            continue
        if key in seen:
            continue
        group, centers, width = frame_lookup[scan["session"]]
        low, high = bisect_left(centers, timestamp - width), bisect_right(centers, timestamp + width)
        midpoint = bisect_left(centers, timestamp)
        indices = {low + index for index in _even_indices(high - low, 32)}
        indices.update(range(max(low, midpoint - 16), min(high, midpoint + 17)))
        candidates = [group[index] for index in indices if group[index]["low"] <= timestamp <= group[index]["high"]]
        candidates.sort(key=lambda frame: (abs(frame["center"] - timestamp), frame["row"]["image_file"]))
        if not candidates:
            continue
        seen.add(key)
        scans.append((scan, timestamp, candidates[:4]))
    scans.sort(key=lambda item: (item[0]["session"], item[1], str(item[0].get("scan_index", ""))))
    summary["fuzzy_wifi_valid_scan_count"] = len(scans)
    wifi_budget = wifi_budget or WifiRetrievalBudget(wifi_neighbors)
    scans = wifi_budget.representatives(scans)
    summary.update(fuzzy_wifi_representative_scan_count=len(scans),
                   fuzzy_wifi_omitted_scan_count=summary["fuzzy_wifi_valid_scan_count"] - len(scans))
    ranked = []
    for index, (scan, timestamp, _, _) in enumerate(scans):
        for other_index in range(index + 1, len(scans)):
            other, other_time, _, _ = scans[other_index]
            if scan["session"] != other["session"] or abs(timestamp - other_time) < 15000:
                continue
            summary["fuzzy_wifi_pair_comparison_count"] += 1
            try:
                similarity = fingerprint_similarity(scan, other)
            except (TypeError, ValueError, AttributeError):
                continue
            if math.isfinite(similarity["score"]) and similarity["score"] > 0:
                ranked.append((-similarity["score"], index, other_index, similarity))
    degrees = defaultdict(int)
    limit = min(3, max(0, wifi_neighbors))
    for negative_score, index, other_index, similarity in sorted(ranked, key=lambda item: item[:3]):
        if degrees[index] >= limit or degrees[other_index] >= limit:
            continue
        added = 0
        for frame in scans[index][2]:
            for other in scans[other_index][2]:
                if not (frame["fuzzy"] or other["fuzzy"]):
                    continue
                if max(frame["low"], other["low"]) <= min(frame["high"], other["high"]):
                    summary["fuzzy_wifi_ambiguous_window_pair_count"] += 1
                    continue
                score = -negative_score
                weight = .5 * score / (1 + (frame["uncertainty"] + other["uncertainty"]) / 10)
                time_a, time_b = scans[index][1], scans[other_index][1]
                if frame["row"]["image_file"] > other["row"]["image_file"]:
                    time_a, time_b = time_b, time_a
                if add(frame["row"], other["row"], "video_wifi_revisit", weight,
                       _wifi_scan_keys=(scans[index][3], scans[other_index][3]),
                       wifi_retrieval_score=score, wifi_rssi_rms_db=similarity["rssi_rms_db"],
                       wifi_shared_ap_count=similarity["shared_aps"], wifi_ap_overlap=similarity["ap_overlap"],
                       wifi_scan_a_elapsed_ts=time_a, wifi_scan_b_elapsed_ts=time_b,
                       **_evidence(frame, other, digest)):
                    added += 1
        if added:
            degrees[index] += 1
            degrees[other_index] += 1
            summary["fuzzy_wifi_scan_pair_count"] += 1
            summary["fuzzy_wifi_candidate_count"] += added
    return summary
