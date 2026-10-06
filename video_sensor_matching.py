"""Soft matching proposals from explicit clocks or authenticated fuzzy windows."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
import math

from video_sensor_fusion import has_valid_video_sensor_sync
from wifi_observations import (
    MAX_WIFI_AGE_MS, finite_number, fingerprint_similarity, measurement_gap_ms,
)


def _sample_indices(start, stop, limit):
    count = stop - start
    if count <= limit:
        return range(start, stop)
    return sorted({start + round(index * (count - 1) / (limit - 1)) for index in range(limit)})


def build_video_sensor_candidates(rows, existing_pairs=(), *, enabled=True,
                                  wifi_enabled=True, route_distance_m=8.,
                                  route_neighbors=4, wifi_neighbors=3, fuzzy_context_path=None):
    """Add soft candidates; neither route distance nor RSSI certifies a match.

    Route retrieval stays inside one continuous IMU contour. RF retrieval uses
    independent fresh measurements and a bounded sample of frames per scan.
    Optional fuzzy context retains separate uncertainty and assessment counts;
    both modes share pair deduplication, route degree and RF retrieval budgets.
    Missing sensors never remove the ordinary visual/temporal candidate graph.
    """
    eligible = [row for row in rows if enabled and row.get("image_file")
                and row.get("source_type") == "video_base_frame" and has_valid_video_sensor_sync(row)]
    existing = {tuple(sorted(pair)) for pair in existing_pairs}
    edges, route_groups, wifi_groups = [], defaultdict(list), defaultdict(list)
    route_degrees = defaultdict(int)
    from video_sensor_fuzzy_matching import WifiRetrievalBudget, add_fuzzy_video_sensor_candidates, measurement_key
    wifi_budget = WifiRetrievalBudget(wifi_neighbors)

    def add(a, b, reason, score, **evidence):
        pair = tuple(sorted((a["image_file"], b["image_file"])))
        if pair[0] == pair[1] or pair in existing:
            return False
        if reason == "video_imu_route" and any(route_degrees[name] >= route_neighbors for name in pair):
            return False
        scan_keys = evidence.pop("_wifi_scan_keys", None)
        if reason == "video_wifi_revisit" and not wifi_budget.accept(scan_keys, (a["image_file"], b["image_file"])):
            return False
        existing.add(pair)
        if reason == "video_imu_route":
            for name in pair:
                route_degrees[name] += 1
        basis = evidence.pop("edge_weight_basis", "explicit_sync_soft_candidate_requires_visual_geometry")
        edges.append(dict(image_a=pair[0], image_b=pair[1], reason=reason,
                          edge_weight=round(score, 6),
                          edge_weight_basis=basis, **evidence))
        return True

    for row in eligible:
        distance = finite_number(row.get("trajectory_route_distance_m"))
        if (row.get("trajectory_position_source") == "synchronized_video_sensor_timestamp"
                and row.get("trajectory_contour_id") and math.isfinite(distance) and distance >= 0):
            route_groups[(row["session"], row["trajectory_contour_id"])].append(row)
        timestamp = finite_number(row.get("sensor_elapsed_ts"))
        if wifi_enabled and measurement_gap_ms(row, timestamp) <= MAX_WIFI_AGE_MS:
            try:
                key = measurement_key(row)
            except (TypeError, ValueError, AttributeError, KeyError):
                continue
            wifi_groups[key].append(row)

    for group in route_groups.values():
        group.sort(key=lambda row: (float(row["trajectory_route_distance_m"]), float(row["sensor_elapsed_ts"]), row["image_file"]))
        distances = [float(row["trajectory_route_distance_m"]) for row in group]
        for index, row in enumerate(group):
            if route_neighbors <= 0 or route_distance_m <= 0:
                continue
            low = bisect_left(distances, distances[index] - route_distance_m)
            high = bisect_right(distances, distances[index] + route_distance_m)
            candidates = [group[j] for j in _sample_indices(low, high, 32) if j != index]
            candidates.sort(key=lambda other: (abs(float(other["trajectory_route_distance_m"]) - distances[index]),
                                               abs(float(other["sensor_elapsed_ts"]) - float(row["sensor_elapsed_ts"])),
                                               other["image_file"]))
            count = 0
            for other in candidates:
                gap = abs(float(other["trajectory_route_distance_m"]) - distances[index])
                if add(row, other, "video_imu_route", 0.7 / (1. + gap), trajectory_distance_m=gap):
                    count += 1
                if count >= route_neighbors:
                    break

    # Rank independent scans rather than every frame pair. Repeated use of one
    # radio measurement cannot establish a revisit; fingerprint_similarity checks it.
    groups = sorted(wifi_groups.items())
    scan_groups = [sorted(group, key=lambda row: (float(row["sensor_elapsed_ts"]), row["image_file"]))
                   for _, group in groups]
    scan_entries = wifi_budget.representatives(
        [(group[0], float(group[0]["sensor_elapsed_ts"]), group) for group in scan_groups])
    representatives = [item[2] for item in scan_entries]
    ranked = defaultdict(list)
    for index, left in enumerate(representatives):
        for other_index in range(index + 1, len(representatives)):
            right = representatives[other_index]
            a, b = left[0], right[0]
            if a["session"] != b["session"] or abs(float(a["sensor_elapsed_ts"]) - float(b["sensor_elapsed_ts"])) < 15000:
                continue
            try:
                similarity = fingerprint_similarity(a, b)
            except (TypeError, ValueError, AttributeError):
                # Invalid optional RF metadata must not remove temporal/visual
                # matching or turn malformed measurements into a position prior.
                continue
            if not math.isfinite(similarity["score"]) or similarity["score"] <= 0:
                continue
            ranked[index].append((-similarity["score"], other_index, similarity))
            ranked[other_index].append((-similarity["score"], index, similarity))
    scan_pairs = {}
    for index, candidates in ranked.items():
        for negative_score, other, similarity in sorted(candidates, key=lambda item: (item[0], item[1]))[:max(0, wifi_neighbors)]:
            scan_pairs[tuple(sorted((index, other)))] = (-negative_score, similarity)
    for (index, other), (score, similarity) in sorted(scan_pairs.items()):
        left, right = representatives[index], representatives[other]
        for i in _sample_indices(0, len(left), 4):
            for j in _sample_indices(0, len(right), 4):
                add(left[i], right[j], "video_wifi_revisit", score,
                    _wifi_scan_keys=(scan_entries[index][3], scan_entries[other][3]),
                    wifi_retrieval_score=score, wifi_rssi_rms_db=similarity["rssi_rms_db"],
                    wifi_shared_ap_count=similarity["shared_aps"], wifi_ap_overlap=similarity["ap_overlap"])
    fuzzy_summary = add_fuzzy_video_sensor_candidates(
        rows, fuzzy_context_path, add, enabled=enabled, wifi_enabled=wifi_enabled,
        route_distance_m=route_distance_m, route_neighbors=route_neighbors, wifi_neighbors=wifi_neighbors,
        wifi_budget=wifi_budget)
    fuzzy_diagnostics = {key.removeprefix("fuzzy_"): value for key, value in fuzzy_summary.items()}
    fuzzy_diagnostics["retained_frame_count"] = fuzzy_summary["fuzzy_frame_count"]
    fuzzy_diagnostics["context_sha256"] = fuzzy_summary.get("fuzzy_context_sha256", "")
    summary = dict(status="disabled" if not enabled else "evaluated" if eligible else
                   "evaluated_fuzzy_time_windows" if fuzzy_summary["fuzzy_frame_count"] else "no_synchronized_video_frames",
                   synchronized_frame_count=len(eligible), route_group_count=len(route_groups),
                   wifi_scan_count=len(wifi_groups), candidate_pair_count=len(edges),
                   shared_wifi_representative_scan_count=len(wifi_budget.scans),
                   shared_wifi_max_frames_per_scan=wifi_budget.max_frames,
                   shared_wifi_max_neighbors_per_scan=wifi_budget.max_neighbors,
                   route_candidate_count=sum(row["reason"] == "video_imu_route" for row in edges),
                   wifi_candidate_count=sum(row["reason"] == "video_wifi_revisit" for row in edges),
                   route_neighbors_per_frame=route_neighbors, wifi_neighbors_per_scan=wifi_neighbors,
                   max_pairs_per_wifi_scan_pair=16, route_distance_limit_m=route_distance_m,
                   requires_visual_geometric_verification=True, camera_pose_constraints=False,
                   fuzzy=fuzzy_diagnostics,
                   **fuzzy_summary,
                   interpretation="Soft matching assistance only; PDR distances are step-model estimates, RSSI does not measure camera position.")
    return edges, summary
