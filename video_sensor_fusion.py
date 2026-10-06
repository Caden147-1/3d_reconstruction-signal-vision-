"""Attach soft PDR/WiFi evidence using explicit video-to-sensor clock anchors.

Sensor timestamps are elapsed milliseconds; video timestamps are decoded seconds.
These annotations retrieve match candidates, never establish camera poses.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from trajectory_contour import PDR_ROUTE_FIELDS
from video_sensor_fuzzy import FUZZY_FIELDS
from wifi_observations import (
    MAX_WIFI_AGE_MS, WIFI_ASSOCIATION_FIELDS, WIFI_QUALITY_FIELDS,
    measurement_gap_ms, valid_fingerprint,
)


VIDEO_SENSOR_SYNC_FIELDS = [
    "video_sensor_sync_status", "video_sensor_sync_source_sha256",
    "video_sensor_sync_anchor_count", "video_sensor_sync_clock_scale",
    "video_sensor_sync_offset_ms", "video_sensor_sync_max_residual_ms",
    "video_sensor_sync_model", "sensor_elapsed_ts",
]
VIDEO_SENSOR_PRIOR_FIELDS = [
    *VIDEO_SENSOR_SYNC_FIELDS, *FUZZY_FIELDS, "session", *PDR_ROUTE_FIELDS,
    "trajectory_contour_id", "trajectory_route_distance_m",
    "trajectory_contour_distance_m", "trajectory_distance_m",
    "trajectory_position_source", "pdr_x", "pdr_y", "pdr_bind_source",
    "pdr_reference_elapsed_ts", "pdr_interp_status", "pdr_interp_gap_ms",
    "wifi_scan_index", "wifi_elapsed_ts", "wifi_gap_ms", "wifi_label_weight",
    *WIFI_QUALITY_FIELDS, *WIFI_ASSOCIATION_FIELDS,
]
_DECODED_TIME_SOURCES = {"opencv_decoded_timestamp", "ffprobe_frame_pts", "ffmpeg_pts", "decoded_pts"}
_CONTINUOUS_LINKS = {"observed_step_link", "stationary_checkpoint", "documented_checkpoint_pause"}


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def load_video_sensor_sync(path: Path) -> dict[str, dict[str, Any]]:
    """Validate named clock anchors. No filename/time-of-day inference is made.

    One anchor assumes equal clock rates and only solves the offset. Multiple
    anchors fit rate and offset, allowing at most 1% drift and 100 ms residual;
    their usable interval is bounded by the first and last supplied anchor.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Video sensor synchronization CSV not found: {path}")
    fields, rows = _read_csv(path)
    required = {"video_name", "session", "video_time_s", "sensor_elapsed_ts"}
    if not required.issubset(fields) or not rows:
        raise ValueError("Synchronization CSV requires nonempty video_name,session,video_time_s,sensor_elapsed_ts rows")
    grouped: dict[str, list[tuple[str, float, float]]] = {}
    for index, row in enumerate(rows, 2):
        name, session = row["video_name"].strip(), row["session"].strip()
        video, sensor = _number(row["video_time_s"]), _number(row["sensor_elapsed_ts"])
        if not name or not session or not all(math.isfinite(v) and v >= 0 for v in (video, sensor)):
            raise ValueError(f"Invalid synchronization anchor at CSV line {index}")
        grouped.setdefault(name, []).append((session, video, sensor))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {}
    for name, anchors in grouped.items():
        anchors.sort(key=lambda anchor: anchor[1])
        if len({anchor[0] for anchor in anchors}) != 1:
            raise ValueError(f"Video {name} has ambiguous sensor sessions")
        if any(b[1] <= a[1] or b[2] <= a[2] for a, b in zip(anchors, anchors[1:])):
            raise ValueError(f"Video {name} anchors must have distinct, increasing video and sensor times")
        count = len(anchors)
        mean_video = sum(a[1] * 1000 for a in anchors) / count
        mean_sensor = sum(a[2] for a in anchors) / count
        scale = 1.0
        if count > 1:
            if anchors[-1][1] - anchors[0][1] < 1.0:
                raise ValueError(f"Video {name} drift anchors must span at least one second")
            scale = sum((a[1] * 1000 - mean_video) * (a[2] - mean_sensor) for a in anchors) / sum(
                (a[1] * 1000 - mean_video) ** 2 for a in anchors)
        offset = mean_sensor - scale * mean_video
        residual = max(abs(a[2] - (offset + scale * a[1] * 1000)) for a in anchors)
        if not math.isfinite(scale) or not 0.99 <= scale <= 1.01 or residual > 100.0:
            raise ValueError(f"Video {name} synchronization exceeds 1% clock drift or 100 ms anchor residual")
        result[name] = dict(session=anchors[0][0], scale=scale, offset_ms=offset,
                            anchor_count=count, max_residual_ms=residual,
                            video_start_s=anchors[0][1], video_end_s=anchors[-1][1],
                            source_sha256=digest,
                            model="affine_anchors" if count > 1 else "offset_equal_clock_rate_assumption")
    return result


def clear_video_sensor_priors(row: Mapping[str, Any], reason: str = "unavailable") -> dict[str, Any]:
    """Remove stale route/WiFi bindings while retaining image/time provenance."""
    fields = [*VIDEO_SENSOR_PRIOR_FIELDS, "nearest_global_point_index", "left_global_point_index",
              "right_global_point_index", "route_ratio", "point_elapsed_ts", "point_index",
              "global_point_index", "pdr_z", "yaw_seed_deg", "pitch_seed_deg", "roll_seed_deg"]
    output = {**row, **{key: "" for key in fields}, **{key: "" for key in row if key.startswith("ap_")}}
    output.update(bind_type="visual_only", wifi_label_weight=0.0,
                  video_sensor_sync_status=reason, trajectory_position_source="unavailable",
                  pdr_bind_source="video_sensor_sync_unavailable")
    return output


def has_valid_video_sensor_sync(row: Mapping[str, Any]) -> bool:
    """Check the explicit-anchor provenance carried by the annotation step."""
    timestamp, scale = _number(row.get("sensor_elapsed_ts")), _number(row.get("video_sensor_sync_clock_scale"))
    video_time, offset = _number(row.get("video_time_s")), _number(row.get("video_sensor_sync_offset_ms"))
    count, residual = _number(row.get("video_sensor_sync_anchor_count")), _number(row.get("video_sensor_sync_max_residual_ms"))
    digest = str(row.get("video_sensor_sync_source_sha256", ""))
    return (row.get("video_sensor_sync_status") == "explicit_anchors"
            and bool(row.get("session")) and math.isfinite(timestamp) and timestamp >= 0
            and 0.99 <= scale <= 1.01 and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and math.isfinite(count) and count >= 1 and count.is_integer()
            and math.isfinite(video_time) and video_time >= 0 and math.isfinite(offset)
            and abs(timestamp - (offset + scale * video_time * 1000)) <= 1e-3
            and 0 <= residual <= 100
            and row.get("video_sensor_sync_model") == ("affine_anchors" if count > 1
                                                      else "offset_equal_clock_rate_assumption")
            and row.get("video_time_source") in _DECODED_TIME_SOURCES)


def _route_at(vertices: Sequence[Mapping[str, Any]], timestamp: float) -> dict[str, Any]:
    times = [_number(row.get("elapsed_ts")) for row in vertices]
    index = bisect.bisect_left(times, timestamp)
    if index < len(vertices) and abs(times[index] - timestamp) <= 1e-6:
        # Conflicting contour samples at one clock tick are ambiguous.
        end = bisect.bisect_right(times, timestamp)
        if any(vertices[other] != vertices[index] for other in range(index + 1, end)):
            return {}
        left = right = vertices[index]
        fraction, status, gap = 0.0, "exact_sensor_sample", 0.0
    elif 0 < index < len(vertices):
        left, right = vertices[index - 1], vertices[index]
        gap = times[index] - times[index - 1]
        if (not left.get("contour_id") or left.get("contour_id") != right.get("contour_id")
                or right.get("link_status") not in _CONTINUOUS_LINKS or gap <= 0):
            return {}
        # A documented stationary pause may be long; all other gaps are bounded.
        if gap > 10000 and right.get("link_status") != "documented_checkpoint_pause":
            return {}
        fraction, status = (timestamp - times[index - 1]) / gap, "interpolated_continuous_contour"
    else:
        return {}
    if not left.get("contour_id"):
        return {}

    def interpolate(key: str) -> Any:
        a, b = _number(left.get(key)), _number(right.get(key))
        return round(a + fraction * (b - a), 6) if math.isfinite(a) and math.isfinite(b) else ""

    result: dict[str, Any] = dict(
        trajectory_contour_id=left["contour_id"],
        trajectory_contour_distance_m=interpolate("contour_distance_m"),
        trajectory_distance_m=interpolate("trajectory_distance_m"),
        trajectory_position_source="synchronized_video_sensor_timestamp",
        pdr_bind_source="explicit_video_sensor_clock_and_continuous_pdr",
        pdr_reference_elapsed_ts=round(timestamp, 6), pdr_interp_status=status,
        pdr_interp_gap_ms=round(gap, 6),
    )
    for field in PDR_ROUTE_FIELDS:
        result[field] = left.get(field, "") if left.get(field) == right.get(field) else ""
    if left.get("route_distance_status") == right.get("route_distance_status") == "estimated_steps":
        result["route_distance_m"] = interpolate("route_distance_m")
        result["trajectory_route_distance_m"] = interpolate("trajectory_route_distance_m")
    # Handheld/legacy phone headings cannot establish camera translation or yaw.
    if all(vertex.get("pdr_heading_policy") == "phone-aligned"
           and vertex.get("pdr_position_status") == "phone_aligned_assumption" for vertex in (left, right)):
        result.update(pdr_x=interpolate("pdr_x"), pdr_y=interpolate("pdr_y"))
    return result


def annotate_video_sensor_priors(processed_dir: Path, sync_path: Path | None = None) -> dict[str, Any]:
    """Annotate the selected video manifest in place; absent sync stays visual."""
    processed_dir = Path(processed_dir)
    models = load_video_sensor_sync(sync_path) if sync_path is not None else {}
    fields, frames = _read_csv(processed_dir / "video_frames_manifest.csv")
    _, vertices = _read_csv(processed_dir / "pdr_trajectory_contour.csv")
    wifi_fields, scans = _read_csv(processed_dir / "wifi_pdr_features.csv")
    by_session: dict[str, list[dict[str, str]]] = {}
    wifi_by_session: dict[str, list[dict[str, str]]] = {}
    for vertex in vertices:
        if vertex.get("session") and math.isfinite(_number(vertex.get("elapsed_ts"))):
            by_session.setdefault(vertex["session"], []).append(vertex)
    for group in by_session.values():
        group.sort(key=lambda row: _number(row["elapsed_ts"]))
    for scan in scans:
        if scan.get("session") and valid_fingerprint(scan):
            wifi_by_session.setdefault(scan["session"], []).append(scan)
    output = []
    for frame in frames:
        row = clear_video_sensor_priors(frame)
        model = models.get(str(frame.get("video_name", "")))
        video_time = _number(frame.get("video_time_s"))
        if model is None:
            row["video_sensor_sync_status"] = "no_explicit_sync"
        elif frame.get("video_time_source") not in _DECODED_TIME_SOURCES or not math.isfinite(video_time) or video_time < 0:
            row["video_sensor_sync_status"] = "decoded_video_timestamp_unavailable"
        elif model["anchor_count"] > 1 and not model["video_start_s"] <= video_time <= model["video_end_s"]:
            row["video_sensor_sync_status"] = "outside_sync_anchor_interval"
        else:
            timestamp = model["offset_ms"] + model["scale"] * video_time * 1000
            if timestamp < 0:
                row["video_sensor_sync_status"] = "sensor_timestamp_before_clock_origin"
                output.append(row)
                continue
            row.update(session=model["session"], sensor_elapsed_ts=round(timestamp, 6),
                       video_sensor_sync_status="explicit_anchors",
                       video_sensor_sync_source_sha256=model["source_sha256"],
                       video_sensor_sync_anchor_count=model["anchor_count"],
                       video_sensor_sync_clock_scale=model["scale"],
                       video_sensor_sync_offset_ms=model["offset_ms"],
                       video_sensor_sync_max_residual_ms=model["max_residual_ms"],
                       video_sensor_sync_model=model["model"],
                       bind_type="synchronized_video_sensor_priors")
            row.update(_route_at(by_session.get(model["session"], []), timestamp))
            eligible = [(measurement_gap_ms(scan, timestamp), scan)
                        for scan in wifi_by_session.get(model["session"], [])]
            eligible = [pair for pair in eligible if pair[0] <= MAX_WIFI_AGE_MS]
            if eligible:
                gap, scan = min(eligible, key=lambda pair: pair[0])
                row.update({key: scan.get(key, "") for key in [*WIFI_QUALITY_FIELDS, *wifi_fields]
                            if key in WIFI_QUALITY_FIELDS or key.startswith("ap_")})
                row.update(wifi_scan_index=scan.get("scan_index", ""), wifi_elapsed_ts=scan.get("elapsed_ts", ""),
                           wifi_gap_ms=round(gap, 6), wifi_label_weight=1.0,
                           wifi_association_basis="explicit_video_sensor_timestamp_fresh_measurements")
        output.append(row)
    summary = dict(status="annotated" if models else "visual_only_no_sync", frame_count=len(output),
                   sync_source=str(Path(sync_path).resolve()) if sync_path else None,
                   models=models, synchronized_frame_count=sum(has_valid_video_sensor_sync(row) for row in output),
                   route_frame_count=sum(row.get("trajectory_position_source") == "synchronized_video_sensor_timestamp" for row in output),
                   wifi_frame_count=sum(valid_fingerprint(row) for row in output),
                   status_counts=dict(Counter(row["video_sensor_sync_status"] for row in output)),
                   camera_orientation_from_phone=False, requires_visual_geometric_verification=True,
                   interpolation_policy="continuous_contour_only; no cross-break interpolation; no multi-anchor extrapolation")
    if fields:
        output_fields = list(dict.fromkeys([*fields, *VIDEO_SENSOR_PRIOR_FIELDS,
                                           *(key for key in wifi_fields if key.startswith("ap_"))]))
        with (processed_dir / "video_frames_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(output)
    processed_dir.mkdir(parents=True, exist_ok=True)
    sync_copy = processed_dir / "video_sensor_sync.csv"
    if sync_path is not None and Path(sync_path).resolve() != sync_copy.resolve():
        shutil.copy2(sync_path, sync_copy)
    elif sync_path is None and sync_copy.exists():
        sync_copy.unlink()
    (processed_dir / "video_sensor_sync_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("video_sensor_fuzzy_context.json", "video_sensor_fuzzy_summary.json"):
        (processed_dir / name).unlink(missing_ok=True)
    return summary
