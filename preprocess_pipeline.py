"""Prepare video frames, optional WiFi/IMU tables and a COLMAP workspace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from colmap_bridge import prepare_colmap_workspace
from reconstruction_export import export_reconstruction_dataset
from trajectory_contour import PDR_ROUTE_FIELDS, build_trajectory_contour
from video_frames import run_video_frame_pipeline
from wifi_positioning import run_wifi_positioning_pipeline
from wifi_observations import MAX_WIFI_AGE_MS, MIN_WIFI_APS, WIFI_QUALITY_FIELDS, finite_number
from workspace_lock import WorkspaceBusyError, workspace_lock
from workspace_inputs import WorkspaceInputsProtectedError, require_fresh_workspace
from preprocess_workers import run_sensor_stages


FILL_RSSI = -110
DEFAULT_BASE_STEP_M = 0.70
MAX_HEADING_GAP_MS = 1000


class HeadingUnavailableError(ValueError):
    """A detected step has no sufficiently recent finite heading measurement."""


PROFILE_DEFAULTS = {
    "preview": {
        "video_frame_interval_s": 1.5,
        "video_max_frames": 120,
        "video_blur_threshold": 80.0,
        "video_keyframe_min_change": 6.0,
        "video_keyframe_min_gap": 2,
        "colmap_dense_max_image_size": 1200,
        "colmap_feature_max_image_size": 1600,
        "colmap_max_num_features": 6000,
        "colmap_max_num_matches": 32768,
        "colmap_matcher": "constrained",
        "colmap_quality": "high",
        "colmap_dense_profile": "balanced",
        "colmap_num_threads": 0,
        "constrained_max_pdr_distance_m": 8.0,
        "constrained_video_frame_window": 3,
        "image_similarity_max_pairs": 1000,
        "image_similarity_max_image_size": 720,
        "image_similarity_orb_features": 800,
        "image_similarity_min_visual_score": 0.22,
    },
    "balanced": {
        "video_frame_interval_s": 1.5,
        "video_max_frames": 160,
        "video_blur_threshold": 75.0,
        "video_keyframe_min_change": 5.5,
        "video_keyframe_min_gap": 2,
        "colmap_dense_max_image_size": 1400,
        "colmap_feature_max_image_size": 1800,
        "colmap_max_num_features": 8000,
        "colmap_max_num_matches": 32768,
        "colmap_matcher": "constrained",
        "colmap_quality": "high",
        "colmap_dense_profile": "balanced",
        "colmap_num_threads": 0,
        "constrained_max_pdr_distance_m": 8.0,
        "constrained_video_frame_window": 4,
        "image_similarity_max_pairs": 2500,
        "image_similarity_max_image_size": 800,
        "image_similarity_orb_features": 1200,
        "image_similarity_min_visual_score": 0.22,
    },
    "final": {
        "video_frame_interval_s": 1.0,
        "video_max_frames": 240,
        "video_blur_threshold": 60.0,
        "video_keyframe_min_change": 4.0,
        "video_keyframe_min_gap": 1,
        "colmap_dense_max_image_size": 2000,
        "colmap_feature_max_image_size": 2400,
        "colmap_max_num_features": 12000,
        "colmap_max_num_matches": 65536,
        "colmap_matcher": "constrained",
        "colmap_quality": "extreme",
        "colmap_dense_profile": "complete",
        "colmap_num_threads": 0,
        "constrained_max_pdr_distance_m": 12.0,
        "constrained_video_frame_window": 6,
        "image_similarity_max_pairs": 0,
        "image_similarity_max_image_size": 900,
        "image_similarity_orb_features": 1500,
        "image_similarity_min_visual_score": 0.22,
    },
    "rtx4080": {
        "video_frame_interval_s": 0.75,
        "video_max_frames": 360,
        "video_blur_threshold": 50.0,
        "video_keyframe_min_change": 3.0,
        "video_keyframe_min_gap": 1,
        "colmap_dense_max_image_size": 2400,
        "colmap_feature_max_image_size": 2800,
        "colmap_max_num_features": 16000,
        "colmap_max_num_matches": 98304,
        "colmap_matcher": "constrained",
        "colmap_quality": "extreme",
        "colmap_dense_profile": "rtx4080",
        "colmap_num_threads": 0,
        "constrained_max_pdr_distance_m": 14.0,
        "constrained_video_frame_window": 7,
        "image_similarity_max_pairs": 0,
        "image_similarity_max_image_size": 1000,
        "image_similarity_orb_features": 1800,
        "image_similarity_min_visual_score": 0.22,
    },
    "complete": {
        "video_frame_interval_s": 0.5,
        "video_max_frames": 0,
        "video_blur_threshold": 40.0,
        "video_keyframe_min_change": 2.0,
        "video_keyframe_min_gap": 1,
        "colmap_dense_max_image_size": 3200,
        "colmap_feature_max_image_size": 3200,
        "colmap_max_num_features": 20000,
        "colmap_max_num_matches": 131072,
        "colmap_matcher": "constrained",
        "colmap_quality": "extreme",
        "colmap_dense_profile": "complete",
        "colmap_num_threads": 0,
        "constrained_max_pdr_distance_m": 16.0,
        "constrained_video_frame_window": 8,
        "image_similarity_max_pairs": 0,
        "image_similarity_max_image_size": 1100,
        "image_similarity_orb_features": 2000,
        "image_similarity_min_visual_score": 0.22,
    },
}


@dataclass
class Event:
    session: str
    elapsed_ts: int
    epoch_ts: int
    event: str


@dataclass
class Segment:
    session: str
    segment_id: int
    start_elapsed_ts: int
    start_epoch_ts: int
    end_elapsed_ts: int
    end_epoch_ts: int
    start_event: str
    end_event: str

    @property
    def duration_s(self) -> float:
        return (self.end_elapsed_ts - self.start_elapsed_ts) / 1000.0


@dataclass
class SensorSample:
    elapsed_ts: int
    epoch_ts: int
    vals: List[float]
    acc: Optional[int] = None


@dataclass
class Step:
    session: str
    segment_id: int
    step_index: int
    elapsed_ts: int
    epoch_ts: int
    peak_accel: float
    accel_range: float
    step_length_m: float
    heading_rad: float
    heading_deg: float
    pdr_x: float
    pdr_y: float
    route_distance_m: float = 0.0
    pdr_heading_policy: str = "handheld"
    pdr_heading_status: str = "available_uncalibrated"
    pdr_position_status: str = "diagnostic_phone_heading"


def normalize_mac(mac: str) -> str:
    return "".join(ch for ch in mac.upper() if ch.isalnum())


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_tab_json(path: Path) -> Iterable[Tuple[int, int, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                print(f"[warn] skip malformed line {path}:{line_no}")
                continue
            try:
                elapsed_ts = int(parts[0])
                epoch_ts = int(parts[1])
                payload = json.loads(parts[2])
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"[warn] skip unparsable line {path}:{line_no}: {exc}")
                continue
            yield elapsed_ts, epoch_ts, payload


def read_events(session_dir: Path) -> List[Event]:
    rows: List[Event] = []
    for elapsed_ts, epoch_ts, payload in parse_tab_json(session_dir / "movements.dat"):
        rows.append(
            Event(
                session=session_dir.name,
                elapsed_ts=elapsed_ts,
                epoch_ts=epoch_ts,
                event=str(payload),
            )
        )
    return rows


def build_segments(events: Sequence[Event]) -> List[Segment]:
    segments: List[Segment] = []
    start: Optional[Event] = None
    for event in events:
        if event.event in {"START", "RESTART"}:
            start = event
        elif event.event in {"CHECKPOINT_PAUSE", "STOP"} and start is not None:
            segments.append(
                Segment(
                    session=event.session,
                    segment_id=len(segments) + 1,
                    start_elapsed_ts=start.elapsed_ts,
                    start_epoch_ts=start.epoch_ts,
                    end_elapsed_ts=event.elapsed_ts,
                    end_epoch_ts=event.epoch_ts,
                    start_event=start.event,
                    end_event=event.event,
                )
            )
            start = None
    return segments


def read_sensor(session_dir: Path, system_id: int) -> List[SensorSample]:
    samples: List[SensorSample] = []
    path = session_dir / f"{system_id}.dat"
    for elapsed_ts, epoch_ts, payload in parse_tab_json(path):
        vals = payload.get("vals", []) if isinstance(payload, dict) else []
        if not isinstance(vals, list):
            continue
        samples.append(
            SensorSample(
                elapsed_ts=elapsed_ts,
                epoch_ts=epoch_ts,
                vals=[safe_float(v) for v in vals],
                acc=payload.get("acc") if isinstance(payload, dict) else None,
            )
        )
    return samples


def median(values: Sequence[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default


def pstdev(values: Sequence[float], default: float = 0.0) -> float:
    return statistics.pstdev(values) if len(values) > 1 else default


def moving_average(points: Sequence[Tuple[int, int, float]], radius: int = 2) -> List[Tuple[int, int, float]]:
    if not points:
        return []
    smoothed: List[Tuple[int, int, float]] = []
    for idx, (elapsed_ts, epoch_ts, _value) in enumerate(points):
        lo = max(0, idx - radius)
        hi = min(len(points), idx + radius + 1)
        avg = sum(item[2] for item in points[lo:hi]) / (hi - lo)
        smoothed.append((elapsed_ts, epoch_ts, avg))
    return smoothed


def sensor_magnitude(sample: SensorSample) -> Optional[float]:
    if len(sample.vals) < 3 or any(math.isnan(v) for v in sample.vals[:3]):
        return None
    x, y, z = sample.vals[:3]
    return math.sqrt(x * x + y * y + z * z)


def detect_steps_in_segment(
    linear_accel: Sequence[SensorSample],
    segment: Segment,
    min_step_interval_ms: int = 320,
) -> List[Tuple[int, int, float, float]]:
    """Return (elapsed_ts, epoch_ts, peak_accel, accel_range) for detected steps."""
    raw_points: List[Tuple[int, int, float]] = []
    for sample in linear_accel:
        if sample.elapsed_ts < segment.start_elapsed_ts or sample.elapsed_ts > segment.end_elapsed_ts:
            continue
        mag = sensor_magnitude(sample)
        if mag is not None:
            raw_points.append((sample.elapsed_ts, sample.epoch_ts, mag))

    if len(raw_points) < 12:
        return []

    points = moving_average(raw_points, radius=2)
    values = [value for _elapsed_ts, _epoch_ts, value in points]
    med = median(values)
    deviations = [abs(value - med) for value in values]
    mad = median(deviations, default=pstdev(values, default=0.1))
    # Keep the threshold robust: high enough to avoid tiny hand jitter, but still
    # adaptive for short walks where only a few steps are present.
    threshold = med + max(0.65, 1.7 * mad)

    candidates: List[Tuple[int, int, float, float]] = []
    last_step_ts = -10**12
    for idx in range(1, len(points) - 1):
        elapsed_ts, epoch_ts, value = points[idx]
        prev_value = points[idx - 1][2]
        next_value = points[idx + 1][2]
        if value < threshold or value < prev_value or value <= next_value:
            continue
        if elapsed_ts - last_step_ts < min_step_interval_ms:
            if candidates and value > candidates[-1][2]:
                # Replace a weaker peak inside the same step interval.
                window_values = local_window_values(points, idx, width_ms=350)
                accel_range = value - min(window_values)
                candidates[-1] = (elapsed_ts, epoch_ts, value, accel_range)
                last_step_ts = elapsed_ts
            continue
        window_values = local_window_values(points, idx, width_ms=350)
        accel_range = value - min(window_values)
        candidates.append((elapsed_ts, epoch_ts, value, accel_range))
        last_step_ts = elapsed_ts

    return candidates


def local_window_values(points: Sequence[Tuple[int, int, float]], center_idx: int, width_ms: int) -> List[float]:
    center_ts = points[center_idx][0]
    half_width = width_ms // 2
    return [
        value
        for elapsed_ts, _epoch_ts, value in points
        if center_ts - half_width <= elapsed_ts <= center_ts + half_width
    ] or [points[center_idx][2]]


def quaternion_to_yaw(vals: Sequence[float]) -> Optional[float]:
    """Convert Android-style quaternion values [x, y, z, w] to yaw radians."""
    if len(vals) < 4 or any(math.isnan(v) for v in vals[:4]):
        return None
    x, y, z, w = vals[:4]
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        return None
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0 or not math.isfinite(norm):
        return None
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def read_heading_samples(session_dir: Path) -> List[Tuple[int, int, float]]:
    # Prefer GameRotationVector (systemId 11). It is magnetometer-free and often
    # smoother for relative PDR. Fall back to RotationVector (systemId 8).
    heading_samples: List[Tuple[int, int, float]] = []
    for system_id in (11, 8):
        heading_samples.clear()
        for sample in read_sensor(session_dir, system_id):
            yaw = quaternion_to_yaw(sample.vals)
            if yaw is not None and math.isfinite(yaw):
                heading_samples.append((sample.elapsed_ts, sample.epoch_ts, yaw))
        if heading_samples:
            return sorted(heading_samples)
    return []


def interpolate_angle(samples: Sequence[Tuple[int, int, float]], target_elapsed_ts: int, max_gap_ms: int = MAX_HEADING_GAP_MS) -> float:
    if not samples:
        raise HeadingUnavailableError("No valid rotation-vector heading measurements")
    if target_elapsed_ts <= samples[0][0]:
        if samples[0][0] - target_elapsed_ts > max_gap_ms or not math.isfinite(samples[0][2]):
            raise HeadingUnavailableError("Step precedes valid heading coverage")
        return samples[0][2]
    if target_elapsed_ts >= samples[-1][0]:
        if target_elapsed_ts - samples[-1][0] > max_gap_ms or not math.isfinite(samples[-1][2]):
            raise HeadingUnavailableError("Step exceeds valid heading coverage")
        return samples[-1][2]

    lo = 0
    hi = len(samples) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if samples[mid][0] < target_elapsed_ts:
            lo = mid + 1
        else:
            hi = mid - 1

    right = samples[lo]
    if right[0] == target_elapsed_ts and math.isfinite(right[2]):
        return right[2]
    left = samples[lo - 1]
    span = right[0] - left[0]
    if span > max_gap_ms or not all(math.isfinite(value) for value in (left[2], right[2])):
        raise HeadingUnavailableError("Heading sampling gap crosses a detected step")
    if span <= 0:
        return left[2]
    ratio = (target_elapsed_ts - left[0]) / span
    diff = wrap_angle(right[2] - left[2])
    return wrap_angle(left[2] + ratio * diff)


def wrap_angle(angle: float) -> float:
    if not math.isfinite(angle):
        raise ValueError("Heading angle must be finite")
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def compute_pdr(
    session_dir: Path,
    segments: Sequence[Segment],
    base_step_m: float,
    heading_offset_deg: float,
    heading_policy: str = "handheld",
) -> List[Step]:
    if heading_policy not in {"handheld", "phone-aligned"}:
        raise ValueError("Unknown PDR heading policy")
    linear_accel = read_sensor(session_dir, 7)
    headings = read_heading_samples(session_dir)

    detected: List[Tuple[Segment, int, int, float, float]] = []
    for segment in segments:
        for elapsed_ts, epoch_ts, peak_accel, accel_range in detect_steps_in_segment(linear_accel, segment):
            detected.append((segment, elapsed_ts, epoch_ts, peak_accel, accel_range))

    ranges = [max(item[4], 0.05) for item in detected]
    median_range = median(ranges, default=1.0)
    heading_offset = math.radians(heading_offset_deg)
    x = 0.0
    y = 0.0
    steps: List[Step] = []
    route_distance = 0.0
    position_available = True

    for step_index, (segment, elapsed_ts, epoch_ts, peak_accel, accel_range) in enumerate(detected, start=1):
        normalized_range = max(accel_range, 0.05) / max(median_range, 0.05)
        step_length = base_step_m * (normalized_range ** 0.25)
        step_length = min(1.10, max(0.35, step_length))
        route_distance += step_length
        try:
            heading = wrap_angle(interpolate_angle(headings, elapsed_ts) + heading_offset)
        except HeadingUnavailableError:
            heading = math.nan
            position_available = False

        # Phone yaw is not camera orientation or calibrated walking direction.
        # A missing heading makes subsequent accumulated XY unavailable, while
        # step counts and route distance remain usable without any heading.
        if position_available:
            x += step_length * math.sin(heading)
            y += step_length * math.cos(heading)
        else:
            x = y = math.nan
        steps.append(
            Step(
                session=session_dir.name,
                segment_id=segment.segment_id,
                step_index=step_index,
                elapsed_ts=elapsed_ts,
                epoch_ts=epoch_ts,
                peak_accel=peak_accel,
                accel_range=accel_range,
                step_length_m=step_length,
                heading_rad=heading,
                heading_deg=math.degrees(heading),
                pdr_x=x,
                pdr_y=y,
                route_distance_m=route_distance,
                pdr_heading_policy=heading_policy,
                pdr_heading_status="available_uncalibrated" if math.isfinite(heading) else "unavailable",
                pdr_position_status=("phone_aligned_assumption" if heading_policy == "phone-aligned" else "diagnostic_phone_heading") if position_available else "unavailable_heading",
            )
        )
    return steps


def nearest_pdr_position(steps: Sequence[Step], elapsed_ts: int) -> Tuple[float, float, int]:
    before = [step for step in steps if step.elapsed_ts <= elapsed_ts]
    if not before:
        return 0.0, 0.0, 0
    step = before[-1]
    return step.pdr_x, step.pdr_y, step.step_index


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_wifi_tables(
    session_dirs: Sequence[Path],
    output_dir: Path,
    min_detection_rate: float,
    min_mean_rssi: float,
    max_rssi_std: float,
    min_scan_count: int,
) -> Dict[str, Any]:
    long_rows: List[Dict[str, Any]] = []
    scan_rows: List[Dict[str, Any]] = []
    ap_stats_by_session: Dict[str, Dict[str, Dict[str, Any]]] = {}
    selected_by_session: Dict[str, List[str]] = {}
    session_scan_counts: Dict[str, int] = {}
    session_measurement_counts: Dict[str, int] = {}

    for session_dir in session_dirs:
        session = session_dir.name
        wifi_path = session_dir / "wifiScanner.dat"
        scan_count = 0
        stats: Dict[str, Dict[str, Any]] = {}
        seen_measurements: set[Tuple[str, int]] = set()
        measurement_scan_count = 0

        for scan_index, (elapsed_ts, epoch_ts, payload) in enumerate(parse_tab_json(wifi_path), start=1):
            if not isinstance(payload, list):
                continue
            scan_count += 1
            per_scan: Dict[str, float] = {}
            meta_by_mac: Dict[str, Dict[str, Any]] = {}
            measurement_times: Dict[str, int] = {}
            for entry_index, ap in enumerate(payload):
                if not isinstance(ap, dict) or not ap.get("mac"):
                    continue
                mac = normalize_mac(str(ap.get("mac", "")))
                if not mac:
                    continue
                rssi = safe_float(ap.get("dbm"))
                name = str(ap.get("name", ""))
                freq = ap.get("freq", "")
                elap_ts = ap.get("elap_ts", "")
                measurement_ts = finite_number(elap_ts)
                age_ms = elapsed_ts - measurement_ts
                valid = math.isfinite(rssi) and FILL_RSSI < rssi < 0
                if not valid or not math.isfinite(measurement_ts) or measurement_ts < 0:
                    observation_status = "invalid_measurement"
                elif not 0 <= age_ms <= MAX_WIFI_AGE_MS:
                    observation_status = "stale_or_future"
                elif (mac, int(measurement_ts)) in seen_measurements:
                    observation_status = "cached_duplicate"
                else:
                    observation_status = "fresh"
                long_rows.append(
                    {
                        "session": session,
                        "scan_index": scan_index,
                        "entry_index": entry_index,
                        "scan_elapsed_ts": elapsed_ts,
                        "scan_epoch_ts": epoch_ts,
                        "ap_elapsed_ts": elap_ts,
                        "mac": mac,
                        "ssid": name,
                        "freq": freq,
                        "rssi": round(rssi, 3) if math.isfinite(rssi) else "",
                        "measurement_age_ms": age_ms if math.isfinite(age_ms) else "",
                        "observation_status": observation_status,
                    }
                )
                if observation_status != "fresh":
                    continue
                seen_measurements.add((mac, int(measurement_ts)))
                if mac not in per_scan or rssi > per_scan[mac]:
                    per_scan[mac] = rssi
                    meta_by_mac[mac] = {"ssid": name, "freq": freq}
                    measurement_times[mac] = int(measurement_ts)

                stat = stats.setdefault(
                    mac,
                    {
                        "rssis": [],
                        "scan_indices": set(),
                        "names": {},
                        "freqs": {},
                        "first_elapsed_ts": int(measurement_ts),
                        "last_elapsed_ts": int(measurement_ts),
                    },
                )
                stat["rssis"].append(rssi)
                stat["scan_indices"].add(scan_index)
                stat["names"][name] = stat["names"].get(name, 0) + 1
                stat["freqs"][freq] = stat["freqs"].get(freq, 0) + 1
                stat["last_elapsed_ts"] = int(measurement_ts)

            measurement_scan_count += bool(per_scan)
            scan_rows.append(
                {
                    "session": session,
                    "scan_index": scan_index,
                    "elapsed_ts": elapsed_ts,
                    "epoch_ts": epoch_ts,
                    "rssi_by_mac": per_scan,
                    "meta_by_mac": meta_by_mac,
                    "measurement_times": measurement_times,
                }
            )

        session_scan_counts[session] = scan_count
        session_measurement_counts[session] = measurement_scan_count
        ap_stats_by_session[session] = {}
        selected_by_session[session] = []
        for mac, stat in stats.items():
            rssis = stat["rssis"]
            scan_indices = stat["scan_indices"]
            mean_rssi = sum(rssis) / len(rssis)
            std_rssi = pstdev(rssis)
            detection_rate = len(scan_indices) / measurement_scan_count if measurement_scan_count else 0.0
            top_name = most_common_key(stat["names"])
            top_freq = most_common_key(stat["freqs"])
            row = {
                "session": session,
                "mac": mac,
                "ssid": top_name,
                "freq": top_freq,
                "scan_count": len(scan_indices),
                "entry_count": len(rssis),
                "total_scans": measurement_scan_count,
                "detection_rate": detection_rate,
                "mean_rssi": mean_rssi,
                "median_rssi": median(rssis),
                "std_rssi": std_rssi,
                "min_rssi": min(rssis),
                "max_rssi": max(rssis),
                "first_elapsed_ts": stat["first_elapsed_ts"],
                "last_elapsed_ts": stat["last_elapsed_ts"],
            }
            ap_stats_by_session[session][mac] = row
            if (
                len(scan_indices) >= min_scan_count
                and detection_rate >= min_detection_rate
                and mean_rssi >= min_mean_rssi
                and std_rssi <= max_rssi_std
            ):
                selected_by_session[session].append(mac)

        selected_by_session[session].sort(
            key=lambda mac: (
                -ap_stats_by_session[session][mac]["detection_rate"],
                -ap_stats_by_session[session][mac]["mean_rssi"],
                mac,
            )
        )

    ap_stats_rows: List[Dict[str, Any]] = []
    for session, stats in ap_stats_by_session.items():
        for mac, row in sorted(
            stats.items(),
            key=lambda item: (-item[1]["detection_rate"], -item[1]["mean_rssi"], item[0]),
        ):
            ap_stats_rows.append(format_stat_row(row))

    selected_union = sorted({mac for macs in selected_by_session.values() for mac in macs})
    selected_meta: Dict[str, Dict[str, Any]] = {}
    for session, macs in selected_by_session.items():
        for mac in macs:
            stat = ap_stats_by_session[session][mac]
            selected_meta.setdefault(
                mac,
                {
                    "mac": mac,
                    "ssid_by_session": {},
                    "freq_by_session": {},
                    "sessions": [],
                },
            )
            selected_meta[mac]["ssid_by_session"][session] = stat["ssid"]
            selected_meta[mac]["freq_by_session"][session] = stat["freq"]
            selected_meta[mac]["sessions"].append(session)

    write_csv(
        output_dir / "wifi_long.csv",
        [
            "session",
            "scan_index",
            "entry_index",
            "scan_elapsed_ts",
            "scan_epoch_ts",
            "ap_elapsed_ts",
            "mac",
            "ssid",
            "freq",
            "rssi",
            "measurement_age_ms",
            "observation_status",
        ],
        long_rows,
    )

    write_csv(
        output_dir / "ap_stats.csv",
        [
            "session",
            "mac",
            "ssid",
            "freq",
            "scan_count",
            "entry_count",
            "total_scans",
            "detection_rate",
            "mean_rssi",
            "median_rssi",
            "std_rssi",
            "min_rssi",
            "max_rssi",
            "first_elapsed_ts",
            "last_elapsed_ts",
            "selected",
        ],
        (
            {
                **row,
                "selected": int(row["mac"] in selected_by_session.get(row["session"], [])),
            }
            for row in ap_stats_rows
        ),
    )

    wide_fieldnames = ["session", "scan_index", "elapsed_ts", "epoch_ts", *WIFI_QUALITY_FIELDS] + [
        f"ap_{mac}" for mac in selected_union
    ]
    wide_rows: List[Dict[str, Any]] = []
    for scan in scan_rows:
        session = scan["session"]
        selected_for_session = set(selected_by_session.get(session, []))
        row = {
            "session": session,
            "scan_index": scan["scan_index"],
            "elapsed_ts": scan["elapsed_ts"],
            "epoch_ts": scan["epoch_ts"],
        }
        for mac in selected_union:
            row[f"ap_{mac}"] = (
                round(scan["rssi_by_mac"].get(mac, FILL_RSSI), 3)
                if mac in selected_for_session
                else FILL_RSSI
            )
        times = [ts for mac, ts in scan["measurement_times"].items() if mac in selected_for_session]
        row.update({
            "wifi_observation_status": "fresh" if len(times) >= MIN_WIFI_APS else "insufficient_fresh_aps",
            "wifi_measurement_min_elapsed_ts": min(times) if times else "",
            "wifi_measurement_max_elapsed_ts": max(times) if times else "",
            "wifi_fresh_ap_count": len(times),
            "wifi_max_age_ms": scan["elapsed_ts"] - min(times) if times else "",
        })
        wide_rows.append(row)

    write_csv(output_dir / "wifi_wide_selected.csv", wide_fieldnames, wide_rows)

    selection_obj = {
        "thresholds": {
            "min_detection_rate": min_detection_rate,
            "min_mean_rssi": min_mean_rssi,
            "max_rssi_std": max_rssi_std,
            "min_scan_count": min_scan_count,
            "missing_rssi_fill": FILL_RSSI,
            "max_measurement_age_ms": MAX_WIFI_AGE_MS,
            "min_fresh_aps_per_fingerprint": MIN_WIFI_APS,
        },
        "session_scan_counts": session_scan_counts,
        "session_independent_measurement_record_counts": session_measurement_counts,
        "statistics_basis": "unique_fresh_ap_timestamps_not_repeated_logger_records",
        "observation_status_counts": dict(Counter(row["observation_status"] for row in long_rows)),
        "selected_ap_count_by_session": {
            session: len(macs) for session, macs in selected_by_session.items()
        },
        "selected_aps_by_session": selected_by_session,
        "selected_ap_union_count": len(selected_union),
        "selected_ap_union": selected_union,
        "selected_ap_metadata": selected_meta,
    }
    write_json(output_dir / "selected_aps.json", selection_obj)
    return selection_obj


def most_common_key(counts: Dict[Any, int]) -> Any:
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], str(item[0])))[0]


def format_stat_row(row: Dict[str, Any]) -> Dict[str, Any]:
    formatted = dict(row)
    for key in ("detection_rate", "mean_rssi", "median_rssi", "std_rssi", "min_rssi", "max_rssi"):
        formatted[key] = round(float(formatted[key]), 6)
    return formatted


def build_pdr_tables(
    session_dirs: Sequence[Path],
    output_dir: Path,
    base_step_m: float,
    heading_offset_deg: float,
    heading_policy: str = "handheld",
) -> Dict[str, Any]:
    event_rows: List[Dict[str, Any]] = []
    segment_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    anchor_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    for session_dir in session_dirs:
        session = session_dir.name
        events = read_events(session_dir)
        segments = build_segments(events)
        steps = compute_pdr(session_dir, segments, base_step_m, heading_offset_deg, heading_policy)
        pdr_available = bool(steps)

        summary[session] = {
            "movement_event_count": len(events),
            "segment_count": len(segments),
            "step_count": len(steps),
            "distance_m": round(sum(step.step_length_m for step in steps), 6),
            "status": "route_only" if pdr_available and (heading_policy == "handheld" or any(not math.isfinite(step.pdr_x) for step in steps)) else "ok" if pdr_available else "no_detected_steps",
            "error": "",
            "heading_policy": heading_policy,
            "steps_without_heading": sum(step.pdr_heading_status == "unavailable" for step in steps),
            "route_distance_basis": "detected_step_lengths_independent_of_phone_yaw_uncalibrated",
            "max_heading_gap_ms": MAX_HEADING_GAP_MS,
        }

        for event in events:
            event_rows.append(
                {
                    "session": session,
                    "elapsed_ts": event.elapsed_ts,
                    "epoch_ts": event.epoch_ts,
                    "event": event.event,
                }
            )
            x, y, step_index = nearest_pdr_position(steps, event.elapsed_ts)
            latest_step = steps[step_index - 1] if step_index else None
            metadata = {
                "route_distance_m": round(latest_step.route_distance_m, 6) if latest_step else 0.0 if pdr_available else "",
                "route_distance_status": "estimated_steps" if pdr_available else "unavailable",
                "pdr_heading_policy": heading_policy,
                "pdr_heading_status": latest_step.pdr_heading_status if latest_step else "unavailable",
                "pdr_position_status": latest_step.pdr_position_status if latest_step else "unavailable_heading",
            }
            if latest_step is None:
                x = y = math.nan
            anchor_rows.append(
                {
                    "session": session,
                    "anchor_id": f"{session}__{event.event}_{event.elapsed_ts}",
                    "event": event.event,
                    "elapsed_ts": event.elapsed_ts,
                    "epoch_ts": event.epoch_ts,
                    "nearest_step_index": step_index,
                    **metadata,
                    "pdr_x": round(x, 6) if math.isfinite(x) else "",
                    "pdr_y": round(y, 6) if math.isfinite(y) else "",
                    "room_x": "",
                    "room_y": "",
                    "comment": "fill room_x/room_y if this event is a known classroom checkpoint",
                }
            )
            path_rows.append(
                {
                    "session": session,
                    "point_type": event.event,
                    "segment_id": "",
                    "step_index": step_index,
                    "elapsed_ts": event.elapsed_ts,
                    "epoch_ts": event.epoch_ts,
                    **metadata,
                    "pdr_x": round(x, 6) if math.isfinite(x) else "",
                    "pdr_y": round(y, 6) if math.isfinite(y) else "",
                    "step_length_m": "",
                    "heading_deg": "",
                }
            )

        for segment in segments:
            segment_steps = [step for step in steps if step.segment_id == segment.segment_id]
            segment_rows.append(
                {
                    "session": session,
                    "segment_id": segment.segment_id,
                    "start_elapsed_ts": segment.start_elapsed_ts,
                    "start_epoch_ts": segment.start_epoch_ts,
                    "end_elapsed_ts": segment.end_elapsed_ts,
                    "end_epoch_ts": segment.end_epoch_ts,
                    "start_event": segment.start_event,
                    "end_event": segment.end_event,
                    "duration_s": round(segment.duration_s, 3),
                    "detected_steps": len(segment_steps),
                    "distance_m": round(sum(step.step_length_m for step in segment_steps), 6),
                }
            )

        for step in steps:
            metadata = {
                "route_distance_m": round(step.route_distance_m, 6), "route_distance_status": "estimated_steps",
                "pdr_heading_policy": step.pdr_heading_policy, "pdr_heading_status": step.pdr_heading_status,
                "pdr_position_status": step.pdr_position_status,
            }
            row = {
                **metadata,
                "session": session,
                "segment_id": step.segment_id,
                "step_index": step.step_index,
                "elapsed_ts": step.elapsed_ts,
                "epoch_ts": step.epoch_ts,
                "peak_accel": round(step.peak_accel, 6),
                "accel_range": round(step.accel_range, 6),
                "step_length_m": round(step.step_length_m, 6),
                "heading_rad": round(step.heading_rad, 9) if math.isfinite(step.heading_rad) else "",
                "heading_deg": round(step.heading_deg, 6) if math.isfinite(step.heading_deg) else "",
                "pdr_x": round(step.pdr_x, 6) if math.isfinite(step.pdr_x) else "",
                "pdr_y": round(step.pdr_y, 6) if math.isfinite(step.pdr_y) else "",
            }
            step_rows.append(row)
            path_rows.append(
                {
                    "session": session,
                    "point_type": "STEP",
                    **metadata,
                    "segment_id": step.segment_id,
                    "step_index": step.step_index,
                    "elapsed_ts": step.elapsed_ts,
                    "epoch_ts": step.epoch_ts,
                    "pdr_x": row["pdr_x"],
                    "pdr_y": row["pdr_y"],
                    "step_length_m": round(step.step_length_m, 6),
                    "heading_deg": row["heading_deg"],
                }
            )

    write_csv(
        output_dir / "movement_events.csv",
        ["session", "elapsed_ts", "epoch_ts", "event"],
        event_rows,
    )
    write_csv(
        output_dir / "movement_segments.csv",
        [
            "session",
            "segment_id",
            "start_elapsed_ts",
            "start_epoch_ts",
            "end_elapsed_ts",
            "end_epoch_ts",
            "start_event",
            "end_event",
            "duration_s",
            "detected_steps",
            "distance_m",
        ],
        segment_rows,
    )
    write_csv(
        output_dir / "pdr_steps.csv",
        [
            *PDR_ROUTE_FIELDS,
            "session",
            "segment_id",
            "step_index",
            "elapsed_ts",
            "epoch_ts",
            "peak_accel",
            "accel_range",
            "step_length_m",
            "heading_rad",
            "heading_deg",
            "pdr_x",
            "pdr_y",
        ],
        step_rows,
    )
    path_rows.sort(key=lambda row: (row["session"], int(row["elapsed_ts"]), str(row["point_type"])))
    write_csv(
        output_dir / "pdr_path.csv",
        [
            *PDR_ROUTE_FIELDS,
            "session",
            "point_type",
            "segment_id",
            "step_index",
            "elapsed_ts",
            "epoch_ts",
            "pdr_x",
            "pdr_y",
            "step_length_m",
            "heading_deg",
        ],
        path_rows,
    )
    write_csv(
        output_dir / "alignment_anchor_template.csv",
        [
            *PDR_ROUTE_FIELDS,
            "session",
            "anchor_id",
            "event",
            "elapsed_ts",
            "epoch_ts",
            "nearest_step_index",
            "pdr_x",
            "pdr_y",
            "room_x",
            "room_y",
            "comment",
        ],
        anchor_rows,
    )
    write_json(output_dir / "pdr_summary.json", summary)
    return summary


def build_wifi_pdr_feature_table(output_dir: Path) -> Dict[str, Any]:
    wifi_path = output_dir / "wifi_wide_selected.csv"
    contour_path = output_dir / "pdr_trajectory_contour.csv"
    pdr_path = contour_path if contour_path.exists() else output_dir / "pdr_path.csv"
    fused_path = output_dir / "wifi_pdr_features.csv"
    if not wifi_path.exists() or not pdr_path.exists():
        return {"rows": 0, "status": "missing_input"}

    pdr_by_session: Dict[str, List[Dict[str, Any]]] = {}
    with pdr_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                elapsed_ts = int(row["elapsed_ts"])
            except (KeyError, TypeError, ValueError):
                continue
            pdr_x, pdr_y = safe_float(row.get("pdr_x")), safe_float(row.get("pdr_y"))
            has_route = row.get("route_distance_status") == "estimated_steps" and math.isfinite(safe_float(row.get("route_distance_m")))
            if not has_route and not (math.isfinite(pdr_x) and math.isfinite(pdr_y)):
                continue
            pdr_by_session.setdefault(row["session"], []).append(
                {
                    "elapsed_ts": elapsed_ts,
                    "pdr_x": pdr_x,
                    "pdr_y": pdr_y,
                    **{field: row.get(field, "") for field in PDR_ROUTE_FIELDS},
                    "trajectory_route_distance_m": row.get("trajectory_route_distance_m", ""),
                    "point_type": row.get("source_point_type", row.get("point_type", "")),
                    "step_index": row.get("source_step_index", row.get("step_index", "")),
                    "trajectory_contour_id": row.get("contour_id", ""),
                    "trajectory_contour_distance_m": row.get("contour_distance_m", ""),
                    "trajectory_distance_m": row.get("trajectory_distance_m", ""),
                }
            )

    for rows in pdr_by_session.values():
        rows.sort(key=lambda row: row["elapsed_ts"])

    fused_rows: List[Dict[str, Any]] = []
    with wifi_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        wifi_fields = reader.fieldnames or []
        fieldnames = wifi_fields + [
            *PDR_ROUTE_FIELDS, "trajectory_route_distance_m",
            "pdr_x",
            "pdr_y",
            "pdr_interp_status",
            "pdr_interp_gap_ms",
            "pdr_left_type",
            "pdr_right_type",
            "trajectory_contour_id",
            "trajectory_contour_distance_m",
            "trajectory_distance_m",
            "trajectory_interp_status",
        ]
        for row in reader:
            session = row.get("session", "")
            try:
                elapsed_ts = int(row["elapsed_ts"])
            except (KeyError, TypeError, ValueError):
                continue
            pdr_state = interpolate_pdr_state(pdr_by_session.get(session, []), elapsed_ts)
            fused_row = dict(row)
            fused_row.update(
                {
                    **{field: pdr_state.get(field, "") for field in PDR_ROUTE_FIELDS},
                    "trajectory_route_distance_m": pdr_state.get("trajectory_route_distance_m", ""),
                    "pdr_x": "" if pdr_state["pdr_x"] is None else round(float(pdr_state["pdr_x"]), 6),
                    "pdr_y": "" if pdr_state["pdr_y"] is None else round(float(pdr_state["pdr_y"]), 6),
                    "pdr_interp_status": pdr_state["status"],
                    "pdr_interp_gap_ms": pdr_state["gap_ms"],
                    "pdr_left_type": pdr_state["left_type"],
                    "pdr_right_type": pdr_state["right_type"],
                    "trajectory_contour_id": pdr_state["trajectory_contour_id"],
                    "trajectory_contour_distance_m": pdr_state["trajectory_contour_distance_m"],
                    "trajectory_distance_m": pdr_state["trajectory_distance_m"],
                    "trajectory_interp_status": pdr_state["trajectory_status"],
                }
            )
            fused_rows.append(fused_row)

    write_csv(fused_path, fieldnames, fused_rows)
    return {
        "rows": len(fused_rows),
        "status": "ok",
        "pdr_source": pdr_path.name,
        "trajectory_contour_available": contour_path.exists(),
    }


def trajectory_state(
    left: Dict[str, Any],
    right: Dict[str, Any],
    ratio: float,
) -> Tuple[str, str, str, str]:
    left_id = str(left.get("trajectory_contour_id", ""))
    right_id = str(right.get("trajectory_contour_id", ""))
    if not left_id or left_id != right_id:
        return "", "", "", "contour_break_or_unavailable"
    left_contour_distance = safe_float(left.get("trajectory_contour_distance_m"))
    right_contour_distance = safe_float(right.get("trajectory_contour_distance_m"))
    left_trajectory_distance = safe_float(left.get("trajectory_distance_m"))
    right_trajectory_distance = safe_float(right.get("trajectory_distance_m"))
    if not all(
        math.isfinite(value)
        for value in (
            left_contour_distance,
            right_contour_distance,
            left_trajectory_distance,
            right_trajectory_distance,
        )
    ):
        return left_id, "", "", "contour_values_missing"
    contour_distance = left_contour_distance + ratio * (right_contour_distance - left_contour_distance)
    route_distance = left_trajectory_distance + ratio * (right_trajectory_distance - left_trajectory_distance)
    return left_id, f"{contour_distance:.6f}", f"{route_distance:.6f}", "interpolated_contour"


def interpolated_route_metadata(left: Dict[str, Any], right: Dict[str, Any], ratio: float) -> Dict[str, Any]:
    result = {field: left.get(field, "") if left.get(field, "") == right.get(field, "") else "unavailable" for field in PDR_ROUTE_FIELDS}
    for key in ("route_distance_m", "trajectory_route_distance_m"):
        a, b = safe_float(left.get(key)), safe_float(right.get(key))
        result[key] = round(a + ratio * (b - a), 6) if all(math.isfinite(v) for v in (a, b)) and left.get("route_distance_status") == right.get("route_distance_status") == "estimated_steps" else ""
    result["pdr_position_status"] = result.get("pdr_position_status") or "legacy_unknown"
    return result


def interpolate_pdr_state(points: Sequence[Dict[str, Any]], target_elapsed_ts: int) -> Dict[str, Any]:
    empty = {
        **{field: "" for field in PDR_ROUTE_FIELDS},
        "trajectory_route_distance_m": "",
        "pdr_x": None,
        "pdr_y": None,
        "status": "no_pdr_points",
        "gap_ms": "",
        "left_type": "",
        "right_type": "",
        "trajectory_contour_id": "",
        "trajectory_contour_distance_m": "",
        "trajectory_distance_m": "",
        "trajectory_status": "contour_unavailable",
    }
    if not points:
        return empty
    if target_elapsed_ts <= points[0]["elapsed_ts"]:
        point = points[0]
        contour_id, contour_distance, route_distance, contour_status = trajectory_state(point, point, 0.0)
        return {
            **interpolated_route_metadata(point, point, 0.0),
            "pdr_x": point["pdr_x"] if math.isfinite(safe_float(point["pdr_x"])) else None,
            "pdr_y": point["pdr_y"] if math.isfinite(safe_float(point["pdr_y"])) else None,
            "status": "before_path",
            "gap_ms": str(abs(point["elapsed_ts"] - target_elapsed_ts)),
            "left_type": point["point_type"],
            "right_type": point["point_type"],
            "trajectory_contour_id": contour_id,
            "trajectory_contour_distance_m": contour_distance,
            "trajectory_distance_m": route_distance,
            "trajectory_status": contour_status,
        }
    if target_elapsed_ts >= points[-1]["elapsed_ts"]:
        point = points[-1]
        contour_id, contour_distance, route_distance, contour_status = trajectory_state(point, point, 0.0)
        return {
            **interpolated_route_metadata(point, point, 0.0),
            "pdr_x": point["pdr_x"] if math.isfinite(safe_float(point["pdr_x"])) else None,
            "pdr_y": point["pdr_y"] if math.isfinite(safe_float(point["pdr_y"])) else None,
            "status": "after_path",
            "gap_ms": str(abs(target_elapsed_ts - point["elapsed_ts"])),
            "left_type": point["point_type"],
            "right_type": point["point_type"],
            "trajectory_contour_id": contour_id,
            "trajectory_contour_distance_m": contour_distance,
            "trajectory_distance_m": route_distance,
            "trajectory_status": contour_status,
        }

    lo = 0
    hi = len(points) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if points[mid]["elapsed_ts"] < target_elapsed_ts:
            lo = mid + 1
        else:
            hi = mid - 1
    right = points[lo]
    left = right if right["elapsed_ts"] == target_elapsed_ts else points[lo - 1]
    span = right["elapsed_ts"] - left["elapsed_ts"]
    ratio = 0.0 if span <= 0 else (target_elapsed_ts - left["elapsed_ts"]) / span
    contour_id, contour_distance, route_distance, contour_status = trajectory_state(left, right, ratio)
    if left.get("trajectory_contour_id") and right.get("trajectory_contour_id") and left["trajectory_contour_id"] != right["trajectory_contour_id"]:
        return {**empty, "status": "contour_break", "gap_ms": str(span), "trajectory_status": contour_status}
    return {
        **interpolated_route_metadata(left, right, ratio),
        "pdr_x": left["pdr_x"] + ratio * (right["pdr_x"] - left["pdr_x"]) if all(math.isfinite(safe_float(p["pdr_x"])) for p in (left, right)) else None,
        "pdr_y": left["pdr_y"] + ratio * (right["pdr_y"] - left["pdr_y"]) if all(math.isfinite(safe_float(p["pdr_y"])) for p in (left, right)) else None,
        "status": "exact_or_duplicate" if span <= 0 else "interpolated",
        "gap_ms": "0" if span <= 0 else str(span),
        "left_type": left["point_type"],
        "right_type": right["point_type"],
        "trajectory_contour_id": contour_id,
        "trajectory_contour_distance_m": contour_distance,
        "trajectory_distance_m": route_distance,
        "trajectory_status": contour_status,
    }


def find_session_dirs(root: Path) -> List[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and path.suffix == ".sess")


def profile_value(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return PROFILE_DEFAULTS[args.reconstruction_profile][name]


def detect_nvidia_gpu() -> Dict[str, Any]:
    """Probe the GPU for this run. No result is loaded from a previous run."""
    command = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "cuda_enabled": False, "reason": str(exc)}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "cuda_enabled": False, "reason": result.stderr.strip() or "nvidia-smi returned no GPU"}
    fields = [field.strip() for field in result.stdout.strip().splitlines()[0].split(",", 2)]
    try:
        memory_mb = int(fields[1])
    except (IndexError, ValueError):
        memory_mb = 0
    return {"available": True, "cuda_enabled": True, "name": fields[0], "memory_mb": memory_mb,
            "driver_version": fields[2] if len(fields) > 2 else "unknown", "probe": "nvidia-smi"}


def automatic_profile(hardware: Dict[str, Any]) -> str:
    memory_mb = int(hardware.get("memory_mb") or 0)
    if memory_mb >= 22000:
        return "rtx4080"
    if memory_mb >= 14000:
        return "final"
    if memory_mb >= 8000:
        return "balanced"
    return "preview"


def print_workload_summary(
    profile: str,
    colmap_num_threads: int,
    dense_max_image_size: int,
    feature_max_image_size: int,
    max_num_features: int,
    matcher: str,
    dense_profile: str,
    video_max_frames: int,
    image_similarity_enabled: bool,
    image_similarity_max_pairs: int,
    image_similarity_max_image_size: int,
    image_similarity_orb_features: int,
) -> None:
    video_text = "uncapped" if video_max_frames <= 0 else str(video_max_frames)
    similarity_text = (
        "disabled"
        if not image_similarity_enabled
        else ("all constrained pairs" if image_similarity_max_pairs <= 0 else f"top {image_similarity_max_pairs} pairs")
    )
    print(
        "[settings] "
        f"profile={profile}, matcher={matcher}, dense_profile={dense_profile}, "
        f"dense_max_image_size={dense_max_image_size}, feature_max_image_size={feature_max_image_size}, "
        f"max_features={max_num_features}, "
        f"video_max_frames={video_text}, colmap_threads={colmap_num_threads or 'automatic'}, "
        f"image_similarity={similarity_text}, "
        f"image_similarity_size={image_similarity_max_image_size}, image_similarity_orb={image_similarity_orb_features}",
        flush=True,
    )
    if profile in {"final", "rtx4080", "complete"} or matcher in {"hybrid", "exhaustive"} or dense_max_image_size >= 2000:
        print(
            "[info] High-quality reconstruction settings may require a longer runtime.",
            flush=True,
        )


def log_stage(message: str) -> None:
    print(f"[stage] {message}", flush=True)


def log_stage_done(label: str, summary: Any, skipped_reason: Optional[str] = None) -> None:
    if isinstance(summary, dict) and summary.get("status") == "skipped":
        print(f"[skip] {label} {skipped_reason or 'disabled by command-line options'}", flush=True)


def validate_numeric_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid sensor/matching settings before main resets any output."""
    def check(name, predicate, requirement):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or not predicate(value)):
            parser.error('--' + name.replace('_', '-') + ' must be finite and ' + requirement)

    for name in ('wifi_grid_cell_size_m', 'wifi_localization_k', 'base_step_m', 'min_scan_count',
                 'colmap_max_num_matches', 'constrained_video_frame_window',
                 'constrained_max_pdr_distance_m', 'image_similarity_orb_features'):
        check(name, lambda value: value > 0, 'positive')
    for name in ('max_rssi_std', 'image_similarity_max_pairs', 'image_similarity_max_image_size',
                 'visual_recovery_neighbors', 'colmap_feature_max_image_size', 'colmap_max_num_features',
                 'camera_height_m'):
        check(name, lambda value: value >= 0, 'nonnegative')
    for name in ('min_detection_rate', 'image_similarity_min_visual_score'):
        check(name, lambda value: 0 <= value <= 1, 'in [0,1]')
    check('min_mean_rssi', lambda value: value <= 0, 'nonpositive')
    check('heading_offset_deg', lambda value: True, 'a valid angle')
    # COLMAP accepts -1 for native resolution; zero/other negatives are invalid.
    check('colmap_dense_max_image_size', lambda value: value == -1 or value > 0,
          '-1 (native resolution) or positive')


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(video_input_mode="sensor_assisted")
    parser.add_argument('--cpu-workers', type=int, default=0,
                        help='Default CPU budget for decoder, image analysis and COLMAP: 0=all available CPUs; stage overrides take precedence.')
    parser.add_argument('--video-decode-threads', type=int, default=0,
                        help='FFmpeg video decoding threads: 0=inherit CPU budget, positive values override it.')
    parser.add_argument('--video-hw-decode', choices=('auto', 'off'), default='auto',
                        help='Prefer hardware video decoding; Windows auto selects an NVIDIA DXGI adapter when available. CPU workers are unchanged.')
    parser.add_argument('--video-hw-device', type=int, default=-1,
                        help='Video backend adapter index (-1=automatic); this is not a CUDA device index.')
    parser.add_argument('--preprocess-workers', type=int, choices=(0, 1, 2), default=0,
                        help='Independent WiFi/PDR processes: 0=automatic (up to two branches), 1=serial, 2=parallel.')
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Folder containing .sess directories.")
    parser.add_argument("--output", type=Path, default=Path("generated/processed"), help="Output folder.")
    parser.add_argument("--video-only", dest="video_input_mode", action="store_const", const="visual",
                        help="Disable sensor processing unless --video-sensor-sync is supplied.")
    parser.add_argument("--video-sensor-sync", type=Path,
                        help="Explicit clock anchors CSV: video_name,session,video_time_s,sensor_elapsed_ts. Enables timestamp-validated IMU/PDR and fresh WiFi matching assistance.")
    parser.add_argument("--video-sensor-alignment", choices=("fuzzy", "strict", "off"),
                        help="Fuzzy uses embedded recording time plus an uncertainty window for soft candidates; explicit sync CSV takes precedence. Sensor-assisted video defaults to fuzzy.")
    parser.add_argument("--video-sensor-time-uncertainty-s", type=float, default=30.,
                        help="Assumed +/- clock uncertainty for fuzzy matching, in (0,300] seconds (default: 30); not a measured accuracy.")
    parser.add_argument("--video-sensor-offset-s", type=float, default=0.,
                        help="Add this correction to the embedded video wall clock when proposing sensor windows, in [-3600,3600] seconds.")
    parser.add_argument("--run-colmap", action="store_true",
                        help="main.py: explicitly require reconstruction after preprocessing (already the default); incompatible with preparation/export skip options.")
    parser.add_argument(
        "--reconstruction-profile",
        choices=["auto", "preview", "balanced", "final", "rtx4080", "complete"],
        default="auto",
        help="auto probes the current NVIDIA GPU and VRAM on every run; other values explicitly override profile selection.",
    )
    parser.add_argument("--min-detection-rate", type=float, default=0.20)
    parser.add_argument("--min-mean-rssi", type=float, default=-90.0)
    parser.add_argument("--max-rssi-std", type=float, default=10.0)
    parser.add_argument("--min-scan-count", type=int, default=5)
    parser.add_argument("--base-step-m", type=float, default=DEFAULT_BASE_STEP_M)
    parser.add_argument("--heading-offset-deg", type=float, default=0.0)
    parser.add_argument("--pdr-heading-policy", choices=("handheld", "phone-aligned"), default="handheld",
                        help="Use yaw-independent route distances by default; phone-aligned explicitly assumes stable phone-to-walking heading, never camera attitude.")
    parser.add_argument("--video-dir", type=Path, default=Path("video"), help="Folder containing route/base videos.")
    parser.add_argument("--skip-video-frames", action="store_true", help="Do not extract video frames for COLMAP.")
    parser.add_argument("--video-frame-interval-s", type=float)
    parser.add_argument("--video-max-frames", type=int, help="Max frames to extract per video; use 0 for no cap.")
    parser.add_argument("--video-blur-threshold", type=float, help="Laplacian blur threshold (adaptive policy uses it as a ceiling); 0 disables relative blur rejection, while adaptive blank/detail floors remain.")
    parser.add_argument("--video-blur-policy", choices=("fixed", "adaptive"),
                        help="Adaptive sharpness selection is the default; fixed uses an absolute threshold.")
    parser.add_argument("--video-keyframe-max-gap-s", type=float, default=2.0,
                        help="Adaptive selection targets this maximum temporal gap using usable native frames; cannot recover unobserved views.")
    parser.add_argument("--video-keyframe-min-change", type=float, help="Drop near-duplicate video frames below this mean grayscale difference; use 0 to disable.")
    parser.add_argument("--video-keyframe-min-gap", type=int, help="Minimum gap in extracted-frame steps between kept video keyframes.")
    parser.add_argument("--video-refine-window-s", type=float, default=0., help="Native candidate search radius around each sampled time; 0 disables.")
    parser.add_argument("--video-refine-budget-hours", type=float, default=8., help="Native frame selection budget; excludes the rest of reconstruction.")
    parser.add_argument("--video-refine-coverage-search", action=argparse.BooleanOptionalAction, default=True, help="Search further within a sampling bin when the local candidates remain blurred; requires a positive refinement window.")
    parser.add_argument("--scene-completion", choices=("off", "videoscene"), default="off", help="Run gated VideoScene completion after the observed dense final is exported.")
    parser.add_argument("--videoscene-max-clips", type=int, default=3, help="Maximum distinct local 49-view gap proposals, processed sequentially; not full-video restoration.")
    parser.add_argument("--videoscene-budget-hours", type=float, default=4., help="Completion-only time budget; not an end-to-end reconstruction deadline.")
    parser.add_argument("--ffmpeg-exe", type=Path, help="Optional path to ffmpeg.exe.")
    parser.add_argument("--reconstruction-output", type=Path, default=Path("generated/reconstruction_dataset"))
    parser.add_argument("--skip-reconstruction-export", action="store_true")
    parser.add_argument("--camera-height-m", type=float, default=1.50)
    parser.add_argument("--skip-wifi-positioning", action="store_true")
    parser.add_argument("--wifi-grid-cell-size-m", type=float, default=1.0)
    parser.add_argument("--wifi-localization-k", type=int, default=3)
    parser.add_argument("--colmap-workspace", type=Path, default=Path("generated/colmap_workspace"))
    parser.add_argument("--colmap-log-level", choices=("compact", "verbose"), default="compact",
                        help="Console detail for preprocessing and COLMAP; full reports/logs are saved to disk.")
    parser.add_argument("--colmap-dir", type=Path, default=Path("colmap-x64-windows-cuda"))
    parser.add_argument("--colmap-exe", type=Path)
    parser.add_argument(
        "--colmap-mask-dir",
        type=Path,
        help=(
            "Optional COLMAP mask directory for semantic clutter or known reflective-region filtering. "
            "Masks must follow generated image_list names, for example video/frame.png."
        ),
    )
    parser.add_argument("--skip-colmap-prepare", action="store_true")
    parser.add_argument("--visual-only", action="store_true", help="Prepare COLMAP without sensor matching assistance or RF boundary reporting.")
    parser.add_argument("--colmap-camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--colmap-matcher", choices=["constrained", "hybrid", "exhaustive", "sequential"])
    parser.add_argument("--colmap-quality", choices=["low", "medium", "high", "extreme"])
    parser.add_argument("--colmap-dense-max-image-size", type=int)
    parser.add_argument("--colmap-dense-profile", choices=["balanced", "efficient", "rtx4080", "complete"], help="Dense/MVS settings; video-only defaults to efficient, with geometric consistency retained.")
    parser.add_argument("--dense-max-images", type=int, default=1200,
                        help="Hard dense-view limit (default: 1200); adaptive selection can stop earlier. 0 uses all eligible views.")
    parser.add_argument("--dense-selection-mode", choices=("adaptive", "fixed"), default="adaptive",
                        help="Add views until coverage targets or the hard limit are reached; fixed restores count-only selection.")
    parser.add_argument("--dense-initial-images", type=int, default=480,
                        help="Initial adaptive view budget, clamped to the hard limit (default: 480).")
    parser.add_argument("--dense-min-single-coverage", type=float, default=.9,
                        help="Target fraction of original sparse points seen by selected views (default: 0.9).")
    parser.add_argument("--dense-min-three-coverage", type=float, default=.8,
                        help="Target retained fraction of original three-view sparse tracks (default: 0.8).")
    parser.add_argument("--dense-max-video-gap-s", type=float, default=4.,
                        help="Target selected-view time gap; missing source observations remain reported (default: 4s).")
    parser.add_argument("--dense-max-source-images", type=int, default=12,
                        help="Maximum selected source views per dense reference (default: 12).")
    parser.add_argument(
        "--colmap-num-threads",
        type=int,
        help="Thread count for supported COLMAP stages; 0 (default) selects automatic threading.",
    )
    meshing = parser.add_mutually_exclusive_group()
    meshing.add_argument("--colmap-meshing", dest="skip_colmap_meshing", action="store_false", help="Generate and retain Delaunay/Poisson inspection meshes separately from final.ply.")
    meshing.add_argument("--skip-colmap-meshing", dest="skip_colmap_meshing", action="store_true", help="Skip optional inspection meshes (default).")
    parser.set_defaults(skip_colmap_meshing=True)
    parser.add_argument("--keep-generated-inputs", action="store_true", help="Keep main.py's generated video frames and exported image copies after successful reconstruction.")
    parser.add_argument('--skip-startup-cleanup', action='store_true',
                        help='Keep intermediate data in other output directories at startup; selected outputs still follow the normal fresh-run policy.')
    parser.add_argument("--skip-point-cloud-filter", action="store_true", help="Skip the filtered fused point cloud post-processing output.")
    parser.add_argument("--point-filter-method", choices=("observed", "wifi-density", "connected-density"), default="observed",
                        help="Observed preserves unknown/weak surfaces and removes only verified conflicts; density methods are legacy alternatives.")
    parser.add_argument("--point-filter-max-candidates", type=int, default=2000000,
                        help="Maximum points assessed against depth maps (1..2000000); unassessed points are retained in observed mode.")
    parser.add_argument("--point-filter-budget-seconds", type=float, default=120.,
                        help="Observation assessment time budget, excluding input verification and PLY I/O (default: 120s).")
    parser.add_argument("--colmap-feature-max-image-size", type=int)
    parser.add_argument("--colmap-max-num-features", type=int)
    parser.add_argument("--colmap-max-num-matches", type=int)
    parser.add_argument(
        "--constrained-max-pdr-distance-m",
        type=float,
        help=(
            "Maximum along-route distance for synchronized video sensor candidates. "
            "Fuzzy timing windows use their supported route-distance intervals."
        ),
    )
    parser.add_argument("--constrained-video-frame-window", type=int)
    parser.add_argument("--skip-image-similarity", action="store_true", help="Disable P1 selection and visual recovery; retain route/order candidates.")
    parser.add_argument("--skip-wifi-recovery", action="store_true", help="Disable WiFi pair retrieval and RF ranking in targeted recovery; retain IMU/PDR and visual recovery.")
    parser.add_argument("--visual-recovery-neighbors", type=int, default=12, help="Appearance-retrieved candidates per image outside local video windows; require geometric verification. Use 0 to disable.")
    parser.add_argument("--image-similarity-max-pairs", type=int, help="Analyze all constrained pairs by default when profile value is 0; set a positive cap for a faster sample.")
    parser.add_argument("--image-similarity-max-image-size", type=int, help="Resize longest image side before ORB similarity analysis.")
    parser.add_argument("--image-similarity-orb-features", type=int, help="ORB feature count per image for match-pair diagnostics.")
    parser.add_argument('--image-similarity-workers', type=int, default=0,
                        help='Parallel image feature and pair-verification workers: 0=inherit CPU budget, positive values override it.')
    parser.add_argument('--colmap-ba-gpu', choices=('auto', 'on', 'off'), default='auto',
                        help='Probe GPU sparse BA support; auto falls back visibly to CPU, on requires verified support.')
    parser.add_argument('--sparse-strategy', choices=('all', 'backbone'), default='all',
                        help='all keeps every frame in mapping; backbone seeds a protected video subset then expands to all frames.')
    parser.add_argument('--sparse-ba-growth-ratio', type=float, default=1.1,
                        help='Global BA image/point growth trigger; 1.1 keeps native frequency, opt-in 1.2 reduces growth-triggered solves.')
    parser.add_argument('--sparse-keyframe-gap-s', type=float, default=1.,
                        help='Target temporal spacing of backbone frames; weak matching segments retain extra frames.')
    parser.add_argument("--image-similarity-min-visual-score", type=float, help="Score threshold for feature-overlap diagnostic labels.")
    args = parser.parse_args(argv)
    validate_numeric_options(parser, args)
    if args.video_hw_device < -1:
        parser.error('--video-hw-device must be -1 (automatic) or nonnegative')
    import math
    if not math.isfinite(args.sparse_ba_growth_ratio) or args.sparse_ba_growth_ratio <= 1:
        parser.error('--sparse-ba-growth-ratio must be finite and greater than 1')
    if not math.isfinite(args.sparse_keyframe_gap_s) or args.sparse_keyframe_gap_s <= 0:
        parser.error('--sparse-keyframe-gap-s must be finite and positive')
    for name in ('cpu_workers', 'video_decode_threads', 'image_similarity_workers'):
        if getattr(args, name) < 0:
            parser.error('--' + name.replace('_', '-') + ' must be nonnegative')
    args.sensor_processing_enabled = args.video_input_mode == "sensor_assisted" or args.video_sensor_sync is not None
    if args.sensor_processing_enabled and args.video_sensor_sync is None and args.video_sensor_alignment != "off" and not args.visual_only:
        conventional_sync = args.root.resolve() / "video_sensor_sync.csv"
        if conventional_sync.is_file():
            args.video_sensor_sync = conventional_sync
    if args.video_sensor_alignment is None:
        args.video_sensor_alignment = "fuzzy" if args.sensor_processing_enabled else "strict"
    if not math.isfinite(args.video_sensor_time_uncertainty_s) or not 0 < args.video_sensor_time_uncertainty_s <= 300:
        parser.error("--video-sensor-time-uncertainty-s must be finite and in (0,300]")
    if not math.isfinite(args.video_sensor_offset_s) or abs(args.video_sensor_offset_s) > 3600:
        parser.error("--video-sensor-offset-s must be finite and in [-3600,3600]")
    if args.video_sensor_sync is not None:
        if args.video_sensor_alignment == "off":
            parser.error("--video-sensor-sync cannot be combined with --video-sensor-alignment off")
        if args.visual_only or args.skip_video_frames:
            parser.error("--video-sensor-sync cannot be combined with --visual-only or --skip-video-frames")
        try:
            from video_sensor_fusion import load_video_sensor_sync
            load_video_sensor_sync(args.root.resolve() / args.video_sensor_sync)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        args.video_sensor_alignment = "strict"
    if args.skip_video_frames:
        parser.error("--video-only cannot be combined with --skip-video-frames")
    if not args.sensor_processing_enabled:
        args.visual_only = True
        args.skip_wifi_recovery = True
        args.skip_wifi_positioning = True
    if args.colmap_dense_profile is None:
        args.colmap_dense_profile = "efficient"
    # Retain coverage across the whole video instead of a small fixed frame cap.
    if args.video_max_frames is None:
        args.video_max_frames = 0
    if args.video_frame_interval_s is None:
        args.video_frame_interval_s = 0.5
    if args.video_keyframe_min_gap is None:
        args.video_keyframe_min_gap = 1
    if args.video_keyframe_min_change is None:
        args.video_keyframe_min_change = 2.0
    for field, default in (("output", "processed"), ("reconstruction_output", "reconstruction_dataset"),
                           ("colmap_workspace", "colmap_workspace")):
        if getattr(args, field) == Path(default):
            suffix = "_video_sensors" if args.sensor_processing_enabled else "_video_only"
            setattr(args, field, Path(default + suffix))
    if args.video_blur_policy is None:
        args.video_blur_policy = "adaptive"
    if args.dense_max_images != 0 and args.dense_max_images < 3:
        parser.error("--dense-max-images must be 0 (all views) or at least 3")
    if not 2 <= args.dense_max_source_images <= 64:
        parser.error("--dense-max-source-images must be in [2,64]")
    if args.dense_initial_images < 3:
        parser.error("--dense-initial-images must be at least 3")
    for name in ("dense_min_single_coverage", "dense_min_three_coverage"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= 1:
            parser.error("--" + name.replace("_", "-") + " must be finite and in (0,1]")
    if not math.isfinite(args.dense_max_video_gap_s) or args.dense_max_video_gap_s <= 0:
        parser.error("--dense-max-video-gap-s must be finite and positive")
    if not 1 <= args.point_filter_max_candidates <= 2000000:
        parser.error("--point-filter-max-candidates must be in [1,2000000]")
    if not math.isfinite(args.point_filter_budget_seconds) or args.point_filter_budget_seconds <= 0:
        parser.error("--point-filter-budget-seconds must be finite and positive")
    if args.video_frame_interval_s is not None and (
            not math.isfinite(args.video_frame_interval_s) or args.video_frame_interval_s <= 0):
        parser.error("--video-frame-interval-s must be finite and greater than zero")
    if args.video_max_frames is not None and args.video_max_frames < 0:
        parser.error("--video-max-frames must be nonnegative; use 0 for no cap")
    for name in ("video_blur_threshold", "video_keyframe_min_change"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error("--" + name.replace("_", "-") + " must be finite and nonnegative")
    if args.video_keyframe_min_gap is not None and args.video_keyframe_min_gap < 1:
        parser.error("--video-keyframe-min-gap must be at least 1")
    if not math.isfinite(args.video_keyframe_max_gap_s) or args.video_keyframe_max_gap_s <= 0:
        parser.error("--video-keyframe-max-gap-s must be finite and greater than zero")
    if args.colmap_num_threads is not None and args.colmap_num_threads < 0:
        parser.error("--colmap-num-threads must be non-negative; use 0 for automatic selection")
    if args.run_colmap and (args.skip_colmap_prepare or args.skip_reconstruction_export
                          ):
        parser.error("--run-colmap requires reconstruction export and COLMAP preparation")
    if args.scene_completion == "videoscene" and args.skip_point_cloud_filter:
        parser.error("VideoScene requires point-cloud filtering; remove --skip-point-cloud-filter or use --scene-completion off")
    if not math.isfinite(args.video_refine_window_s) or not 0 <= args.video_refine_window_s <= 2:
        parser.error("--video-refine-window-s must be finite and in [0,2]")
    if not 1 <= args.videoscene_max_clips <= 8:
        parser.error("--videoscene-max-clips must be in [1,8]")
    if not math.isfinite(args.videoscene_budget_hours) or not 0 < args.videoscene_budget_hours <= 20:
        parser.error("--videoscene-budget-hours must be in (0,20]")
    if not math.isfinite(args.video_refine_budget_hours) or not 0 < args.video_refine_budget_hours <= 20:
        parser.error("--video-refine-budget-hours must be in (0,20]")
    return args


class PipelineStageError(RuntimeError):
    pass


def require_stage(stage, summary, allowed=("ok", "skipped")):
    status = summary.get("status")
    if status not in allowed:
        raise PipelineStageError(f"{stage} failed: {status}; downstream stages were not started")
    if status not in ("ok", "skipped"):
        print(f"[warn] {stage}: {status}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    workspace = root / args.colmap_workspace
    try:
        # Different reconstruction workspaces must not concurrently overwrite
        # a shared preprocessing/export directory. Acquire all locks before work.
        directories = {workspace.resolve(), (root / args.output).resolve()}
        if not args.skip_reconstruction_export:
            directories.add((root / args.reconstruction_output).resolve())
        with ExitStack() as locks:
            for directory in sorted(directories):
                locks.enter_context(workspace_lock(directory))
            try:
                return run_pipeline(args)
            except WorkspaceInputsProtectedError as exc:
                # A refused preparation must not overwrite even the old summary.
                print(f"[error] {exc}", flush=True)
                return 2
            except (PipelineStageError, OSError, ValueError) as exc:
                try:
                    write_json(args.root.resolve() / args.output / "processing_summary.json",
                               {"status": "failed", "error": str(exc)})
                except OSError as report_error:
                    print(f"[warn] Could not record pipeline failure: {report_error}", flush=True)
                print(f"[error] {exc}", flush=True)
                return 2
    except WorkspaceBusyError as exc:
        print(f"[error] {exc}", flush=True)
        return 2


def run_pipeline(args) -> int:
    require_fresh_workspace(args.root.resolve() / args.colmap_workspace)
    requested_profile = args.reconstruction_profile
    hardware = detect_nvidia_gpu()
    from gpu_image_ops import configure_gpu_image_ops, sharpness_diagnostics
    configure_gpu_image_ops(prefer_nvidia=bool(hardware.get('cuda_enabled')))
    from compute_resources import pipeline_resources
    resources = pipeline_resources(args)
    if requested_profile == "auto":
        args.reconstruction_profile = automatic_profile(hardware)
    root = args.root.resolve()
    output_dir = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    sensor_processing_enabled = args.sensor_processing_enabled
    # Number the active video and optional sensor preprocessing stages.
    stage_order = [1, 2, 3, 5, 6, 7, 8] if sensor_processing_enabled else [5, 6, 8]
    stage_labels = {number: f"{index}/{len(stage_order)}"
                    for index, number in enumerate(stage_order, 1)}
    session_dirs = find_session_dirs(root) if sensor_processing_enabled else []
    if not session_dirs and sensor_processing_enabled:
        raise PipelineStageError(f"No .sess directories found under {root}")

    verbose = args.colmap_log_level == "verbose"
    if verbose:
        print(f"[info] root: {root}", flush=True)
        print(f"[info] output: {output_dir}", flush=True)
        print(f"[info] sessions: {', '.join(path.name for path in session_dirs)}", flush=True)
    if hardware.get("available"):
        print(f"[hardware] GPU detected now: {hardware['name']}, VRAM={hardware['memory_mb']} MiB, driver={hardware['driver_version']}", flush=True)
    else:
        print(f"[hardware] NVIDIA CUDA GPU not detected: {hardware.get('reason', 'unknown reason')}", flush=True)
    selection_source = "automatic current-hardware selection" if requested_profile == "auto" else "explicit command-line override"
    print(f"[info] reconstruction profile: {args.reconstruction_profile} ({selection_source})", flush=True)

    video_frame_interval_s = profile_value(args, "video_frame_interval_s")
    video_max_frames = profile_value(args, "video_max_frames")
    video_blur_threshold = profile_value(args, "video_blur_threshold")
    video_keyframe_min_change = profile_value(args, "video_keyframe_min_change")
    video_keyframe_min_gap = profile_value(args, "video_keyframe_min_gap")
    print(f"[video] Primary input: {1.0 / video_frame_interval_s:g} candidate frames/s, "
          f"budget={video_max_frames or 'whole clip'}, blur_policy={args.video_blur_policy}, "
          f"target_gap={args.video_keyframe_max_gap_s:g}s; native frame resolution is preserved.", flush=True)
    colmap_dense_max_image_size = profile_value(args, "colmap_dense_max_image_size")
    colmap_dense_profile = profile_value(args, "colmap_dense_profile")
    colmap_matcher = profile_value(args, "colmap_matcher")
    colmap_quality = profile_value(args, "colmap_quality")
    colmap_feature_max_image_size = profile_value(args, "colmap_feature_max_image_size")
    colmap_max_num_features = profile_value(args, "colmap_max_num_features")
    colmap_max_num_matches = profile_value(args, "colmap_max_num_matches")
    colmap_num_threads = resources['colmap_threads']
    print(f"[resources] Available CPUs={resources['available_cpus']}; default budget={resources['cpu_workers']}; "
          f"COLMAP threads={colmap_num_threads}; image analysis workers={resources['image_similarity_workers']}; "
          f"video decoder threads={resources['video_decode_threads']}.", flush=True)
    constrained_max_pdr_distance_m = profile_value(args, "constrained_max_pdr_distance_m")
    constrained_video_frame_window = profile_value(args, "constrained_video_frame_window")
    image_similarity_max_pairs = profile_value(args, "image_similarity_max_pairs")
    image_similarity_max_image_size = profile_value(args, "image_similarity_max_image_size")
    image_similarity_orb_features = profile_value(args, "image_similarity_orb_features")
    if verbose:
        print_workload_summary(
            profile=args.reconstruction_profile,
            colmap_num_threads=colmap_num_threads,
            dense_max_image_size=colmap_dense_max_image_size,
            feature_max_image_size=colmap_feature_max_image_size,
            max_num_features=colmap_max_num_features,
            matcher=colmap_matcher,
            dense_profile=colmap_dense_profile,
            video_max_frames=video_max_frames,
            image_similarity_enabled=not args.skip_image_similarity,
            image_similarity_max_pairs=image_similarity_max_pairs,
            image_similarity_max_image_size=image_similarity_max_image_size,
            image_similarity_orb_features=image_similarity_orb_features,
        )

    if not sensor_processing_enabled:
        print("[input] Video-only reconstruction: WiFi and IMU priors are disabled.", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        sensor_results = {
            "wifi": {"selected_ap_count_by_session": {}, "selected_ap_union_count": 0},
            "pdr": {"pdr": {}, "trajectory_contour": {"status": "skipped", "reason": "video_only"}},
            "execution": {"status": "skipped", "reason": "video_only"},
        }
    else:
        print(f"[input] Video with WiFi and IMU/PDR processing enabled. Video/sensor alignment={args.video_sensor_alignment}.", flush=True)
        if args.video_sensor_alignment == "fuzzy":
            print(f"[sensors] Approximate recording-time association: +/-{args.video_sensor_time_uncertainty_s:g}s assumed uncertainty; "
                  "sensor candidates require visual geometry checks.", flush=True)
        log_stage(f"1-2/{len(stage_order)} WiFi tables and PDR trajectory ({resources['sensor_workers']} workers)")
        sensor_results = run_sensor_stages(
            session_dirs, output_dir, workers=resources['sensor_workers'],
            wifi_options=dict(min_detection_rate=args.min_detection_rate,
                              min_mean_rssi=args.min_mean_rssi, max_rssi_std=args.max_rssi_std,
                              min_scan_count=args.min_scan_count),
            pdr_options=dict(base_step_m=args.base_step_m, heading_offset_deg=args.heading_offset_deg,
                             heading_policy=args.pdr_heading_policy))
    selection = sensor_results['wifi']
    pdr_summary = sensor_results['pdr']['pdr']
    trajectory_contour_summary = sensor_results['pdr']['trajectory_contour']
    if sensor_processing_enabled:
        log_stage(f"{stage_labels[3]} fusing WiFi fingerprints with PDR positions")
    fused_summary = build_wifi_pdr_feature_table(output_dir) if sensor_processing_enabled else {"status": "skipped", "reason": "video_only"}
    if sensor_processing_enabled:
        log_stage_done(stage_labels[3], fused_summary)
    require_stage("WiFi/PDR feature table", fused_summary, ("ok", "skipped"))
    log_stage(f"{stage_labels[5]} extracting reconstruction video frames")
    video_frames_summary = (
        {"status": "skipped"}
        if args.skip_video_frames
        else run_video_frame_pipeline(
            root=root,
            processed_dir=output_dir,
            video_dir=args.video_dir,
            frame_interval_s=video_frame_interval_s,
            max_frames=video_max_frames,
            ffmpeg_exe=args.ffmpeg_exe,
            blur_threshold=video_blur_threshold,
            keyframe_min_change=video_keyframe_min_change,
            keyframe_min_gap=video_keyframe_min_gap,
            blur_policy=args.video_blur_policy,
            keyframe_max_gap_s=args.video_keyframe_max_gap_s,
            refine_window_s=args.video_refine_window_s,
            refine_budget_hours=args.video_refine_budget_hours,
            refine_coverage_search=args.video_refine_coverage_search,
            decode_threads=resources['video_decode_threads'],
            decode_hardware=args.video_hw_decode,
            decode_device=args.video_hw_device,
        )
    )
    log_stage_done(
        stage_labels[5],
        video_frames_summary,
        "video extraction disabled by --skip-video-frames",
    )
    require_stage("Video-only frame extraction", video_frames_summary, ("ok",))
    if video_frames_summary.get("frame_count", 0) < 3:
        raise PipelineStageError("Video-only reconstruction requires at least three retained video frames")
    video_sensor_summary = {"status": "disabled", "reason": "no_explicit_video_sensor_sync"}
    if args.video_sensor_sync is not None:
        from video_sensor_fusion import annotate_video_sensor_priors
        video_sensor_summary = annotate_video_sensor_priors(output_dir, root / args.video_sensor_sync)
        print(f"[sensors] Synchronized {video_sensor_summary.get('synchronized_frame_count', 0)}/"
              f"{video_sensor_summary.get('frame_count', 0)} frames; "
              f"route={video_sensor_summary.get('route_frame_count', 0)}, "
              f"fresh WiFi={video_sensor_summary.get('wifi_frame_count', 0)}. "
              "These are matching priors, not measured camera poses.", flush=True)
        if not video_sensor_summary.get("synchronized_frame_count"):
            print("[warn] No video frame could be synchronized; retaining visual reconstruction. "
                  "Inspect video_sensor_sync_summary.json.", flush=True)
    elif sensor_processing_enabled and args.video_sensor_alignment == "fuzzy" and not args.visual_only:
        from video_sensor_fuzzy import annotate_fuzzy_video_sensor_priors
        video_sensor_summary = annotate_fuzzy_video_sensor_priors(
            output_dir, root, args.video_dir,
            uncertainty_s=args.video_sensor_time_uncertainty_s, offset_s=args.video_sensor_offset_s,
            ffmpeg_exe=args.ffmpeg_exe,
        )
        print(f"[sensors] Fuzzy time windows for {video_sensor_summary.get('fuzzy_frame_count', 0)}/"
              f"{video_sensor_summary.get('frame_count', 0)} frames; exact synchronization is not assumed. "
              "WiFi/IMU only propose visually verified matches.", flush=True)
        if not video_sensor_summary.get("fuzzy_frame_count"):
            print("[warn] No unambiguous recording-time overlap for fuzzy sensor matching; sensor tables are retained. "
                  "Inspect video_sensor_fuzzy_summary.json.", flush=True)
    elif sensor_processing_enabled:
        from video_sensor_fusion import annotate_video_sensor_priors
        video_sensor_summary = annotate_video_sensor_priors(output_dir)
        print("[sensors] Sensor tables processed; frame association is disabled or awaiting explicit synchronization.", flush=True)
    reconstruction_output_dir = (
        (root / args.reconstruction_output).resolve()
        if not args.reconstruction_output.is_absolute()
        else args.reconstruction_output
    )
    log_stage(f"{stage_labels[6]} exporting reconstruction image dataset")
    reconstruction_summary = (
        {"status": "skipped"}
        if args.skip_reconstruction_export
        else export_reconstruction_dataset(
            root=root,
            processed_dir=output_dir,
            output_dir=reconstruction_output_dir,
            camera_height_m=args.camera_height_m,
            copy_images=True,
            include_video_frames=not args.skip_video_frames,
            include_sensor_priors=sensor_processing_enabled,
        )
    )
    log_stage_done(
        stage_labels[6],
        reconstruction_summary,
        "reconstruction export disabled by --skip-reconstruction-export",
    )
    require_stage("Reconstruction export", reconstruction_summary, ("ok", "skipped"))
    if 7 in stage_labels:
        log_stage(f"{stage_labels[7]} building WiFi positioning artifacts")
    wifi_positioning_summary = (
        {"status": "skipped"}
        if args.skip_wifi_positioning or reconstruction_summary.get("status") == "skipped"
        else run_wifi_positioning_pipeline(
            fingerprint_path=reconstruction_output_dir / "wifi_fingerprint_library.csv",
            output_dir=reconstruction_output_dir,
            grid_cell_size_m=args.wifi_grid_cell_size_m,
            localization_k=args.wifi_localization_k,
        )
    )
    if 7 in stage_labels:
        log_stage_done(stage_labels[7], wifi_positioning_summary, "WiFi positioning disabled or reconstruction export was skipped")
    require_stage("WiFi positioning", wifi_positioning_summary, ("ok", "skipped", "no_valid_fingerprints", "no_usable_position_labels"))
    colmap_workspace_dir = (
        (root / args.colmap_workspace).resolve()
        if not args.colmap_workspace.is_absolute()
        else args.colmap_workspace
    )
    log_stage(f"{stage_labels[8]} preparing COLMAP workspace and run scripts")
    colmap_summary = (
        {"status": "skipped"}
        if args.skip_colmap_prepare or reconstruction_summary.get("status") == "skipped"
        else prepare_colmap_workspace(
            root=root,
            reconstruction_dir=reconstruction_output_dir,
            workspace_dir=colmap_workspace_dir,
            colmap_dir=args.colmap_dir,
            colmap_exe=args.colmap_exe,
            colmap_mask_dir=args.colmap_mask_dir,
            camera_model=args.colmap_camera_model,
            single_camera=True,
            use_gpu=bool(hardware.get("cuda_enabled")),
            matcher=colmap_matcher,
            quality=colmap_quality,
            dense_max_image_size=colmap_dense_max_image_size,
            visual_only=args.visual_only,
            scene_completion=args.scene_completion,
            videoscene_max_clips=args.videoscene_max_clips,
            videoscene_budget_hours=args.videoscene_budget_hours,
            feature_max_image_size=colmap_feature_max_image_size,
            max_num_features=colmap_max_num_features,
            max_num_matches=colmap_max_num_matches,
            colmap_num_threads=colmap_num_threads,
            dense_profile=colmap_dense_profile,
            colmap_ba_gpu=args.colmap_ba_gpu,
            sparse_strategy=args.sparse_strategy,
            sparse_ba_growth_ratio=args.sparse_ba_growth_ratio,
            sparse_keyframe_gap_s=args.sparse_keyframe_gap_s,
            dense_max_images=args.dense_max_images,
            dense_max_source_images=args.dense_max_source_images,
            dense_coverage_policy=({
                "mode": "adaptive",
                "initial_images": min(args.dense_initial_images, args.dense_max_images) if args.dense_max_images else args.dense_initial_images,
                "min_single_view_coverage": args.dense_min_single_coverage,
                "min_three_view_coverage": args.dense_min_three_coverage,
                "max_video_gap_s": args.dense_max_video_gap_s,
            } if args.dense_selection_mode == "adaptive" else None),
            point_filter_method=args.point_filter_method,
            point_filter_max_candidates=args.point_filter_max_candidates,
            point_filter_budget_seconds=args.point_filter_budget_seconds,
            run_meshing=not args.skip_colmap_meshing,
            run_point_cloud_filter=not args.skip_point_cloud_filter,
            constrained_max_pdr_distance_m=constrained_max_pdr_distance_m,
            constrained_video_frame_window=constrained_video_frame_window,
            run_image_similarity=not args.skip_image_similarity,
            wifi_recovery_enabled=not args.skip_wifi_recovery,
            image_similarity_max_pairs=image_similarity_max_pairs,
            image_similarity_max_image_size=image_similarity_max_image_size,
            image_similarity_orb_features=image_similarity_orb_features,
            image_similarity_min_visual_score=profile_value(args, "image_similarity_min_visual_score"),
            image_similarity_workers=resources['image_similarity_workers'],
            visual_recovery_neighbors=args.visual_recovery_neighbors,
        )
    )
    log_stage_done(stage_labels[8], colmap_summary, "COLMAP preparation disabled or reconstruction export was skipped")
    require_stage("COLMAP workspace preparation", colmap_summary)
    write_json(
        output_dir / "processing_summary.json",
        {
            "status": "ok",
            "input_mode": "video_only",
            "sensor_processing_enabled": sensor_processing_enabled,
            "video_sensor_alignment": args.video_sensor_alignment,
            "root": str(root),
            "sessions": [path.name for path in session_dirs],
            "requested_reconstruction_profile": requested_profile,
            "reconstruction_profile": args.reconstruction_profile,
            "hardware_probe": hardware,
            "compute_resources": resources,
            "sharpness_backend": sharpness_diagnostics(),
            "profile_defaults": PROFILE_DEFAULTS[args.reconstruction_profile],
            "sensor_execution": sensor_results['execution'],
            "video_sensor_assistance": video_sensor_summary,
            "wifi": {
                "selected_ap_count_by_session": selection["selected_ap_count_by_session"],
                "selected_ap_union_count": selection["selected_ap_union_count"],
            },
            "pdr": pdr_summary,
            "trajectory_contour": trajectory_contour_summary,
            "wifi_pdr_features": fused_summary,
            "video_frames": video_frames_summary,
            "reconstruction_export": reconstruction_summary,
            "wifi_positioning": wifi_positioning_summary,
            "colmap_workspace": colmap_summary,
        },
    )

    if verbose:
        print("[info] selected APs:", selection["selected_ap_count_by_session"])
        print("[info] selected AP union:", selection["selected_ap_union_count"])
        for session, summary in pdr_summary.items():
            print(
                f"[info] {session}: {summary['segment_count']} segments, "
                f"{summary['step_count']} steps, {summary['distance_m']} m"
            )
        print("[info] trajectory_contour:", trajectory_contour_summary)
        print("[info] wifi_pdr_features:", fused_summary)
        print("[info] video_frames:", video_frames_summary)
        print("[info] reconstruction_export:", reconstruction_summary)
        print("[info] wifi_positioning:", wifi_positioning_summary)
        print("[info] colmap_workspace:", colmap_summary)
    print(f"[done] Preprocessing complete | sessions={len(session_dirs)} | "
          f"selected APs={selection['selected_ap_union_count']} | "
          f"COLMAP images={colmap_summary.get('image_count', 0)}", flush=True)
    print(f"[report] {output_dir / 'processing_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
