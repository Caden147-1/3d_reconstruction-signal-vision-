"""Preprocess classroom raw .sess data into WiFi, PDR, and image-ready tables.

The script intentionally uses only the Python standard library so it can run on
the raw-data folder without dependency setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from colmap_bridge import prepare_colmap_workspace
from image_mapping import build_image_wifi_tables
from reconstruction_export import export_reconstruction_dataset
from trajectory_contour import build_trajectory_contour
from video_frames import run_video_frame_pipeline
from wifi_positioning import run_wifi_positioning_pipeline


FILL_RSSI = -110
DEFAULT_BASE_STEP_M = 0.70


PROFILE_DEFAULTS = {
    "preview": {
        "max_colmap_anchor_images_per_point": 10,
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
        "colmap_load_profile": "low-peak",
        "colmap_num_threads": 2,
        "colmap_stage_cooldown_s": 10,
        "constrained_point_window": 1,
        "constrained_slot_window": 2,
        "constrained_max_pdr_distance_m": 8.0,
        "constrained_video_frame_window": 3,
        "constrained_video_anchor_point_window": 1,
        "constrained_video_anchor_limit": 12,
        "image_similarity_max_pairs": 1000,
        "image_similarity_max_image_size": 720,
        "image_similarity_orb_features": 800,
        "image_similarity_min_visual_score": 0.22,
    },
    "balanced": {
        "max_colmap_anchor_images_per_point": 12,
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
        "colmap_load_profile": "low-peak",
        "colmap_num_threads": 2,
        "colmap_stage_cooldown_s": 15,
        "constrained_point_window": 1,
        "constrained_slot_window": 2,
        "constrained_max_pdr_distance_m": 8.0,
        "constrained_video_frame_window": 4,
        "constrained_video_anchor_point_window": 1,
        "constrained_video_anchor_limit": 12,
        "image_similarity_max_pairs": 2500,
        "image_similarity_max_image_size": 800,
        "image_similarity_orb_features": 1200,
        "image_similarity_min_visual_score": 0.22,
    },
    "final": {
        "max_colmap_anchor_images_per_point": 0,
        "video_frame_interval_s": 1.0,
        "video_max_frames": 240,
        "video_blur_threshold": 60.0,
        "video_keyframe_min_change": 4.0,
        "video_keyframe_min_gap": 1,
        "colmap_dense_max_image_size": 2000,
        "colmap_feature_max_image_size": 2400,
        "colmap_max_num_features": 12000,
        "colmap_max_num_matches": 65536,
        "colmap_matcher": "hybrid",
        "colmap_quality": "extreme",
        "colmap_dense_profile": "complete",
        "colmap_load_profile": "low-peak",
        "colmap_num_threads": 2,
        "colmap_stage_cooldown_s": 20,
        "constrained_point_window": 2,
        "constrained_slot_window": 3,
        "constrained_max_pdr_distance_m": 12.0,
        "constrained_video_frame_window": 6,
        "constrained_video_anchor_point_window": 2,
        "constrained_video_anchor_limit": 24,
        "image_similarity_max_pairs": 0,
        "image_similarity_max_image_size": 900,
        "image_similarity_orb_features": 1500,
        "image_similarity_min_visual_score": 0.22,
    },
    "rtx4080": {
        "max_colmap_anchor_images_per_point": 0,
        "video_frame_interval_s": 0.75,
        "video_max_frames": 360,
        "video_blur_threshold": 50.0,
        "video_keyframe_min_change": 3.0,
        "video_keyframe_min_gap": 1,
        "colmap_dense_max_image_size": 2400,
        "colmap_feature_max_image_size": 2800,
        "colmap_max_num_features": 16000,
        "colmap_max_num_matches": 98304,
        "colmap_matcher": "hybrid",
        "colmap_quality": "extreme",
        "colmap_dense_profile": "rtx4080",
        "colmap_load_profile": "low-peak",
        "colmap_num_threads": 3,
        "colmap_stage_cooldown_s": 20,
        "constrained_point_window": 2,
        "constrained_slot_window": 3,
        "constrained_max_pdr_distance_m": 14.0,
        "constrained_video_frame_window": 7,
        "constrained_video_anchor_point_window": 2,
        "constrained_video_anchor_limit": 30,
        "image_similarity_max_pairs": 0,
        "image_similarity_max_image_size": 1000,
        "image_similarity_orb_features": 1800,
        "image_similarity_min_visual_score": 0.22,
    },
    "complete": {
        "max_colmap_anchor_images_per_point": 0,
        "video_frame_interval_s": 0.5,
        "video_max_frames": 0,
        "video_blur_threshold": 40.0,
        "video_keyframe_min_change": 2.0,
        "video_keyframe_min_gap": 1,
        "colmap_dense_max_image_size": 3200,
        "colmap_feature_max_image_size": 3200,
        "colmap_max_num_features": 20000,
        "colmap_max_num_matches": 131072,
        "colmap_matcher": "hybrid",
        "colmap_quality": "extreme",
        "colmap_dense_profile": "complete",
        "colmap_load_profile": "low-peak",
        "colmap_num_threads": 3,
        "colmap_stage_cooldown_s": 30,
        "constrained_point_window": 3,
        "constrained_slot_window": 4,
        "constrained_max_pdr_distance_m": 16.0,
        "constrained_video_frame_window": 8,
        "constrained_video_anchor_point_window": 3,
        "constrained_video_anchor_limit": 36,
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
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
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
            if yaw is not None:
                heading_samples.append((sample.elapsed_ts, sample.epoch_ts, yaw))
        if heading_samples:
            return heading_samples
    return []


def interpolate_angle(samples: Sequence[Tuple[int, int, float]], target_elapsed_ts: int) -> float:
    if not samples:
        return 0.0
    if target_elapsed_ts <= samples[0][0]:
        return samples[0][2]
    if target_elapsed_ts >= samples[-1][0]:
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
    left = samples[lo - 1]
    span = right[0] - left[0]
    if span <= 0:
        return left[2]
    ratio = (target_elapsed_ts - left[0]) / span
    diff = wrap_angle(right[2] - left[2])
    return wrap_angle(left[2] + ratio * diff)


def wrap_angle(angle: float) -> float:
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
) -> List[Step]:
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

    for step_index, (segment, elapsed_ts, epoch_ts, peak_accel, accel_range) in enumerate(detected, start=1):
        normalized_range = max(accel_range, 0.05) / max(median_range, 0.05)
        step_length = base_step_m * (normalized_range ** 0.25)
        step_length = min(1.10, max(0.35, step_length))
        heading = wrap_angle(interpolate_angle(headings, elapsed_ts) + heading_offset)

        # Android yaw zero is roughly "north"; this convention leaves the final
        # frame arbitrary until checkpoint anchors rotate/translate it.
        x += step_length * math.sin(heading)
        y += step_length * math.cos(heading)
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

    for session_dir in session_dirs:
        session = session_dir.name
        wifi_path = session_dir / "wifiScanner.dat"
        scan_count = 0
        stats: Dict[str, Dict[str, Any]] = {}

        for scan_index, (elapsed_ts, epoch_ts, payload) in enumerate(parse_tab_json(wifi_path), start=1):
            if not isinstance(payload, list):
                continue
            scan_count += 1
            per_scan: Dict[str, float] = {}
            meta_by_mac: Dict[str, Dict[str, Any]] = {}
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
                        "rssi": "" if math.isnan(rssi) else round(rssi, 3),
                    }
                )
                if math.isnan(rssi):
                    continue
                if mac not in per_scan or rssi > per_scan[mac]:
                    per_scan[mac] = rssi
                    meta_by_mac[mac] = {"ssid": name, "freq": freq}

                stat = stats.setdefault(
                    mac,
                    {
                        "rssis": [],
                        "scan_indices": set(),
                        "names": {},
                        "freqs": {},
                        "first_elapsed_ts": elapsed_ts,
                        "last_elapsed_ts": elapsed_ts,
                    },
                )
                stat["rssis"].append(rssi)
                stat["scan_indices"].add(scan_index)
                stat["names"][name] = stat["names"].get(name, 0) + 1
                stat["freqs"][freq] = stat["freqs"].get(freq, 0) + 1
                stat["last_elapsed_ts"] = elapsed_ts

            scan_rows.append(
                {
                    "session": session,
                    "scan_index": scan_index,
                    "elapsed_ts": elapsed_ts,
                    "epoch_ts": epoch_ts,
                    "rssi_by_mac": per_scan,
                    "meta_by_mac": meta_by_mac,
                }
            )

        session_scan_counts[session] = scan_count
        ap_stats_by_session[session] = {}
        selected_by_session[session] = []
        for mac, stat in stats.items():
            rssis = stat["rssis"]
            scan_indices = stat["scan_indices"]
            mean_rssi = sum(rssis) / len(rssis)
            std_rssi = pstdev(rssis)
            detection_rate = len(scan_indices) / scan_count if scan_count else 0.0
            top_name = most_common_key(stat["names"])
            top_freq = most_common_key(stat["freqs"])
            row = {
                "session": session,
                "mac": mac,
                "ssid": top_name,
                "freq": top_freq,
                "scan_count": len(scan_indices),
                "entry_count": len(rssis),
                "total_scans": scan_count,
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

    wide_fieldnames = ["session", "scan_index", "elapsed_ts", "epoch_ts"] + [
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
        wide_rows.append(row)

    write_csv(output_dir / "wifi_wide_selected.csv", wide_fieldnames, wide_rows)

    selection_obj = {
        "thresholds": {
            "min_detection_rate": min_detection_rate,
            "min_mean_rssi": min_mean_rssi,
            "max_rssi_std": max_rssi_std,
            "min_scan_count": min_scan_count,
            "missing_rssi_fill": FILL_RSSI,
        },
        "session_scan_counts": session_scan_counts,
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
        steps = compute_pdr(session_dir, segments, base_step_m, heading_offset_deg)

        summary[session] = {
            "movement_event_count": len(events),
            "segment_count": len(segments),
            "step_count": len(steps),
            "distance_m": round(sum(step.step_length_m for step in steps), 6),
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
            anchor_rows.append(
                {
                    "session": session,
                    "anchor_id": f"{session}__{event.event}_{event.elapsed_ts}",
                    "event": event.event,
                    "elapsed_ts": event.elapsed_ts,
                    "epoch_ts": event.epoch_ts,
                    "nearest_step_index": step_index,
                    "pdr_x": round(x, 6),
                    "pdr_y": round(y, 6),
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
                    "pdr_x": round(x, 6),
                    "pdr_y": round(y, 6),
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
            row = {
                "session": session,
                "segment_id": step.segment_id,
                "step_index": step.step_index,
                "elapsed_ts": step.elapsed_ts,
                "epoch_ts": step.epoch_ts,
                "peak_accel": round(step.peak_accel, 6),
                "accel_range": round(step.accel_range, 6),
                "step_length_m": round(step.step_length_m, 6),
                "heading_rad": round(step.heading_rad, 9),
                "heading_deg": round(step.heading_deg, 6),
                "pdr_x": round(step.pdr_x, 6),
                "pdr_y": round(step.pdr_y, 6),
            }
            step_rows.append(row)
            path_rows.append(
                {
                    "session": session,
                    "point_type": "STEP",
                    "segment_id": step.segment_id,
                    "step_index": step.step_index,
                    "elapsed_ts": step.elapsed_ts,
                    "epoch_ts": step.epoch_ts,
                    "pdr_x": round(step.pdr_x, 6),
                    "pdr_y": round(step.pdr_y, 6),
                    "step_length_m": round(step.step_length_m, 6),
                    "heading_deg": round(step.heading_deg, 6),
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


def write_readme(output_dir: Path) -> None:
    content = """# Processed Classroom Raw Data

Generated by `main.py`.

## Main Outputs

- `wifi_long.csv`: one row per observed AP entry in every WiFi scan.
- `ap_stats.csv`: AP frequency, RSSI strength/stability, and selected flag.
- `selected_aps.json`: AP filtering thresholds and selected MAC lists.
- `wifi_wide_selected.csv`: one row per WiFi scan, selected AP RSSI columns only; missing RSSI is `-110`.
- `movement_events.csv`: START / RESTART / CHECKPOINT_PAUSE / STOP events.
- `movement_segments.csv`: active walking segments inferred from movement events.
- `pdr_steps.csv`: detected PDR steps with heading, step length, and relative position.
- `pdr_path.csv`: event and step points in one relative trajectory table.
- `pdr_trajectory_contour.csv`: time-ordered PDR centerline with a bounded route corridor.
- `pdr_trajectory_edges.csv`: observed links in that continuous route contour.
- `wifi_pdr_features.csv`: selected WiFi RSSI table plus interpolated relative PDR position per scan.
- `image_wifi_long.csv`: one row per ordered image, with its bound WiFi checkpoint feature row.
- `image_wifi_points.csv`: one row per WiFi checkpoint, with the ordered checkpoint photos in columns.
- `image_wifi_summary.json`: image/WiFi count checks and mapping status.
- `alignment_anchor_template.csv`: fill `room_x` and `room_y` for known classroom checkpoints, then use those anchors to align/repair coordinates.

## Image/WiFi Mapping

Photos are sorted by filename number. By default every `CHECKPOINT_PAUSE` event
is treated as one WiFi point, and every point consumes `--images-per-wifi-point`
photos from `pic` (default: 20 photos covering one full rotation).
The nearest WiFi scan row is copied into the image mapping outputs.

## Coordinate Note

The raw files in this folder do not contain explicit classroom `x/y` labels. The PDR
trajectory is therefore relative. To unify it into the classroom coordinate system,
fill the anchor template with known checkpoint coordinates, then fit a similarity
or affine transform from `pdr_x,pdr_y` to `room_x,room_y`.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(content, encoding="utf-8")


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
                pdr_x = float(row["pdr_x"])
                pdr_y = float(row["pdr_y"])
            except (KeyError, TypeError, ValueError):
                continue
            pdr_by_session.setdefault(row["session"], []).append(
                {
                    "elapsed_ts": elapsed_ts,
                    "pdr_x": pdr_x,
                    "pdr_y": pdr_y,
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


def interpolate_pdr_state(points: Sequence[Dict[str, Any]], target_elapsed_ts: int) -> Dict[str, Any]:
    empty = {
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
            "pdr_x": point["pdr_x"],
            "pdr_y": point["pdr_y"],
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
            "pdr_x": point["pdr_x"],
            "pdr_y": point["pdr_y"],
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
    left = points[lo - 1]
    span = right["elapsed_ts"] - left["elapsed_ts"]
    ratio = 0.0 if span <= 0 else (target_elapsed_ts - left["elapsed_ts"]) / span
    contour_id, contour_distance, route_distance, contour_status = trajectory_state(left, right, ratio)
    return {
        "pdr_x": left["pdr_x"] + ratio * (right["pdr_x"] - left["pdr_x"]),
        "pdr_y": left["pdr_y"] + ratio * (right["pdr_y"] - left["pdr_y"]),
        "status": "exact_or_duplicate" if span <= 0 else "interpolated",
        "gap_ms": "0" if span <= 0 else str(span),
        "left_type": left["point_type"],
        "right_type": right["point_type"],
        "trajectory_contour_id": contour_id,
        "trajectory_contour_distance_m": contour_distance,
        "trajectory_distance_m": route_distance,
        "trajectory_status": contour_status,
    }


def interpolate_pdr_position(
    points: Sequence[Dict[str, Any]],
    target_elapsed_ts: int,
) -> Tuple[Optional[float], Optional[float], str, str, str, str]:
    state = interpolate_pdr_state(points, target_elapsed_ts)
    return (
        state["pdr_x"],
        state["pdr_y"],
        str(state["status"]),
        str(state["gap_ms"]),
        str(state["left_type"]),
        str(state["right_type"]),
    )


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
    load_profile: str,
    colmap_num_threads: int,
    stage_cooldown_s: int,
    dense_max_image_size: int,
    feature_max_image_size: int,
    max_num_features: int,
    matcher: str,
    dense_profile: str,
    video_max_frames: int,
    max_anchor_images_per_point: int,
    image_similarity_enabled: bool,
    image_similarity_max_pairs: int,
    image_similarity_max_image_size: int,
    image_similarity_orb_features: int,
) -> None:
    anchor_text = "all" if max_anchor_images_per_point <= 0 else str(max_anchor_images_per_point)
    video_text = "uncapped" if video_max_frames <= 0 else str(video_max_frames)
    similarity_text = (
        "disabled"
        if not image_similarity_enabled
        else ("all constrained pairs" if image_similarity_max_pairs <= 0 else f"top {image_similarity_max_pairs} pairs")
    )
    print(
        "[load] "
        f"profile={profile}, load_profile={load_profile}, matcher={matcher}, dense_profile={dense_profile}, "
        f"dense_max_image_size={dense_max_image_size}, feature_max_image_size={feature_max_image_size}, "
        f"max_features={max_num_features}, anchor_images_per_point={anchor_text}, "
        f"video_max_frames={video_text}, colmap_threads={colmap_num_threads}, "
        f"stage_cooldown_s={stage_cooldown_s}, image_similarity={similarity_text}, "
        f"image_similarity_size={image_similarity_max_image_size}, image_similarity_orb={image_similarity_orb_features}",
        flush=True,
    )
    if profile in {"final", "rtx4080", "complete"} or matcher in {"hybrid", "exhaustive"} or dense_max_image_size >= 2000:
        print(
            "[warn] This is a high-quality reconstruction profile. Peak load is limited by "
            "--colmap-load-profile/--colmap-num-threads, so it may run longer.",
            flush=True,
        )


def prepare_colmap_workspace_from_args(
    args: argparse.Namespace,
    load_profile_override: Optional[str] = None,
    colmap_num_threads_override: Optional[int] = None,
    stage_cooldown_s_override: Optional[int] = None,
) -> Dict[str, Any]:
    effective_args = argparse.Namespace(**vars(args))
    hardware = detect_nvidia_gpu()
    if effective_args.reconstruction_profile == "auto":
        effective_args.reconstruction_profile = automatic_profile(hardware)

    root = effective_args.root.resolve()
    reconstruction_output_dir = (
        (root / effective_args.reconstruction_output).resolve()
        if not effective_args.reconstruction_output.is_absolute()
        else effective_args.reconstruction_output
    )
    colmap_workspace_dir = (
        (root / effective_args.colmap_workspace).resolve()
        if not effective_args.colmap_workspace.is_absolute()
        else effective_args.colmap_workspace
    )

    colmap_dense_max_image_size = profile_value(effective_args, "colmap_dense_max_image_size")
    colmap_dense_profile = profile_value(effective_args, "colmap_dense_profile")
    colmap_matcher = profile_value(effective_args, "colmap_matcher")
    colmap_quality = profile_value(effective_args, "colmap_quality")
    colmap_feature_max_image_size = profile_value(effective_args, "colmap_feature_max_image_size")
    colmap_max_num_features = profile_value(effective_args, "colmap_max_num_features")
    colmap_max_num_matches = profile_value(effective_args, "colmap_max_num_matches")
    colmap_load_profile = load_profile_override or profile_value(effective_args, "colmap_load_profile")
    colmap_num_threads = (
        colmap_num_threads_override
        if colmap_num_threads_override is not None
        else profile_value(effective_args, "colmap_num_threads")
    )
    colmap_stage_cooldown_s = (
        stage_cooldown_s_override
        if stage_cooldown_s_override is not None
        else profile_value(effective_args, "colmap_stage_cooldown_s")
    )
    constrained_point_window = profile_value(effective_args, "constrained_point_window")
    constrained_slot_window = profile_value(effective_args, "constrained_slot_window")
    constrained_max_pdr_distance_m = profile_value(effective_args, "constrained_max_pdr_distance_m")
    constrained_video_frame_window = profile_value(effective_args, "constrained_video_frame_window")
    constrained_video_anchor_point_window = profile_value(effective_args, "constrained_video_anchor_point_window")
    constrained_video_anchor_limit = profile_value(effective_args, "constrained_video_anchor_limit")
    image_similarity_max_pairs = profile_value(effective_args, "image_similarity_max_pairs")
    image_similarity_max_image_size = profile_value(effective_args, "image_similarity_max_image_size")
    image_similarity_orb_features = profile_value(effective_args, "image_similarity_orb_features")
    image_similarity_min_visual_score = profile_value(effective_args, "image_similarity_min_visual_score")

    print(
        "[info] preparing COLMAP scripts for "
        f"load_profile={colmap_load_profile}, colmap_threads={colmap_num_threads}, "
        f"stage_cooldown_s={colmap_stage_cooldown_s}",
        flush=True,
    )
    return prepare_colmap_workspace(
        root=root,
        reconstruction_dir=reconstruction_output_dir,
        workspace_dir=colmap_workspace_dir,
        colmap_dir=effective_args.colmap_dir,
        colmap_exe=effective_args.colmap_exe,
        colmap_mask_dir=effective_args.colmap_mask_dir,
        camera_model=effective_args.colmap_camera_model,
        single_camera=True,
        use_gpu=bool(hardware.get("cuda_enabled")),
        matcher=colmap_matcher,
        quality=colmap_quality,
        dense_max_image_size=colmap_dense_max_image_size,
        visual_only=effective_args.visual_only,
        feature_max_image_size=colmap_feature_max_image_size,
        max_num_features=colmap_max_num_features,
        max_num_matches=colmap_max_num_matches,
        load_profile=colmap_load_profile,
        colmap_num_threads=colmap_num_threads,
        stage_cooldown_s=colmap_stage_cooldown_s,
        dense_profile=colmap_dense_profile,
        run_meshing=not effective_args.skip_colmap_meshing,
        run_point_cloud_filter=not effective_args.skip_point_cloud_filter,
        constrained_point_window=constrained_point_window,
        constrained_slot_window=constrained_slot_window,
        constrained_max_pdr_distance_m=constrained_max_pdr_distance_m,
        constrained_video_frame_window=constrained_video_frame_window,
        constrained_video_anchor_point_window=constrained_video_anchor_point_window,
        constrained_video_anchor_limit=constrained_video_anchor_limit,
        run_image_similarity=not effective_args.skip_image_similarity,
        image_similarity_max_pairs=image_similarity_max_pairs,
        image_similarity_max_image_size=image_similarity_max_image_size,
        image_similarity_orb_features=image_similarity_orb_features,
        image_similarity_min_visual_score=image_similarity_min_visual_score,
    )


def log_stage(message: str) -> None:
    print(f"[stage] {message}", flush=True)


def log_stage_done(number: int, summary: Any, skipped_reason: Optional[str] = None) -> None:
    if isinstance(summary, dict) and summary.get("status") == "skipped":
        print(f"[skip] {number}/8 {skipped_reason or 'disabled by command-line options'}", flush=True)
    else:
        print(f"[done] {number}/8 completed", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Folder containing .sess directories.")
    parser.add_argument("--output", type=Path, default=Path("processed"), help="Output folder.")
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
    parser.add_argument("--pic-dir", type=Path, default=Path("pic"), help="Folder containing ordered photos.")
    parser.add_argument("--images-per-wifi-point", type=int, default=20)
    parser.add_argument("--point-event", default="CHECKPOINT_PAUSE")
    parser.add_argument("--skip-image-wifi", action="store_true", help="Do not build image/WiFi mapping outputs.")
    parser.add_argument("--video-dir", type=Path, default=Path("video"), help="Folder containing route/base videos.")
    parser.add_argument("--skip-video-frames", action="store_true", help="Do not extract video frames for COLMAP.")
    parser.add_argument("--video-frame-interval-s", type=float)
    parser.add_argument("--video-max-frames", type=int, help="Max frames to extract per video; use 0 for no cap.")
    parser.add_argument("--video-blur-threshold", type=float, help="Drop video frames below this Laplacian blur score; use 0 to disable.")
    parser.add_argument("--video-keyframe-min-change", type=float, help="Drop near-duplicate video frames below this mean grayscale difference; use 0 to disable.")
    parser.add_argument("--video-keyframe-min-gap", type=int, help="Minimum gap in extracted-frame steps between kept video keyframes.")
    parser.add_argument("--ffmpeg-exe", type=Path, help="Optional path to ffmpeg.exe.")
    parser.add_argument("--reconstruction-output", type=Path, default=Path("reconstruction_dataset"))
    parser.add_argument("--skip-reconstruction-export", action="store_true")
    parser.add_argument(
        "--max-colmap-anchor-images-per-point",
        type=int,
        help="Limit exported point photos per WiFi/checkpoint; balanced defaults to circular 12-of-20, final defaults to all.",
    )
    parser.add_argument("--camera-height-m", type=float, default=1.50)
    parser.add_argument("--slot-yaw-start-deg", type=float, default=0.0)
    parser.add_argument(
        "--slot-yaw-step-deg",
        type=float,
        help="Yaw step between ordered point photos. Defaults to 360 / images-per-wifi-point.",
    )
    parser.add_argument("--skip-wifi-positioning", action="store_true")
    parser.add_argument("--wifi-grid-cell-size-m", type=float, default=1.0)
    parser.add_argument("--wifi-localization-k", type=int, default=3)
    parser.add_argument("--colmap-workspace", type=Path, default=Path("colmap_workspace"))
    parser.add_argument("--colmap-dir", type=Path, default=Path("colmap-x64-windows-cuda"))
    parser.add_argument("--colmap-exe", type=Path)
    parser.add_argument(
        "--colmap-mask-dir",
        type=Path,
        help=(
            "Optional COLMAP mask directory for semantic clutter filtering. "
            "Masks should follow image_list names, for example image.jpg.png or image.png."
        ),
    )
    parser.add_argument("--skip-colmap-prepare", action="store_true")
    parser.add_argument("--visual-only", action="store_true", help="Prepare COLMAP without PDR distance fallback or PDR model alignment.")
    parser.add_argument("--colmap-camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--colmap-matcher", choices=["constrained", "hybrid", "exhaustive", "sequential"])
    parser.add_argument("--colmap-quality", choices=["low", "medium", "high", "extreme"])
    parser.add_argument("--colmap-dense-max-image-size", type=int)
    parser.add_argument("--colmap-dense-profile", choices=["balanced", "rtx4080", "complete"], help="Dense/MVS parameter profile written into run_colmap_dense.bat.")
    parser.add_argument(
        "--colmap-load-profile",
        choices=["low-peak", "standard", "max"],
        help="Limit per-stage peak load without changing reconstruction quality settings.",
    )
    parser.add_argument(
        "--colmap-num-threads",
        type=int,
        help="Thread cap passed to supported COLMAP stages; use 0 for COLMAP defaults.",
    )
    parser.add_argument(
        "--colmap-stage-cooldown-s",
        type=int,
        help="Seconds to wait between heavy COLMAP stages to lower sustained peak load.",
    )
    parser.add_argument("--skip-colmap-meshing", action="store_true", help="Skip Delaunay/Poisson mesh commands in the generated dense script.")
    parser.add_argument("--skip-point-cloud-filter", action="store_true", help="Skip the filtered fused point cloud post-processing output.")
    parser.add_argument("--colmap-feature-max-image-size", type=int)
    parser.add_argument("--colmap-max-num-features", type=int)
    parser.add_argument("--colmap-max-num-matches", type=int)
    parser.add_argument("--constrained-point-window", type=int)
    parser.add_argument("--constrained-slot-window", type=int)
    parser.add_argument(
        "--constrained-max-pdr-distance-m",
        type=float,
        help=(
            "Maximum along-route PDR contour distance for extra checkpoint matches. "
            "Older exports without contour metadata fall back to XY PDR proximity."
        ),
    )
    parser.add_argument("--constrained-video-frame-window", type=int)
    parser.add_argument("--constrained-video-anchor-point-window", type=int)
    parser.add_argument("--constrained-video-anchor-limit", type=int)
    parser.add_argument("--skip-image-similarity", action="store_true", help="Skip P1 image-similarity diagnostics for constrained match pairs.")
    parser.add_argument("--image-similarity-max-pairs", type=int, help="Analyze all constrained pairs by default when profile value is 0; set a positive cap for a faster sample.")
    parser.add_argument("--image-similarity-max-image-size", type=int, help="Resize longest image side before ORB similarity analysis.")
    parser.add_argument("--image-similarity-orb-features", type=int, help="ORB feature count per image for match-pair diagnostics.")
    parser.add_argument("--image-similarity-min-visual-score", type=float, help="Score threshold for feature-overlap diagnostic labels.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    requested_profile = args.reconstruction_profile
    hardware = detect_nvidia_gpu()
    if requested_profile == "auto":
        args.reconstruction_profile = automatic_profile(hardware)
    root = args.root.resolve()
    output_dir = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    session_dirs = find_session_dirs(root)
    if not session_dirs:
        raise SystemExit(f"No .sess directories found under {root}")

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
    max_colmap_anchor_images_per_point = profile_value(args, "max_colmap_anchor_images_per_point")
    colmap_dense_max_image_size = profile_value(args, "colmap_dense_max_image_size")
    colmap_dense_profile = profile_value(args, "colmap_dense_profile")
    colmap_matcher = profile_value(args, "colmap_matcher")
    colmap_quality = profile_value(args, "colmap_quality")
    colmap_feature_max_image_size = profile_value(args, "colmap_feature_max_image_size")
    colmap_max_num_features = profile_value(args, "colmap_max_num_features")
    colmap_max_num_matches = profile_value(args, "colmap_max_num_matches")
    colmap_load_profile = profile_value(args, "colmap_load_profile")
    colmap_num_threads = profile_value(args, "colmap_num_threads")
    colmap_stage_cooldown_s = profile_value(args, "colmap_stage_cooldown_s")
    constrained_point_window = profile_value(args, "constrained_point_window")
    constrained_slot_window = profile_value(args, "constrained_slot_window")
    constrained_max_pdr_distance_m = profile_value(args, "constrained_max_pdr_distance_m")
    constrained_video_frame_window = profile_value(args, "constrained_video_frame_window")
    constrained_video_anchor_point_window = profile_value(args, "constrained_video_anchor_point_window")
    constrained_video_anchor_limit = profile_value(args, "constrained_video_anchor_limit")
    image_similarity_max_pairs = profile_value(args, "image_similarity_max_pairs")
    image_similarity_max_image_size = profile_value(args, "image_similarity_max_image_size")
    image_similarity_orb_features = profile_value(args, "image_similarity_orb_features")
    print_workload_summary(
        profile=args.reconstruction_profile,
        load_profile=colmap_load_profile,
        colmap_num_threads=colmap_num_threads,
        stage_cooldown_s=colmap_stage_cooldown_s,
        dense_max_image_size=colmap_dense_max_image_size,
        feature_max_image_size=colmap_feature_max_image_size,
        max_num_features=colmap_max_num_features,
        matcher=colmap_matcher,
        dense_profile=colmap_dense_profile,
        video_max_frames=video_max_frames,
        max_anchor_images_per_point=max_colmap_anchor_images_per_point,
        image_similarity_enabled=not args.skip_image_similarity,
        image_similarity_max_pairs=image_similarity_max_pairs,
        image_similarity_max_image_size=image_similarity_max_image_size,
        image_similarity_orb_features=image_similarity_orb_features,
    )

    log_stage("1/8 parsing WiFi scans and selecting stable APs")
    selection = build_wifi_tables(
        session_dirs=session_dirs,
        output_dir=output_dir,
        min_detection_rate=args.min_detection_rate,
        min_mean_rssi=args.min_mean_rssi,
        max_rssi_std=args.max_rssi_std,
        min_scan_count=args.min_scan_count,
    )
    log_stage_done(1, selection)
    log_stage("2/8 computing IMU/PDR movement tables and route contour")
    pdr_summary = build_pdr_tables(
        session_dirs=session_dirs,
        output_dir=output_dir,
        base_step_m=args.base_step_m,
        heading_offset_deg=args.heading_offset_deg,
    )
    trajectory_contour_summary = build_trajectory_contour(
        input_path=output_dir / "pdr_path.csv",
        contour_path=output_dir / "pdr_trajectory_contour.csv",
        edges_path=output_dir / "pdr_trajectory_edges.csv",
        summary_path=output_dir / "pdr_trajectory_contour_summary.json",
    )
    log_stage_done(2, {"pdr": pdr_summary, "trajectory_contour": trajectory_contour_summary})
    log_stage("3/8 fusing WiFi fingerprints with PDR positions")
    fused_summary = build_wifi_pdr_feature_table(output_dir)
    log_stage_done(3, fused_summary)
    log_stage("4/8 mapping checkpoint photos to WiFi/PDR points")
    image_summary = (
        {"status": "skipped"}
        if args.skip_image_wifi
        else build_image_wifi_tables(
            root=root,
            session_dirs=session_dirs,
            output_dir=output_dir,
            pic_dir=args.pic_dir,
            images_per_point=args.images_per_wifi_point,
            point_event=args.point_event,
        )
    )
    log_stage_done(4, image_summary, "checkpoint photo mapping disabled by --skip-image-wifi")
    log_stage("5/8 extracting route video bridge frames")
    video_frames_summary = (
        {"status": "skipped"}
        if args.skip_video_frames or args.skip_image_wifi
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
        )
    )
    log_stage_done(
        5,
        video_frames_summary,
        "video extraction disabled by --skip-video-frames or because image/WiFi mapping is disabled",
    )
    reconstruction_output_dir = (
        (root / args.reconstruction_output).resolve()
        if not args.reconstruction_output.is_absolute()
        else args.reconstruction_output
    )
    slot_yaw_step_deg = (
        args.slot_yaw_step_deg
        if args.slot_yaw_step_deg is not None
        else 360.0 / max(1, args.images_per_wifi_point)
    )
    log_stage("6/8 exporting reconstruction image dataset")
    reconstruction_summary = (
        {"status": "skipped"}
        if args.skip_reconstruction_export or args.skip_image_wifi
        else export_reconstruction_dataset(
            root=root,
            processed_dir=output_dir,
            output_dir=reconstruction_output_dir,
            camera_height_m=args.camera_height_m,
            slot_yaw_start_deg=args.slot_yaw_start_deg,
            slot_yaw_step_deg=slot_yaw_step_deg,
            copy_images=True,
            max_anchor_images_per_point=max_colmap_anchor_images_per_point,
            include_video_frames=not args.skip_video_frames,
        )
    )
    log_stage_done(
        6,
        reconstruction_summary,
        "reconstruction export disabled by --skip-reconstruction-export or because image/WiFi mapping is disabled",
    )
    log_stage("7/8 building WiFi positioning artifacts")
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
    log_stage_done(
        7,
        wifi_positioning_summary,
        "WiFi positioning disabled or reconstruction export was skipped",
    )
    colmap_workspace_dir = (
        (root / args.colmap_workspace).resolve()
        if not args.colmap_workspace.is_absolute()
        else args.colmap_workspace
    )
    log_stage("8/8 preparing COLMAP workspace and run scripts")
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
            feature_max_image_size=colmap_feature_max_image_size,
            max_num_features=colmap_max_num_features,
            max_num_matches=colmap_max_num_matches,
            load_profile=colmap_load_profile,
            colmap_num_threads=colmap_num_threads,
            stage_cooldown_s=colmap_stage_cooldown_s,
            dense_profile=colmap_dense_profile,
            run_meshing=not args.skip_colmap_meshing,
            run_point_cloud_filter=not args.skip_point_cloud_filter,
            constrained_point_window=constrained_point_window,
            constrained_slot_window=constrained_slot_window,
            constrained_max_pdr_distance_m=constrained_max_pdr_distance_m,
            constrained_video_frame_window=constrained_video_frame_window,
            constrained_video_anchor_point_window=constrained_video_anchor_point_window,
            constrained_video_anchor_limit=constrained_video_anchor_limit,
            run_image_similarity=not args.skip_image_similarity,
            image_similarity_max_pairs=image_similarity_max_pairs,
            image_similarity_max_image_size=image_similarity_max_image_size,
            image_similarity_orb_features=image_similarity_orb_features,
            image_similarity_min_visual_score=profile_value(args, "image_similarity_min_visual_score"),
        )
    )
    log_stage_done(8, colmap_summary, "COLMAP preparation disabled or reconstruction export was skipped")
    write_readme(output_dir)
    write_json(
        output_dir / "processing_summary.json",
        {
            "root": str(root),
            "sessions": [path.name for path in session_dirs],
            "requested_reconstruction_profile": requested_profile,
            "reconstruction_profile": args.reconstruction_profile,
            "hardware_probe": hardware,
            "profile_defaults": PROFILE_DEFAULTS[args.reconstruction_profile],
            "wifi": {
                "selected_ap_count_by_session": selection["selected_ap_count_by_session"],
                "selected_ap_union_count": selection["selected_ap_union_count"],
            },
            "pdr": pdr_summary,
            "trajectory_contour": trajectory_contour_summary,
            "wifi_pdr_features": fused_summary,
            "image_wifi": image_summary,
            "video_frames": video_frames_summary,
            "reconstruction_export": reconstruction_summary,
            "wifi_positioning": wifi_positioning_summary,
            "colmap_workspace": colmap_summary,
        },
    )

    print("[info] selected APs:", selection["selected_ap_count_by_session"])
    print("[info] selected AP union:", selection["selected_ap_union_count"])
    for session, summary in pdr_summary.items():
        print(
            f"[info] {session}: {summary['segment_count']} segments, "
            f"{summary['step_count']} steps, {summary['distance_m']} m"
        )
    print("[info] trajectory_contour:", trajectory_contour_summary)
    print("[info] wifi_pdr_features:", fused_summary)
    print("[info] image_wifi:", image_summary)
    print("[info] video_frames:", video_frames_summary)
    print("[info] reconstruction_export:", reconstruction_summary)
    print("[info] wifi_positioning:", wifi_positioning_summary)
    print("[info] colmap_workspace:", colmap_summary)
    print("[done] preprocessing complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
