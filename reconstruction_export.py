"""Export processed captures into a reconstruction-friendly dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from wifi_observations import WIFI_QUALITY_FIELDS, WIFI_ASSOCIATION_FIELDS, valid_fingerprint, fingerprint_confidence
from video_frames import VIDEO_FRAME_PROVENANCE_FIELDS, bind_frame_to_route


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def resolve_source_image(root: Path, image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else root / path


def exported_image_name(row: Dict[str, str]) -> str:
    point_index = safe_int(row.get("global_point_index"))
    slot = safe_int(row.get("image_slot"))
    original_name = Path(row.get("image_name", "")).name
    return f"point_{point_index:03d}_slot_{slot:02d}_{original_name}"


def safe_name_part(value: str, default: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or default


def exported_video_frame_name(row: Dict[str, str]) -> str:
    sequence_index = safe_int(row.get("video_sequence_index"), safe_int(row.get("video_frame_index")))
    video_stem = safe_name_part(Path(row.get("video_name", "video")).stem, default="video")
    frame_name = Path(row.get("frame_name", "")).name
    return f"video_base_{sequence_index:06d}_{video_stem}_{frame_name}"


def audit_unassigned_images(
    root: Path,
    processed_dir: Path,
    mapped_image_rows: Sequence[Dict[str, str]],
    include_unassigned_images: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Select decodable extra photos without inventing checkpoint metadata.

    Eligibility establishes file integrity and exact-byte uniqueness only. It
    does not establish that a photo belongs to the reconstructed scene; later
    visual geometric verification must establish usable overlap.
    """
    root = root.resolve()
    _, candidates = read_csv(processed_dir / "unassigned_images.csv")
    capture_dir = root / "pic"
    mapping_summary_path = processed_dir / "image_wifi_summary.json"
    if mapping_summary_path.is_file():
        with mapping_summary_path.open(encoding="utf-8") as handle:
            mapping_summary = json.load(handle)
        if mapping_summary.get("pic_dir"):
            capture_dir = resolve_source_image(root, str(mapping_summary["pic_dir"]))
    capture_dir = capture_dir.resolve()
    mapped_paths = {
        resolve_source_image(root, row["image_path"]).resolve()
        for row in mapped_image_rows if row.get("image_path")
    }
    accepted: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    pending: List[Tuple[Path, Dict[str, Any]]] = []
    seen_paths: set[Path] = set()
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for index, row in enumerate(candidates, start=1):
        record: Dict[str, Any] = {
            "candidate_index": index,
            "image_name": row.get("image_name", ""),
            "source_image_path": row.get("image_path", ""),
            "original_unassigned_reason": row.get("reason", ""),
            "status": "rejected",
            "reason": "",
            "exported_image_file": "",
            "source_sha256": "",
            "source_size_bytes": "",
            "width": "",
            "height": "",
            "duplicate_of": "",
        }
        audit.append(record)
        if not include_unassigned_images:
            record["reason"] = "disabled_by_option"
            continue
        if not row.get("image_path", "").strip():
            record["reason"] = "missing_source_path"
            continue
        try:
            source = resolve_source_image(root, row["image_path"]).resolve()
            if not source.is_relative_to(capture_dir):
                record["reason"] = "outside_capture_directory"
                continue
            record["source_image_path"] = relative_path(source, root)
            if source in mapped_paths:
                record["reason"] = "already_mapped_source"
                continue
            if source in seen_paths:
                record["reason"] = "duplicate_source_path"
                continue
            seen_paths.add(source)
            if source.suffix.lower() not in suffixes:
                record["reason"] = "unsupported_image_extension"
                continue
            if not source.is_file():
                record["reason"] = "source_not_found_or_not_file"
                continue
            record["source_size_bytes"] = source.stat().st_size
            pending.append((source, record))
        except (OSError, RuntimeError, ValueError) as error:
            record["reason"] = "invalid_source_path"
            record["error"] = str(error)

    # Identical bytes have identical file sizes. Hash only mapped sources that
    # could duplicate an extra photo, avoiding full decoding of mapped images.
    candidate_sizes = {record["source_size_bytes"] for _, record in pending}
    known_content: Dict[str, str] = {}
    duplicate_reference_failures: List[str] = []
    for source in sorted(mapped_paths):
        try:
            if source.is_file() and source.stat().st_size in candidate_sizes:
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                known_content.setdefault(digest, relative_path(source, root))
        except OSError:
            duplicate_reference_failures.append(relative_path(source, root))
    export_names: set[str] = set()
    for source, record in pending:
        try:
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            record["source_sha256"] = digest
            record["source_size_bytes"] = len(data)
            if digest in known_content:
                record["reason"] = "duplicate_file_content"
                record["duplicate_of"] = known_content[digest]
                continue
            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
            if decoded is None or decoded.ndim != 3 or min(decoded.shape[:2]) < 2:
                record["reason"] = "unreadable_or_invalid_image"
                continue
            record["height"], record["width"] = map(int, decoded.shape[:2])
            identity = source.relative_to(capture_dir).as_posix()
            identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            export_name = f"visual_only_{identity_digest}_{safe_name_part(source.stem)[:80]}{source.suffix.lower()}"
            if export_name in export_names:
                record["reason"] = "export_name_collision"
                continue
            record.update(status="accepted", reason="readable_unique_visual_candidate", exported_image_file=export_name)
            accepted.append({"source": source, "image_file": export_name,
                             "source_sha256": digest, "source_size_bytes": len(data),
                             "width": record["width"], "height": record["height"]})
            known_content[digest] = relative_path(source, root)
            export_names.add(export_name)
        except (OSError, ValueError, cv2.error) as error:
            record["reason"] = "source_read_or_decode_failed"
            record["error"] = str(error)
    summary = {
        "enabled": include_unassigned_images,
        "capture_directory": str(capture_dir),
        "input_candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(candidates)-len(accepted),
        "rejection_reason_counts": dict(sorted(Counter(row["reason"] for row in audit if row["status"] == "rejected").items())),
        "duplicate_reference_read_failures": duplicate_reference_failures,
        "duplicate_policy": "resolved_source_path_and_exact_file_sha256_against_mapped_and_accepted_images",
        "eligibility_scope": "Readable unique capture files; scene membership and geometric overlap require visual verification.",
        "output": "unassigned_image_audit.csv",
    }
    return accepted, audit, summary


def pose_seed_for_row(
    row: Dict[str, str],
    camera_height_m: float,
    slot_yaw_start_deg: float,
    slot_yaw_step_deg: Optional[float],
) -> Dict[str, Any]:
    slot = safe_int(row.get("image_slot"), 1)
    if slot_yaw_step_deg is None:
        slot_yaw_step_deg = 360.0 / max(1, safe_int(row.get("image_group_size"), 20))
    yaw_deg = slot_yaw_start_deg + (slot - 1) * slot_yaw_step_deg
    return {
        "pdr_x": row.get("pdr_x", ""),
        "pdr_y": row.get("pdr_y", ""),
        "pdr_z": camera_height_m,
        "yaw_seed_deg": round(yaw_deg, 6),
        "pitch_seed_deg": 0.0,
        "roll_seed_deg": 0.0,
        "pose_seed_source": "pdr_xy_plus_ordered_anchor_view_yaw",
    }


def pose_seed_for_video_row(row: Dict[str, str], camera_height_m: float) -> Dict[str, Any]:
    return {
        "pdr_x": row.get("pdr_x", ""),
        "pdr_y": row.get("pdr_y", ""),
        "pdr_z": camera_height_m,
        "yaw_seed_deg": "",
        "pitch_seed_deg": "",
        "roll_seed_deg": "",
        "pose_seed_source": row.get("pdr_bind_source", "video_base_frame"),
    }


def read_expected_images_per_point(processed_dir: Path) -> Optional[int]:
    summary_path = processed_dir / "image_wifi_summary.json"
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    value = summary.get("images_per_wifi_point")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return list(unique)

    def cross(origin: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: List[Tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: List[Tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def point_key(row: Dict[str, str]) -> str:
    return f"{row.get('session', '')}|{row.get('global_point_index', '')}"


def select_evenly_spaced_rows(rows: Sequence[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    return [rows[idx * len(rows) // limit] for idx in range(limit)]


def select_anchor_image_rows(
    image_rows: Sequence[Dict[str, str]],
    max_images_per_point: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    if max_images_per_point <= 0:
        selected_slots_by_point: Dict[str, List[int]] = {}
        for row in image_rows:
            selected_slots_by_point.setdefault(point_key(row), []).append(safe_int(row.get("image_slot")))
        for slots in selected_slots_by_point.values():
            slots.sort()
        return list(image_rows), {
            "enabled": False,
            "max_images_per_point": 0,
            "input_rows": len(image_rows),
            "selected_rows": len(image_rows),
            "dropped_rows": 0,
            "capped_point_count": 0,
            "selection_strategy": "all_anchor_images",
            "selected_slots_by_point": selected_slots_by_point,
        }

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in image_rows:
        grouped.setdefault(point_key(row), []).append(row)

    selected: List[Dict[str, str]] = []
    capped_point_count = 0
    selected_slots_by_point: Dict[str, List[int]] = {}
    for key in sorted(
        grouped,
        key=lambda item: (
            grouped[item][0].get("session", ""),
            safe_int(grouped[item][0].get("global_point_index")),
        ),
    ):
        point_rows = sorted(grouped[key], key=lambda row: safe_int(row.get("image_slot")))
        kept_rows = select_evenly_spaced_rows(point_rows, max_images_per_point)
        if len(kept_rows) < len(point_rows):
            capped_point_count += 1
        selected.extend(kept_rows)
        selected_slots_by_point[key] = [safe_int(row.get("image_slot")) for row in kept_rows]

    return selected, {
        "enabled": True,
        "max_images_per_point": max_images_per_point,
        "input_rows": len(image_rows),
        "selected_rows": len(selected),
        "dropped_rows": len(image_rows) - len(selected),
        "point_count": len(grouped),
        "capped_point_count": capped_point_count,
        "selection_strategy": "circular_even_slots",
        "selected_slots_by_point": selected_slots_by_point,
    }


def build_fingerprint_rows(point_rows: Sequence[Dict[str, str]], ap_fields: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in point_rows:
        output = {
            "session": row.get("session", ""),
            "point_index": row.get("point_index", ""),
            "global_point_index": row.get("global_point_index", ""),
            "pdr_x": row.get("pdr_x", ""),
            "pdr_y": row.get("pdr_y", ""),
            "wifi_scan_index": row.get("wifi_scan_index", ""),
            "wifi_elapsed_ts": row.get("wifi_elapsed_ts", ""),
            "wifi_gap_ms": row.get("wifi_gap_ms", ""),
            "mapping_status": row.get("mapping_status", ""),
        }
        for field in ap_fields:
            output[field] = row.get(field, "")
        for field in [*WIFI_QUALITY_FIELDS, *WIFI_ASSOCIATION_FIELDS]:
            output[field] = row.get(field, "")
        rows.append(output)
    return rows


def build_boundary_rows(point_rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    coord_to_rows: Dict[Tuple[float, float], List[Dict[str, str]]] = {}
    for row in point_rows:
        x = safe_float(row.get("pdr_x"))
        y = safe_float(row.get("pdr_y"))
        if math.isnan(x) or math.isnan(y):
            continue
        coord_to_rows.setdefault((x, y), []).append(row)

    points = list(coord_to_rows)
    hull = convex_hull(points)
    boundary_rows: List[Dict[str, Any]] = []
    for order, coord in enumerate(hull, start=1):
        source = coord_to_rows[coord][0]
        boundary_rows.append(
            {
                "boundary_order": order,
                "session": source.get("session", ""),
                "point_index": source.get("point_index", ""),
                "global_point_index": source.get("global_point_index", ""),
                "pdr_x": round(coord[0], 6),
                "pdr_y": round(coord[1], 6),
                "source": "convex_hull_of_wifi_checkpoint_samples_not_route_contour",
            }
        )

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    summary = {
        "point_count": len(points),
        "hull_point_count": len(hull),
        "min_x": round(min(xs), 6) if xs else None,
        "max_x": round(max(xs), 6) if xs else None,
        "min_y": round(min(ys), 6) if ys else None,
        "max_y": round(max(ys), 6) if ys else None,
    }
    return boundary_rows, summary


def write_colmap_image_list(path: Path, image_paths: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for image_path in image_paths:
            handle.write(f"{image_path.replace(chr(92), '/')}\n")



def export_reconstruction_dataset(
    root: Path,
    processed_dir: Path,
    output_dir: Path,
    camera_height_m: float = 1.50,
    slot_yaw_start_deg: float = 0.0,
    slot_yaw_step_deg: Optional[float] = None,
    copy_images: bool = True,
    max_anchor_images_per_point: int = 0,
    include_video_frames: bool = True,
    include_unassigned_images: bool = True,
) -> Dict[str, Any]:
    _long_fields, image_rows = read_csv(processed_dir / "image_wifi_long.csv")
    point_fields, point_rows = read_csv(processed_dir / "image_wifi_points.csv")
    _video_fields, video_rows = (
        read_csv(processed_dir / "video_frames_manifest.csv")
        if include_video_frames
        else ([], [])
    )
    unassigned_images, unassigned_audit, unassigned_summary = audit_unassigned_images(
        root, processed_dir, image_rows, include_unassigned_images=include_unassigned_images,
    )
    write_csv(output_dir / "unassigned_image_audit.csv", [
        "candidate_index", "image_name", "source_image_path", "original_unassigned_reason",
        "status", "reason", "exported_image_file", "source_sha256", "source_size_bytes",
        "width", "height", "duplicate_of", "error",
    ], unassigned_audit)
    expected_images_per_point = read_expected_images_per_point(processed_dir)
    expected_by_point = {
        point_key(row): safe_int(row.get("image_group_size"), expected_images_per_point or 0)
        for row in point_rows
        if row.get("image_group_size") not in (None, "") or expected_images_per_point is not None
    }
    if not image_rows and not video_rows and not unassigned_images:
        summary = {
            "status": "missing_input",
            "processed_dir": str(processed_dir),
            "output_dir": str(output_dir),
            "image_rows": len(image_rows),
            "point_rows": len(point_rows),
            "include_unassigned_images": include_unassigned_images,
            "unassigned_image_audit": unassigned_summary,
        }
        write_json(output_dir / "export_summary.json", summary)
        return summary
    original_image_row_count = len(image_rows)
    image_rows, anchor_selection_summary = select_anchor_image_rows(image_rows, max_anchor_images_per_point)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if images_dir.resolve().parent != output_dir.resolve():
        raise ValueError("Export image directory must resolve directly inside the output directory")
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    ap_fields = [field for field in point_fields if field.startswith("ap_")]
    manifest_rows: List[Dict[str, Any]] = []
    pose_rows: List[Dict[str, Any]] = []
    colmap_image_paths: List[str] = []
    missing_sources: List[str] = []

    for row in image_rows:
        if not row.get("image_name"):
            continue
        source = resolve_source_image(root, row.get("image_path", ""))
        export_name = exported_image_name(row)
        exported_rel_path = str(Path("images") / export_name)
        exported_path = output_dir / exported_rel_path
        if source.exists():
            if copy_images:
                shutil.copy2(source, exported_path)
            colmap_image_paths.append(exported_rel_path)
        else:
            missing_sources.append(row.get("image_path", ""))

        pose_seed = pose_seed_for_row(
            {**row, "image_group_size": row.get("image_group_size") or expected_images_per_point or 20},
            camera_height_m=camera_height_m,
            slot_yaw_start_deg=slot_yaw_start_deg,
            slot_yaw_step_deg=slot_yaw_step_deg,
        )
        common = {
            "image_file": export_name,
            "image_rel_path": exported_rel_path,
            "source_image_path": relative_path(source, root),
            "source_type": "wifi_anchor_photo",
            "bind_type": ("checkpoint_photo_with_weighted_wifi" if row.get("wifi_observation_status") == "aggregated_checkpoint"
                          else "checkpoint_photo_with_fresh_wifi" if valid_fingerprint(row) else "checkpoint_photo_without_valid_wifi"),
            "wifi_label_weight": fingerprint_confidence(row),
            **{field: row.get(field, "") for field in [*WIFI_QUALITY_FIELDS, *WIFI_ASSOCIATION_FIELDS, *ap_fields]},
            "image_association_basis": row.get("image_association_basis", "ordered_checkpoint_same_location"),
            "image_grouping_method": row.get("image_grouping_method", ""),
            "session": row.get("session", ""),
            "point_index": row.get("point_index", ""),
            "global_point_index": row.get("global_point_index", ""),
            "image_slot": row.get("image_slot", ""),
            "anchor_slot_count": row.get("image_group_size") or expected_images_per_point or "",
            "image_number": row.get("image_number", ""),
            "point_elapsed_ts": row.get("point_elapsed_ts", ""),
            "pdr_reference_elapsed_ts": row.get("pdr_reference_elapsed_ts", ""),
            "pdr_interp_status": row.get("pdr_interp_status", ""),
            "wifi_scan_index": row.get("wifi_scan_index", ""),
            "wifi_gap_ms": row.get("wifi_gap_ms", ""),
            "mapping_status": row.get("mapping_status", ""),
            "source_exists": int(source.exists()),
            "video_name": "",
            "video_frame_index": "",
            "video_sequence_index": "",
            "video_time_s": "",
            "nearest_global_point_index": row.get("global_point_index", ""),
            "left_global_point_index": row.get("global_point_index", ""),
            "right_global_point_index": row.get("global_point_index", ""),
            "route_ratio": "",
            "trajectory_contour_id": row.get("trajectory_contour_id", ""),
            "trajectory_contour_distance_m": row.get("trajectory_contour_distance_m", ""),
            "trajectory_distance_m": row.get("trajectory_distance_m", ""),
            "trajectory_position_source": (
                "checkpoint_sensor_timestamp" if row.get("pdr_interp_status") == "exact_checkpoint"
                else "unavailable" if row.get("pdr_interp_status") == "checkpoint_pdr_unavailable"
                else "checkpoint_nearest_log_timestamp"
            ),
        }
        manifest_rows.append({**common, **pose_seed})
        pose_rows.append(
            {
                "image_file": export_name,
                "image_rel_path": exported_rel_path,
                "source_type": "wifi_anchor_photo",
                "bind_type": common["bind_type"],
                "session": row.get("session", ""),
                "global_point_index": row.get("global_point_index", ""),
                "image_slot": row.get("image_slot", ""),
                "anchor_slot_count": common["anchor_slot_count"],
                "video_name": "",
                "video_frame_index": "",
                "video_sequence_index": "",
                **pose_seed,
            }
        )

    for row in video_rows:
        if not row.get("frame_path"):
            continue
        if row.get("trajectory_position_source") != "synchronized_video_sensor_timestamp":
            row = {**row, **bind_frame_to_route([], 0, 0)}
        source = resolve_source_image(root, row.get("frame_path", ""))
        export_name = exported_video_frame_name(row)
        exported_rel_path = str(Path("images") / export_name)
        exported_path = output_dir / exported_rel_path
        if source.exists():
            if copy_images:
                shutil.copy2(source, exported_path)
            colmap_image_paths.append(exported_rel_path)
        else:
            missing_sources.append(row.get("frame_path", ""))

        pose_seed = pose_seed_for_video_row(row, camera_height_m=camera_height_m)
        common = {
            "image_file": export_name,
            "image_rel_path": exported_rel_path,
            "source_image_path": relative_path(source, root),
            "source_type": row.get("source_type", "video_base_frame"),
            "bind_type": row.get("bind_type", "video_base_frame"),
            "wifi_label_weight": 0.0,
            "session": "",
            "point_index": "",
            "global_point_index": "",
            "image_slot": "",
            "anchor_slot_count": "",
            "image_number": "",
            "point_elapsed_ts": "",
            "wifi_scan_index": "",
            "wifi_gap_ms": "",
            "mapping_status": row.get("bind_type", "video_base_frame"),
            "source_exists": int(source.exists()),
            "video_name": row.get("video_name", ""),
            "video_frame_index": row.get("video_frame_index", ""),
            "video_sequence_index": row.get("video_sequence_index", ""),
            "video_time_s": row.get("video_time_s", ""),
            "video_time_source": row.get("video_time_source", "unavailable"),
            **{field: row.get(field, "") for field in VIDEO_FRAME_PROVENANCE_FIELDS},
            "nearest_global_point_index": row.get("nearest_global_point_index", ""),
            "left_global_point_index": row.get("left_global_point_index", ""),
            "right_global_point_index": row.get("right_global_point_index", ""),
            "route_ratio": row.get("route_ratio", ""),
            "trajectory_contour_id": row.get("trajectory_contour_id", ""),
            "trajectory_contour_distance_m": row.get("trajectory_contour_distance_m", ""),
            "trajectory_distance_m": row.get("trajectory_distance_m", ""),
            "trajectory_position_source": row.get("trajectory_position_source", "unavailable"),
        }
        manifest_rows.append({**common, **pose_seed})
        pose_rows.append(
            {
                "image_file": export_name,
                "image_rel_path": exported_rel_path,
                "source_type": row.get("source_type", "video_base_frame"),
                "bind_type": row.get("bind_type", "video_base_frame"),
                "session": "",
                "global_point_index": "",
                "image_slot": "",
                "anchor_slot_count": "",
                "video_name": row.get("video_name", ""),
                "video_frame_index": row.get("video_frame_index", ""),
                "video_sequence_index": row.get("video_sequence_index", ""),
                **pose_seed,
            }
        )

    for image in unassigned_images:
        source = image["source"]
        export_name = image["image_file"]
        exported_rel_path = str(Path("images") / export_name)
        if copy_images:
            shutil.copy2(source, output_dir / exported_rel_path)
        colmap_image_paths.append(exported_rel_path)
        manifest_rows.append({
            "image_file": export_name,
            "image_rel_path": exported_rel_path,
            "source_image_path": relative_path(source, root),
            "source_type": "visual_only_photo",
            "bind_type": "unassigned_visual_only",
            "wifi_label_weight": 0.0,
            "mapping_status": "unassigned_visual_only",
            "source_exists": 1,
            "source_sha256": image["source_sha256"],
            "source_size_bytes": image["source_size_bytes"],
            "image_width": image["width"],
            "image_height": image["height"],
        })

    fingerprint_rows = build_fingerprint_rows(point_rows, ap_fields)
    boundary_rows, boundary_summary = build_boundary_rows(point_rows)

    manifest_fields = [
        "image_file",
        "image_rel_path",
        "source_image_path",
        "source_sha256",
        "source_size_bytes",
        "image_width",
        "image_height",
        "source_type",
        "bind_type",
        "wifi_label_weight",
        "image_association_basis",
        "image_grouping_method",
        *WIFI_ASSOCIATION_FIELDS,
        *WIFI_QUALITY_FIELDS,
        *ap_fields,
        "session",
        "point_index",
        "global_point_index",
        "image_slot",
        "anchor_slot_count",
        "image_number",
        "point_elapsed_ts",
        "pdr_reference_elapsed_ts",
        "pdr_interp_status",
        "wifi_scan_index",
        "wifi_gap_ms",
        "mapping_status",
        "source_exists",
        "video_name",
        "video_frame_index",
        "video_sequence_index",
        "video_time_s",
        "video_time_source",
        *VIDEO_FRAME_PROVENANCE_FIELDS,
        "nearest_global_point_index",
        "left_global_point_index",
        "right_global_point_index",
        "route_ratio",
        "trajectory_contour_id",
        "trajectory_contour_distance_m",
        "trajectory_distance_m",
        "trajectory_position_source",
        "pdr_x",
        "pdr_y",
        "pdr_z",
        "yaw_seed_deg",
        "pitch_seed_deg",
        "roll_seed_deg",
        "pose_seed_source",
    ]
    pose_fields = [
        "image_file",
        "image_rel_path",
        "source_type",
        "bind_type",
        "session",
        "global_point_index",
        "image_slot",
        "anchor_slot_count",
        "video_name",
        "video_frame_index",
        "video_sequence_index",
        "pdr_x",
        "pdr_y",
        "pdr_z",
        "yaw_seed_deg",
        "pitch_seed_deg",
        "roll_seed_deg",
        "pose_seed_source",
    ]
    fingerprint_fields = [
        "session",
        "point_index",
        "global_point_index",
        "pdr_x",
        "pdr_y",
        "wifi_scan_index",
        "wifi_elapsed_ts",
        "wifi_gap_ms",
        "mapping_status",
        *WIFI_ASSOCIATION_FIELDS,
        *WIFI_QUALITY_FIELDS,
        *ap_fields,
    ]
    boundary_fields = [
        "boundary_order",
        "session",
        "point_index",
        "global_point_index",
        "pdr_x",
        "pdr_y",
        "source",
    ]

    write_csv(output_dir / "images_manifest.csv", manifest_fields, manifest_rows)
    write_csv(output_dir / "camera_pose_seed.csv", pose_fields, pose_rows)
    write_csv(output_dir / "wifi_fingerprint_library.csv", fingerprint_fields, fingerprint_rows)
    write_csv(output_dir / "wifi_boundary_seed.csv", boundary_fields, boundary_rows)
    write_colmap_image_list(output_dir / "colmap_image_list.txt", colmap_image_paths)
    copied_optional_metadata: List[str] = []
    if include_video_frames:
        for optional_name in ("video_frames_manifest.csv", "video_frames_summary.json"):
            source = processed_dir / optional_name
            if source.exists():
                shutil.copy2(source, output_dir / optional_name)
                copied_optional_metadata.append(optional_name)
    else:
        for optional_name in ("video_frames_manifest.csv", "video_frames_summary.json"):
            stale_path = output_dir / optional_name
            if stale_path.exists():
                stale_path.unlink()

    for optional_name in (
        "pdr_trajectory_contour.csv",
        "pdr_trajectory_edges.csv",
        "pdr_trajectory_contour_summary.json",
    ):
        source = processed_dir / optional_name
        if source.exists():
            shutil.copy2(source, output_dir / optional_name)
            copied_optional_metadata.append(optional_name)

    point_image_counts: Dict[str, int] = {point_key(row): 0 for row in point_rows}
    for row in manifest_rows:
        if row.get("source_type") != "wifi_anchor_photo":
            continue
        key = point_key(row)
        point_image_counts[key] = point_image_counts.get(key, 0) + 1
    expected_count = expected_images_per_point or max(expected_by_point.values() or point_image_counts.values() or [0])
    exported_expected_count = (
        min(expected_count, max_anchor_images_per_point)
        if max_anchor_images_per_point > 0
        else expected_count
    )
    exported_expected_by_point = {
        key: min(count, max_anchor_images_per_point) if max_anchor_images_per_point > 0 else count
        for key, count in expected_by_point.items()
    }
    incomplete_points = [key for key, count in point_image_counts.items()
                         if count != exported_expected_by_point.get(key, exported_expected_count)]

    copied_count = len(list(images_dir.iterdir())) if images_dir.exists() else 0
    status = "ok"
    if missing_sources:
        status = "missing_source_images"
    elif incomplete_points:
        status = "incomplete_points"

    summary = {
        "status": status,
        "root": str(root),
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "copy_images": copy_images,
        "camera_height_m": camera_height_m,
        "slot_yaw_start_deg": slot_yaw_start_deg,
        "slot_yaw_step_deg": slot_yaw_step_deg,
        "include_video_frames": include_video_frames,
        "include_unassigned_images": include_unassigned_images,
        "image_rows": original_image_row_count,
        "selected_image_rows": len(image_rows),
        "anchor_image_selection": anchor_selection_summary,
        "anchor_image_rows": sum(1 for row in manifest_rows if row.get("source_type") == "wifi_anchor_photo"),
        "video_frame_rows": sum(1 for row in manifest_rows if row.get("source_type") == "video_base_frame"),
        "visual_only_image_rows": sum(1 for row in manifest_rows if row.get("source_type") == "visual_only_photo"),
        "source_type_counts": dict(sorted(Counter(row["source_type"] for row in manifest_rows).items())),
        "unassigned_image_audit": unassigned_summary,
        "point_rows": len(point_rows),
        "expected_images_per_wifi_point": (expected_count if len(set(expected_by_point.values())) <= 1 else None),
        "expected_exported_images_per_wifi_point": (exported_expected_count if len(set(exported_expected_by_point.values())) <= 1 else None),
        "expected_exported_images_by_point": exported_expected_by_point,
        "exported_image_rows": len(manifest_rows),
        "copied_image_file_count": copied_count,
        "colmap_image_count": len(colmap_image_paths),
        "wifi_ap_feature_count": len(ap_fields),
        "missing_source_image_count": len(missing_sources),
        "missing_source_images": missing_sources[:20],
        "incomplete_point_count": len(incomplete_points),
        "incomplete_points": incomplete_points[:20],
        "copied_optional_metadata": copied_optional_metadata,
        "boundary": boundary_summary,
        "outputs": {
            "images": "images",
            "colmap_image_list": "colmap_image_list.txt",
            "images_manifest": "images_manifest.csv",
            "unassigned_image_audit": "unassigned_image_audit.csv",
            "camera_pose_seed": "camera_pose_seed.csv",
            "wifi_fingerprint_library": "wifi_fingerprint_library.csv",
            "wifi_boundary_seed": "wifi_boundary_seed.csv",
            "trajectory_contour": "pdr_trajectory_contour.csv",
            "trajectory_edges": "pdr_trajectory_edges.csv",
            "video_frames_manifest": "video_frames_manifest.csv",
        },
    }
    write_json(output_dir / "export_summary.json", summary)
    return summary
