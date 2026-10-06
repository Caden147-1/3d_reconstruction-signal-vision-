"""Export processed captures into a reconstruction-friendly dataset."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


from wifi_observations import WIFI_QUALITY_FIELDS, WIFI_ASSOCIATION_FIELDS, fingerprint_confidence
from video_frames import VIDEO_FRAME_PROVENANCE_FIELDS
from trajectory_contour import PDR_ROUTE_FIELDS
from video_sensor_fusion import (
    VIDEO_SENSOR_SYNC_FIELDS, clear_video_sensor_priors, has_valid_video_sensor_sync,
)
from video_sensor_fuzzy import FUZZY_FIELDS, has_fuzzy_video_sensor_window


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


def safe_name_part(value: str, default: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or default


def exported_video_frame_name(row: Dict[str, str]) -> str:
    sequence_index = safe_int(row.get("video_sequence_index"), safe_int(row.get("video_frame_index")))
    video_stem = safe_name_part(Path(row.get("video_name", "video")).stem, default="video")
    frame_name = Path(row.get("frame_name", "")).name
    return f"video_base_{sequence_index:06d}_{video_stem}_{frame_name}"


def pose_seed_for_video_row(row: Dict[str, str], camera_height_m: float) -> Dict[str, Any]:
    return {
        **{field: row.get(field, "") for field in PDR_ROUTE_FIELDS},
        "pdr_position_status": row.get("pdr_position_status") or "legacy_unknown",
        "pdr_x": row.get("pdr_x", ""),
        "pdr_y": row.get("pdr_y", ""),
        "pdr_z": camera_height_m if row.get("pdr_heading_policy") == "phone-aligned" and row.get("pdr_position_status") == "phone_aligned_assumption" else "",
        "yaw_seed_deg": "",
        "pitch_seed_deg": "",
        "roll_seed_deg": "",
        "pose_seed_source": row.get("pdr_bind_source", "video_base_frame"),
    }


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


def build_sensor_fingerprint_rows(sensor_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Keep independent sensor observations without inventing camera positions.

    WiFi/PDR fusion uses the sensor clock. Its rows remain useful without any
    association to a video frame, and unknown coordinates must remain unknown.
    Preserve the original scan/route fields alongside the library's aliases.
    """
    return [
        {
            **row,
            "point_index": "",
            "global_point_index": "",
            "wifi_scan_index": row.get("wifi_scan_index") or row.get("scan_index", ""),
            "wifi_elapsed_ts": row.get("wifi_elapsed_ts") or row.get("elapsed_ts", ""),
            "wifi_gap_ms": row.get("wifi_gap_ms", ""),
            "pdr_position_status": row.get("pdr_position_status") or "legacy_unknown",
            "mapping_status": "independent_sensor_observation",
        }
        for row in sensor_rows
    ]


def build_boundary_rows(point_rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    coord_to_rows: Dict[Tuple[float, float], List[Dict[str, str]]] = {}
    for row in point_rows:
        if row.get("pdr_heading_policy") != "phone-aligned" or row.get("pdr_position_status") != "phone_aligned_assumption":
            continue
        x = safe_float(row.get("pdr_x"))
        y = safe_float(row.get("pdr_y"))
        if not math.isfinite(x) or not math.isfinite(y):
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
    copy_images: bool = True,
    include_video_frames: bool = True,
    include_sensor_priors: bool = True,
) -> Dict[str, Any]:
    """Export video frames and independent sensor observations."""
    sensor_fields, sensor_rows = (
        read_csv(processed_dir / "wifi_pdr_features.csv") if include_sensor_priors else ([], [])
    )
    _video_fields, video_rows = (
        read_csv(processed_dir / "video_frames_manifest.csv")
        if include_video_frames
        else ([], [])
    )
    if any(row.get("source_type", "video_base_frame") != "video_base_frame" or not row.get("video_name") for row in video_rows):
        raise ValueError("Video export requires extracted frames with source_type=video_base_frame and video_name")
    sanitized_video_rows = []
    for row in video_rows:
        if include_sensor_priors and has_valid_video_sensor_sync(row):
            sanitized_video_rows.append(row)
        elif include_sensor_priors and has_fuzzy_video_sensor_window(row):
            # A time window authorizes soft retrieval context, not one scan,
            # exact sensor time, interpolated PDR coordinate, or camera pose.
            cleaned = clear_video_sensor_priors(row, "fuzzy_time_window")
            cleaned.update({field: row.get(field, "") for field in FUZZY_FIELDS})
            cleaned.update(session=row["session"], bind_type="fuzzy_video_sensor_window")
            sanitized_video_rows.append(cleaned)
        else:
            sanitized_video_rows.append(clear_video_sensor_priors(
                row, "sensor_priors_disabled" if not include_sensor_priors
                else row.get("video_sensor_sync_status") or "no_explicit_sync"))
    video_rows = sanitized_video_rows
    if not video_rows:
        summary = {"status": "missing_input", "processed_dir": str(processed_dir),
                   "output_dir": str(output_dir), "video_frame_rows": 0,
                   "include_sensor_priors": include_sensor_priors}
        write_json(output_dir / "export_summary.json", summary)
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if images_dir.resolve().parent != output_dir.resolve():
        raise ValueError("Export image directory must resolve directly inside the output directory")
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    ap_fields = list(dict.fromkeys(field for field in [*sensor_fields,
                                                      *(_video_fields if include_sensor_priors else [])]
                                  if field.startswith("ap_")))
    manifest_rows: List[Dict[str, Any]] = []
    pose_rows: List[Dict[str, Any]] = []
    colmap_image_paths: List[str] = []
    missing_sources: List[str] = []

    for row in video_rows:
        if not row.get("frame_path"):
            continue
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
            "wifi_label_weight": fingerprint_confidence(row),
            **{field: row.get(field, "") for field in [*WIFI_QUALITY_FIELDS, *WIFI_ASSOCIATION_FIELDS, *ap_fields,
                                                        *VIDEO_SENSOR_SYNC_FIELDS, *FUZZY_FIELDS]},
            "session": row.get("session", ""),
            "point_index": "",
            "global_point_index": "",
            "point_elapsed_ts": "",
            "pdr_reference_elapsed_ts": row.get("pdr_reference_elapsed_ts", ""),
            "pdr_interp_status": row.get("pdr_interp_status", ""),
            "wifi_scan_index": row.get("wifi_scan_index", ""),
            "wifi_elapsed_ts": row.get("wifi_elapsed_ts", ""),
            "wifi_gap_ms": row.get("wifi_gap_ms", ""),
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
            "trajectory_route_distance_m": row.get("trajectory_route_distance_m", ""),
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
                "session": row.get("session", ""),
                **{field: row.get(field, "") for field in VIDEO_SENSOR_SYNC_FIELDS},
                **{field: row.get(field, "") for field in FUZZY_FIELDS},
                "global_point_index": "",
                "video_name": row.get("video_name", ""),
                "video_frame_index": row.get("video_frame_index", ""),
                "video_sequence_index": row.get("video_sequence_index", ""),
                **pose_seed,
            }
        )

    fingerprint_rows = build_sensor_fingerprint_rows(sensor_rows)
    boundary_rows, boundary_summary = build_boundary_rows(fingerprint_rows)
    for row in boundary_rows:
        row["source"] = "convex_hull_of_wifi_sensor_scans_not_camera_map"

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
        *WIFI_ASSOCIATION_FIELDS,
        *WIFI_QUALITY_FIELDS,
        *ap_fields,
        "session",
        "point_index",
        "global_point_index",
        "point_elapsed_ts",
        "pdr_reference_elapsed_ts",
        "pdr_interp_status",
        "wifi_scan_index",
        "wifi_elapsed_ts",
        "wifi_gap_ms",
        "mapping_status",
        "source_exists",
        "video_name",
        "video_frame_index",
        "video_sequence_index",
        "video_time_s",
        "video_time_source",
        *VIDEO_FRAME_PROVENANCE_FIELDS,
        *VIDEO_SENSOR_SYNC_FIELDS,
        *FUZZY_FIELDS,
        "nearest_global_point_index",
        "left_global_point_index",
        "right_global_point_index",
        "route_ratio",
        "trajectory_contour_id",
        "trajectory_route_distance_m",
        *PDR_ROUTE_FIELDS,
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
        *PDR_ROUTE_FIELDS,
        *VIDEO_SENSOR_SYNC_FIELDS,
        *FUZZY_FIELDS,
        "image_file",
        "image_rel_path",
        "source_type",
        "bind_type",
        "session",
        "global_point_index",
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
        *PDR_ROUTE_FIELDS,
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
    if include_sensor_priors:
        fingerprint_fields = list(dict.fromkeys([*fingerprint_fields, *sensor_fields]))
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
                if optional_name == "video_frames_manifest.csv":
                    write_csv(output_dir / optional_name,
                              list(dict.fromkeys([*_video_fields, *VIDEO_SENSOR_SYNC_FIELDS, *FUZZY_FIELDS])), video_rows)
                else:
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
        "wifi_pdr_features.csv",
        "video_sensor_sync.csv",
        "video_sensor_sync_summary.json",
        "video_sensor_fuzzy_context.json",
        "video_sensor_fuzzy_summary.json",
    ):
        source = processed_dir / optional_name
        if include_sensor_priors and source.exists():
            shutil.copy2(source, output_dir / optional_name)
            copied_optional_metadata.append(optional_name)
        elif (output_dir / optional_name).exists():
            (output_dir / optional_name).unlink()

    copied_count = len(list(images_dir.iterdir())) if images_dir.exists() else 0
    status = "ok"
    if missing_sources:
        status = "missing_source_images"

    summary = {
        "status": status,
        "root": str(root),
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "copy_images": copy_images,
        "camera_height_m": camera_height_m,
        "include_video_frames": include_video_frames,
        "include_sensor_priors": include_sensor_priors,
        "synchronized_video_frame_rows": sum(has_valid_video_sensor_sync(row) for row in manifest_rows),
        "fuzzy_video_frame_rows": sum(has_fuzzy_video_sensor_window(row) for row in manifest_rows),
        "video_frame_rows": sum(1 for row in manifest_rows if row.get("source_type") == "video_base_frame"),
        "source_type_counts": dict(sorted(Counter(row["source_type"] for row in manifest_rows).items())),
        "independent_sensor_observation_rows": len(sensor_rows),
        "wifi_fingerprint_rows": len(fingerprint_rows),
        "wifi_fingerprint_source": "wifi_pdr_features.csv",
        "exported_image_rows": len(manifest_rows),
        "copied_image_file_count": copied_count,
        "colmap_image_count": len(colmap_image_paths),
        "wifi_ap_feature_count": len(ap_fields),
        "missing_source_image_count": len(missing_sources),
        "missing_source_images": missing_sources[:20],
        "copied_optional_metadata": copied_optional_metadata,
        "boundary": boundary_summary,
        "outputs": {
            "images": "images",
            "colmap_image_list": "colmap_image_list.txt",
            "images_manifest": "images_manifest.csv",
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
