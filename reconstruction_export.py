"""Export processed captures into a reconstruction-friendly dataset."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def pose_seed_for_row(
    row: Dict[str, str],
    camera_height_m: float,
    slot_yaw_start_deg: float,
    slot_yaw_step_deg: float,
) -> Dict[str, Any]:
    slot = safe_int(row.get("image_slot"), 1)
    yaw_deg = slot_yaw_start_deg + (slot - 1) * slot_yaw_step_deg
    return {
        "pdr_x": row.get("pdr_x", ""),
        "pdr_y": row.get("pdr_y", ""),
        "pdr_z": camera_height_m,
        "yaw_seed_deg": round(yaw_deg, 6),
        "pitch_seed_deg": 0.0,
        "roll_seed_deg": 0.0,
        "pose_seed_source": "pdr_xy_plus_ordered_8_view_yaw",
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


def point_key(row: Dict[str, str]) -> str:
    return f"{row.get('session', '')}|{row.get('global_point_index', '')}"


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
                "source": "convex_hull_of_wifi_points",
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


def write_readme(path: Path) -> None:
    content = """# Reconstruction Dataset

This folder is generated from `processed/image_wifi_long.csv` and
`processed/image_wifi_points.csv`.

## Files

- `images/`: copied and renamed photos for SfM/MVS tools.
- `colmap_image_list.txt`: image list that can be passed to COLMAP feature extraction.
- `images_manifest.csv`: each copied image with source file, WiFi point, and PDR metadata.
- `camera_pose_seed.csv`: approximate camera pose seed. The XY position comes from PDR,
  Z is a configurable camera height, and yaw is assigned from image slot order.
- `wifi_fingerprint_library.csv`: one row per WiFi point, with selected AP RSSI columns.
- `wifi_boundary_seed.csv`: convex hull of WiFi/PDR point positions, useful as a rough
  boundary seed rather than a final room outline.
- `wifi_localization_model.json`: generated later by the WiFi positioning step.
- `wifi_localization_eval.csv`: leave-one-out fingerprint localization evaluation.
- `wifi_signal_grid.csv`: interpolated RSSI field over the sampled area.
- `wifi_signal_boundary.csv`: boundary points with RSSI coverage metrics.
- `export_summary.json`: counts and quality checks for this export.

## Suggested Use

Use `images/` or `colmap_image_list.txt` as the photo input for COLMAP/COMPLab.
Use `camera_pose_seed.csv` as metadata for alignment and debugging. It is not a
calibrated camera model. Use `wifi_fingerprint_library.csv` for later indoor
localization, and use `wifi_boundary_seed.csv` as a soft spatial constraint.
The WiFi positioning step turns this fingerprint library into a KNN localization
model and an RSSI boundary/grid model.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def export_reconstruction_dataset(
    root: Path,
    processed_dir: Path,
    output_dir: Path,
    camera_height_m: float = 1.50,
    slot_yaw_start_deg: float = 0.0,
    slot_yaw_step_deg: float = 45.0,
    copy_images: bool = True,
) -> Dict[str, Any]:
    long_fields, image_rows = read_csv(processed_dir / "image_wifi_long.csv")
    point_fields, point_rows = read_csv(processed_dir / "image_wifi_points.csv")
    if not image_rows or not point_rows:
        summary = {
            "status": "missing_input",
            "processed_dir": str(processed_dir),
            "output_dir": str(output_dir),
            "image_rows": len(image_rows),
            "point_rows": len(point_rows),
        }
        write_json(output_dir / "export_summary.json", summary)
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
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
            row,
            camera_height_m=camera_height_m,
            slot_yaw_start_deg=slot_yaw_start_deg,
            slot_yaw_step_deg=slot_yaw_step_deg,
        )
        common = {
            "image_file": export_name,
            "image_rel_path": exported_rel_path,
            "source_image_path": relative_path(source, root),
            "session": row.get("session", ""),
            "point_index": row.get("point_index", ""),
            "global_point_index": row.get("global_point_index", ""),
            "image_slot": row.get("image_slot", ""),
            "image_number": row.get("image_number", ""),
            "point_elapsed_ts": row.get("point_elapsed_ts", ""),
            "wifi_scan_index": row.get("wifi_scan_index", ""),
            "wifi_gap_ms": row.get("wifi_gap_ms", ""),
            "mapping_status": row.get("mapping_status", ""),
            "source_exists": int(source.exists()),
        }
        manifest_rows.append({**common, **pose_seed})
        pose_rows.append(
            {
                "image_file": export_name,
                "image_rel_path": exported_rel_path,
                "session": row.get("session", ""),
                "global_point_index": row.get("global_point_index", ""),
                "image_slot": row.get("image_slot", ""),
                **pose_seed,
            }
        )

    fingerprint_rows = build_fingerprint_rows(point_rows, ap_fields)
    boundary_rows, boundary_summary = build_boundary_rows(point_rows)

    manifest_fields = [
        "image_file",
        "image_rel_path",
        "source_image_path",
        "session",
        "point_index",
        "global_point_index",
        "image_slot",
        "image_number",
        "point_elapsed_ts",
        "wifi_scan_index",
        "wifi_gap_ms",
        "mapping_status",
        "source_exists",
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
        "session",
        "global_point_index",
        "image_slot",
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
    write_readme(output_dir / "README.md")

    point_image_counts: Dict[str, int] = {}
    for row in manifest_rows:
        key = point_key(row)
        point_image_counts[key] = point_image_counts.get(key, 0) + 1
    incomplete_points = [key for key, count in point_image_counts.items() if count != 8]

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
        "image_rows": len(image_rows),
        "point_rows": len(point_rows),
        "exported_image_rows": len(manifest_rows),
        "copied_image_file_count": copied_count,
        "colmap_image_count": len(colmap_image_paths),
        "wifi_ap_feature_count": len(ap_fields),
        "missing_source_image_count": len(missing_sources),
        "missing_source_images": missing_sources[:20],
        "incomplete_point_count": len(incomplete_points),
        "incomplete_points": incomplete_points[:20],
        "boundary": boundary_summary,
        "outputs": {
            "images": "images",
            "colmap_image_list": "colmap_image_list.txt",
            "images_manifest": "images_manifest.csv",
            "camera_pose_seed": "camera_pose_seed.csv",
            "wifi_fingerprint_library": "wifi_fingerprint_library.csv",
            "wifi_boundary_seed": "wifi_boundary_seed.csv",
        },
    }
    write_json(output_dir / "export_summary.json", summary)
    return summary
