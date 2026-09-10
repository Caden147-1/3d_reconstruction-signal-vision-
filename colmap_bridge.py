"""Prepare a COLMAP workspace from the exported reconstruction dataset."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from image_similarity import analyze_workspace_image_similarity


METADATA_FILES = [
    "images_manifest.csv",
    "camera_pose_seed.csv",
    "wifi_fingerprint_library.csv",
    "wifi_boundary_seed.csv",
    "wifi_localization_model.json",
    "wifi_localization_eval.csv",
    "wifi_positioning_summary.json",
    "wifi_signal_boundary.csv",
    "wifi_signal_grid.csv",
    "video_frames_manifest.csv",
    "video_frames_summary.json",
    "pdr_trajectory_contour.csv",
    "pdr_trajectory_edges.csv",
    "pdr_trajectory_contour_summary.json",
    "export_summary.json",
]


DENSE_PROFILES: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "description": "Balanced indoor reconstruction with bounded dense and post-processing cost.",
        "patch_match": {
            "geom_consistency": 1,
            "window_radius": 5,
            "window_step": 1,
            "num_samples": 15,
            "num_iterations": 5,
            "cache_size": 8,
        },
        "fusion": {
            "min_num_pixels": 4,
            "check_num_images": 60,
        },
        "run_meshing": True,
        "cleanup_intermediate_ply": True,
        "point_filter_max_points": 600000,
    },
    "rtx4080": {
        "description": (
            "RTX 4080-oriented dense reconstruction. It keeps the completeness "
            "bias of the final profile while trimming PatchMatch cost a little."
        ),
        "patch_match": {
            "geom_consistency": 1,
            "window_radius": 7,
            "window_step": 1,
            "num_samples": 20,
            "num_iterations": 6,
            "filter": 1,
            "filter_min_ncc": 0.05,
            "cache_size": 12,
        },
        "fusion": {
            "min_num_pixels": 3,
            "max_reproj_error": 2.5,
            "max_depth_error": 0.02,
            "max_normal_error": 15,
            "check_num_images": 100,
        },
        "run_meshing": True,
        "cleanup_intermediate_ply": True,
        "point_filter_max_points": 900000,
    },
    "complete": {
        "description": (
            "Higher-completeness indoor reconstruction. It keeps geometric "
            "consistency, uses a larger PatchMatch support window, and relaxes "
            "fusion filters so weak-texture floors survive instead of being "
            "over-pruned."
        ),
        "patch_match": {
            "geom_consistency": 1,
            "window_radius": 9,
            "window_step": 1,
            "num_samples": 25,
            "num_iterations": 7,
            "filter": 1,
            "filter_min_ncc": 0.05,
            "cache_size": 16,
        },
        "fusion": {
            "min_num_pixels": 3,
            "max_reproj_error": 2.5,
            "max_depth_error": 0.02,
            "max_normal_error": 15,
            "check_num_images": 120,
        },
        "run_meshing": True,
        "cleanup_intermediate_ply": True,
        "point_filter_max_points": 1200000,
    },
}


LOAD_PROFILES: Dict[str, Dict[str, Any]] = {
    "low-peak": {
        "description": "Limit supported COLMAP stages to a few threads and rest between heavy stages.",
        "num_threads": 2,
        "stage_cooldown_s": 20,
    },
    "standard": {
        "description": "Moderate throttling for normal desktop use.",
        "num_threads": 4,
        "stage_cooldown_s": 5,
    },
    "max": {
        "description": "Use COLMAP defaults with no artificial cooldown.",
        "num_threads": 0,
        "stage_cooldown_s": 0,
    },
}


REASON_BASE_WEIGHTS: Dict[str, float] = {
    "same_wifi_point_nearby_slots": 1.00,
    "nearby_wifi_points": 0.88,
    "nearby_trajectory_points": 0.82,
    "nearby_pdr_points": 0.74,
    "video_sequence": 0.70,
    "video_to_wifi_anchor": 0.62,
}

SOURCE_PAIR_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("wifi_anchor_photo", "wifi_anchor_photo"): 1.00,
    ("video_base_frame", "video_base_frame"): 0.82,
    ("wifi_anchor_photo", "video_base_frame"): 0.72,
}


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def windows_path(path: Path) -> str:
    return str(path.resolve())


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


def find_colmap_exe(root: Path, colmap_dir: Optional[Path] = None, colmap_exe: Optional[Path] = None) -> Path:
    candidates: List[Path] = []
    if colmap_exe is not None:
        candidates.append(colmap_exe if colmap_exe.is_absolute() else root / colmap_exe)
    if colmap_dir is not None:
        resolved_dir = colmap_dir if colmap_dir.is_absolute() else root / colmap_dir
        candidates.extend([resolved_dir / "bin" / "colmap.exe", resolved_dir / "COLMAP.bat"])
    candidates.extend(
        [
            root / "colmap-x64-windows-cuda" / "bin" / "colmap.exe",
            root / "colmap-x64-windows-cuda" / "COLMAP.bat",
            root / "colmap" / "bin" / "colmap.exe",
            root / "colmap.exe",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else (root / "colmap.exe").resolve()


def copy_images(
    reconstruction_dir: Path,
    workspace_dir: Path,
    manifest_rows: Sequence[Dict[str, str]],
) -> Tuple[int, List[str], List[str], List[str]]:
    source_images_dir = reconstruction_dir / "images"
    target_images_dir = workspace_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    image_names: List[str] = []
    copied_image_names: List[str] = []
    missing: List[str] = []
    copied = 0
    for row in manifest_rows:
        image_file = row.get("image_file", "")
        if not image_file:
            continue
        source = source_images_dir / image_file
        target = target_images_dir / image_file
        image_names.append(image_file)
        if not source.exists():
            missing.append(str(source))
            continue
        shutil.copy2(source, target)
        copied_image_names.append(image_file)
        copied += 1

    expected_names = set(copied_image_names)
    removed_stale: List[str] = []
    for target in target_images_dir.iterdir():
        if not target.is_file() or target.name in expected_names:
            continue
        try:
            target.unlink()
        except OSError:
            continue
        removed_stale.append(target.name)
    return copied, image_names, missing, removed_stale


def copy_metadata(reconstruction_dir: Path, workspace_dir: Path) -> List[str]:
    metadata_dir = workspace_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in METADATA_FILES:
        source = reconstruction_dir / name
        if not source.exists():
            continue
        shutil.copy2(source, metadata_dir / name)
        copied.append(name)
    return copied


def write_image_list(workspace_dir: Path, image_names: Sequence[str]) -> None:
    with (workspace_dir / "image_list.txt").open("w", encoding="utf-8") as handle:
        for image_name in image_names:
            handle.write(f"{image_name}\n")


def circular_slot_distance(slot_a: int, slot_b: int, slot_count: int) -> int:
    diff = abs(slot_a - slot_b)
    return min(diff, max(0, slot_count - diff))


def anchor_slot_count(rows: Sequence[Dict[str, str]]) -> int:
    counts = [safe_int(row.get("anchor_slot_count")) for row in rows if safe_int(row.get("anchor_slot_count")) > 0]
    if counts:
        return max(counts)
    return max((safe_int(row.get("image_slot")) for row in rows), default=len(rows))


def point_distance(row_a: Dict[str, str], row_b: Dict[str, str]) -> float:
    ax = safe_float(row_a.get("pdr_x"))
    ay = safe_float(row_a.get("pdr_y"))
    bx = safe_float(row_b.get("pdr_x"))
    by = safe_float(row_b.get("pdr_y"))
    if any(math.isnan(value) for value in (ax, ay, bx, by)):
        return math.inf
    return math.hypot(ax - bx, ay - by)


def trajectory_coordinate(row: Dict[str, str]) -> Optional[Tuple[str, float]]:
    contour_id = str(row.get("trajectory_contour_id", "")).strip()
    distance_m = safe_float(row.get("trajectory_contour_distance_m"))
    if not contour_id or not math.isfinite(distance_m):
        return None
    return contour_id, distance_m


def trajectory_distance(row_a: Dict[str, str], row_b: Dict[str, str]) -> float:
    left = trajectory_coordinate(row_a)
    right = trajectory_coordinate(row_b)
    if left is None or right is None or left[0] != right[0]:
        return math.inf
    return abs(left[1] - right[1])


def bounded_score(value: Optional[float], scale: float) -> float:
    if value is None or scale <= 0 or math.isnan(value) or math.isinf(value):
        return 1.0
    return 1.0 / (1.0 + max(0.0, value) / scale)


def row_point_index(row: Dict[str, str]) -> int:
    return safe_int(row.get("global_point_index") or row.get("nearest_global_point_index"), -1)


def row_slot(row: Dict[str, str]) -> int:
    return safe_int(row.get("image_slot"), -1)


def row_video_sequence(row: Dict[str, str]) -> int:
    return safe_int(row.get("video_sequence_index"), -1)


def pair_point_gap(row_a: Dict[str, str], row_b: Dict[str, str]) -> Optional[int]:
    point_a = row_point_index(row_a)
    point_b = row_point_index(row_b)
    if point_a < 0 or point_b < 0:
        return None
    return abs(point_a - point_b)


def pair_slot_gap(row_a: Dict[str, str], row_b: Dict[str, str]) -> Optional[int]:
    slot_a = row_slot(row_a)
    slot_b = row_slot(row_b)
    if slot_a < 0 or slot_b < 0:
        return None
    slot_count = max(
        safe_int(row_a.get("anchor_slot_count")),
        safe_int(row_b.get("anchor_slot_count")),
        slot_a,
        slot_b,
    )
    if slot_count <= 0:
        return abs(slot_a - slot_b)
    return circular_slot_distance(slot_a, slot_b, slot_count)


def pair_video_sequence_gap(row_a: Dict[str, str], row_b: Dict[str, str]) -> Optional[int]:
    sequence_a = row_video_sequence(row_a)
    sequence_b = row_video_sequence(row_b)
    if sequence_a < 0 or sequence_b < 0:
        return None
    return abs(sequence_a - sequence_b)


def source_pair_weight(row_a: Dict[str, str], row_b: Dict[str, str]) -> float:
    source_a = row_a.get("source_type", "")
    source_b = row_b.get("source_type", "")
    return SOURCE_PAIR_WEIGHTS.get((source_a, source_b), SOURCE_PAIR_WEIGHTS.get((source_b, source_a), 0.75))


def score_match_edge(
    row_a: Dict[str, str],
    row_b: Dict[str, str],
    reason: str,
    pdr_distance: float,
    trajectory_distance_m: float,
    point_window: int,
    slot_window: int,
    max_pdr_distance_m: float,
    video_frame_window: int,
    visual_only: bool,
) -> Dict[str, Any]:
    point_gap = pair_point_gap(row_a, row_b)
    slot_gap = pair_slot_gap(row_a, row_b)
    video_gap = pair_video_sequence_gap(row_a, row_b)
    pdr_score = 1.0 if visual_only else bounded_score(pdr_distance, max_pdr_distance_m)
    trajectory_score = 1.0 if visual_only else bounded_score(trajectory_distance_m, max_pdr_distance_m)
    slot_score = bounded_score(float(slot_gap) if slot_gap is not None else None, max(1.0, float(slot_window)))
    point_score = bounded_score(float(point_gap) if point_gap is not None else None, max(1.0, float(point_window + 1)))
    video_score = bounded_score(float(video_gap) if video_gap is not None else None, max(1.0, float(video_frame_window)))

    if reason == "same_wifi_point_nearby_slots":
        combined = 0.20 * pdr_score + 0.20 * trajectory_score + 0.40 * slot_score + 0.20 * point_score
    elif reason == "nearby_trajectory_points":
        combined = 0.10 * pdr_score + 0.65 * trajectory_score + 0.15 * slot_score + 0.10 * point_score
    elif reason == "video_sequence":
        combined = 0.20 * pdr_score + 0.25 * trajectory_score + 0.15 * point_score + 0.40 * video_score
    elif reason == "video_to_wifi_anchor":
        combined = 0.20 * pdr_score + 0.25 * trajectory_score + 0.35 * point_score + 0.20 * video_score
    else:
        combined = 0.30 * pdr_score + 0.35 * trajectory_score + 0.20 * slot_score + 0.15 * point_score

    reason_weight = REASON_BASE_WEIGHTS.get(reason, 0.60)
    pair_weight = source_pair_weight(row_a, row_b)
    edge_weight = max(0.01, min(1.0, reason_weight * pair_weight * combined))
    return {
        "edge_weight": round(edge_weight, 6),
        "reason_weight": round(reason_weight, 6),
        "source_pair_weight": round(pair_weight, 6),
        "pdr_score": round(pdr_score, 6),
        "trajectory_score": round(trajectory_score, 6),
        "slot_score": round(slot_score, 6),
        "point_score": round(point_score, 6),
        "video_score": round(video_score, 6),
        "point_gap": "" if point_gap is None else point_gap,
        "slot_gap": "" if slot_gap is None else slot_gap,
        "video_sequence_gap": "" if video_gap is None else video_gap,
        "edge_weight_basis": "reason_source_pdr_trajectory_slot_point_video",
    }


def is_video_frame_row(row: Dict[str, str]) -> bool:
    return row.get("source_type") == "video_base_frame" or bool(row.get("video_sequence_index"))


def select_evenly_spaced(rows: Sequence[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    selected: List[Dict[str, str]] = []
    used: set[int] = set()
    for idx in range(limit):
        pos = round(idx * (len(rows) - 1) / max(1, limit - 1))
        if pos not in used:
            selected.append(rows[pos])
            used.add(pos)
    return selected


def build_constrained_match_pairs(
    manifest_rows: Sequence[Dict[str, str]],
    point_window: int,
    slot_window: int,
    max_pdr_distance_m: float,
    video_frame_window: int,
    video_anchor_point_window: int,
    video_anchor_limit: int,
    visual_only: bool = False,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
    video_frame_window = max(1, video_frame_window)
    video_anchor_point_window = max(0, video_anchor_point_window)
    video_anchor_limit = max(0, video_anchor_limit)
    rows = [
        row
        for row in manifest_rows
        if row.get("image_file")
        and row.get("global_point_index")
        and row.get("image_slot")
        and not is_video_frame_row(row)
    ]
    rows.sort(key=lambda row: (safe_int(row.get("global_point_index")), safe_int(row.get("image_slot"))))

    by_point: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        by_point.setdefault(safe_int(row.get("global_point_index")), []).append(row)
    for point_rows in by_point.values():
        point_rows.sort(key=lambda row: safe_int(row.get("image_slot")))

    pairs: Dict[Tuple[str, str], str] = {}

    def add_pair(row_a: Dict[str, str], row_b: Dict[str, str], reason: str) -> None:
        image_a = row_a.get("image_file", "")
        image_b = row_b.get("image_file", "")
        if not image_a or not image_b or image_a == image_b:
            return
        key = tuple(sorted((image_a, image_b)))
        existing_reason = pairs.get(key)
        if existing_reason is None or REASON_BASE_WEIGHTS.get(reason, 0.0) > REASON_BASE_WEIGHTS.get(existing_reason, 0.0):
            pairs[key] = reason

    for point, point_rows in by_point.items():
        slot_count = anchor_slot_count(point_rows)
        for idx, row_a in enumerate(point_rows):
            for row_b in point_rows[idx + 1 :]:
                distance = circular_slot_distance(
                    safe_int(row_a.get("image_slot")),
                    safe_int(row_b.get("image_slot")),
                    slot_count,
                )
                if distance <= slot_window:
                    add_pair(row_a, row_b, "same_wifi_point_nearby_slots")

    points = sorted(by_point)
    point_row_by_id = {point: by_point[point][0] for point in points if by_point[point]}

    def row_xy(row: Dict[str, str]) -> Optional[Tuple[float, float]]:
        x = safe_float(row.get("pdr_x"))
        y = safe_float(row.get("pdr_y"))
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return None
        return x, y

    trajectory_constraint_available = (
        not visual_only
        and len(point_row_by_id) >= 2
        and all(trajectory_coordinate(row) is not None for row in point_row_by_id.values())
    )
    pdr_cell_size = max(1e-6, max_pdr_distance_m)
    pdr_grid: Dict[Tuple[int, int], List[int]] = {}
    if not visual_only and not trajectory_constraint_available and max_pdr_distance_m > 0:
        for point, row in point_row_by_id.items():
            xy = row_xy(row)
            if xy is None:
                continue
            cell = (math.floor(xy[0] / pdr_cell_size), math.floor(xy[1] / pdr_cell_size))
            pdr_grid.setdefault(cell, []).append(point)

    def nearby_trajectory_points(row: Dict[str, str]) -> List[int]:
        if not trajectory_constraint_available or max_pdr_distance_m <= 0:
            return []
        coordinate = trajectory_coordinate(row)
        if coordinate is None:
            return []
        contour_id, route_distance_m = coordinate
        return [
            point
            for point, point_row in point_row_by_id.items()
            if point_row.get("session") == row.get("session")
            and (other := trajectory_coordinate(point_row)) is not None
            and other[0] == contour_id
            and abs(other[1] - route_distance_m) <= max_pdr_distance_m
        ]

    def nearby_pdr_points(row: Dict[str, str]) -> List[int]:
        if visual_only or trajectory_constraint_available or max_pdr_distance_m <= 0:
            return []
        xy = row_xy(row)
        if xy is None:
            return []
        base_cell = (math.floor(xy[0] / pdr_cell_size), math.floor(xy[1] / pdr_cell_size))
        nearby: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for point in pdr_grid.get((base_cell[0] + dx, base_cell[1] + dy), []):
                    point_row = point_row_by_id.get(point)
                    if point_row is not None and point_distance(row, point_row) <= max_pdr_distance_m:
                        nearby.append(point)
        return nearby

    candidate_point_reasons: Dict[Tuple[int, int], str] = {}

    def add_candidate(point_a: int, point_b: int, reason: str) -> None:
        if point_a == point_b:
            return
        left, right = sorted((point_a, point_b))
        row_a = point_row_by_id.get(left)
        row_b = point_row_by_id.get(right)
        if row_a is None or row_b is None or row_a.get("session") != row_b.get("session"):
            return
        key = (left, right)
        current_reason = candidate_point_reasons.get(key)
        if current_reason is None or REASON_BASE_WEIGHTS.get(reason, 0.0) > REASON_BASE_WEIGHTS.get(current_reason, 0.0):
            candidate_point_reasons[key] = reason

    for left_index, point_a in enumerate(points):
        for point_b in points[left_index + 1 :]:
            if point_b - point_a > point_window:
                break
            add_candidate(point_a, point_b, "nearby_wifi_points")
        if trajectory_constraint_available:
            for point_b in nearby_trajectory_points(point_row_by_id[point_a]):
                add_candidate(point_a, point_b, "nearby_trajectory_points")
        else:
            for point_b in nearby_pdr_points(point_row_by_id[point_a]):
                add_candidate(point_a, point_b, "nearby_pdr_points")

    for (point_a, point_b), reason in sorted(candidate_point_reasons.items()):
        rows_a = by_point[point_a]
        rows_b = by_point[point_b]
        slot_count = anchor_slot_count(rows_a + rows_b)
        for row_a in rows_a:
            for row_b in rows_b:
                slot_gap = circular_slot_distance(
                    safe_int(row_a.get("image_slot")),
                    safe_int(row_b.get("image_slot")),
                    slot_count,
                )
                if slot_gap <= slot_window:
                    add_pair(row_a, row_b, reason)

    video_rows = [row for row in manifest_rows if row.get("image_file") and is_video_frame_row(row)]
    by_video: Dict[str, List[Dict[str, str]]] = {}
    for row in video_rows:
        by_video.setdefault(row.get("video_name", "video"), []).append(row)
    for video_name, sequence_rows in by_video.items():
        sequence_rows.sort(
            key=lambda row: (
                safe_int(row.get("video_sequence_index")),
                safe_int(row.get("video_frame_index")),
                row.get("image_file", ""),
            )
        )
        for idx, row_a in enumerate(sequence_rows):
            upper = min(len(sequence_rows), idx + max(1, video_frame_window) + 1)
            for row_b in sequence_rows[idx + 1 : upper]:
                add_pair(row_a, row_b, "video_sequence")

    for video_row in video_rows:
        candidate_points: set[int] = set()
        for field in ("nearest_global_point_index", "left_global_point_index", "right_global_point_index"):
            point = safe_int(video_row.get(field), -1)
            if point < 0:
                continue
            for neighbor in range(point - video_anchor_point_window, point + video_anchor_point_window + 1):
                if neighbor in by_point:
                    candidate_points.add(neighbor)

        if not candidate_points and not visual_only:
            if trajectory_constraint_available:
                candidate_points.update(nearby_trajectory_points(video_row))
            else:
                candidate_points.update(nearby_pdr_points(video_row))

        candidate_anchor_rows: List[Dict[str, str]] = []
        for point in sorted(candidate_points):
            candidate_anchor_rows.extend(by_point.get(point, []))
        candidate_anchor_rows.sort(
            key=lambda row: (
                point_distance(video_row, row),
                safe_int(row.get("global_point_index")),
                safe_int(row.get("image_slot")),
            )
        )
        for anchor_row in select_evenly_spaced(candidate_anchor_rows, video_anchor_limit):
            add_pair(video_row, anchor_row, "video_to_wifi_anchor")

    sorted_pairs = sorted(pairs)
    row_by_image = {
        row.get("image_file", ""): row
        for row in manifest_rows
        if row.get("image_file")
    }
    edge_rows: List[Dict[str, Any]] = []
    for image_a, image_b in sorted_pairs:
        row_a = row_by_image.get(image_a, {})
        row_b = row_by_image.get(image_b, {})
        reason = pairs[(image_a, image_b)]
        pdr_distance = point_distance(row_a, row_b)
        route_distance_m = trajectory_distance(row_a, row_b)
        edge_scores = score_match_edge(
            row_a=row_a,
            row_b=row_b,
            reason=reason,
            pdr_distance=pdr_distance,
            trajectory_distance_m=route_distance_m,
            point_window=point_window,
            slot_window=slot_window,
            max_pdr_distance_m=max_pdr_distance_m,
            video_frame_window=video_frame_window,
            visual_only=visual_only,
        )
        edge_rows.append(
            {
                "image_a": image_a,
                "image_b": image_b,
                "reason": reason,
                "source_type_a": row_a.get("source_type", ""),
                "source_type_b": row_b.get("source_type", ""),
                "global_point_index_a": row_a.get("global_point_index") or row_a.get("nearest_global_point_index", ""),
                "global_point_index_b": row_b.get("global_point_index") or row_b.get("nearest_global_point_index", ""),
                "image_slot_a": row_a.get("image_slot", ""),
                "image_slot_b": row_b.get("image_slot", ""),
                "anchor_slot_count_a": row_a.get("anchor_slot_count", ""),
                "anchor_slot_count_b": row_b.get("anchor_slot_count", ""),
                "video_sequence_index_a": row_a.get("video_sequence_index", ""),
                "video_sequence_index_b": row_b.get("video_sequence_index", ""),
                "pdr_distance_m": "" if visual_only or math.isinf(pdr_distance) else round(pdr_distance, 6),
                "trajectory_contour_id_a": row_a.get("trajectory_contour_id", ""),
                "trajectory_contour_id_b": row_b.get("trajectory_contour_id", ""),
                "trajectory_distance_m": "" if visual_only or math.isinf(route_distance_m) else round(route_distance_m, 6),
                **edge_scores,
            }
        )
    summary = {
        "point_count": len(by_point),
        "image_count": len(rows),
        "video_frame_count": len(video_rows),
        "pair_count": len(sorted_pairs),
        "visual_only": visual_only,
        "point_window": point_window,
        "slot_window": slot_window,
        "max_pdr_distance_m": None if visual_only else max_pdr_distance_m,
        "max_trajectory_distance_m": None if visual_only else max_pdr_distance_m,
        "trajectory_constraint_available": trajectory_constraint_available,
        "trajectory_constraint_mode": "ordered_contour" if trajectory_constraint_available else "legacy_pdr_proximity",
        "video_frame_window": video_frame_window,
        "video_anchor_point_window": video_anchor_point_window,
        "video_anchor_limit": video_anchor_limit,
        "candidate_point_pair_count": len(candidate_point_reasons),
        "reason_counts": {},
        "weighted_graph": True,
        "edge_weight": {"min": None, "mean": None, "max": None},
        "weight_by_reason": {},
    }
    weights: List[float] = []
    weights_by_reason: Dict[str, List[float]] = {}
    for row in edge_rows:
        reason = str(row.get("reason", "unknown"))
        summary["reason_counts"][reason] = summary["reason_counts"].get(reason, 0) + 1
        weight = safe_float(row.get("edge_weight"), math.nan)
        if math.isnan(weight):
            continue
        weights.append(weight)
        weights_by_reason.setdefault(reason, []).append(weight)
    if weights:
        summary["edge_weight"] = {
            "min": round(min(weights), 6),
            "mean": round(sum(weights) / len(weights), 6),
            "max": round(max(weights), 6),
        }
    for reason, reason_weights in sorted(weights_by_reason.items()):
        summary["weight_by_reason"][reason] = {
            "count": len(reason_weights),
            "min": round(min(reason_weights), 6),
            "mean": round(sum(reason_weights) / len(reason_weights), 6),
            "max": round(max(reason_weights), 6),
        }
    return sorted_pairs, edge_rows, summary


def write_match_pairs(workspace_dir: Path, pairs: Sequence[Tuple[str, str]]) -> None:
    with (workspace_dir / "match_pairs.txt").open("w", encoding="utf-8") as handle:
        for image_a, image_b in pairs:
            handle.write(f"{image_a} {image_b}\n")


def write_match_graph_edges(workspace_dir: Path, edge_rows: Sequence[Dict[str, Any]]) -> None:
    write_csv_rows(
        workspace_dir / "match_graph_edges.csv",
        [
            "image_a",
            "image_b",
            "reason",
            "edge_weight",
            "reason_weight",
            "source_pair_weight",
            "pdr_score",
            "trajectory_score",
            "slot_score",
            "point_score",
            "video_score",
            "point_gap",
            "slot_gap",
            "video_sequence_gap",
            "edge_weight_basis",
            "source_type_a",
            "source_type_b",
            "global_point_index_a",
            "global_point_index_b",
            "image_slot_a",
            "image_slot_b",
            "anchor_slot_count_a",
            "anchor_slot_count_b",
            "video_sequence_index_a",
            "video_sequence_index_b",
            "pdr_distance_m",
            "trajectory_contour_id_a",
            "trajectory_contour_id_b",
            "trajectory_distance_m",
        ],
        edge_rows,
    )


def write_pdr_alignment_refs(workspace_dir: Path, manifest_rows: Sequence[Dict[str, str]]) -> int:
    written = 0
    seen: set[str] = set()
    with (workspace_dir / "pdr_alignment_refs.txt").open("w", encoding="utf-8") as handle:
        for row in sorted(
            manifest_rows,
            key=lambda item: (safe_int(item.get("global_point_index")), safe_int(item.get("image_slot"))),
        ):
            image_file = row.get("image_file", "")
            if row.get("source_type") == "video_base_frame":
                continue
            if not image_file or image_file in seen:
                continue
            x = safe_float(row.get("pdr_x"))
            y = safe_float(row.get("pdr_y"))
            z = safe_float(row.get("pdr_z"), 1.5)
            if any(math.isnan(value) for value in (x, y, z)):
                continue
            handle.write(f"{image_file} {x:.6f} {y:.6f} {z:.6f}\n")
            seen.add(image_file)
            written += 1
    return written


def remove_pdr_alignment_refs(workspace_dir: Path) -> None:
    refs_path = workspace_dir / "pdr_alignment_refs.txt"
    if refs_path.exists():
        refs_path.unlink()


def batch_header(root: Path, workspace_dir: Path, colmap_exe: Path) -> str:
    return f"""@echo off
setlocal
set "ROOT={windows_path(root)}"
set "WORKSPACE={windows_path(workspace_dir)}"
set "COLMAP_EXE={windows_path(colmap_exe)}"
set "IMAGE_PATH=%WORKSPACE%\\images"
set "IMAGE_LIST=%WORKSPACE%\\image_list.txt"
set "DATABASE=%WORKSPACE%\\database.db"

if not exist "%COLMAP_EXE%" (
  echo [error] COLMAP not found: %COLMAP_EXE%
  exit /b 1
)
"""


def batch_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def colmap_command(command: str, args: Sequence[str]) -> str:
    if not args:
        return f'"%COLMAP_EXE%" {command}'
    return f'"%COLMAP_EXE%" {command} ^\n' + " ^\n".join(args)


def prefixed_options(prefix: str, options: Dict[str, Any]) -> List[str]:
    return [
        f"  --{prefix}.{name} {batch_value(value)}"
        for name, value in options.items()
        if value is not None
    ]


def dense_profile_config(dense_profile: str) -> Dict[str, Any]:
    if dense_profile not in DENSE_PROFILES:
        return DENSE_PROFILES["balanced"]
    return DENSE_PROFILES[dense_profile]


def load_profile_config(load_profile: str) -> Dict[str, Any]:
    if load_profile not in LOAD_PROFILES:
        return LOAD_PROFILES["low-peak"]
    return LOAD_PROFILES[load_profile]


def resolve_load_settings(
    load_profile: str,
    colmap_num_threads: Optional[int],
    stage_cooldown_s: Optional[int],
) -> Tuple[str, Dict[str, Any], int, int]:
    profile_name = load_profile if load_profile in LOAD_PROFILES else "low-peak"
    config = load_profile_config(profile_name)
    default_threads = int(config.get("num_threads") or 0)
    default_cooldown = int(config.get("stage_cooldown_s") or 0)
    threads = default_threads if colmap_num_threads is None else max(0, int(colmap_num_threads))
    cooldown = default_cooldown if stage_cooldown_s is None else max(0, int(stage_cooldown_s))
    return profile_name, config, threads, cooldown


def thread_option(prefix: str, num_threads: int) -> str:
    return f" ^\n  --{prefix}.num_threads {num_threads}" if num_threads > 0 else ""


def prefixed_thread_option(prefix: str, num_threads: int) -> List[str]:
    return [f"  --{prefix}.num_threads {num_threads}"] if num_threads > 0 else []


def cooldown_block(stage_cooldown_s: int, label: str) -> str:
    if stage_cooldown_s <= 0:
        return ""
    return f"""
echo [cooldown] Waiting {stage_cooldown_s}s after {label} to reduce sustained peak load...
timeout /t {stage_cooldown_s} /nobreak >nul
"""


def thread_environment_block(num_threads: int) -> str:
    if num_threads <= 0:
        return ""
    return f"""
set "OMP_NUM_THREADS={num_threads}"
set "OPENBLAS_NUM_THREADS={num_threads}"
set "MKL_NUM_THREADS={num_threads}"
set "NUMEXPR_NUM_THREADS={num_threads}"
set "OPENCV_FOR_THREADS_NUM={num_threads}"
echo [load] Low-peak thread cap: {num_threads}
"""


def remove_stale_quality_reports(workspace_dir: Path) -> List[str]:
    removed: List[str] = []
    for name in ("quality_report.json", "QUALITY_REPORT.md"):
        path = workspace_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def remove_stale_image_similarity_outputs(workspace_dir: Path) -> List[str]:
    removed: List[str] = []
    for name in ("image_similarity_edges.csv", "image_similarity_summary.json"):
        path = workspace_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def resolve_optional_path(root: Path, path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else root / path


def mask_candidates(mask_dir: Path, image_name: str) -> List[Path]:
    rel = Path(image_name)
    candidates = [mask_dir / rel]
    candidates.append(mask_dir / f"{image_name}.png")
    if rel.suffix:
        candidates.append(mask_dir / rel.with_suffix(".png"))
    else:
        candidates.extend(mask_dir / f"{image_name}{suffix}" for suffix in MASK_EXTENSIONS)
    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def mask_dir_summary(mask_dir: Optional[Path], image_names: Sequence[str]) -> Dict[str, Any]:
    if mask_dir is None:
        return {"enabled": False, "path": "", "exists": False, "matched_image_count": 0, "missing_image_count": 0}
    matched = 0
    missing: List[str] = []
    if mask_dir.exists():
        for image_name in image_names:
            if any(candidate.exists() for candidate in mask_candidates(mask_dir, image_name)):
                matched += 1
            elif len(missing) < 20:
                missing.append(image_name)
    return {
        "enabled": mask_dir.exists(),
        "path": str(mask_dir),
        "exists": mask_dir.exists(),
        "matched_image_count": matched,
        "missing_image_count": max(0, len(image_names) - matched) if mask_dir.exists() else len(image_names),
        "missing_images": missing,
        "accepted_name_patterns": [
            "same relative path as image",
            "image name plus .png",
            "same stem with .png",
        ],
    }


def write_sparse_batch(
    root: Path,
    workspace_dir: Path,
    colmap_exe: Path,
    mask_dir: Optional[Path],
    camera_model: str,
    single_camera: bool,
    use_gpu: bool,
    matcher: str,
    align_to_pdr: bool,
    feature_max_image_size: int,
    max_num_features: int,
    max_num_matches: int,
    num_threads: int,
    stage_cooldown_s: int,
) -> None:
    matcher_command = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    single_camera_flag = "1" if single_camera else "0"
    gpu_flag = "1" if use_gpu else "0"
    feature_extraction_threads = thread_option("FeatureExtraction", num_threads)
    feature_matching_threads = thread_option("FeatureMatching", num_threads)
    mapper_threads = thread_option("Mapper", num_threads)
    after_feature_cooldown = cooldown_block(stage_cooldown_s, "feature extraction")
    after_matching_cooldown = cooldown_block(stage_cooldown_s, "feature matching")
    after_mapper_cooldown = cooldown_block(stage_cooldown_s, "sparse mapping")
    thread_env = thread_environment_block(num_threads)
    constrained_matching_block = f"""echo [2/4] Matching constrained image pairs from visual/order priors...
if not exist "%WORKSPACE%\\match_pairs.txt" (
  echo [error] Missing constrained pair list: %WORKSPACE%\\match_pairs.txt
  exit /b 1
)
"%COLMAP_EXE%" matches_importer ^
  --database_path "%DATABASE%" ^
  --match_list_path "%WORKSPACE%\\match_pairs.txt" ^
  --match_type pairs ^
  --FeatureMatching.use_gpu {gpu_flag} ^
  --FeatureMatching.guided_matching 1 ^
  --FeatureMatching.max_num_matches {max_num_matches}{feature_matching_threads}
if errorlevel 1 exit /b %errorlevel%
{after_matching_cooldown}
"""
    if matcher == "constrained":
        matching_block = constrained_matching_block
    elif matcher == "hybrid":
        matching_block = constrained_matching_block + f"""
echo [2b/4] Adding exhaustive visual bridge matches...
"%COLMAP_EXE%" exhaustive_matcher ^
  --database_path "%DATABASE%" ^
  --FeatureMatching.use_gpu {gpu_flag} ^
  --FeatureMatching.guided_matching 1 ^
  --FeatureMatching.max_num_matches {max_num_matches}{feature_matching_threads}
if errorlevel 1 exit /b %errorlevel%
{after_matching_cooldown}
"""
    else:
        matching_block = f"""echo [2/4] Matching features with {matcher_command}...
"%COLMAP_EXE%" {matcher_command} ^
  --database_path "%DATABASE%" ^
  --FeatureMatching.use_gpu {gpu_flag} ^
  --FeatureMatching.guided_matching 1 ^
  --FeatureMatching.max_num_matches {max_num_matches}{feature_matching_threads}
if errorlevel 1 exit /b %errorlevel%
{after_matching_cooldown}
"""
    feature_limits = ""
    if feature_max_image_size > 0:
        feature_limits += f" ^\n  --SiftExtraction.max_image_size {feature_max_image_size}"
    if max_num_features > 0:
        feature_limits += f" ^\n  --SiftExtraction.max_num_features {max_num_features}"
    mask_option = ""
    if mask_dir is not None:
        mask_option = f' ^\n  --ImageReader.mask_path "{windows_path(mask_dir)}"'
    alignment_block = ""
    if align_to_pdr:
        alignment_block = """
echo [4/4] Aligning sparse model to PDR coordinate seeds...
if exist "%WORKSPACE%\\pdr_alignment_refs.txt" (
  if exist "%WORKSPACE%\\sparse\\0" (
    if not exist "%WORKSPACE%\\sparse_pdr_aligned" mkdir "%WORKSPACE%\\sparse_pdr_aligned"
    if not exist "%WORKSPACE%\\sparse_pdr_aligned\\0" mkdir "%WORKSPACE%\\sparse_pdr_aligned\\0"
    "%COLMAP_EXE%" model_aligner ^
      --input_path "%WORKSPACE%\\sparse\\0" ^
      --output_path "%WORKSPACE%\\sparse_pdr_aligned\\0" ^
      --ref_images_path "%WORKSPACE%\\pdr_alignment_refs.txt" ^
      --ref_is_gps 0 ^
      --alignment_type custom ^
      --min_common_images 3 ^
      --alignment_max_error 10
    if errorlevel 1 (
      echo [warn] PDR alignment failed; dense step will use raw sparse model.
    ) else (
      if exist "%WORKSPACE%\\sparse_pdr_aligned\\0\\images.bin" (
        echo [done] PDR-aligned sparse model: %WORKSPACE%\\sparse_pdr_aligned\\0
      ) else (
        if exist "%WORKSPACE%\\sparse_pdr_aligned\\0\\images.txt" (
          echo [done] PDR-aligned sparse model: %WORKSPACE%\\sparse_pdr_aligned\\0
        ) else (
          echo [warn] PDR alignment did not write a model; dense step will use raw sparse model.
        )
      )
    )
  )
)
"""
    else:
        alignment_block = """
echo [4/4] Skipping PDR alignment for visual-only reconstruction.
"""
    content = batch_header(root, workspace_dir, colmap_exe) + f"""{thread_env}
if exist "%DATABASE%" (
  echo [warn] Existing database found: %DATABASE%
  echo [warn] If you changed matching constraints, run reset_colmap_outputs.bat first.
)
if not exist "%WORKSPACE%\\sparse" mkdir "%WORKSPACE%\\sparse"

echo [1/4] Extracting features...
"%COLMAP_EXE%" feature_extractor ^
  --database_path "%DATABASE%" ^
  --image_path "%IMAGE_PATH%" ^
  --image_list_path "%IMAGE_LIST%" ^
  --ImageReader.camera_model {camera_model} ^
  --ImageReader.single_camera {single_camera_flag} ^
  --FeatureExtraction.use_gpu {gpu_flag}{mask_option}{feature_limits}{feature_extraction_threads}
if errorlevel 1 exit /b %errorlevel%
{after_feature_cooldown}

{matching_block}

echo [3/4] Building sparse model...
"%COLMAP_EXE%" mapper ^
  --database_path "%DATABASE%" ^
  --image_path "%IMAGE_PATH%" ^
  --output_path "%WORKSPACE%\\sparse" ^
  --Mapper.multiple_models 0{mapper_threads}
if errorlevel 1 exit /b %errorlevel%
{after_mapper_cooldown}

{alignment_block}
python "%ROOT%\\reconstruction_quality.py" --workspace "%WORKSPACE%" --summary "%WORKSPACE%\\quality_report.json" --markdown "%WORKSPACE%\\QUALITY_REPORT.md"
if errorlevel 1 echo [warn] Sparse quality report failed; reconstruction outputs are still available.

echo [done] Sparse model candidates written under %WORKSPACE%\\sparse
"""
    write_text(workspace_dir / "run_colmap_sparse.bat", content)


def write_dense_batch(
    root: Path,
    workspace_dir: Path,
    colmap_exe: Path,
    mask_dir: Optional[Path],
    dense_max_image_size: int,
    align_to_pdr: bool,
    dense_profile: str,
    run_meshing: bool,
    run_point_cloud_filter: bool,
    num_threads: int,
    stage_cooldown_s: int,
) -> None:
    profile = dense_profile_config(dense_profile)
    point_filter_max_points = int(profile.get("point_filter_max_points", 600000))
    cleanup_intermediate_ply = bool(profile.get("cleanup_intermediate_ply", True))
    after_undistort_cooldown = cooldown_block(stage_cooldown_s, "image undistortion")
    after_patchmatch_cooldown = cooldown_block(stage_cooldown_s, "PatchMatch stereo")
    after_fusion_cooldown = cooldown_block(stage_cooldown_s, "stereo fusion")
    after_postprocess_cooldown = cooldown_block(stage_cooldown_s, "post-processing")
    thread_env = thread_environment_block(num_threads)
    patch_match_args = [
        '  --workspace_path "%WORKSPACE%\\dense"',
        "  --workspace_format COLMAP",
        *prefixed_options("PatchMatchStereo", profile["patch_match"]),
    ]
    fusion_args = [
        '  --workspace_path "%WORKSPACE%\\dense"',
        "  --workspace_format COLMAP",
        "  --input_type geometric",
        *prefixed_options("StereoFusion", profile["fusion"]),
        *prefixed_thread_option("StereoFusion", num_threads),
        *([f'  --StereoFusion.mask_path "{windows_path(mask_dir)}"'] if mask_dir is not None else []),
        '  --output_path "%WORKSPACE%\\dense\\fused.ply"',
    ]
    undistorter_command = colmap_command(
        "image_undistorter",
        [
            '  --image_path "%IMAGE_PATH%"',
            '  --input_path "%SPARSE_MODEL%"',
            '  --output_path "%WORKSPACE%\\dense"',
            "  --output_type COLMAP",
            f"  --max_image_size {dense_max_image_size}",
        ],
    )
    patch_match_command = colmap_command("patch_match_stereo", patch_match_args)
    fusion_command = colmap_command("stereo_fusion", fusion_args)
    sparse_selector = """set "SPARSE_MODEL=%WORKSPACE%\\sparse\\0"
if exist "%WORKSPACE%\\sparse_pdr_aligned\\0\\cameras.bin" set "SPARSE_MODEL=%WORKSPACE%\\sparse_pdr_aligned\\0"
if exist "%WORKSPACE%\\sparse_pdr_aligned\\0\\cameras.txt" set "SPARSE_MODEL=%WORKSPACE%\\sparse_pdr_aligned\\0"
echo [info] Dense input sparse model: %SPARSE_MODEL%
"""
    point_filter_block = ""
    if run_point_cloud_filter:
        point_filter_block = f"""
echo [4/8] Filtering dense point cloud outliers...
python "%ROOT%\\point_cloud_filter.py" ^
  --input "%WORKSPACE%\\dense\\fused.ply" ^
  --output "%WORKSPACE%\\dense\\fused_filtered.ply" ^
  --summary "%WORKSPACE%\\dense\\point_filter_summary.json" ^
  --max-points {point_filter_max_points}
if errorlevel 1 (
  echo [warn] Point cloud filtering failed; raw fused.ply is still available.
) else (
  set "DENSE_POINT_CLOUD=%WORKSPACE%\\dense\\fused_filtered.ply"
)
{after_postprocess_cooldown}
"""
    mesh_block = ""
    if run_meshing and profile.get("run_meshing", True):
        delaunay_command = colmap_command(
            "delaunay_mesher",
            [
                '  --input_path "%WORKSPACE%\\dense"',
                '  --output_path "%WORKSPACE%\\dense\\meshed-delaunay.ply"',
            ],
        )
        poisson_command = colmap_command(
            "poisson_mesher",
            [
                '  --input_path "%DENSE_POINT_CLOUD%"',
                '  --output_path "%WORKSPACE%\\dense\\meshed-poisson.ply"',
            ],
        )
        mesh_block = f"""
echo [5/8] Building a Delaunay mesh for robust MeshLab inspection...
{delaunay_command}
if errorlevel 1 echo [warn] Delaunay meshing failed; continuing with fused point cloud.
{after_postprocess_cooldown}

echo [6/8] Building a Poisson mesh for smoother surface inspection...
{poisson_command}
if errorlevel 1 echo [warn] Poisson meshing failed; continuing with fused point cloud.
{after_postprocess_cooldown}
"""
    quality_block = """
python "%ROOT%\\reconstruction_quality.py" --workspace "%WORKSPACE%" --summary "%WORKSPACE%\\quality_report.json" --markdown "%WORKSPACE%\\QUALITY_REPORT.md"
if errorlevel 1 echo [warn] Final quality report failed; reconstruction outputs are still available.
"""
    keep_intermediate_flag = " ^\n  --keep-intermediate" if not cleanup_intermediate_ply else ""
    finalize_block = f"""
echo [finalize] Selecting dense\\final.ply and pruning intermediate PLY files...
python "%ROOT%\\finalize_dense_outputs.py" ^
  --dense-dir "%WORKSPACE%\\dense" ^
  --summary "%WORKSPACE%\\dense\\final_output_summary.json"{keep_intermediate_flag}
if errorlevel 1 exit /b %errorlevel%
{after_postprocess_cooldown}
"""
    pdr_note = "PDR-aligned sparse models are preferred when present." if align_to_pdr else "Visual sparse models are used because visual-only mode is enabled."
    content = batch_header(root, workspace_dir, colmap_exe) + f"""{thread_env}
{sparse_selector}if not exist "%SPARSE_MODEL%" (
  echo [error] Sparse model not found: %SPARSE_MODEL%
  echo Run run_colmap_sparse.bat first, or check which sparse model folder COLMAP created.
  exit /b 1
)
if not exist "%WORKSPACE%\\dense" mkdir "%WORKSPACE%\\dense"
set "DENSE_POINT_CLOUD=%WORKSPACE%\\dense\\fused.ply"
if exist "%WORKSPACE%\\dense\\final.ply" del "%WORKSPACE%\\dense\\final.ply"
if exist "%WORKSPACE%\\dense\\final_output_summary.json" del "%WORKSPACE%\\dense\\final_output_summary.json"
if exist "%WORKSPACE%\\dense\\meshed-poisson.ply" del "%WORKSPACE%\\dense\\meshed-poisson.ply"
if exist "%WORKSPACE%\\dense\\meshed-delaunay.ply" del "%WORKSPACE%\\dense\\meshed-delaunay.ply"
if exist "%WORKSPACE%\\dense\\fused_filtered.ply" del "%WORKSPACE%\\dense\\fused_filtered.ply"
if exist "%WORKSPACE%\\dense\\point_filter_summary.json" del "%WORKSPACE%\\dense\\point_filter_summary.json"
if exist "%WORKSPACE%\\dense\\fused.ply.vis" del "%WORKSPACE%\\dense\\fused.ply.vis"

echo [1/8] Undistorting images...
{undistorter_command}
if errorlevel 1 exit /b %errorlevel%
{after_undistort_cooldown}

echo [2/8] Running CUDA PatchMatch stereo...
{patch_match_command}
if errorlevel 1 exit /b %errorlevel%
{after_patchmatch_cooldown}

echo [3/8] Fusing dense point cloud...
{fusion_command}
if errorlevel 1 exit /b %errorlevel%
{after_fusion_cooldown}

{point_filter_block}{mesh_block}{finalize_block}{quality_block}
if exist "%WORKSPACE%\\dense\\final.ply" echo [done] Final dense model: %WORKSPACE%\\dense\\final.ply
"""
    write_text(workspace_dir / "run_colmap_dense.bat", content)


def write_full_batch(workspace_dir: Path) -> None:
    content = """@echo off
setlocal
call "%~dp0run_colmap_sparse.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0run_colmap_dense.bat"
if errorlevel 1 exit /b %errorlevel%
"""
    write_text(workspace_dir / "run_colmap_full.bat", content)


def write_automatic_batch(
    root: Path,
    workspace_dir: Path,
    colmap_exe: Path,
    camera_model: str,
    single_camera: bool,
    use_gpu: bool,
    quality: str,
) -> None:
    single_camera_flag = "1" if single_camera else "0"
    gpu_flag = "1" if use_gpu else "0"
    content = batch_header(root, workspace_dir, colmap_exe) + f"""
echo [info] Running COLMAP automatic_reconstructor...
"%COLMAP_EXE%" automatic_reconstructor ^
  --workspace_path "%WORKSPACE%" ^
  --image_path "%IMAGE_PATH%" ^
  --image_list_path "%IMAGE_LIST%" ^
  --camera_model {camera_model} ^
  --single_camera {single_camera_flag} ^
  --use_gpu {gpu_flag} ^
  --quality {quality} ^
  --dense 1
if errorlevel 1 exit /b %errorlevel%

echo [done] Automatic reconstruction complete.
"""
    write_text(workspace_dir / "run_colmap_automatic.bat", content)


def write_gui_batch(root: Path, workspace_dir: Path, colmap_exe: Path) -> None:
    content = batch_header(root, workspace_dir, colmap_exe) + """
echo [info] Opening COLMAP GUI...
"%COLMAP_EXE%" gui
"""
    write_text(workspace_dir / "open_colmap_gui.bat", content)


def write_reset_batch(root: Path, workspace_dir: Path) -> None:
    content = f"""@echo off
setlocal
set "ROOT={windows_path(root)}"
set "WORKSPACE=%~dp0"
echo This will delete database.db, sparse, PDR-aligned sparse, dense, and automatic output folders under:
echo %WORKSPACE%
set /p CONFIRM=Type YES to reset COLMAP outputs:
if /I not "%CONFIRM%"=="YES" (
  echo cancelled
  exit /b 0
)
if exist "%WORKSPACE%database.db" del "%WORKSPACE%database.db"
if exist "%WORKSPACE%sparse" rmdir /s /q "%WORKSPACE%sparse"
if exist "%WORKSPACE%sparse_pdr_aligned" rmdir /s /q "%WORKSPACE%sparse_pdr_aligned"
if exist "%WORKSPACE%dense" rmdir /s /q "%WORKSPACE%dense"
if exist "%WORKSPACE%stereo" rmdir /s /q "%WORKSPACE%stereo"
if exist "%WORKSPACE%final.ply" del "%WORKSPACE%final.ply"
if exist "%WORKSPACE%final_output_summary.json" del "%WORKSPACE%final_output_summary.json"
if exist "%WORKSPACE%meshed-poisson.ply" del "%WORKSPACE%meshed-poisson.ply"
if exist "%WORKSPACE%meshed-delaunay.ply" del "%WORKSPACE%meshed-delaunay.ply"
if exist "%WORKSPACE%fused_filtered.ply" del "%WORKSPACE%fused_filtered.ply"
if exist "%WORKSPACE%point_filter_summary.json" del "%WORKSPACE%point_filter_summary.json"
if exist "%WORKSPACE%quality_report.json" del "%WORKSPACE%quality_report.json"
if exist "%WORKSPACE%QUALITY_REPORT.md" del "%WORKSPACE%QUALITY_REPORT.md"
echo [done] COLMAP outputs reset.
"""
    write_text(workspace_dir / "reset_colmap_outputs.bat", content)


def write_readme(
    workspace_dir: Path,
    visual_only: bool,
    matcher: str,
    dense_profile: str,
    load_profile: str,
    num_threads: int,
    stage_cooldown_s: int,
) -> None:
    alignment_note = (
        "This workspace is visual-only: no `pdr_alignment_refs.txt` is generated, and sparse models are not aligned to PDR.\n"
        if visual_only
        else "After sparse mapping, the script tries to align the model to PDR coordinates and dense reconstruction prefers that aligned model.\n"
    )
    pdr_ref_line = (
        "- `pdr_alignment_refs.txt`: disabled in visual-only mode.\n"
        if visual_only
        else "- `pdr_alignment_refs.txt`: image position seeds used to align sparse output.\n"
    )
    content = f"""# COLMAP Workspace

This folder is prepared from the current WiFi/IMU/photo capture.

## Start Here

1. Run `run_colmap_sparse.bat` to build sparse SfM model candidates.
2. Run `run_colmap_dense.bat` to generate the dense result and reports.
3. Open `dense/final.ply` in MeshLab for the final reconstruction view.

For normal rebuilds, prefer `python run_colmap.py full` from the project root; it clears old COLMAP outputs automatically before running sparse and dense reconstruction. If you run these generated `.bat` scripts directly and they report existing outputs, run `reset_colmap_outputs.bat` first; it asks you to type `YES` before deleting old outputs.

The generated scripts favor reconstruction quality over speed. The sparse matcher is `{matcher}` and the dense profile is `{dense_profile}`.
The load profile is `{load_profile}` with COLMAP thread cap `{num_threads}` and `{stage_cooldown_s}` seconds of cooldown between heavy stages.

## Why This Setup

- Hybrid matching first uses explainable WiFi checkpoint, ordered-PDR-contour, and video-order pairs, then adds COLMAP exhaustive visual bridge matches to reduce disconnected sparse components. `match_graph_edges.csv` explains the constrained pairs; exhaustive bridge matches are generated internally by COLMAP.
- `image_similarity_edges.csv` checks the constrained pairs with perceptual hash, color histogram, ORB matches, and fundamental-matrix inliers. This is diagnostic evidence; it does not delete COLMAP pairs automatically.
- Dense fusion uses a completeness-oriented profile for indoor weak-texture floor and wall regions.
- `metadata/pdr_trajectory_contour.csv` is the time-ordered IMU/PDR route corridor used as a soft matching constraint. It is not a room boundary, mesh-completion source, or pose optimizer.
- If `--colmap-mask-dir` was provided during preprocessing and the directory exists, feature extraction and dense fusion use those masks to suppress semantic clutter such as desks and chairs.
- Mesh outputs are finalized into `dense/final.ply` so the result directory has one obvious model to open.
- Intermediate dense PLY files are removed by default after `final.ply` is selected; set `KEEP_INTERMEDIATE_PLY=1` before running the batch if you need debug artifacts.

## Inputs

- `images/`: renamed WiFi-point photos plus optional video base frames.
- `image_list.txt`: exact image list passed to COLMAP.
- `metadata/camera_pose_seed.csv`: PDR XY plus ordered point-photo yaw seed.
- `metadata/wifi_fingerprint_library.csv`: WiFi fingerprint table.
- `metadata/wifi_boundary_seed.csv`: WiFi interpolation envelope from checkpoint samples, not a route contour or room boundary.
- `metadata/pdr_trajectory_contour.csv`: ordered PDR route centerline plus a bounded coverage corridor.
- `metadata/wifi_localization_model.json`: KNN WiFi fingerprint localizer.
- `metadata/wifi_signal_grid.csv`: interpolated RSSI field inside the sampled area.
- `metadata/wifi_signal_boundary.csv`: RSSI metrics on the WiFi/PDR boundary.
- `metadata/video_frames_manifest.csv`: optional video frames used as a weak visual base.
- `match_pairs.txt`: constrained COLMAP image pairs from WiFi checkpoint order,
  neighboring angular slots, ordered PDR-contour distance, and video-frame sequence order.
- `match_graph_edges.csv`: diagnostic weighted edge list with pair reasons, scores, and original slot counts.
- `image_similarity_edges.csv` / `image_similarity_summary.json`: P1 visual-support diagnostics for the constrained match graph.
{pdr_ref_line}

## Outputs After Running COLMAP

- `QUALITY_REPORT.md` / `quality_report.json`: sparse component counts, largest registered-image count, and dense-output availability.
- `dense/final.ply`: final dense model to open; Poisson mesh is preferred, with Delaunay or point-cloud fallback if meshing fails.
- `dense/final_output_summary.json`: source chosen for `final.ply` and intermediate files removed during cleanup.

{alignment_note}
"""
    write_text(workspace_dir / "README.md", content)


def prepare_colmap_workspace(
    root: Path,
    reconstruction_dir: Path,
    workspace_dir: Path,
    colmap_dir: Optional[Path] = None,
    colmap_exe: Optional[Path] = None,
    colmap_mask_dir: Optional[Path] = None,
    camera_model: str = "SIMPLE_RADIAL",
    single_camera: bool = True,
    use_gpu: bool = True,
    matcher: str = "constrained",
    quality: str = "high",
    dense_max_image_size: int = 1200,
    visual_only: bool = False,
    feature_max_image_size: int = 3200,
    max_num_features: int = 20000,
    max_num_matches: int = 131072,
    load_profile: str = "low-peak",
    colmap_num_threads: Optional[int] = None,
    stage_cooldown_s: Optional[int] = None,
    dense_profile: str = "complete",
    run_meshing: bool = True,
    run_point_cloud_filter: bool = True,
    constrained_point_window: int = 2,
    constrained_slot_window: int = 3,
    constrained_max_pdr_distance_m: float = 12.0,
    constrained_video_frame_window: int = 6,
    constrained_video_anchor_point_window: int = 2,
    constrained_video_anchor_limit: int = 24,
    run_image_similarity: bool = True,
    image_similarity_max_pairs: int = 0,
    image_similarity_max_image_size: int = 900,
    image_similarity_orb_features: int = 1500,
    image_similarity_min_visual_score: float = 0.22,
) -> Dict[str, Any]:
    _fields, manifest_rows = read_csv(reconstruction_dir / "images_manifest.csv")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    existing_colmap_outputs = [
        name
        for name in ("database.db", "sparse", "sparse_pdr_aligned", "sparse_pose_prior", "dense")
        if (workspace_dir / name).exists()
    ]
    removed_stale_reports = remove_stale_quality_reports(workspace_dir)
    removed_stale_image_similarity = remove_stale_image_similarity_outputs(workspace_dir)
    resolved_colmap_exe = find_colmap_exe(root, colmap_dir=colmap_dir, colmap_exe=colmap_exe)
    resolved_mask_dir = resolve_optional_path(root, colmap_mask_dir)
    dense_profile = dense_profile if dense_profile in DENSE_PROFILES else "balanced"
    profile_config = DENSE_PROFILES[dense_profile]
    load_profile, load_config, effective_num_threads, effective_stage_cooldown_s = resolve_load_settings(
        load_profile,
        colmap_num_threads,
        stage_cooldown_s,
    )
    effective_run_meshing = run_meshing and bool(profile_config.get("run_meshing", True))

    copied_images, image_names, missing_images, removed_stale_workspace_images = copy_images(
        reconstruction_dir,
        workspace_dir,
        manifest_rows,
    )
    copied_metadata = copy_metadata(reconstruction_dir, workspace_dir)
    mask_summary = mask_dir_summary(resolved_mask_dir, image_names)
    active_mask_dir = resolved_mask_dir if mask_summary.get("enabled") else None
    write_image_list(workspace_dir, image_names)
    match_pairs, match_edge_rows, match_pair_summary = build_constrained_match_pairs(
        manifest_rows=manifest_rows,
        point_window=constrained_point_window,
        slot_window=constrained_slot_window,
        max_pdr_distance_m=constrained_max_pdr_distance_m,
        video_frame_window=constrained_video_frame_window,
        video_anchor_point_window=constrained_video_anchor_point_window,
        video_anchor_limit=constrained_video_anchor_limit,
        visual_only=visual_only,
    )
    write_match_pairs(workspace_dir, match_pairs)
    write_match_graph_edges(workspace_dir, match_edge_rows)
    if run_image_similarity:
        try:
            image_similarity_summary = analyze_workspace_image_similarity(
                workspace_dir=workspace_dir,
                edge_rows=match_edge_rows,
                max_pairs=image_similarity_max_pairs,
                max_image_size=image_similarity_max_image_size,
                orb_features=image_similarity_orb_features,
                min_visual_score=image_similarity_min_visual_score,
            )
        except Exception as exc:
            image_similarity_summary = {
                "status": "failed",
                "reason": str(exc),
                "diagnostic_only": True,
                "input_pair_count": len(match_edge_rows),
                "output_csv": str(workspace_dir / "image_similarity_edges.csv"),
            }
            write_json(workspace_dir / "image_similarity_summary.json", image_similarity_summary)
    else:
        image_similarity_summary = {
            "status": "disabled",
            "reason": "disabled_by_command_line",
            "diagnostic_only": True,
            "input_pair_count": len(match_edge_rows),
            "output_csv": str(workspace_dir / "image_similarity_edges.csv"),
        }
        write_json(workspace_dir / "image_similarity_summary.json", image_similarity_summary)
    align_to_pdr = not visual_only
    if align_to_pdr:
        pdr_alignment_ref_count = write_pdr_alignment_refs(workspace_dir, manifest_rows)
    else:
        remove_pdr_alignment_refs(workspace_dir)
        pdr_alignment_ref_count = 0

    matcher = matcher if matcher in {"constrained", "hybrid", "exhaustive", "sequential"} else "constrained"
    write_sparse_batch(
        root=root,
        workspace_dir=workspace_dir,
        colmap_exe=resolved_colmap_exe,
        mask_dir=active_mask_dir,
        camera_model=camera_model,
        single_camera=single_camera,
        use_gpu=use_gpu,
        matcher=matcher,
        align_to_pdr=align_to_pdr,
        feature_max_image_size=feature_max_image_size,
        max_num_features=max_num_features,
        max_num_matches=max_num_matches,
        num_threads=effective_num_threads,
        stage_cooldown_s=effective_stage_cooldown_s,
    )
    write_dense_batch(
        root=root,
        workspace_dir=workspace_dir,
        colmap_exe=resolved_colmap_exe,
        mask_dir=active_mask_dir,
        dense_max_image_size=dense_max_image_size,
        align_to_pdr=align_to_pdr,
        dense_profile=dense_profile,
        run_meshing=run_meshing,
        run_point_cloud_filter=run_point_cloud_filter,
        num_threads=effective_num_threads,
        stage_cooldown_s=effective_stage_cooldown_s,
    )
    write_full_batch(workspace_dir)
    write_automatic_batch(
        root=root,
        workspace_dir=workspace_dir,
        colmap_exe=resolved_colmap_exe,
        camera_model=camera_model,
        single_camera=single_camera,
        use_gpu=use_gpu,
        quality=quality,
    )
    write_gui_batch(root=root, workspace_dir=workspace_dir, colmap_exe=resolved_colmap_exe)
    write_reset_batch(root, workspace_dir)
    write_readme(
        workspace_dir,
        visual_only=visual_only,
        matcher=matcher,
        dense_profile=dense_profile,
        load_profile=load_profile,
        num_threads=effective_num_threads,
        stage_cooldown_s=effective_stage_cooldown_s,
    )

    summary_status = "ok" if resolved_colmap_exe.exists() else "missing_colmap_exe"
    summary = {
        "status": summary_status,
        "workspace_dir": str(workspace_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "colmap_exe": str(resolved_colmap_exe),
        "colmap_exists": resolved_colmap_exe.exists(),
        "image_count": len(image_names),
        "copied_image_count": copied_images,
        "removed_stale_workspace_image_count": len(removed_stale_workspace_images),
        "removed_stale_workspace_images": removed_stale_workspace_images[:20],
        "missing_image_count": len(missing_images),
        "missing_images": missing_images[:20],
        "existing_colmap_outputs": existing_colmap_outputs,
        "colmap_reset_recommended": bool(existing_colmap_outputs),
        "removed_stale_image_similarity_outputs": removed_stale_image_similarity,
        "metadata_files": copied_metadata,
        "colmap_mask": mask_summary,
        "camera_model": camera_model,
        "single_camera": single_camera,
        "use_gpu": use_gpu,
        "matcher": matcher,
        "visual_only": visual_only,
        "pdr_alignment_enabled": align_to_pdr,
        "feature_max_image_size": feature_max_image_size,
        "max_num_features": max_num_features,
        "max_num_matches": max_num_matches,
        "load_profile": load_profile,
        "load_profile_config": load_config,
        "colmap_num_threads": effective_num_threads,
        "stage_cooldown_s": effective_stage_cooldown_s,
        "constrained_match_pairs": match_pair_summary,
        "match_graph_edges": "match_graph_edges.csv",
        "image_similarity": image_similarity_summary,
        "image_similarity_edges": "image_similarity_edges.csv",
        "pdr_alignment_ref_count": pdr_alignment_ref_count,
        "quality": quality,
        "dense_max_image_size": dense_max_image_size,
        "dense_profile": dense_profile,
        "dense_profile_config": profile_config,
        "run_meshing": effective_run_meshing,
        "run_point_cloud_filter": run_point_cloud_filter,
        "cleanup_intermediate_ply": bool(profile_config.get("cleanup_intermediate_ply", True)),
        "requested_run_meshing": run_meshing,
        "sparse_model_selection": (
            "run_colmap_dense.bat prefers sparse_pdr_aligned/0 when it exists, otherwise it uses sparse/0."
        ),
        "scripts": [
            "run_colmap_sparse.bat",
            "run_colmap_dense.bat",
            "run_colmap_full.bat",
            "run_colmap_automatic.bat",
            "open_colmap_gui.bat",
            "reset_colmap_outputs.bat",
        ],
    }
    write_json(workspace_dir / "colmap_workspace_summary.json", summary)
    return summary
