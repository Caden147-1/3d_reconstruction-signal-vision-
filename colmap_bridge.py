"""Prepare a COLMAP workspace from the exported reconstruction dataset."""

from __future__ import annotations

import csv
from bisect import bisect_left
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from image_similarity import analyze_workspace_image_similarity
from workspace_lock import locked_workspace
from workspace_inputs import require_fresh_workspace


METADATA_FILES = [
    "images_manifest.csv",
    "camera_pose_seed.csv",
    "wifi_fingerprint_library.csv",
    "wifi_pdr_features.csv",
    "wifi_boundary_seed.csv",
    "wifi_localization_model.json",
    "wifi_localization_eval.csv",
    "wifi_positioning_summary.json",
    "wifi_signal_boundary.csv",
    "wifi_signal_grid.csv",
    "video_frames_manifest.csv",
    "video_frames_summary.json",
    "video_sensor_sync_summary.json",
    "video_sensor_sync.csv",
    "video_sensor_fuzzy_context.json",
    "video_sensor_fuzzy_summary.json",
    "pdr_trajectory_contour.csv",
    "pdr_trajectory_edges.csv",
    "pdr_trajectory_contour_summary.json",
    "export_summary.json",
    "unassigned_image_audit.csv",
]


DENSE_PROFILES: Dict[str, Dict[str, Any]] = {
    "efficient": {
        "description": "Bounded video MVS with moderate sampling and original geometric validation thresholds.",
        "patch_match": {
            "geom_consistency": 1, "window_radius": 7, "window_step": 1,
            "num_samples": 15, "num_iterations": 5, "filter": 1,
            "filter_min_ncc": 0.05, "cache_size": 8,
        },
        "fusion": {
            "min_num_pixels": 3, "max_reproj_error": 2.5,
            "max_depth_error": 0.02, "max_normal_error": 15,
            "check_num_images": 60,
        },
        "run_meshing": True,
        "cleanup_intermediate_ply": True,
        "point_filter_max_points": 0,
    },
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
        "point_filter_max_points": 0,
    },
}


P2_CONTOUR_REASONS = {"video_imu_route"}


P1_VISUAL_SUPPORT_DECISIONS = {
    "geometry_supported",
    "feature_overlap",
    "near_duplicate",
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
        return (colmap_exe if colmap_exe.is_absolute() else root / colmap_exe).resolve()
    if colmap_dir is not None:
        resolved_dir = colmap_dir if colmap_dir.is_absolute() else root / colmap_dir
        candidates.extend([resolved_dir / "bin" / "colmap.exe", resolved_dir / "COLMAP.bat"] if os.name == 'nt'
                          else [resolved_dir / 'bin' / 'colmap', resolved_dir / 'colmap'])
    if os.name != 'nt':
        candidates.extend([root / 'colmap' / 'bin' / 'colmap', root / 'colmap'])
    if os.name == 'nt':
        candidates.extend([
            root / "colmap-x64-windows-cuda" / "bin" / "colmap.exe",
            root / "colmap-x64-windows-cuda" / "COLMAP.bat",
            root / "colmap" / "bin" / "colmap.exe",
            root / "colmap.exe",
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    found = shutil.which("colmap")
    if found:
        return Path(found).resolve()
    return candidates[0].resolve() if candidates else (root / ("colmap.exe" if os.name == 'nt' else "colmap")).resolve()


def workspace_image_group(row: Dict[str, str]) -> str:
    """Only extracted video frames may enter a new reconstruction."""
    if row.get("source_type") != "video_base_frame" or not row.get("video_name"):
        raise ValueError("Reconstruction inputs must be extracted video frames with video_name metadata")
    return "video"


def workspace_image_name(row: Dict[str, str], *, group: Optional[str] = None) -> str:
    workspace_image_group(row)
    source_name = Path(row.get("image_file", ""))
    if not source_name.name or source_name.is_absolute() or ".." in source_name.parts:
        raise ValueError(f"Unsafe or missing reconstruction image name: {source_name}")
    return (Path(group or workspace_image_group(row)) / source_name.name).as_posix()


def build_workspace_manifest(manifest_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Map exported filenames to source-specific workspace folders.

    Each video clip/resolution gets its own camera when multiple video groups
    exist. Keep the original video/ layout for a single video group, including
    older exports without decoded dimensions. Existing workspace manifests and
    reconstruction files are not rewritten by this preparation helper.
    """
    rows: List[Dict[str, str]] = []
    seen: set[str] = set()

    def video_key(row: Dict[str, str]) -> Tuple[str, int, int]:
        width, height = safe_int(row.get("image_width")), safe_int(row.get("image_height"))
        if width <= 0 or height <= 0:
            width = height = 0
        return (row.get("video_name", "").replace("\\", "/"), width, height)

    video_keys = {video_key(row) for row in manifest_rows
                  if row.get("image_file") and row.get("source_type") == "video_base_frame"}
    video_groups = {key: "video_" + hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()[:16]
                    for key in video_keys} if len(video_keys) > 1 else {}
    for source_row in manifest_rows:
        source_name = source_row.get("image_file", "")
        if not source_name:
            continue
        row = dict(source_row)
        group = (video_groups.get(video_key(row), "video")
                 if row.get("source_type") == "video_base_frame" else workspace_image_group(row))
        target_name = workspace_image_name(row, group=group)
        if target_name in seen:
            raise ValueError(f"Duplicate workspace image name: {target_name}")
        seen.add(target_name)
        row["source_image_file"] = source_name
        row["image_file"] = target_name
        row["workspace_image_group"] = group
        rows.append(row)
    return rows


def camera_grouping_summary(manifest_rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    groups: Dict[str, int] = {}
    for row in manifest_rows:
        image_name = row.get("image_file", "")
        if not image_name:
            continue
        group = row.get("workspace_image_group") or workspace_image_group(row)
        groups[group] = groups.get(group, 0) + 1
    use_folder_groups = len(groups) > 1
    return {
        "mode": "source_folder_groups" if use_folder_groups else "single_camera",
        "groups": dict(sorted(groups.items())),
        "single_camera": not use_folder_groups,
        "single_camera_per_folder": use_folder_groups,
        "explanation": (
            "Video clips have separate camera groups; "
            "groups also separate known decoded resolutions. Shared intrinsics "
            "within a group assume a common camera/lens setting."
            if use_folder_groups
            else "All current inputs belong to one capture-source group."
        ),
    }


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
        source_image_file = row.get("source_image_file") or image_file
        source = source_images_dir / source_image_file
        target = target_images_dir / image_file
        image_names.append(image_file)
        if not source.exists():
            missing.append(str(source))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_image_names.append(image_file)
        copied += 1

    expected_names = set(copied_image_names)
    removed_stale: List[str] = []
    for target in target_images_dir.rglob("*"):
        if not target.is_file():
            continue
        target_name = target.relative_to(target_images_dir).as_posix()
        if target_name in expected_names:
            continue
        try:
            target.unlink()
        except OSError:
            continue
        removed_stale.append(target_name)
    for directory in sorted(
        (path for path in target_images_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            continue
    return copied, image_names, missing, removed_stale


def copy_metadata(reconstruction_dir: Path, workspace_dir: Path) -> List[str]:
    metadata_dir = workspace_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in METADATA_FILES:
        source = reconstruction_dir / name
        if not source.exists():
            # The current export is authoritative, including when sensor modes
            # change in a workspace that has only been prepared, not reconstructed.
            (metadata_dir / name).unlink(missing_ok=True)
            continue
        shutil.copy2(source, metadata_dir / name)
        copied.append(name)
    return copied


def write_image_list(workspace_dir: Path, image_names: Sequence[str]) -> None:
    with (workspace_dir / "image_list.txt").open("w", encoding="utf-8") as handle:
        for image_name in image_names:
            handle.write(f"{image_name}\n")


def bounded_score(value: Optional[float], scale: float) -> float:
    if value is None or scale <= 0 or math.isnan(value) or math.isinf(value):
        return 1.0
    return 1.0 / (1.0 + max(0.0, value) / scale)


def row_video_sequence(row: Dict[str, str]) -> int:
    return safe_int(row.get("video_sequence_index"), -1)


def pair_video_sequence_gap(row_a: Dict[str, str], row_b: Dict[str, str]) -> Optional[int]:
    sequence_a = row_video_sequence(row_a)
    sequence_b = row_video_sequence(row_b)
    if sequence_a < 0 or sequence_b < 0:
        return None
    return abs(sequence_a - sequence_b)


def is_video_frame_row(row: Dict[str, str]) -> bool:
    return row.get("source_type") == "video_base_frame" or bool(row.get("video_sequence_index"))


VIDEO_MATCH_TIME_OFFSETS_S = (1.0, 2.0, 4.0, 8.0, 16.0)


def video_sequence_pair_indices(
    sequence_rows: Sequence[Dict[str, str]], frame_window: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """Preserve local continuity and add bounded, wider-baseline candidates.

    Timestamps refer to the decoded video, not a sensor clock. Sampling more
    densely must not shrink every matching baseline to the next few frames.
    These are candidates only; COLMAP still performs SIFT geometric verification.
    """
    window = max(1, frame_window)
    count = len(sequence_rows)
    local_pairs = {
        (index, other)
        for index in range(count)
        for other in range(index + 1, min(count, index + window + 1))
    }
    pairs = set(local_pairs)
    times = [safe_float(row.get("video_time_s")) for row in sequence_rows]
    usable_times = (
        bool(times) and all(math.isfinite(value) and value >= 0 for value in times)
        and all(left < right for left, right in zip(times, times[1:]))
    )
    fallback_offsets = [window * multiplier for multiplier in (2, 4, 8)]
    for index in range(count):
        if usable_times:
            for offset in VIDEO_MATCH_TIME_OFFSETS_S:
                target = times[index] + offset
                insertion = bisect_left(times, target, lo=index + 1)
                neighbors = [
                    other for other in (insertion - 1, insertion)
                    if index < other < count
                ]
                if not neighbors:
                    continue
                closest = min(neighbors, key=lambda other: abs(times[other] - target))
                # Do not call a distant frame across a capture gap a 1s neighbor.
                if abs(times[closest] - target) <= max(0.1, offset * 0.35):
                    pairs.add((index, closest))
        else:
            for offset in fallback_offsets:
                if index + offset < count:
                    pairs.add((index, index + offset))
    return sorted(pairs), {
        "mode": "decoded_video_time" if usable_times else "frame_rank_fallback",
        "frame_count": count,
        "local_pair_count": len(local_pairs),
        "additional_multiscale_pair_count": len(pairs) - len(local_pairs),
        "pair_count": len(pairs),
        "time_offsets_s": list(VIDEO_MATCH_TIME_OFFSETS_S) if usable_times else [],
        "fallback_rank_offsets": [] if usable_times else fallback_offsets,
        "requires_colmap_geometric_verification": True,
    }


def build_constrained_match_pairs(
    manifest_rows: Sequence[Dict[str, str]],
    video_frame_window: int,
    visual_only: bool = False,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
    """Propose local and multiscale video neighbors; sensors are added separately."""
    video_frame_window = max(1, video_frame_window)
    by_video: Dict[str, List[Dict[str, str]]] = {}
    for row in manifest_rows:
        workspace_image_group(row)
        if row.get("image_file"):
            by_video.setdefault(row["video_name"], []).append(row)
    pairs, edges, clips = set(), [], []
    for video_name, rows in sorted(by_video.items()):
        rows.sort(key=lambda row: (safe_int(row.get("video_sequence_index")),
                                  safe_int(row.get("video_frame_index")), row["image_file"]))
        indices, timing = video_sequence_pair_indices(rows, video_frame_window)
        clips.append({"video_name": video_name, **timing})
        for a, b in indices:
            left, right = rows[a], rows[b]
            pair = tuple(sorted((left["image_file"], right["image_file"])))
            if pair[0] == pair[1] or pair in pairs:
                continue
            pairs.add(pair)
            gap = pair_video_sequence_gap(left, right)
            score = bounded_score(float(gap) if gap is not None else None, float(video_frame_window))
            edges.append({"image_a": pair[0], "image_b": pair[1], "reason": "video_sequence",
                          "edge_weight": round(max(.01, .70 * .82 * (.60 + .40 * score)), 6),
                          "video_score": score, "video_sequence_gap": gap,
                          "source_type_a": "video_base_frame", "source_type_b": "video_base_frame",
                          "video_sequence_index_a": left.get("video_sequence_index", ""),
                          "video_sequence_index_b": right.get("video_sequence_index", ""),
                          "edge_weight_basis": "video_temporal_neighbors_require_colmap_geometry"})
    weights = [row["edge_weight"] for row in edges]
    summary = {"image_count": len(manifest_rows), "video_frame_count": len(manifest_rows),
               "pair_count": len(pairs), "visual_only": visual_only,
               "candidate_basis": "video_order_and_appearance_retrieval",
               "camera_orientation_from_phone": False, "video_frame_window": video_frame_window,
               "reason_counts": {"video_sequence": len(pairs)}, "weighted_graph": True,
               "video_sequence_pairing": {"policy": "local_neighbors_plus_bounded_multiscale_video_time",
                   "clips": clips, "pair_budget_per_frame": video_frame_window + len(VIDEO_MATCH_TIME_OFFSETS_S),
                   "requires_colmap_geometric_verification": True},
               "edge_weight": {"min": min(weights) if weights else None,
                               "max": max(weights) if weights else None,
                               "mean": sum(weights) / len(weights) if weights else None}}
    return sorted(pairs), sorted(edges, key=lambda row: (row["image_a"], row["image_b"])), summary


def write_match_pairs(workspace_dir: Path, pairs: Sequence[Tuple[str, str]]) -> None:
    with (workspace_dir / "match_pairs.txt").open("w", encoding="utf-8") as handle:
        for image_a, image_b in pairs:
            handle.write(f"{image_a} {image_b}\n")


def write_named_match_pairs(workspace_dir: Path, filename: str, pairs: Sequence[Tuple[str, str]]) -> None:
    with (workspace_dir / filename).open("w", encoding="utf-8") as handle:
        for image_a, image_b in pairs:
            handle.write(f"{image_a} {image_b}\n")


def select_visually_supported_match_pairs(
    pairs: Sequence[Tuple[str, str]],
    edge_rows: Sequence[Dict[str, Any]],
    similarity_rows: Sequence[Dict[str, str]],
    similarity_summary: Dict[str, Any],
) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
    """Apply P1 decisions to the P2-constrained COLMAP pair list.

    COLMAP's ``matches_importer`` accepts a pair list, not per-pair prior
    weights. P2 therefore supplies the route-topology candidate graph and P1
    turns evaluated visual evidence into a keep/drop decision. Unevaluated route
    pairs remain candidates; visual recovery always requires geometric evidence.
    Temporal video candidates are passed to COLMAP even when the smaller ORB
    preview finds weak support; only COLMAP's SIFT geometry can accept them.
    """
    edge_by_pair = {
        tuple(sorted((str(row.get("image_a", "")), str(row.get("image_b", ""))))): row
        for row in edge_rows
        if row.get("image_a") and row.get("image_b")
    }
    similarity_by_pair = {
        tuple(sorted((str(row.get("image_a", "")), str(row.get("image_b", ""))))): row
        for row in similarity_rows
        if row.get("image_a") and row.get("image_b")
    }
    p1_status = str(similarity_summary.get("status", "missing"))
    p1_available = p1_status == "ok"
    selected: List[Tuple[str, str]] = []
    selection_rows: List[Dict[str, Any]] = []
    selection_reason_counts: Dict[str, int] = {}
    rejected_decision_counts: Dict[str, int] = {}
    unreviewed_count = 0

    for image_a, image_b in pairs:
        key = tuple(sorted((image_a, image_b)))
        edge = edge_by_pair.get(key, {})
        similarity = similarity_by_pair.get(key)
        decision = "not_evaluated" if similarity is None else str(similarity.get("visual_decision", "not_evaluated"))
        pair_status = "not_evaluated" if similarity is None else str(similarity.get("pair_status", "not_evaluated"))
        selected_pair = True
        selection_reason = "p1_unavailable_kept"
        if edge.get("reason") in {"visual_recovery", "wifi_fingerprint_recovery", "video_imu_route", "video_wifi_revisit"}:
            selected_pair = p1_available and pair_status == "ok" and decision == "geometry_supported"
            selection_reason = "p1_recovery_geometry_supported" if selected_pair else "p1_recovery_unverified_rejected"
            if not selected_pair:
                rejected_decision_counts[decision] = rejected_decision_counts.get(decision, 0) + 1
        elif edge.get("reason") == "video_sequence":
            selection_reason = "video_sequence_pending_colmap_geometry"
        elif p1_available:
            if similarity is None:
                selection_reason = "p1_not_evaluated_kept"
                unreviewed_count += 1
            elif pair_status != "ok":
                selection_reason = "p1_unreadable_kept"
            elif decision in P1_VISUAL_SUPPORT_DECISIONS:
                selection_reason = "p1_visual_supported"
            else:
                selected_pair = False
                selection_reason = f"p1_rejected_{decision}"
                rejected_decision_counts[decision] = rejected_decision_counts.get(decision, 0) + 1
        selection_reason_counts[selection_reason] = selection_reason_counts.get(selection_reason, 0) + 1
        if selected_pair:
            selected.append((image_a, image_b))
        selection_rows.append(
            {
                "image_a": image_a,
                "image_b": image_b,
                "reason": edge.get("reason", ""),
                "edge_weight": edge.get("edge_weight", ""),
                "sensor_association_mode": edge.get("sensor_association_mode", ""),
                "sensor_time_uncertainty_s": edge.get("sensor_time_uncertainty_s", ""),
                "visual_decision": decision,
                "pair_status": pair_status,
                "visual_overlap_score": "" if similarity is None else similarity.get("visual_overlap_score", ""),
                "selected": int(selected_pair),
                "selection_reason": selection_reason,
            }
        )

    summary = {
        "status": "empty_selection" if pairs and not selected else "applied" if p1_available else "not_applied",
        "p1_status": p1_status,
        "input_pair_count": len(pairs),
        "selected_pair_count": len(selected),
        "rejected_pair_count": len(pairs) - len(selected),
        "unreviewed_pair_count": unreviewed_count,
        "fallback_used": False,
        "selected_p2_contour_pair_count": sum(row["selected"] for row in selection_rows if row["reason"] in P2_CONTOUR_REASONS),
        "selected_visual_recovery_pair_count": sum(row["selected"] for row in selection_rows if row["reason"] == "visual_recovery"),
        "selected_wifi_recovery_pair_count": sum(row["selected"] for row in selection_rows if row["reason"] == "wifi_fingerprint_recovery"),
        "selected_video_sequence_pair_count": sum(row["selected"] for row in selection_rows if row["reason"] == "video_sequence"),
        "selected_video_sensor_pair_count": sum(row["selected"] for row in selection_rows if row["reason"] in {"video_imu_route", "video_wifi_revisit"}),
        "selected_fuzzy_sensor_pair_count": sum(row["selected"] for row in selection_rows if row.get("sensor_association_mode") == "fuzzy_time_window"),
        "video_sequence_kept_despite_p1_rejection_count": sum(
            row["reason"] == "video_sequence" and p1_available
            and row["pair_status"] == "ok" and row["visual_decision"] not in P1_VISUAL_SUPPORT_DECISIONS
            for row in selection_rows
        ),
        "selection_reason_counts": dict(sorted(selection_reason_counts.items())),
        "rejected_visual_decision_counts": dict(sorted(rejected_decision_counts.items())),
        "explanation": (
            "P2 determines the route-contour candidate graph. P1 keeps evaluated "
            "route pairs with visual support and rejects evaluated weak/no-overlap route pairs; "
            "unevaluated route pairs remain candidates. Temporal video pairs are always "
            "candidates pending COLMAP SIFT geometric verification, not verified matches. "
            "Recovery pairs require preview geometry, "
            "and an all-rejected graph stays empty instead of restoring rejected pairs."
        ),
    }
    return selected, selection_rows, summary


def write_match_pair_filter(workspace_dir: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    write_csv_rows(
        workspace_dir / "match_pair_filter.csv",
        [
            "image_a",
            "image_b",
            "reason",
            "edge_weight",
            "visual_decision",
            "sensor_association_mode",
            "sensor_time_uncertainty_s",
            "pair_status",
            "visual_overlap_score",
            "selected",
            "selection_reason",
        ],
        rows,
    )
    write_json(workspace_dir / "match_pair_filter_summary.json", summary)


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
            "point_score",
            "video_score",
            "point_gap",
            "video_sequence_gap",
            "edge_weight_basis",
            "sensor_association_mode",
            "sensor_time_uncertainty_s",
            "sensor_time_uncertainty_a_s",
            "sensor_time_uncertainty_b_s",
            "sensor_window_a_start_ms",
            "sensor_window_a_end_ms",
            "sensor_window_b_start_ms",
            "sensor_window_b_end_ms",
            "video_sensor_fuzzy_source_sha256",
            "temporal_uncertainty_is_assumed",
            "trajectory_distance_min_m",
            "trajectory_distance_max_m",
            "wifi_scan_a_elapsed_ts",
            "wifi_scan_b_elapsed_ts",
            "wifi_rssi_rms_db",
            "wifi_retrieval_score",
            "wifi_fingerprint_score",
            "wifi_sequence_multiplier",
            "wifi_sequence_status",
            "wifi_sequence_direction",
            "wifi_sequence_neighbor_pairs",
            "wifi_shared_ap_count",
            "wifi_ap_overlap",
            "wifi_pair_budget",
            "source_type_a",
            "source_type_b",
            "global_point_index_a",
            "global_point_index_b",
            "video_sequence_index_a",
            "video_sequence_index_b",
            "pdr_distance_m",
            "trajectory_contour_id_a",
            "trajectory_contour_id_b",
            "trajectory_distance_m",
        ],
        edge_rows,
    )


def remove_pdr_alignment_refs(workspace_dir: Path) -> None:
    refs_path = workspace_dir / "pdr_alignment_refs.txt"
    if refs_path.exists():
        refs_path.unlink()


def batch_lock_guard(mode: str) -> str:
    return f'''
python "%ROOT%\\workspace_lock.py" --workspace "%WORKSPACE%" --check-owned
if "%errorlevel%"=="0" goto :workspace_locked
python "%ROOT%\\run_colmap.py" {mode} --workspace "%WORKSPACE%"
exit /b %errorlevel%
:workspace_locked
'''


def batch_header(root: Path, workspace_dir: Path, colmap_exe: Path, mode: str = "gui") -> str:
    return f"""@echo off
rem portable-stage-v1
setlocal
set "ROOT={windows_path(root)}"
set "WORKSPACE={windows_path(workspace_dir)}"
set "COLMAP_EXE={windows_path(colmap_exe)}"
set "IMAGE_PATH=%WORKSPACE%\\images"
set "IMAGE_LIST=%WORKSPACE%\\image_list.txt"
set "DATABASE=%WORKSPACE%\\database.db"
{batch_lock_guard(mode)}
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


def thread_option(prefix: str, num_threads: int) -> str:
    return f" ^\n  --{prefix}.num_threads {num_threads}" if num_threads > 0 else ""


def prefixed_thread_option(prefix: str, num_threads: int) -> List[str]:
    return [f"  --{prefix}.num_threads {num_threads}"] if num_threads > 0 else []


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
    for name in (
        "image_similarity_edges.csv",
        "image_similarity_summary.json",
        "match_pair_filter.csv",
        "match_pair_filter_summary.json",
    ):
        path = workspace_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def resolve_optional_path(root: Path, path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else root / path


def mask_candidates(mask_dir: Path, image_name: str) -> List[Path]:
    return [mask_dir / (image_name.replace("\\", "/") + ".png")]


def mask_dir_summary(mask_dir: Optional[Path], image_names: Sequence[str]) -> Dict[str, Any]:
    if mask_dir is None:
        return {"enabled": False, "path": "", "exists": False, "matched_image_count": 0, "missing_image_count": 0}
    matched = 0
    missing: List[str] = []
    if mask_dir.is_dir():
        for image_name in image_names:
            if any(candidate.is_file() for candidate in mask_candidates(mask_dir, image_name)):
                matched += 1
            elif len(missing) < 20:
                missing.append(image_name)
    return {
        "enabled": mask_dir.is_dir(),
        "path": str(mask_dir),
        "exists": mask_dir.is_dir(),
        "matched_image_count": matched,
        "missing_image_count": max(0, len(image_names) - matched),
        "missing_images": missing,
        "accepted_name_patterns": [
            "same relative image path including original extension, followed by .png",
        ],
    }


def write_sparse_batch(
    root: Path,
    workspace_dir: Path,
    colmap_exe: Path,
    mask_dir: Optional[Path],
    camera_model: str,
    single_camera: bool,
    single_camera_per_folder: bool,
    use_gpu: bool,
    matcher: str,
    align_to_pdr: bool,
    feature_max_image_size: int,
    max_num_features: int,
    max_num_matches: int,
    num_threads: int,
    ba_gpu: str = 'auto',
    sparse_strategy: str = 'all',
    sparse_ba_growth_ratio: float = 1.1,
    sparse_keyframe_gap_s: float = 1.0,
) -> None:
    if ba_gpu not in {'auto', 'on', 'off'}:
        raise ValueError('Unknown GPU BA policy')
    import math
    if sparse_strategy not in {'all', 'backbone'}:
        raise ValueError('Unknown sparse frame strategy')
    if not math.isfinite(sparse_ba_growth_ratio) or sparse_ba_growth_ratio <= 1:
        raise ValueError('Sparse BA growth ratio must be finite and greater than 1')
    if not math.isfinite(sparse_keyframe_gap_s) or sparse_keyframe_gap_s <= 0:
        raise ValueError('Sparse keyframe gap must be finite and positive')
    matcher_command = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    single_camera_flag = "1" if single_camera else "0"
    single_camera_per_folder_flag = "1" if single_camera_per_folder else "0"
    gpu_flag = "1" if use_gpu else "0"
    feature_extraction_threads = thread_option("FeatureExtraction", num_threads)
    feature_matching_threads = thread_option("FeatureMatching", num_threads)
    constrained_matching_block = f"""echo [2/4] Matching selected route pairs and geometry-supported visual recovery pairs...
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
if not "%errorlevel%"=="0" exit /b %errorlevel%
"""
    if matcher == "constrained":
        matching_block = constrained_matching_block
    elif matcher == "hybrid":
        matching_block = constrained_matching_block + f"""
echo [2b/4] Adding opt-in exhaustive visual bridge matches outside the P1/P2 pair gate...
"%COLMAP_EXE%" exhaustive_matcher ^
  --database_path "%DATABASE%" ^
  --FeatureMatching.use_gpu {gpu_flag} ^
  --FeatureMatching.guided_matching 1 ^
  --FeatureMatching.max_num_matches {max_num_matches}{feature_matching_threads}
if not "%errorlevel%"=="0" exit /b %errorlevel%
"""
    else:
        matching_block = f"""echo [2/4] Matching features with {matcher_command}...
"%COLMAP_EXE%" {matcher_command} ^
  --database_path "%DATABASE%" ^
  --FeatureMatching.use_gpu {gpu_flag} ^
  --FeatureMatching.guided_matching 1 ^
  --FeatureMatching.max_num_matches {max_num_matches}{feature_matching_threads}
if not "%errorlevel%"=="0" exit /b %errorlevel%
"""
    feature_limits = ""
    if feature_max_image_size > 0:
        feature_limits += f" ^\n  --SiftExtraction.max_image_size {feature_max_image_size}"
    if max_num_features > 0:
        feature_limits += f" ^\n  --SiftExtraction.max_num_features {max_num_features}"
    mask_option = ""
    if mask_dir is not None:
        mask_option = f' ^\n  --ImageReader.mask_path "{windows_path(mask_dir)}"'
    alignment_disabled = "" if align_to_pdr else " --disabled"
    alignment_block = f"""
echo [4/4] Recording visual sparse model selection...
python "%ROOT%\\pdr_alignment.py" --workspace "%WORKSPACE%" --colmap-exe "%COLMAP_EXE%" --capture-mode off{alignment_disabled}
if not "%errorlevel%"=="0" exit /b %errorlevel%
"""
    content = batch_header(root, workspace_dir, colmap_exe, "sparse") + f"""
echo [preflight] Checking that sparse reconstruction starts from the current input set...
python "%ROOT%\\colmap_workspace_check.py" --workspace "%WORKSPACE%" --require-clean-database
if not "%errorlevel%"=="0" exit /b %errorlevel%
python "%ROOT%\\colmap_workspace_check.py" --workspace "%WORKSPACE%" --record-sparse-start
if not "%errorlevel%"=="0" exit /b %errorlevel%

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
  --ImageReader.single_camera_per_folder {single_camera_per_folder_flag} ^
  --FeatureExtraction.use_gpu {gpu_flag}{mask_option}{feature_limits}{feature_extraction_threads}
if not "%errorlevel%"=="0" exit /b %errorlevel%

{matching_block}

echo [3/4] Building sparse model...
if not defined FYP_COLMAP_RUN_ID set "FYP_COLMAP_RUN_ID=manual_%RANDOM%_%RANDOM%"
set "SPARSE_SNAPSHOTS=%WORKSPACE%\\sparse_snapshots\\%FYP_COLMAP_RUN_ID%"
mkdir "%SPARSE_SNAPSHOTS%"
if not "%errorlevel%"=="0" exit /b %errorlevel%
echo [snapshot] Recovery candidates: %SPARSE_SNAPSHOTS%
python "%ROOT%\\sparse_optimization.py" ^
  --workspace "%WORKSPACE%" ^
  --colmap-exe "%COLMAP_EXE%" ^
  --snapshots "%SPARSE_SNAPSHOTS%" ^
  --num-threads {num_threads} ^
  --ba-gpu {ba_gpu} ^
  --strategy {sparse_strategy} ^
  --ba-growth-ratio {sparse_ba_growth_ratio} ^
  --keyframe-gap-s {sparse_keyframe_gap_s}
if not "%errorlevel%"=="0" exit /b %errorlevel%
python "%ROOT%\\sparse_recovery.py" ^
  --workspace "%WORKSPACE%" ^
  --colmap-exe "%COLMAP_EXE%" ^
  --num-threads {num_threads}
if not "%errorlevel%"=="0" exit /b %errorlevel%
echo [sparse-log] mapper complete run=%FYP_COLMAP_RUN_ID%

{alignment_block}
python "%ROOT%\\colmap_workspace_check.py" --workspace "%WORKSPACE%" --record-sparse-complete
if not "%errorlevel%"=="0" exit /b %errorlevel%
echo [report] Generating reconstruction quality report...
python "%ROOT%\\reconstruction_quality.py" --workspace "%WORKSPACE%" --summary "%WORKSPACE%\\quality_report.json"
if not "%errorlevel%"=="0" echo [warn] Sparse quality report failed; reconstruction outputs are still available.

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
    scene_completion: str = "off",
    videoscene_max_clips: int = 3,
    videoscene_budget_hours: float = 4.,
    dense_selection_enabled: bool = False,
    point_filter_method: str = "observed",
    point_filter_max_candidates: int = 2000000,
    point_filter_budget_seconds: float = 120.,
) -> None:
    if point_filter_method not in {"observed", "wifi-density", "connected-density"}:
        raise ValueError("Unknown point-cloud filter method")
    if type(point_filter_max_candidates) is not int or not 1 <= point_filter_max_candidates <= 2000000 or not math.isfinite(point_filter_budget_seconds) or point_filter_budget_seconds <= 0:
        raise ValueError("Invalid point-cloud observation budget")
    if scene_completion not in {"off", "videoscene"}:
        raise ValueError("Unknown scene completion method")
    if scene_completion == "videoscene" and not run_point_cloud_filter:
        raise ValueError("VideoScene completion requires point-cloud filtering")
    if not 1 <= videoscene_max_clips <= 8 or not math.isfinite(videoscene_budget_hours) or not 0 < videoscene_budget_hours <= 20:
        raise ValueError("Invalid VideoScene completion budget")
    profile = dense_profile_config(dense_profile)
    depth_planner = "dense_keyframes.py" if dense_selection_enabled else "mvs_observations.py"
    point_filter_max_points = 0 if point_filter_method == "observed" else int(profile.get("point_filter_max_points", 600000))
    cleanup_intermediate_ply = bool(profile.get("cleanup_intermediate_ply", True))
    patch_match_args = [
        '  --workspace_path "%WORKSPACE%\\dense"',
        "  --workspace_format COLMAP",
        # COLMAP 3.13 may retain native-size zero-distortion cameras even
        # after resizing their images. Apply the same cap to MVS calibration.
        *prefixed_options("PatchMatchStereo", {**profile["patch_match"], "max_image_size": dense_max_image_size}),
    ]
    fusion_args = [
        '  --workspace_path "%WORKSPACE%\\dense"',
        "  --workspace_format COLMAP",
        "  --input_type geometric",
        *prefixed_options("StereoFusion", {**profile["fusion"], "max_image_size": dense_max_image_size}),
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
    sparse_selector = """set "SPARSE_MODEL="
for /f "usebackq delims=" %%I in (`python "%ROOT%\\colmap_workspace_check.py" --workspace "%WORKSPACE%" --print-dense-input`) do set "SPARSE_MODEL=%%I"
if not defined SPARSE_MODEL exit /b 1
echo [info] Dense input sparse model: %SPARSE_MODEL%
"""
    point_filter_block = ""
    filter_preflight = ""
    # Keep the receipt until filtering replaces it: exact-source observation
    # layers may still be useful when the dense cache is reused. The filter
    # checks the fusion identity and reconstruction binding before reuse.
    filter_summary_cleanup = (
        '' if run_point_cloud_filter else
        'if exist "%WORKSPACE%\\dense\\point_filter_summary.json" del "%WORKSPACE%\\dense\\point_filter_summary.json"\n'
    )
    if run_point_cloud_filter:
        filter_preflight = """python -c "import numpy"
if not "%errorlevel%"=="0" (
  echo [error] Point cloud filtering requires NumPy in this Python environment.
  exit /b 1
)
"""
        point_filter_block = f"""
echo [4/8] Filtering dense point cloud outliers...
python "%ROOT%\\point_cloud_filter.py" ^
  --input "%WORKSPACE%\\dense\\fused.ply" ^
  --output "%WORKSPACE%\\dense\\fused_filtered.ply" ^
  --summary "%WORKSPACE%\\dense\\point_filter_summary.json" ^
  --method {point_filter_method} ^
  --workspace "%WORKSPACE%" ^
  --observation-max-candidates {point_filter_max_candidates} ^
  --observation-budget-seconds {point_filter_budget_seconds} ^
  --max-points {point_filter_max_points}
if not "%errorlevel%"=="0" (
  echo [error] Point cloud filtering failed; stopping before final promotion. Raw fused.ply is still available.
  exit /b 1
) else (
  set "DENSE_POINT_CLOUD=%WORKSPACE%\\dense\\fused_filtered.ply"
)
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
if not "%errorlevel%"=="0" echo [warn] Delaunay meshing failed; continuing with fused point cloud.

echo [6/8] Building a Poisson mesh for smoother surface inspection...
{poisson_command}
if not "%errorlevel%"=="0" echo [warn] Poisson meshing failed; continuing with fused point cloud.
"""
    quality_block = """
echo [report] Generating reconstruction quality report...
python "%ROOT%\\reconstruction_quality.py" --workspace "%WORKSPACE%" --summary "%WORKSPACE%\\quality_report.json"
if not "%errorlevel%"=="0" echo [warn] Final quality report failed; reconstruction outputs are still available.
"""
    if align_to_pdr:
        quality_block = """
python "%ROOT%\\wifi_boundary.py" --workspace "%WORKSPACE%"
if not "%errorlevel%"=="0" echo [warn] WiFi boundary hypotheses unavailable; observed reconstruction is retained.
""" + quality_block
    keep_intermediate_default = "--keep-intermediate" if mesh_block or not cleanup_intermediate_ply else ""
    # This runner has just produced a new fusion. Without filtering there is no
    # filtered source to distinguish it from raw fusion retained by an older run.
    raw_replacement = " --allow-raw-replacement" if not run_point_cloud_filter else ""
    finalize_block = f"""
set "FINALIZE_KEEP={keep_intermediate_default}"
if "%KEEP_INTERMEDIATE_PLY%"=="1" set "FINALIZE_KEEP=--keep-intermediate"
echo [finalize] Validating and selecting dense\\final.ply; applying the intermediate retention setting...
python "%ROOT%\\finalize_dense_outputs.py" ^
  --dense-dir "%WORKSPACE%\\dense" ^
  --summary "%WORKSPACE%\\dense\\final_output_summary.json" %FINALIZE_KEEP%{raw_replacement}
if not "%errorlevel%"=="0" exit /b %errorlevel%
"""
    pdr_note = "Visual sparse models are retained; sensor timing windows do not establish a metric camera transform."
    completion_block = ""
    if scene_completion == "videoscene":
        completion_block = f"""
echo [7/8] Completing supported local holes with VideoScene...
python "%ROOT%\\videoscene_completion.py" --workspace "%WORKSPACE%" --max-clips {videoscene_max_clips} --budget-hours {videoscene_budget_hours} --skip-quality-report
if not "%errorlevel%"=="0" exit /b %errorlevel%
"""
    content = batch_header(root, workspace_dir, colmap_exe, "dense") + f"""
{filter_preflight}{sparse_selector}if not exist "%SPARSE_MODEL%" (
  echo [error] Sparse model not found: %SPARSE_MODEL%
  echo Run run_colmap_sparse.bat first, or check which sparse model folder COLMAP created.
  exit /b 1
)
echo [preflight] Verifying that database.db covers the current image list...
python "%ROOT%\\colmap_workspace_check.py" --workspace "%WORKSPACE%" --require-database-coverage --require-sparse-model
if not "%errorlevel%"=="0" exit /b %errorlevel%
if not exist "%WORKSPACE%\\dense" mkdir "%WORKSPACE%\\dense"
python "%ROOT%\\dense_cache.py" prepare --workspace "%WORKSPACE%"
if not "%errorlevel%"=="0" exit /b %errorlevel%
set "DENSE_POINT_CLOUD=%WORKSPACE%\\dense\\fused.ply"
if exist "%WORKSPACE%\\dense\\meshed-poisson.ply" del "%WORKSPACE%\\dense\\meshed-poisson.ply"
if exist "%WORKSPACE%\\dense\\meshed-delaunay.ply" del "%WORKSPACE%\\dense\\meshed-delaunay.ply"
if exist "%WORKSPACE%\\dense\\fused_filtered.ply" del "%WORKSPACE%\\dense\\fused_filtered.ply"
{filter_summary_cleanup.rstrip()}
if exist "%WORKSPACE%\\dense\\fused.ply.vis" del "%WORKSPACE%\\dense\\fused.ply.vis"

python "%ROOT%\\dense_cache.py" undistorted --workspace "%WORKSPACE%"
if not "%errorlevel%"=="0" if not "%errorlevel%"=="1" exit /b %errorlevel%
if "%errorlevel%"=="0" goto :undistortion_ready
echo [1/8] Undistorting images...
{undistorter_command}
if not "%errorlevel%"=="0" exit /b %errorlevel%
python "%ROOT%\\{depth_planner}" --workspace "%WORKSPACE%" --apply
if not "%errorlevel%"=="0" exit /b %errorlevel%
python "%ROOT%\\dense_cache.py" checkpoint --workspace "%WORKSPACE%"
if not "%errorlevel%"=="0" exit /b %errorlevel%
:undistortion_ready

python "%ROOT%\\dense_cache.py" matched --workspace "%WORKSPACE%"
if not "%errorlevel%"=="0" if not "%errorlevel%"=="1" exit /b %errorlevel%
if "%errorlevel%"=="0" goto :patchmatch_ready
echo [2/8] Running CUDA PatchMatch stereo...
{patch_match_command}
if not "%errorlevel%"=="0" exit /b %errorlevel%
:patchmatch_ready
python "%ROOT%\\dense_cache.py" complete --workspace "%WORKSPACE%"
if not "%errorlevel%"=="0" exit /b %errorlevel%

echo [3/8] Fusing dense point cloud...
{fusion_command}
if not "%errorlevel%"=="0" exit /b %errorlevel%
python "%ROOT%\\reconstruction_stages.py" record-fusion --workspace "%WORKSPACE%"
if not "%errorlevel%"=="0" exit /b %errorlevel%

{point_filter_block}{mesh_block}{finalize_block}{completion_block}{quality_block}
if exist "%WORKSPACE%\\dense\\final.ply" echo [done] Final dense model: %WORKSPACE%\\dense\\final.ply
"""
    write_text(workspace_dir / "run_colmap_dense.bat", content)
    compatibility = f'''@echo off
setlocal
echo [info] Resuming the verified configured reconstruction pipeline.
python "{windows_path(root / 'run_colmap.py')}" resume --workspace "{windows_path(workspace_dir)}" %*
exit /b %errorlevel%
'''
    write_text(workspace_dir / "run_colmap_dense_high.bat", compatibility)


def write_full_batch(workspace_dir: Path, root: Optional[Path] = None) -> None:
    root = root or Path(__file__).resolve().parent
    content = f"""@echo off
setlocal
set "ROOT={windows_path(root)}"
set "WORKSPACE={windows_path(workspace_dir)}"
{batch_lock_guard("full")}
call "%~dp0run_colmap_sparse.bat"
if not "%errorlevel%"=="0" exit /b %errorlevel%
call "%~dp0run_colmap_dense.bat"
if not "%errorlevel%"=="0" exit /b %errorlevel%
"""
    write_text(workspace_dir / "run_colmap_full.bat", content)


def write_automatic_batch(
    root: Path,
    workspace_dir: Path,
    colmap_exe: Path,
    camera_model: str,
    single_camera: bool,
    single_camera_per_folder: bool,
    use_gpu: bool,
    quality: str,
) -> None:
    single_camera_flag = "1" if single_camera else "0"
    single_camera_per_folder_flag = "1" if single_camera_per_folder else "0"
    gpu_flag = "1" if use_gpu else "0"
    content = batch_header(root, workspace_dir, colmap_exe, "automatic") + f"""
echo [preflight] Checking that automatic reconstruction starts from the current input set...
python "%ROOT%\\colmap_workspace_check.py" --workspace "%WORKSPACE%" --require-clean-database
if not "%errorlevel%"=="0" exit /b %errorlevel%
python "%ROOT%\\colmap_workspace_check.py" --workspace "%WORKSPACE%" --record-sparse-start --runner run_colmap_automatic.bat
if not "%errorlevel%"=="0" exit /b %errorlevel%

echo [info] Running COLMAP automatic_reconstructor...
"%COLMAP_EXE%" automatic_reconstructor ^
  --workspace_path "%WORKSPACE%" ^
  --image_path "%IMAGE_PATH%" ^
  --image_list_path "%IMAGE_LIST%" ^
  --camera_model {camera_model} ^
  --single_camera {single_camera_flag} ^
  --single_camera_per_folder {single_camera_per_folder_flag} ^
  --use_gpu {gpu_flag} ^
  --quality {quality} ^
  --dense 1
if not "%errorlevel%"=="0" exit /b %errorlevel%
echo [sparse-log] mapper complete run=%FYP_COLMAP_RUN_ID%

python "%ROOT%\\colmap_workspace_check.py" --workspace "%WORKSPACE%" --record-sparse-complete
if not "%errorlevel%"=="0" exit /b %errorlevel%
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
set "WORKSPACE={windows_path(workspace_dir)}"
echo This will delete database.db, sparse, PDR-aligned sparse, dense, and automatic output folders under:
echo %WORKSPACE%
set /p CONFIRM=Type YES to reset COLMAP outputs:
if /I not "%CONFIRM%"=="YES" (
  echo cancelled
  exit /b 0
)
python "%ROOT%\\run_colmap.py" reset --workspace "%WORKSPACE%"
exit /b %errorlevel%
"""
    write_text(workspace_dir / "reset_colmap_outputs.bat", content)


@locked_workspace("workspace_dir")
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
    colmap_num_threads: Optional[int] = None,
    colmap_ba_gpu: str = 'auto',
    sparse_strategy: str = 'all',
    sparse_ba_growth_ratio: float = 1.1,
    sparse_keyframe_gap_s: float = 1.0,
    dense_profile: str = "complete",
    run_meshing: bool = False,
    run_point_cloud_filter: bool = True,
    constrained_max_pdr_distance_m: float = 12.0,
    constrained_video_frame_window: int = 6,
    run_image_similarity: bool = True,
    image_similarity_max_pairs: int = 0,
    image_similarity_max_image_size: int = 900,
    image_similarity_orb_features: int = 1500,
    image_similarity_min_visual_score: float = 0.22,
    visual_recovery_neighbors: int = 12,
    wifi_recovery_enabled: bool = True,
    scene_completion: str = "off",
    videoscene_max_clips: int = 3,
    videoscene_budget_hours: float = 4.,
    image_similarity_workers: int = 0,
    dense_max_images: Optional[int] = None,
    dense_max_source_images: int = 12,
    dense_coverage_policy: Optional[Dict[str, Any]] = None,
    point_filter_method: str = "observed",
    point_filter_max_candidates: int = 2000000,
    point_filter_budget_seconds: float = 120.,
) -> Dict[str, Any]:
    if dense_max_images is not None and dense_max_images != 0 and dense_max_images < 3:
        raise ValueError("Dense view budget must be 0 or at least 3")
    if not 2 <= dense_max_source_images <= 64:
        raise ValueError("Dense source view budget must be in [2,64]")
    selection_settings = None
    if dense_max_images is not None:
        selection_settings = {"enabled": True, "max_images": dense_max_images, "max_source_images": dense_max_source_images}
        if dense_coverage_policy is not None:
            selection_settings["coverage_policy"] = dict(dense_coverage_policy)
        from dense_keyframes import selection_config
        selection_config({"dense_selection_config": selection_settings})
    elif dense_coverage_policy is not None:
        raise ValueError("Adaptive dense coverage requires a dense view budget")
    if point_filter_method not in {"observed", "wifi-density", "connected-density"}:
        raise ValueError("Unknown point-cloud filter method")
    if type(point_filter_max_candidates) is not int or not 1 <= point_filter_max_candidates <= 2000000 or not math.isfinite(point_filter_budget_seconds) or point_filter_budget_seconds <= 0:
        raise ValueError("Invalid point-cloud observation budget")
    effective_num_threads = 0 if colmap_num_threads is None else int(colmap_num_threads)
    if effective_num_threads < 0:
        raise ValueError("COLMAP thread count must be non-negative; use 0 for automatic selection")
    matcher = matcher if matcher in {"constrained", "hybrid", "exhaustive", "sequential"} else "constrained"
    if scene_completion == "videoscene" and not run_point_cloud_filter:
        raise ValueError("VideoScene completion requires point-cloud filtering")
    require_fresh_workspace(workspace_dir)
    manifest_fields, manifest_rows = read_csv(reconstruction_dir / "images_manifest.csv")
    workspace_manifest_rows = build_workspace_manifest(manifest_rows)
    camera_groups = camera_grouping_summary(workspace_manifest_rows)
    effective_single_camera = bool(single_camera and camera_groups["single_camera"])
    effective_single_camera_per_folder = bool(camera_groups["single_camera_per_folder"])
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
    profile_config = {**DENSE_PROFILES[dense_profile],
                      "patch_match": {**DENSE_PROFILES[dense_profile]["patch_match"],
                                      "max_image_size": dense_max_image_size},
                      "fusion": {**DENSE_PROFILES[dense_profile]["fusion"],
                                 "max_image_size": dense_max_image_size}}
    effective_run_meshing = run_meshing and bool(profile_config.get("run_meshing", True))

    copied_images, image_names, missing_images, removed_stale_workspace_images = copy_images(
        reconstruction_dir,
        workspace_dir,
        workspace_manifest_rows,
    )
    copied_metadata = copy_metadata(reconstruction_dir, workspace_dir)
    workspace_manifest_fields = [
        *manifest_fields,
        *[
            field
            for field in ("source_image_file", "workspace_image_group")
            if field not in manifest_fields
        ],
    ]
    write_csv_rows(
        workspace_dir / "metadata" / "images_manifest.csv",
        workspace_manifest_fields,
        workspace_manifest_rows,
    )
    mask_summary = mask_dir_summary(resolved_mask_dir, image_names)
    if resolved_mask_dir is not None and (not mask_summary["enabled"] or (image_names and not mask_summary["matched_image_count"])):
        raise ValueError("No usable mask paths: expected a mask directory containing <relative-image-name>.png, for example video/a.jpg.png")
    active_mask_dir = resolved_mask_dir if mask_summary.get("enabled") else None
    write_image_list(workspace_dir, image_names)
    match_pairs, match_edge_rows, match_pair_summary = build_constrained_match_pairs(
        workspace_manifest_rows, constrained_video_frame_window, visual_only)
    from video_sensor_matching import build_video_sensor_candidates
    sensor_edges, video_sensor_summary = build_video_sensor_candidates(
        workspace_manifest_rows, match_pairs, enabled=not visual_only and run_image_similarity,
        wifi_enabled=wifi_recovery_enabled, route_distance_m=constrained_max_pdr_distance_m,
        fuzzy_context_path=workspace_dir / "metadata" / "video_sensor_fuzzy_context.json",
    )
    match_edge_rows.extend(sensor_edges)
    match_pairs = sorted(set(match_pairs) | {(row["image_a"], row["image_b"]) for row in sensor_edges})
    match_pair_summary["video_sensor_assistance"] = video_sensor_summary
    route_pair_count = len(match_pairs)
    write_named_match_pairs(workspace_dir, "match_pairs_route.txt", match_pairs)
    if run_image_similarity:
        try:
            image_similarity_summary = analyze_workspace_image_similarity(
                workspace_dir=workspace_dir,
                edge_rows=match_edge_rows,
                max_pairs=image_similarity_max_pairs,
                max_image_size=image_similarity_max_image_size,
                orb_features=image_similarity_orb_features,
                feature_workers=image_similarity_workers,
                min_visual_score=image_similarity_min_visual_score,
                recovery_image_names=image_names,
                recovery_neighbors=visual_recovery_neighbors,
                recovery_metadata={row["image_file"]: row for row in workspace_manifest_rows},
                wifi_recovery_enabled=wifi_recovery_enabled and not visual_only,
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
    _similarity_fields, similarity_rows = read_csv(workspace_dir / "image_similarity_edges.csv")
    source_by_image = {row["image_file"]: row.get("source_type", "") for row in workspace_manifest_rows}
    recovery_edges = [
        {
            "image_a": row["image_a"], "image_b": row["image_b"], "reason": "visual_recovery",
            "source_type_a": source_by_image.get(row["image_a"], ""),
            "source_type_b": source_by_image.get(row["image_b"], ""),
            "edge_weight": row.get("edge_weight", ""),
            "edge_weight_basis": "appearance_retrieval_only_requires_geometry",
        }
        for row in similarity_rows if row.get("reason") == "visual_recovery"
    ]
    match_edge_rows.extend(recovery_edges)
    match_pairs = sorted(set(match_pairs) | {(row["image_a"], row["image_b"]) for row in recovery_edges})
    write_named_match_pairs(workspace_dir, "match_pairs_unfiltered.txt", match_pairs)
    write_match_graph_edges(workspace_dir, match_edge_rows)
    colmap_match_pairs, match_pair_filter_rows, match_pair_filter_summary = select_visually_supported_match_pairs(
        pairs=match_pairs,
        edge_rows=match_edge_rows,
        similarity_rows=similarity_rows,
        similarity_summary=image_similarity_summary,
    )
    write_match_pairs(workspace_dir, colmap_match_pairs)
    match_pair_filter_summary.update({
        "matcher": matcher,
        "pair_list_used_by_sparse_matcher": matcher in {"constrained", "hybrid"},
        "pair_gate_enforced": matcher == "constrained",
    })
    video_sensor_summary.update(
        selected_pair_count=match_pair_filter_summary["selected_video_sensor_pair_count"],
        selected_fuzzy_pair_count=match_pair_filter_summary["selected_fuzzy_sensor_pair_count"],
        pair_list_used_by_sparse_matcher=matcher in {"constrained", "hybrid"},
    )
    if video_sensor_summary.get("candidate_pair_count"):
        print(f"[sensors] Proposed {video_sensor_summary['candidate_pair_count']} sensor pairs; "
              f"{video_sensor_summary['selected_pair_count']} passed visual geometry "
              f"({video_sensor_summary['selected_fuzzy_pair_count']} from fuzzy time windows).", flush=True)
    write_match_pair_filter(workspace_dir, match_pair_filter_rows, match_pair_filter_summary)
    match_pair_summary.update(
        {
            "unfiltered_pair_count": len(match_pairs),
            "pair_count": len(match_pairs),
            "route_candidate_pair_count": route_pair_count,
            "visual_recovery_candidate_pair_count": len(recovery_edges),
            "wifi_recovery": video_sensor_summary,
            "weight_statistics_scope": "route_candidates_only",
            "colmap_pair_count": len(colmap_match_pairs),
            "p1_pair_filter": match_pair_filter_summary,
        }
    )
    if recovery_edges:
        match_pair_summary["reason_counts"]["visual_recovery"] = len(recovery_edges)
    image_similarity_summary["used_for_colmap_pair_selection"] = (
        match_pair_filter_summary.get("status") == "applied" and matcher in {"constrained", "hybrid"}
    )
    image_similarity_summary["diagnostic_only"] = not image_similarity_summary["used_for_colmap_pair_selection"]
    write_json(workspace_dir / "image_similarity_summary.json", image_similarity_summary)
    if not colmap_match_pairs and matcher in {"constrained", "hybrid"}:
        raise ValueError("No supported match pairs remain. Inspect match_pair_filter_summary.json and image coverage before reconstruction.")
    align_to_pdr = not visual_only  # Also controls optional sensor boundary reporting.
    remove_pdr_alignment_refs(workspace_dir)

    write_sparse_batch(
        root=root,
        workspace_dir=workspace_dir,
        colmap_exe=resolved_colmap_exe,
        mask_dir=active_mask_dir,
        camera_model=camera_model,
        single_camera=effective_single_camera,
        single_camera_per_folder=effective_single_camera_per_folder,
        use_gpu=use_gpu,
        matcher=matcher,
        align_to_pdr=align_to_pdr,
        feature_max_image_size=feature_max_image_size,
        max_num_features=max_num_features,
        max_num_matches=max_num_matches,
        num_threads=effective_num_threads,
        ba_gpu=colmap_ba_gpu,
        sparse_strategy=sparse_strategy,
        sparse_ba_growth_ratio=sparse_ba_growth_ratio,
        sparse_keyframe_gap_s=sparse_keyframe_gap_s,
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
        scene_completion=scene_completion,
        videoscene_max_clips=videoscene_max_clips,
        videoscene_budget_hours=videoscene_budget_hours,
        dense_selection_enabled=dense_max_images is not None,
        point_filter_method=point_filter_method,
        point_filter_max_candidates=point_filter_max_candidates,
        point_filter_budget_seconds=point_filter_budget_seconds,
    )
    write_full_batch(workspace_dir, root=root)
    write_automatic_batch(
        root=root,
        workspace_dir=workspace_dir,
        colmap_exe=resolved_colmap_exe,
        camera_model=camera_model,
        single_camera=effective_single_camera,
        single_camera_per_folder=effective_single_camera_per_folder,
        use_gpu=use_gpu,
        quality=quality,
    )
    write_gui_batch(root=root, workspace_dir=workspace_dir, colmap_exe=resolved_colmap_exe)
    write_reset_batch(root, workspace_dir)

    summary_status = "ok" if resolved_colmap_exe.exists() else "missing_colmap_exe"
    if missing_images:
        summary_status = "missing_source_images"
    elif not image_names:
        summary_status = "no_images"
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
        "single_camera": effective_single_camera,
        "single_camera_per_folder": effective_single_camera_per_folder,
        "camera_grouping": camera_groups,
        "use_gpu": use_gpu,
        "matcher": matcher,
        "visual_only": visual_only,
        "pdr_alignment_enabled": False,
        "pdr_capture_mode": "off",
        "pdr_alignment_policy": "visual_geometry_with_soft_sensor_matching",
        "pdr_alignment_decision": "pdr_alignment_decision.json",
        "wifi_recovery_enabled": wifi_recovery_enabled and not visual_only and run_image_similarity,
        "visual_pairing_mode": ("video_order_plus_appearance" if visual_only else "video_order_appearance_and_sensor_candidates"),
        "input_mode": "video_only",
        "video_sensor_assistance": video_sensor_summary,
        "image_similarity_config": {
            "enabled": run_image_similarity, "max_pairs": image_similarity_max_pairs,
            "max_image_size": image_similarity_max_image_size, "orb_features": image_similarity_orb_features,
            "feature_workers": image_similarity_workers,
            "min_visual_score": image_similarity_min_visual_score, "recovery_neighbors": visual_recovery_neighbors,
        },
        "constrained_pairing_config": {
            "max_pdr_distance_m": constrained_max_pdr_distance_m,
            "video_frame_window": constrained_video_frame_window,
        },
        "feature_max_image_size": feature_max_image_size,
        "max_num_features": max_num_features,
        "max_num_matches": max_num_matches,
        "colmap_num_threads": effective_num_threads,
        "colmap_ba_gpu": colmap_ba_gpu,
        "sparse_strategy": sparse_strategy,
        "sparse_ba_growth_ratio": sparse_ba_growth_ratio,
        "sparse_keyframe_gap_s": sparse_keyframe_gap_s,
        "execution_backend": "portable_stage_v1",
        "constrained_match_pairs": match_pair_summary,
        "match_graph_edges": "match_graph_edges.csv",
        "image_similarity": image_similarity_summary,
        "image_similarity_edges": "image_similarity_edges.csv",
        "quality": quality,
        "dense_max_image_size": dense_max_image_size,
        "dense_profile": dense_profile,
        "dense_profile_config": profile_config,
        **({"dense_selection_config": selection_settings} if selection_settings is not None else {}),
        "point_filter_config": {"method": point_filter_method,
                                "max_candidates": point_filter_max_candidates,
                                "budget_seconds": point_filter_budget_seconds},
        "run_meshing": effective_run_meshing,
        "run_point_cloud_filter": run_point_cloud_filter,
        "scene_completion": scene_completion,
        "videoscene_max_clips": videoscene_max_clips,
        "videoscene_budget_hours": videoscene_budget_hours,
        "cleanup_intermediate_ply": bool(profile_config.get("cleanup_intermediate_ply", True)) and not effective_run_meshing,
        "cleanup_runtime_override": "KEEP_INTERMEDIATE_PLY=1 retains intermediate PLY files",
        "requested_run_meshing": run_meshing,
        "sparse_model_selection": (
            "Select the valid sparse component with the most registered images (then sparse points); use sparse_pdr_aligned/0 only when authenticated to that component."
        ),
        "scripts": [
            "run_colmap_sparse.bat",
            "run_colmap_dense.bat",
            "run_colmap_dense_high.bat",
            "run_colmap_full.bat",
            "run_colmap_automatic.bat",
            "open_colmap_gui.bat",
            "reset_colmap_outputs.bat",
        ],
    }
    write_json(workspace_dir / "colmap_workspace_summary.json", summary)
    return summary
