#!/usr/bin/env python3
"""Generate a compact quality report for a prepared COLMAP workspace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from colmap_workspace_check import dense_input_model, inspect_sparse_model, inspect_workspace, file_identity, contained_path, write_provenance, open_readonly_database
from colmap_workspace_check import dense_readiness_failures
from dense_cache import cache_identity, reconstruction_binding, dense_selection_config, validate_dense_selection
from point_cloud_filter import sampling_quality_warning
from ply_io import validate_ply
from workspace_lock import WorkspaceBusyError, workspace_lock
from reconstruction_run_diagnostics import (
    database_feature_diagnostics, recorded_log_diagnostics, sparse_run_diagnostics,
)


SPARSE_ROOTS = ("sparse",)
ALIGNED_SPARSE_ROOTS = ("sparse_pdr_aligned", "sparse_pose_prior")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_provenance(path, obj)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_numbers(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) * 0.5
    return {
        "min": round(min(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "median": round(median, 6),
        "max": round(max(values), 6),
    }


def file_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def read_image_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.strip() for line in handle if line.strip()]


def sparse_model_stats(model_dir: Path, root_name: str) -> Dict[str, Any]:
    error = ""
    registered_images = sparse_points = 0
    try:
        model = inspect_sparse_model(model_dir)
        registered_images, sparse_points = len(model["names"]), model["point_count"]
    except (OSError, ValueError, struct.error) as exc:
        error = str(exc)
    exists = not error
    return {
        "model": model_dir.name,
        "root": root_name,
        "label": f"{root_name}/{model_dir.name}",
        "path": str(model_dir),
        "registered_images": registered_images,
        "sparse_points": sparse_points,
        "exists": exists,
        "error": error,
    }


def collect_sparse_models(
    workspace_dir: Path,
    input_image_count: int,
    sparse_roots: Sequence[str] = SPARSE_ROOTS,
    aligned_roots: Sequence[str] = ALIGNED_SPARSE_ROOTS,
) -> Dict[str, Any]:
    raw_models: List[Dict[str, Any]] = []
    for root_name in sparse_roots:
        sparse_dir = workspace_dir / root_name
        if not sparse_dir.exists():
            continue
        for model_dir in sorted((path for path in sparse_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            raw_models.append(sparse_model_stats(model_dir, root_name))

    aligned_models: List[Dict[str, Any]] = []
    for root_name in aligned_roots:
        aligned_dir = workspace_dir / root_name
        if not aligned_dir.exists():
            continue
        for model_dir in sorted((path for path in aligned_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            stats = sparse_model_stats(model_dir, root_name)
            if stats["exists"]:
                aligned_models.append(stats)

    valid_raw = [row for row in raw_models if row["exists"]]
    largest = max(valid_raw, key=lambda row: (row["registered_images"], row["sparse_points"]), default=None)
    dense_path = dense_input_model(workspace_dir)
    selected = next((row for row in [*raw_models, *aligned_models] if Path(row["path"]) == dense_path), None)
    largest_registered = (largest or {}).get("registered_images") or 0
    selected_registered = (selected or {}).get("registered_images") or 0
    camera_support = {"status": "unavailable", "reason": "no_selected_sparse_model"}
    if selected and selected.get("exists"):
        from reconstruction_metrics import Snapshot, sparse_metrics
        snapshot = Snapshot()
        try:
            camera_support = sparse_metrics(Path(selected["path"]), snapshot)["camera_support"]
            snapshot.verify_unchanged()
        except (OSError, ValueError, KeyError, struct.error) as exc:
            camera_support = {"status": "unavailable", "reason": str(exc)}
    return {
        "component_count": len(valid_raw),
        "models": raw_models,
        "largest_component": largest,
        "registered_image_ratio": (
            round(largest_registered / input_image_count, 6) if input_image_count > 0 else None
        ),
        "aligned_models": aligned_models,
        "pdr_aligned_exists": dense_path == workspace_dir / "sparse_pdr_aligned/0",
        "pdr_alignment_decision": read_json(workspace_dir / "pdr_alignment_decision.json"),
        "selected_model": selected,
        "selected_camera_support": camera_support,
        "selected_registered_image_ratio": (
            round(selected_registered / input_image_count, 6) if input_image_count > 0 else None
        ),
    }


def alignment_decision_matches_selected(sparse: Dict[str, Any]) -> bool:
    selected = sparse.get("selected_model") or {}
    decision = sparse.get("pdr_alignment_decision") or {}
    return bool(selected.get("label")) and decision.get("selected_model") == selected["label"]


def video_sampling_diagnostics(workspace_dir: Path) -> Dict[str, Any]:
    """Assess the sampling receipt copied with this workspace's input images.

    Never follow processed/output paths in the receipt: those may now describe
    a newer extraction than the images used by this reconstruction.
    """
    source = workspace_dir / "metadata/video_frames_summary.json"
    summary = read_json(source)
    result: Dict[str, Any] = {
        "status": "unavailable", "source": str(source), "videos": [],
        "warnings": [], "flagged_video_count": 0,
        "thresholds": {"minimum_retained_fraction": 0.5, "gap_interval_multiplier": 3.0,
                       "minimum_gap_warning_s": 5.0},
        "interpretation": (
            "Sampling retention and temporal gaps are input diagnostics, not measurements of 3D surface "
            "coverage or sensor synchronization. Thresholds are review heuristics; dense resolution "
            "cannot recover views absent from the selected sparse model."
        ),
    }
    entries = summary.get("videos")
    if not isinstance(entries, list) or not entries:
        result["reason"] = "workspace_sampling_receipt_missing_or_without_video_details"
        return result

    def nonnegative(value):
        number = safe_float(value)
        return number if math.isfinite(number) and number >= 0 else None

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        keyframes = entry.get("keyframe_filter") or {}
        coverage = entry.get("temporal_coverage") or {}
        if not isinstance(keyframes, dict) or not isinstance(coverage, dict):
            continue
        sampled = nonnegative(keyframes.get("input_frame_count", entry.get("extracted_frame_count")))
        retained = nonnegative(keyframes.get("kept_frame_count", entry.get("frame_count")))
        fraction = retained / sampled if sampled and retained is not None and retained <= sampled else None
        interval = nonnegative(entry.get("effective_frame_interval_s", summary.get("frame_interval_s")))
        gap_threshold = max(5.0, 3.0 * (interval or 0.0))
        gaps = {key: nonnegative(coverage.get(key)) for key in ("max_gap_s", "start_gap_s", "end_gap_s")}
        maximum_gap = max((value for value in gaps.values() if value is not None), default=None)
        flags = []
        if fraction is not None and fraction < 0.5:
            flags.append("low_frame_retention")
        if maximum_gap is not None and maximum_gap > gap_threshold:
            flags.append("long_temporal_gap")
        video = str(entry.get("video") or "unnamed_video")
        row = {
            "video": video, "sampled_frame_count": sampled, "retained_frame_count": retained,
            "retained_fraction": fraction, "rejected_blurry_count": nonnegative(keyframes.get("rejected_blurry_count")),
            "blur_threshold": nonnegative(keyframes.get("blur_threshold", summary.get("blur_threshold"))),
            "effective_frame_interval_s": interval, "video_duration_s": nonnegative(entry.get("video_duration_s")),
            **gaps, "largest_reported_gap_s": maximum_gap, "gap_warning_threshold_s": gap_threshold,
            "flags": flags,
        }
        result["videos"].append(row)
        if flags:
            result["flagged_video_count"] += 1
            findings = []
            if "low_frame_retention" in flags:
                findings.append(f"retained only {int(retained)}/{int(sampled)} sampled frames ({fraction:.1%})")
            if "long_temporal_gap" in flags:
                findings.append(f"has a {maximum_gap:.2f}s temporal gap (review threshold {gap_threshold:.2f}s)")
            result["warnings"].append(
                f"Video {video} " + " and ".join(findings) + "; inspect video_sampling before increasing dense resolution."
            )
    if result["videos"]:
        assessed = any(row["retained_fraction"] is not None or row["largest_reported_gap_s"] is not None
                       for row in result["videos"])
        result["status"] = "needs_review" if result["warnings"] else "no_issues_detected" if assessed else "unavailable"
    return result


def inspect_dense_ply(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "elements": {}, "error": ""}
    try:
        elements = validate_ply(path)
    except (OSError, ValueError) as exc:
        return {"status": "invalid", "elements": {}, "error": str(exc)}
    return {"status": "validated", "elements": elements, "error": ""}


def completion_status(final_summary: dict, binding_status: str, attempt: dict) -> dict:
    """The committed final receipt wins over a missing/stale progress receipt."""
    if binding_status != "verified" or final_summary.get("source_kind") != "videoscene_completed_point_cloud":
        return attempt
    committed = {
        "status": "completed", "promoted": True,
        "generated_points": final_summary.get("generated_vertex_count"),
        "observed_points": final_summary.get("observed_vertex_count"),
        "generated_source": final_summary.get("generated_source"),
        "generated_identity": final_summary.get("generated_identity"),
        "completed_identity": final_summary.get("final_identity"),
    }
    if all(attempt.get(key) == value for key, value in committed.items()):
        return attempt
    return {**committed, "status_source": "verified_final_output_summary", "status_receipt_stale": True,
            "recorded_attempt": {key: attempt.get(key) for key in ("status", "run", "error")}}


def summarize_dense(workspace_dir: Path) -> Dict[str, Any]:
    dense_dir = workspace_dir / "dense"
    final = dense_dir / "final.ply"
    fused = dense_dir / "fused.ply"
    filtered = dense_dir / "fused_filtered.ply"
    final_summary = read_json(dense_dir / "final_output_summary.json")
    filter_path = Path(final_summary.get("filter_receipt") or dense_dir / "point_filter_summary.json").resolve()
    if not filter_path.is_relative_to(dense_dir.resolve()):
        raise ValueError("Filter receipt points outside its dense workspace")
    validation = {name: inspect_dense_ply(path) for name, path in (("final", final), ("fused", fused), ("filtered", filtered))}
    binding_status = "missing" if not final.exists() else "legacy_unverified"
    if final.exists() and final_summary.get("final_identity") and final_summary.get("reconstruction_binding"):
        try:
            binding_status = "verified" if (
                file_identity(final) == final_summary["final_identity"] and
                reconstruction_binding(dense_dir) == final_summary["reconstruction_binding"] and
                cache_identity(workspace_dir) == final_summary["reconstruction_binding"].get("cache_identity") and
                validation["final"]["elements"].get("vertex") == final_summary.get("vertex_count")
            ) else "mismatch"
        except (OSError, ValueError):
            binding_status = "mismatch"
    return {
        "final_ply": str(final),
        "final_exists": final.exists(),
        "final_vertex_count": validation["final"]["elements"].get("vertex"),
        "final_face_count": validation["final"]["elements"].get("face", 0) if validation["final"]["status"] == "validated" else None,
        "ply_validation": validation,
        "final_output_summary": final_summary,
        "final_source_kind": final_summary.get("source_kind"),
        "final_reconstruction_binding": binding_status,
        "generated_vertex_count": final_summary.get("generated_vertex_count", 0) if binding_status == "verified" else None,
        "observed_vertex_count": final_summary.get("observed_vertex_count", validation["final"]["elements"].get("vertex")) if binding_status == "verified" else None,
        "scene_completion": completion_status(final_summary, binding_status, read_json(dense_dir / "videoscene_summary.json")),
        "scene_completion_applied": binding_status == "verified" and final_summary.get("source_kind") == "videoscene_completed_point_cloud",
        "intermediate_ply_cleanup": bool(final_summary) and not bool(final_summary.get("keep_intermediate")),
        "fused_ply": str(fused),
        "fused_exists": fused.exists(),
        "fused_vertex_count": validation["fused"]["elements"].get("vertex"),
        "filtered_ply": str(filtered),
        "filtered_exists": filtered.exists(),
        "filtered_vertex_count": validation["filtered"]["elements"].get("vertex"),
        "point_filter_summary": read_json(filter_path),
        "meshed_delaunay_exists": (dense_dir / "meshed-delaunay.ply").exists(),
        "meshed_poisson_exists": (dense_dir / "meshed-poisson.ply").exists(),
    }


def dense_selection_diagnostics(workspace: Path) -> Dict[str, Any]:
    """Report current, authenticated selection evidence separately from success."""
    dense = workspace / "dense"
    path = dense / "depth_source_summary.json"
    plan = read_json(path)
    selection = plan.get("dense_selection")
    result: Dict[str, Any] = dict(status="unavailable", binding_status="unavailable", warnings=[],
                                  coverage_report={}, coverage_targets={}, selected_view_count=None,
                                  registered_view_count=None, report=str(path),
                                  interpretation="Selection coverage measures retained sparse evidence, not reconstructed surface completeness or camera accuracy.")
    if not selection:
        return result
    try:
        config = dense_selection_config(workspace)
        if config is None:
            raise ValueError("Selection receipt has no matching enabled workspace configuration")
        model = dense / "sparse"
        names = inspect_sparse_model(model)["names"]
        lines = [line.strip() for line in (dense / "stereo/patch-match.cfg").read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        references = set(lines[::2])
        if len(lines) % 2 or len(references) != len(lines[::2]):
            raise ValueError("Selection receipt has invalid or duplicate stereo references")
        validate_dense_selection(workspace, plan, references, names, config)
        identities = plan.get("inputs", {})
        for stem in ("cameras", "images", "points3D"):
            model_path = next((model / (stem + suffix) for suffix in (".bin", ".txt")
                               if (model / (stem + suffix)).is_file()), None)
            if model_path is None or identities.get(str(model_path.resolve())) != file_identity(model_path):
                raise ValueError("Selection receipt does not authenticate the current undistorted model")
        if any(file_identity(Path(name)) != identity for name, identity in identities.items()):
            raise ValueError("Selection receipt inputs or planning code changed")
        for reference, sources in zip(lines[::2], lines[1::2]):
            if sources.split(", ") != [row["source"] for row in plan["sources"][reference]]:
                raise ValueError("Selection receipt differs from PatchMatch source configuration")
        fusion = (dense / "stereo/fusion.cfg").read_text(encoding="utf-8").splitlines()
        if len(fusion) != len(references) or set(fusion) != references:
            raise ValueError("Selection receipt differs from fusion configuration")
        cache = read_json(dense / "mvs_cache_identity.json")
        expected = cache.get("inputs", {}).get("files", {})
        if cache.get("status") == "complete":
            for name in ("depth_source_summary.json", "stereo/patch-match.cfg", "stereo/fusion.cfg"):
                if expected.get(name) != file_identity(dense / name):
                    raise ValueError("Selection receipt differs from the completed MVS run")
        result.update(binding_status="verified", status="no_issues_detected", config=config,
                      selected_view_count=selection["selected_view_count"],
                      registered_view_count=selection["registered_view_count"],
                      coverage_report=selection.get("coverage_report", {}),
                      coverage_targets=selection.get("coverage_targets", {}))
        result["warnings"] = ["Dense selection: " + warning for warning in selection.get("warnings", [])
                              if isinstance(warning, str) and warning]
        if result["coverage_targets"].get("target_met") is False:
            result["warnings"].append("Dense selection coverage targets were not reached. Successful stereo execution does not establish complete geometry; inspect the retained evidence and local gaps.")
        if result["warnings"]:
            result["status"] = "needs_review"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result.update(status="needs_review", binding_status="stale_or_unverified", reason=str(exc),
                      warnings=["Dense selection coverage cannot be authenticated against current inputs: " + str(exc)])
    return result


def post_filter_diagnostics(workspace: Path, dense: Dict[str, Any]) -> Dict[str, Any]:
    """Check coverage/unknown-removal claims against their original PLY bytes."""
    receipt = dense.get("point_filter_summary", {})
    result: Dict[str, Any] = dict(status="unavailable", binding_status="unavailable", warnings=[],
                                  spatial_coverage={"status": "not_assessed"}, unknown_removed_vertex_count=None)
    if not receipt:
        return result
    folder = (workspace / "dense").resolve()
    try:
        source = Path(receipt.get("input", "")).resolve()
        output = Path(receipt.get("output", "")).resolve()
        if not source.is_relative_to(folder) or not output.is_relative_to(folder):
            raise ValueError("Filter receipt paths are outside the dense workspace")
        if not receipt.get("input_identity") or file_identity(source) != receipt["input_identity"]:
            raise ValueError("Filter input PLY identity is unavailable or changed")
        output_identity = receipt.get("output_identity")
        final_summary = dense.get("final_output_summary", {})
        final = folder / "final.ply"
        if not output_identity or not (
                file_identity(output) == output_identity or
                (final_summary.get("source_kind") == "filtered_point_cloud"
                 and Path(final_summary.get("source", "")).resolve() == output
                 and final_summary.get("final_identity") == output_identity
                 and file_identity(final) == output_identity)):
            raise ValueError("Filter output does not match its retained PLY or authenticated final copy")
        if receipt.get("reconstruction_binding"):
            if reconstruction_binding(folder) != receipt["reconstruction_binding"]:
                raise ValueError("Filter receipt reconstruction inputs changed")
        result.update(status="no_issues_detected", binding_status="verified", method=receipt.get("method"),
                      spatial_coverage=receipt.get("spatial_coverage") or {"status": "not_assessed"},
                      assessment=receipt.get("assessment", {}))
        warnings = result["warnings"]
        warning = receipt.get("quality_warning")
        if isinstance(warning, str) and warning:
            warnings.append("Post-filter: " + warning)
        if receipt.get("method") == "wifi_connected_density":
            protection = receipt.get("observation_protection", {})
            unknown = max(0, int(protection.get("unknown_candidate_count", 0)))
            unassessed = max(0, int(protection.get("unevaluated_candidate_count", 0)))
            removed = max(0, int(receipt.get("geometry_removed_vertex_count", 0)))
            unknown_removed = min(removed, unknown + unassessed)
            result["unknown_removed_vertex_count"] = unknown_removed
            if unknown_removed:
                warnings.append(f"Post-filter removed {unknown_removed:,} original vertices classified as unknown or not assessed by observation checks. Their removal is a density decision, not proof that they are erroneous; compare raw fusion and retained geometry.")
        coverage = result["spatial_coverage"]
        if isinstance(coverage, dict):
            if coverage.get("retained_fraction") is not None:
                fraction = safe_float(coverage.get("retained_fraction"))
                initial, retained = coverage.get("input_occupied_cells"), coverage.get("output_occupied_cells")
                origin = coverage.get("origin")
                cell_size = safe_float(coverage.get("cell_size_scene_units"))
                if (type(initial) is not int or type(retained) is not int or initial <= 0
                        or not 0 <= retained <= initial or not math.isfinite(fraction)
                        or not math.isclose(fraction, retained / initial, rel_tol=1e-6, abs_tol=1e-6)
                        or not math.isfinite(cell_size) or cell_size <= 0
                        or not isinstance(origin, list) or len(origin) != 3
                        or any(not math.isfinite(safe_float(value)) for value in origin)):
                    raise ValueError("Filter spatial coverage counts/fraction are inconsistent")
                if fraction < .9:
                    warnings.append(f"Post-filter retained {fraction:.1%} of occupied spatial cells on the reported common grid. Point-count retention alone hides this local coverage loss; the 90% threshold is a review heuristic, not a surface-accuracy measurement.")
            for warning in coverage.get("warnings", []):
                if isinstance(warning, str) and warning:
                    warnings.append("Post-filter spatial coverage: " + warning)
        if warnings:
            result["status"] = "needs_review"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result.update(status="needs_review", binding_status="stale_or_unverified", reason=str(exc),
                      spatial_coverage={"status": "unverified"}, unknown_removed_vertex_count=None,
                      warnings=["Post-filter coverage cannot be authenticated against the retained source/output: " + str(exc)])
    return result


def summarize_match_graph(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "edge_count": 0, "reason_counts": {}, "edge_weight": None}

    reason_counts: Dict[str, int] = {}
    source_pair_counts: Dict[str, int] = {}
    weights: List[float] = []
    edge_count = 0
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            edge_count += 1
            reason = row.get("reason") or "unknown"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            source_key = f"{row.get('source_type_a', '')}->{row.get('source_type_b', '')}"
            source_pair_counts[source_key] = source_pair_counts.get(source_key, 0) + 1
            weight = safe_float(row.get("edge_weight"))
            if not math.isnan(weight):
                weights.append(weight)

    edge_weight = None
    if weights:
        edge_weight = {
            "min": round(min(weights), 6),
            "mean": round(sum(weights) / len(weights), 6),
            "max": round(max(weights), 6),
        }
    return {
        "exists": True,
        "edge_count": edge_count,
        "fieldnames": fieldnames,
        "weighted": bool(weights),
        "reason_counts": dict(sorted(reason_counts.items())),
        "source_pair_counts": dict(sorted(source_pair_counts.items())),
        "edge_weight": edge_weight,
    }


def summarize_image_similarity(workspace_dir: Path) -> Dict[str, Any]:
    summary_path = workspace_dir / "image_similarity_summary.json"
    csv_path = workspace_dir / "image_similarity_edges.csv"
    summary = read_json(summary_path)
    if summary:
        summary.setdefault("exists", csv_path.exists() or summary_path.exists())
        summary.setdefault("summary_path", str(summary_path))
        summary.setdefault("output_csv", str(csv_path))
        return summary
    if not csv_path.exists():
        return {
            "exists": False,
            "status": "missing",
            "evaluated_pair_count": 0,
            "visual_supported_pair_count": 0,
            "visual_decision_counts": {},
            "visual_overlap_score": None,
        }

    decision_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    scores: List[float] = []
    row_count = 0
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            decision = row.get("visual_decision") or "unknown"
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            pair_status = row.get("pair_status") or "unknown"
            status_counts[pair_status] = status_counts.get(pair_status, 0) + 1
            score = safe_float(row.get("visual_overlap_score"))
            if not math.isnan(score):
                scores.append(score)

    visual_supported = sum(
        decision_counts.get(name, 0)
        for name in ("geometry_supported", "feature_overlap", "near_duplicate")
    )
    return {
        "exists": True,
        "status": "ok",
        "summary_path": str(summary_path),
        "output_csv": str(csv_path),
        "evaluated_pair_count": row_count,
        "visual_supported_pair_count": visual_supported,
        "visual_decision_counts": dict(sorted(decision_counts.items())),
        "pair_status_counts": dict(sorted(status_counts.items())),
        "visual_overlap_score": summarize_numbers(scores),
    }


def summarize_match_pair_filter(workspace_dir: Path, workspace_summary: Dict[str, Any]) -> Dict[str, Any]:
    summary_path = workspace_dir / "match_pair_filter_summary.json"
    csv_path = workspace_dir / "match_pair_filter.csv"
    summary = read_json(summary_path)
    if not summary:
        constrained = workspace_summary.get("constrained_match_pairs", {})
        embedded = constrained.get("p1_pair_filter", {}) if isinstance(constrained, dict) else {}
        summary = dict(embedded) if isinstance(embedded, dict) else {}
    if summary:
        summary.setdefault("exists", summary_path.exists() or csv_path.exists())
        summary.setdefault("summary_path", str(summary_path))
        summary.setdefault("output_csv", str(csv_path))
        return summary
    return {
        "exists": False,
        "status": "missing",
        "input_pair_count": 0,
        "selected_pair_count": 0,
        "rejected_pair_count": 0,
        "unreviewed_pair_count": 0,
        "fallback_used": False,
        "selection_reason_counts": {},
    }


def database_status(workspace: Path) -> Dict[str, Any]:
    database = workspace / "database.db"
    if not database.exists():
        return {"exists": False, "image_count": 0, "error": ""}
    try:
        connection = open_readonly_database(database)
        try:
            image_count = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return {"exists": True, "image_count": 0, "error": str(exc)}
    return {"exists": True, "image_count": int(image_count), "error": ""}


def image_directory_status(workspace: Path, image_list: Sequence[str]) -> Dict[str, Any]:
    image_dir = workspace / "images"
    listed = set(image_list)
    files: List[str] = []
    if image_dir.exists():
        files = [
            str(path.relative_to(image_dir)).replace("\\", "/")
            for path in image_dir.rglob("*")
            if path.is_file()
        ]
    actual = set(files)
    missing = sorted(listed - actual)
    extra = sorted(actual - listed)
    return {
        "exists": image_dir.exists(),
        "file_count": len(files),
        "listed_count": len(image_list),
        "missing_listed_count": len(missing),
        "missing_listed": missing[:20],
        "extra_file_count": len(extra),
        "extra_files": extra[:20],
    }


def relative_status(
    workspace_summary: Dict[str, Any],
    database: Dict[str, Any],
    input_coverage: Dict[str, Any],
    image_directory: Dict[str, Any],
    image_similarity: Dict[str, Any],
    match_pair_filter: Dict[str, Any],
    sparse: Dict[str, Any],
    dense: Dict[str, Any],
    input_image_count: int,
) -> Tuple[str, List[str], List[str]]:
    warnings: List[str] = []
    notes: List[str] = []

    if not image_directory["exists"]:
        warnings.append("COLMAP images directory is missing.")
    if image_directory["missing_listed_count"]:
        warnings.append("Some image_list.txt entries are missing from the COLMAP images directory.")
    if image_directory["extra_file_count"]:
        warnings.append("COLMAP images directory contains files that are not listed in image_list.txt.")

    missing_source_count = int(workspace_summary.get("missing_image_count") or 0)
    if missing_source_count:
        warnings.append(f"Workspace preparation could not copy {missing_source_count} source image file(s).")

    if database.get("error"):
        warnings.append(f"Could not inspect COLMAP database: {database['error']}")
    database_coverage = input_coverage.get("database", {})
    sparse_coverage = input_coverage.get("dense_input_sparse", {})
    provenance = input_coverage.get("provenance", {})
    provenance_status = provenance.get("status", "unverified")
    if (database["exists"] or sparse_coverage.get("exists")) and provenance_status != "verified":
        warnings.append(f"Sparse input provenance is {provenance_status}: {provenance.get('error', 'no recorded content identity')}")
    if sparse_coverage.get("error"):
        warnings.append(f"Dense input sparse model is invalid: {sparse_coverage['error']}")
    if sparse_coverage.get("extra_names"):
        warnings.append("Dense input sparse model includes images outside the current input list.")
    database_name_mismatch = bool(
        database_coverage.get("missing_names") or database_coverage.get("extra_names")
    )
    if database["exists"] and input_image_count and database["image_count"] != input_image_count:
        warnings.append("COLMAP database image count does not match image_list.txt; rebuild sparse from a reset workspace.")
    elif database_name_mismatch:
        warnings.append("COLMAP database image names do not match image_list.txt; rebuild sparse from a reset workspace.")

    expected_video = database_coverage.get("source_coverage", {}).get("video_base_frame", {})
    database_video = expected_video.get("present_count", 0)
    video_count = expected_video.get("expected_count", 0)
    sparse_video = sparse_coverage.get("source_coverage", {}).get("video_base_frame", {}).get("present_count", 0)
    if video_count and database_video < video_count:
        warnings.append(
            f"Only {database_video}/{video_count} video frame(s) reached the COLMAP database; dense output cannot use the missing frames."
        )
    elif video_count and sparse_coverage.get("exists") and sparse_video == 0:
        warnings.append(
            "Video frames reached the COLMAP database but none registered in the selected sparse model; they currently provide no dense coverage."
        )

    if sparse["component_count"] == 0:
        warnings.append("No sparse COLMAP model was found yet.")
    elif input_image_count and (sparse["registered_image_ratio"] or 0) < 0.75:
        warnings.append("Sparse registration ratio is below 75%; dense reconstruction will cover only the registered views.")

    if sparse["component_count"] > 1:
        warnings.append("Multiple sparse components exist; more overlap or wider hybrid matching may improve coverage.")

    dense_blockers = dense_readiness_failures(input_coverage)
    warnings.extend(dense_blockers)

    if workspace_summary.get("pdr_alignment_enabled") and not sparse.get("pdr_aligned_exists"):
        decision = sparse.get("pdr_alignment_decision", {})
        decision_matches = alignment_decision_matches_selected(sparse)
        if decision and not decision_matches:
            warnings.append("The PDR decision belongs to another sparse component; it does not describe the currently selected visual model.")
        elif provenance_status == "verified" and decision.get("status") == "withheld":
            notes.append("PDR alignment was intentionally withheld by its reliability gate; raw visual coordinates are retained. Rebuilding unchanged inputs will not resolve unreliable PDR.")
        elif provenance_status == "verified" and decision.get("status") == "failed":
            warnings.append("PDR alignment failed validation; the verified raw sparse model is selected. Inspect pdr_alignment_decision.json for the cause.")
        else:
            warnings.append("No authenticated PDR-aligned model is selected; dense reconstruction uses the primary raw sparse component.")

    cleanup_intermediate_ply = bool(
        workspace_summary.get("cleanup_intermediate_ply") or dense.get("intermediate_ply_cleanup")
    )
    ply_validation = dense.get("ply_validation", {})
    dense_available = any(ply_validation.get(name, {}).get("status") == "validated" for name in ("final", "fused"))
    invalid_dense = False
    if dense.get("final_reconstruction_binding") == "mismatch":
        invalid_dense = True
        warnings.append("Final PLY no longer matches its recorded MVS inputs/cameras or file hash; do not run completion on it.")
    elif dense.get("final_reconstruction_binding") == "legacy_unverified":
        notes.append("Legacy final has no authenticated dense-run binding; it remains viewable but is not eligible for automatic VideoScene promotion.")
    if dense.get("scene_completion", {}).get("status_receipt_stale"):
        warnings.append("VideoScene completion is committed in the verified final receipt, but its progress receipt is stale or describes another attempt; the completed final is retained.")
    completion = dense.get("scene_completion", {})
    if completion.get("inference_attempted") is False:
        selection = completion.get("selection", {})
        notes.append("VideoScene inference did not run: " + (selection.get("reason") or completion.get("error") or completion.get("status", "unknown")))
    for name, result in ply_validation.items():
        if result["status"] == "invalid":
            invalid_dense = True
            warnings.append(f"Dense {name} PLY failed full payload validation: {result['error']}")
    filter_warning = sampling_quality_warning(dense.get("point_filter_summary", {}))
    if filter_warning:
        warnings.append(filter_warning)
    observation_summary = dense.get("point_filter_summary", {})
    unknown_count = 0
    if observation_summary.get("method") == "observation_support":
        classification = observation_summary.get("classification", {})
        unknown_count = int(classification.get("unknown", 0))
        supported_count = int(classification.get("supported", 0))
        notes.append(f"Observation assessment: {supported_count:,} supported, {unknown_count:,} unknown; internal support does not establish real-world accuracy.")
        if unknown_count:
            warnings.append("Unknown observation points remain in the retained geometry. Inspect the supported and unknown layers separately before judging reconstruction quality.")
    elif observation_summary.get('method') == 'wifi_connected_density':
        notes.append('The display cloud uses connected density with a soft WiFi extent prior. Removed points remain in raw fusion; this selection does not certify poses or wall locations.')
        prior = observation_summary.get('wifi_boundary', {})
        if prior.get('status') == 'unavailable':
            warnings.append('WiFi extent was unavailable; the density baseline was used: ' + prior.get('reason', 'unknown'))
        protection = observation_summary.get('observation_protection', {})
        if protection.get('observations_checked'):
            notes.append(
                f"Original-depth protection assessed {protection.get('assessed_candidate_count', 0):,} "
                f"of {protection.get('candidate_count', 0):,} density rejects and rescued "
                f"{protection.get('rescued_vertex_count', 0):,} original vertices before the export budget. "
                "Unassessed points retain the density decision; this is a bounded check, not full-cloud validation.")
        elif protection.get('status') == 'unavailable':
            notes.append('Original-depth protection was unavailable: ' + protection.get('reason', 'unknown'))
    if not dense_available:
        warnings.append("No fully validated final or fused dense PLY is available.")
    elif not dense["final_exists"]:
        warnings.append("Dense output exists, but final.ply has not been selected yet.")
    if (
        dense["fused_exists"]
        and workspace_summary.get("run_point_cloud_filter")
        and not dense["filtered_exists"]
        and not cleanup_intermediate_ply
    ):
        warnings.append("Filtered dense point cloud is not present yet.")
    if workspace_summary.get("matcher") == "hybrid":
        notes.append("Hybrid matching uses constrained WiFi/checkpoint, ordered-PDR-contour, and video-order pairs plus COLMAP visual bridge matches.")
    if workspace_summary.get("colmap_reset_recommended"):
        notes.append("Workspace preparation previously saw old COLMAP outputs; this report uses the files present now.")
    if image_similarity.get("status") == "ok":
        notes.append(
            "P1 image-similarity diagnostics evaluated "
            f"{image_similarity.get('evaluated_pair_count', 0)} constrained pairs; "
            f"{image_similarity.get('visual_supported_pair_count', 0)} have visual support labels."
        )
    elif image_similarity.get("status") in {"skipped", "failed"}:
        notes.append(
            "P1 image-similarity diagnostics are "
            f"{image_similarity.get('status')}: {image_similarity.get('reason', '')}"
        )
    if match_pair_filter.get("status") == "applied":
        notes.append(
            "P1 prepared "
            f"{match_pair_filter.get('selected_pair_count', 0)}/"
            f"{match_pair_filter.get('input_pair_count', 0)} route/recovery candidates in match_pairs.txt "
            f"and rejected {match_pair_filter.get('rejected_pair_count', 0)} candidates. These are input-list counts, not COLMAP-verified matches."
        )
    elif match_pair_filter.get("status") == "empty_selection":
        warnings.append("P1 rejected all candidates. The empty match list must be resolved before constrained reconstruction.")
    elif match_pair_filter.get("status") in {"not_applied", "missing"}:
        notes.append(
            "P1 pair selection was not recorded or was disabled; no visual selection benefit can be inferred."
        )
    if match_pair_filter.get("pair_gate_enforced") is False:
        notes.append("The selected matcher can generate matches outside this pair list; P1/P2 selection does not bound its final match graph.")
    trajectory_constraint = workspace_summary.get("constrained_match_pairs", {})
    if trajectory_constraint.get("camera_orientation_from_phone") is False:
        notes.append("Phone yaw is not used as camera orientation. Matching uses checkpoint order and available step-based route distance; diagnostic phone XY is excluded.")
    if trajectory_constraint.get("trajectory_constraint_available"):
        p2_contour_edges = trajectory_constraint.get("p2_contour_edge_count")
        notes.append(
            "P2 uses the ordered IMU/PDR route contour as a soft match-pair constraint and contributed "
            f"{p2_contour_edges if p2_contour_edges is not None else 'unreported'} route-contour candidate edge(s). "
            "This counts construction reasons, not incremental edges or a measured reconstruction gain; P2 adds no synthetic geometry."
        )
        if p2_contour_edges == 0:
            warnings.append(
                "P2 contour metadata is available but produced no route-contour candidate edges; inspect match_graph_edges.csv before relying on the constraint."
            )
    elif workspace_summary.get("video_sensor_assistance", {}).get("fuzzy_frame_count", 0):
        notes.append("Video sensor assistance uses uncertain time windows to propose visually checked pairs; exact frame synchronization and absolute PDR camera poses are not assumed.")
    elif not workspace_summary.get("visual_only") and workspace_summary:
        warnings.append(
            "Trajectory-contour metadata is unavailable, so constrained matching falls back to XY PDR proximity."
        )

    needs_rerun = (
        bool(database.get("error"))
        or (database["exists"] and input_image_count and database["image_count"] != input_image_count)
        or database_name_mismatch
        or bool(image_directory["missing_listed_count"])
        or bool(sparse_coverage.get("error"))
        or bool(sparse_coverage.get("extra_names"))
        or provenance_status in {"input_mismatch", "output_mismatch", "invalid_record", "incomplete_run"}
    )
    if needs_rerun:
        status = "needs_colmap_rerun"
    elif sparse["component_count"] == 0:
        status = "missing_sparse"
    elif dense_blockers:
        status = "sparse_reconstruction_failed"
    elif invalid_dense:
        status = "dense_needs_review"
    elif provenance_status != "verified":
        status = "unverified_reconstruction_inputs"
    elif not dense_available:
        status = "sparse_available_dense_missing"
    elif filter_warning or unknown_count:
        status = "dense_needs_review"
    else:
        status = "ok"
    return status, warnings, notes


def build_next_actions(report: Dict[str, Any]) -> List[str]:
    actions: List[str] = []
    database = report["database"]
    images = report["image_directory"]
    sparse = report["sparse"]
    dense = report["dense"]
    if report.get("dense_selection", {}).get("status") == "needs_review":
        actions.append("Inspect dense_selection coverage, local gaps and its identity check. Keep the full sparse reconstruction and compare an independently staged dense selection with a larger hard budget when targets remain unmet; never rewrite an old depth-cache receipt to claim the new plan generated it. Missing coverage is not proof that excluded images are invalid.")
    if report.get("post_filter", {}).get("status") == "needs_review":
        actions.append("Compare the bound raw fused cloud with the filtered cloud and spatial_coverage diagnostics. Preserve unknown/unassessed observed points in a separate layer or a conservative export; they are not confirmed outliers. Regenerate only the post-filter stage when source depths and fusion are unchanged and authenticated.")
    if report.get("video_sampling", {}).get("status") == "needs_review":
        actions.append("Inspect video_sampling and the original clip around long gaps. Re-extract more temporally distributed original frames, review the sharpness threshold and preserve overlap; compare retained coverage and sparse registration/components before repeating dense reconstruction. Increasing dense resolution alone cannot restore missing views.")
    if report.get("run_diagnostics", {}).get("linear_solver_failure_count", 0):
        actions.append("Review the recorded sparse log's solver warnings together with camera calibration, overlap and reprojection statistics. Preserve the current model; do not infer a root cause or change bundle-adjustment thresholds from the warning count alone.")
    feature_diagnostics = report.get("database", {}).get("features", {})
    if feature_diagnostics.get("zero_feature_count", 0) or feature_diagnostics.get("missing_feature_count", 0):
        actions.append("Inspect database.features for images with zero or missing keypoints. Check the original images and any extraction masks before deciding whether to replace, remove or re-extract an image; do not reassign checkpoint groups automatically.")

    if sampling_quality_warning(dense.get("point_filter_summary", {})):
        actions.append("Inspect fused.ply and rerun only post-filtering with the corrected sampler; if fused.ply was deleted, re-fuse the retained depth maps first.")
    observation_summary = dense.get("point_filter_summary", {})
    if (observation_summary.get("method") == "observation_support"
            and observation_summary.get("classification", {}).get("unknown", 0)):
        actions.append("Compare the supported and unknown layer paths in point_filter_summary.json; unknown points are not confirmed errors. Review camera parallax and cross-checkpoint observations before changing WiFi weights or removing geometry.")
    input_count = report["input_image_count"]
    coverage = report.get("input_coverage", {})
    database_coverage = coverage.get("database", {})
    sparse_coverage = coverage.get("dense_input_sparse", {})
    provenance_status = coverage.get("provenance", {}).get("status", "unverified")
    dense_blockers = dense_readiness_failures(coverage)
    if dense_blockers:
        actions.append("Stop before dense reconstruction: " + "; ".join(dense_blockers) + ". Preserve the database/features and failed models for diagnosis; a complete re-extraction is not automatically required.")
    if any(item.get("status") == "invalid" for item in dense.get("ply_validation", {}).values()):
        actions.append("Inspect the PLY validation errors; preserve inputs and regenerate only the failed dense/post-filter output from valid sources.")
    if sparse["component_count"] and provenance_status == "unverified":
        actions.append("The old sparse result has no run-time content record. Preserve the current PLY for viewing/post-filtering; verified dense execution requires a newly recorded sparse run, not a retroactive stamp.")
    needs_clean_sparse = (
        (database["exists"] and input_count and database["image_count"] != input_count)
        or bool(database_coverage.get("missing_names"))
        or bool(database_coverage.get("extra_names"))
        or bool(images["extra_file_count"])
        or bool(images["missing_listed_count"])
        or provenance_status in {"input_mismatch", "output_mismatch", "invalid_record", "incomplete_run"}
    )

    if needs_clean_sparse:
        actions.append("Run `python run_colmap.py sparse`; the launcher resets stale COLMAP outputs before rebuilding.")
    elif sparse["component_count"] == 0:
        actions.append("Run `python run_colmap.py sparse` to create a sparse model.")
    elif not dense_blockers and not sparse.get("pdr_aligned_exists") and report["workspace_summary"].get("pdr_alignment_enabled"):
        decision = sparse.get("pdr_alignment_decision", {})
        if decision and not alignment_decision_matches_selected(sparse):
            actions.append("The PDR decision belongs to another sparse component. Continue with the selected visual model; reassess PDR on that component before applying alignment.")
        elif provenance_status == "verified" and decision.get("status") == "withheld":
            actions.append("Continue with the verified visual model. Review checkpoint consistency in pdr_alignment_decision.json before changing PDR inputs; a sparse rerun alone is not a reliability fix.")
        elif provenance_status == "verified" and decision.get("status") == "failed":
            actions.append("Inspect the recorded alignment failure in pdr_alignment_decision.json; dense reconstruction can continue with the verified raw sparse model.")
        else:
            actions.append("Inspect PDR alignment provenance and checkpoint reliability before deciding whether a rebuild is necessary.")
    expected_video = database_coverage.get("source_coverage", {}).get("video_base_frame", {})
    selected_video = sparse_coverage.get("source_coverage", {}).get("video_base_frame", {})
    if expected_video.get("expected_count", 0) and selected_video.get("present_count", 0) == 0 and not needs_clean_sparse:
        actions.append("Video frames are not registered in the selected sparse model; inspect video/anchor overlap and P1 diagnostics before judging dense completeness.")
    if (sparse["component_count"] > 0 and not dense["final_exists"]
            and not dense_blockers and not needs_clean_sparse and provenance_status == "verified"):
        actions.append("Run `python run_colmap.py dense` to create `colmap_workspace/dense/final.ply`.")
    elif (not dense_blockers and not needs_clean_sparse and provenance_status == "verified"
          and dense.get("final_reconstruction_binding") == "mismatch"):
        actions.append("Preserve the old final for reference and run `python run_colmap.py dense` to rebuild dense output for the currently selected model; do not reuse the mismatched final for completion.")
    return actions


def build_report(root: Path, processed_dir: Path, reconstruction_dir: Path, workspace_dir: Path) -> Dict[str, Any]:
    processing_summary = read_json(processed_dir / "processing_summary.json")
    export_summary = read_json(reconstruction_dir / "export_summary.json")
    workspace_summary = read_json(workspace_dir / "colmap_workspace_summary.json")

    image_list = read_image_list(workspace_dir / "image_list.txt")
    input_image_count = int(
        workspace_summary.get("image_count")
        or export_summary.get("colmap_image_count")
        or len(image_list)
        or file_line_count(workspace_dir / "image_list.txt")
    )
    match_graph = summarize_match_graph(workspace_dir / "match_graph_edges.csv")
    image_similarity = summarize_image_similarity(workspace_dir)
    match_pair_filter = summarize_match_pair_filter(workspace_dir, workspace_summary)
    sparse = collect_sparse_models(workspace_dir, input_image_count)
    dense = summarize_dense(workspace_dir)
    database = database_status(workspace_dir)
    image_directory = image_directory_status(workspace_dir, image_list)
    input_coverage = inspect_workspace(workspace_dir)
    run_diagnostics = sparse_run_diagnostics(
        workspace_dir, input_coverage.get("provenance", {}).get("status", "unverified"))
    database["features"] = database_feature_diagnostics(workspace_dir)
    status, warnings, notes = relative_status(
        workspace_summary=workspace_summary,
        database=database,
        input_coverage=input_coverage,
        image_directory=image_directory,
        image_similarity=image_similarity,
        match_pair_filter=match_pair_filter,
        sparse=sparse,
        dense=dense,
        input_image_count=input_image_count,
    )
    camera_support = sparse.get("selected_camera_support", {})
    video_sampling = video_sampling_diagnostics(workspace_dir)
    warnings.extend(video_sampling["warnings"])
    video_sensor_assistance = workspace_summary.get("video_sensor_assistance", {})
    if video_sensor_assistance.get("fuzzy_frame_count", 0):
        notes.append("Video/sensor timing uses assumed uncertainty windows. Sensor pairs require visual geometry; fuzzy association is not exact synchronization or camera-position accuracy.")
    dense_selection = dense_selection_diagnostics(workspace_dir)
    post_filter = post_filter_diagnostics(workspace_dir, dense)
    warnings.extend(dense_selection["warnings"])
    warnings.extend(post_filter["warnings"])
    if sparse.get("selected_model") and camera_support.get("status") != "available":
        warnings.append("Per-camera sparse support could not be evaluated: " + camera_support.get("reason", "unknown"))
    elif camera_support.get("views_with_at_most_two_occupied_cells"):
        notes.append(f"{len(camera_support['views_with_at_most_two_occupied_cells'])} registered view(s) have linked features in at most 2 of 64 image cells; inspect local support before treating registration as reliable pose coverage.")
    solver_failures = run_diagnostics.get("linear_solver_failure_count", 0)
    if solver_failures:
        warnings.append(f"The authenticated sparse run logged {solver_failures} failed linear-solver steps. Process success does not certify bundle-adjustment convergence; inspect run_diagnostics.")
    elif run_diagnostics["status"] != "verified":
        notes.append("Sparse numerical convergence is not assessed: no unchanged run-bound log is available for the current verified outputs. Historical log timestamps do not establish this association.")
    features = database["features"]
    if features.get("zero_feature_count", 0):
        warnings.append(f"{features['zero_feature_count']} database image(s) have zero keypoints and cannot contribute feature matches; inspect database.features.zero_feature_images.")
    if features.get("missing_feature_count", 0):
        warnings.append(f"{features['missing_feature_count']} database image(s) have no keypoint records; inspect database.features.missing_feature_images.")
    quality_review = {
        "status": "needs_review" if warnings else (
            "no_issues_detected" if run_diagnostics["status"] == "verified" and features["status"] == "available" else "not_assessed"),
        "warning_count": len(warnings),
        "numerical_log_status": run_diagnostics["status"],
        "dense_selection_status": dense_selection["status"],
        "dense_coverage_target_met": dense_selection["coverage_targets"].get("target_met"),
        "post_filter_status": post_filter["status"],
        "interpretation": "Pipeline completion and review findings are separate; neither an exit code nor absence of detected warnings proves geometry accuracy.",
    }
    report = {
        "status": status,
        "root": str(root),
        "processed_dir": str(processed_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "workspace_dir": str(workspace_dir),
        "input_image_count": input_image_count,
        "video_sampling": video_sampling,
        "video_sensor_assistance": video_sensor_assistance,
        "processing_status": processing_summary.get("status", "available" if processing_summary else "missing"),
        "export_status": export_summary.get("status", "missing"),
        "workspace_status": workspace_summary.get("status", "missing"),
        "workspace_summary": workspace_summary,
        "colmap_reset_recommended": bool(workspace_summary.get("colmap_reset_recommended")),
        "image_directory": image_directory,
        "database": database,
        "run_diagnostics": run_diagnostics,
        "quality_review": quality_review,
        "input_coverage": input_coverage,
        "match_graph": match_graph,
        "image_similarity": image_similarity,
        "match_pair_filter": match_pair_filter,
        "video_intrinsics_initialization": read_json(workspace_dir / "video_intrinsics_summary.json"),
        "sparse_recovery": read_json(workspace_dir / "sparse_recovery_summary.json"),
        "sparse": sparse,
        "dense": dense,
        "dense_selection": dense_selection,
        "post_filter": post_filter,
        "warnings": warnings,
        "notes": notes,
    }
    report["next_actions"] = build_next_actions(report)
    return report


def quality_source_binding(workspace: Path, root: Path, processed_dir: Path, reconstruction_dir: Path):
    """Bind every report source to its current bytes, including absent artifacts.

    No old quality report, file timestamp, or point count substitutes for an
    input/output identity. The returned snapshot checks concurrent changes.
    """
    from reconstruction_metrics import Snapshot, digest
    from runner_migration import migration_dependency_paths

    snapshot = Snapshot()
    paths = [processed_dir / "processing_summary.json", reconstruction_dir / "export_summary.json"]
    paths.extend(migration_dependency_paths(workspace))
    paths.extend(workspace / name for name in (
        "colmap_workspace_summary.json", "image_list.txt", "database.db", "database.db-wal",
        "metadata/images_manifest.csv", "metadata/video_frames_summary.json", "match_graph_edges.csv", "image_similarity_edges.csv",
        "metadata/video_sensor_fuzzy_context.json", "metadata/video_sensor_fuzzy_summary.json",
        "metadata/video_sensor_sync.csv", "metadata/video_sensor_sync_summary.json",
        "video_intrinsics_summary.json", "sparse_recovery_summary.json",
        "image_similarity_summary.json", "match_pair_filter.csv", "match_pair_filter_summary.json",
        "match_pairs.txt", "pdr_alignment_refs.txt", "sparse_run_provenance.json", "sparse_run_inputs.json",
        "sparse_import/source_sparse_run_provenance.json", "sparse_import_pending.json",
        "pdr_alignment_decision.json", "pdr_alignment_reviewed_refs.txt",
        "run_colmap_sparse.bat", "run_colmap_automatic.bat", "run_colmap_dense.bat",
        "dense/final_output_summary.json", "dense/point_filter_summary.json", "dense/videoscene_summary.json",
        "dense/mvs_cache_identity.json", "dense/final.ply", "dense/fused.ply", "dense/fused_filtered.ply",
        "dense/depth_source_summary.json", "dense/stereo/patch-match.cfg", "dense/stereo/fusion.cfg",
        "dense/meshed-poisson.ply", "dense/meshed-delaunay.ply",
    ))
    for folder in ("sparse", "sparse_pdr_aligned", "sparse_pose_prior", "dense/sparse"):
        paths.extend(path for path in (workspace / folder).rglob("*") if path.is_file() and path.suffix in (".bin", ".txt"))
    for name in read_image_list(workspace / "image_list.txt"):
        paths.append(contained_path(workspace / "images", name))
    paths.extend(path for path in (workspace / "images").rglob("*") if path.is_file())
    settings = snapshot.read_json(workspace / "colmap_workspace_summary.json")
    if settings.get("colmap_exe"):
        executable = Path(settings["colmap_exe"])
        paths.append(executable if executable.is_absolute() else workspace / executable)
    mask = settings.get("colmap_mask", {})
    if mask.get("enabled") and mask.get("path"):
        mask_root = Path(mask["path"])
        paths.extend(contained_path(mask_root, name + ".png") for name in read_image_list(workspace / "image_list.txt"))
    final = snapshot.read_json(workspace / "dense/final_output_summary.json")
    for name in ("observed_source", "generated_source", "filter_receipt"):
        if final.get(name):
            path = Path(final[name]).resolve()
            if not path.is_relative_to((workspace / "dense").resolve()):
                raise ValueError("Final report source points outside its dense workspace")
            paths.append(path)
    filter_path = Path(final.get("filter_receipt") or workspace / "dense/point_filter_summary.json").resolve()
    if not filter_path.is_relative_to((workspace / "dense").resolve()):
        raise ValueError("Filter receipt points outside its dense workspace")
    filter_record = snapshot.read_json(filter_path)
    if filter_record.get('method') == 'observation_support':
        for artifact in [filter_record.get('evidence', {}), *filter_record.get('layers', {}).values()]:
            if artifact.get('path'):
                path = Path(artifact['path']).resolve()
                if not path.is_relative_to((workspace / 'dense').resolve()):
                    raise ValueError('Observation evidence points outside its dense workspace')
                paths.append(path)
    completion = snapshot.read_json(workspace / 'dense/videoscene_summary.json')
    if completion.get('status') == 'hypotheses_only':
        for key in ('generated_source', 'hypothesis_preview'):
            if completion.get(key):
                path = Path(completion[key]).resolve()
                if not path.is_relative_to((workspace/'dense').resolve()):
                    raise ValueError('Hypothesis evidence points outside its dense workspace')
                paths.append(path)
    for name in ("reconstruction_quality.py", "reconstruction_run_diagnostics.py", "runner_migration.py", "sparse_reuse.py", "colmap_workspace_check.py", "sparse_quality.py", "dense_cache.py", "dense_keyframes.py", "ply_io.py", "reconstruction_metrics.py", "observation_filter.py", "mvs_observations.py"):
        paths.append(Path(__file__).with_name(name))
    files = {str(path.resolve()): snapshot.identity(path) for path in sorted(set(paths), key=str)}
    binding = {"schema_version": 1, "workspace": str(workspace), "root": str(root),
               "processed_dir": str(processed_dir), "reconstruction_dir": str(reconstruction_dir), "files": files,
               "sparse_log_prefix": recorded_log_diagnostics(workspace)}
    binding["identity"] = digest(binding)
    return binding, snapshot


def _quality_locations(workspace, root=None, processed_dir=None, reconstruction_dir=None, output=None):
    workspace = Path(workspace).resolve()
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    settings = read_json(workspace / "colmap_workspace_summary.json")
    reconstruction = Path(reconstruction_dir or settings.get("reconstruction_dir") or root / "reconstruction_dataset")
    reconstruction = (reconstruction if reconstruction.is_absolute() else root / reconstruction).resolve()
    exported = read_json(reconstruction / "export_summary.json")
    processed = Path(processed_dir or exported.get("processed_dir") or root / "processed")
    processed = (processed if processed.is_absolute() else root / processed).resolve()
    output = Path(output).resolve() if output is not None else workspace / "quality_report.json"
    return workspace, root, processed, reconstruction, output


def _quality_report_current(report, binding):
    from reconstruction_metrics import digest
    if not report:
        return {"status": "missing", "reason": "No quality report exists."}
    if not report.get("result_binding") or not report.get("report_identity"):
        return {"status": "stale", "reason": "Legacy report has no input/output identity binding."}
    unsigned = {key: value for key, value in report.items() if key != "report_identity"}
    if digest(unsigned) != report["report_identity"]:
        return {"status": "stale", "reason": "Quality report content changed after its snapshot."}
    if report["result_binding"] != binding:
        return {"status": "stale", "reason": "Current input, configuration, model, output, or evaluation code differs from the report snapshot."}
    return {"status": "current", "reason": "Report hashes match current inputs and outputs."}


def quality_report_status(workspace, output=None, *, root=None, processed_dir=None, reconstruction_dir=None):
    """Read-only freshness check; a legacy unbound report is always stale."""
    workspace, root, processed, reconstruction, output = _quality_locations(workspace, root, processed_dir, reconstruction_dir, output)
    binding, snapshot = quality_source_binding(workspace, root, processed, reconstruction)
    snapshot.identity(output)
    report = read_json(output)
    result = _quality_report_current(report, binding)
    if not report and output.exists():
        result = {"status": "stale", "reason": "Existing quality report is empty or invalid JSON."}
    snapshot.verify_unchanged()
    if recorded_log_diagnostics(workspace) != binding["sparse_log_prefix"]:
        raise ValueError("Sparse log evidence changed during quality evaluation")
    return {**result, "report": str(output), "current_identity": binding["identity"]}


def refresh_quality_report(workspace, output=None, *, root=None, processed_dir=None, reconstruction_dir=None, force=False):
    """Refresh stale quality data atomically while the workspace is locked."""
    from reconstruction_metrics import digest
    workspace, root, processed, reconstruction, output = _quality_locations(workspace, root, processed_dir, reconstruction_dir, output)
    with workspace_lock(workspace):
        binding, snapshot = quality_source_binding(workspace, root, processed, reconstruction)
        snapshot.identity(output)
        previous = read_json(output)
        if not force and _quality_report_current(previous, binding)["status"] == "current":
            snapshot.verify_unchanged()
            if recorded_log_diagnostics(workspace) != binding["sparse_log_prefix"]:
                raise ValueError("Sparse log evidence changed during quality evaluation")
            return previous
        report = build_report(root, processed, reconstruction, workspace)
        snapshot.verify_unchanged()
        if recorded_log_diagnostics(workspace) != binding["sparse_log_prefix"]:
            raise ValueError("Sparse log evidence changed during quality evaluation")
        report.update(generated_utc=datetime.now(timezone.utc).isoformat(), result_binding=binding)
        report["report_identity"] = digest(report)
        write_json(output, report)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--processed", type=Path, help="Override the processed directory recorded by this workspace's dataset.")
    parser.add_argument("--reconstruction", type=Path, help="Override the dataset directory recorded by this workspace.")
    parser.add_argument("--workspace", type=Path, default=Path("colmap_workspace"))
    parser.add_argument(
        "--summary",
        "--json-output",
        dest="summary",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--markdown",
        "--markdown-output",
        dest="markdown",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--check-freshness", action="store_true", help="Check report input/output hashes without writing; exit 2 if stale.")
    parser.add_argument("--refresh-if-stale", action="store_true", help="Keep an already current report unchanged.")
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    processed_dir = resolve_path(root, args.processed).resolve() if args.processed else None
    reconstruction_dir = resolve_path(root, args.reconstruction).resolve() if args.reconstruction else None
    workspace_dir = resolve_path(root, args.workspace).resolve()
    summary_output = resolve_path(root, args.summary) if args.summary else workspace_dir / "quality_report.json"
    try:
        if args.check_freshness:
            result = quality_report_status(workspace_dir, summary_output, root=root, processed_dir=processed_dir, reconstruction_dir=reconstruction_dir)
            print(json.dumps(result))
            return 0 if result["status"] == "current" else 2
        print("[report] Generating reconstruction quality report...", flush=True)
        report = refresh_quality_report(workspace_dir, summary_output, root=root, processed_dir=processed_dir,
                                        reconstruction_dir=reconstruction_dir, force=not args.refresh_if_stale)
    except (OSError, ValueError, KeyError, WorkspaceBusyError) as exc:
        print(f"[quality error] {exc}")
        return 2
    print(f"[done] wrote {summary_output}")
    if args.markdown:
        print("[info] Legacy --markdown argument ignored; generated reports are JSON-only.")
    print(f"[info] status: {report['status']}")
    review = report["quality_review"]
    print(f"[quality] review={review['status']}; warnings={review['warning_count']}; "
          f"solver_failures={report['run_diagnostics'].get('linear_solver_failure_count', 'unverified')}; "
          f"zero_feature_images={report['database']['features'].get('zero_feature_count', 'unavailable')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
