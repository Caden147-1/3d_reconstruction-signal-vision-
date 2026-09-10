#!/usr/bin/env python3
"""Generate a compact quality report for a prepared COLMAP workspace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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


def read_uint64_prefix(path: Path) -> Optional[int]:
    if not path.exists() or path.stat().st_size < 8:
        return None
    with path.open("rb") as handle:
        data = handle.read(8)
    return struct.unpack("<Q", data)[0]


def sparse_model_has_files(model_dir: Path) -> bool:
    binary = (model_dir / "cameras.bin").exists() and (model_dir / "images.bin").exists()
    text = (model_dir / "cameras.txt").exists() and (model_dir / "images.txt").exists()
    return binary or text


def sparse_model_stats(model_dir: Path, root_name: str) -> Dict[str, Any]:
    registered_images = read_uint64_prefix(model_dir / "images.bin")
    sparse_points = read_uint64_prefix(model_dir / "points3D.bin")
    exists = sparse_model_has_files(model_dir)
    return {
        "model": model_dir.name,
        "root": root_name,
        "label": f"{root_name}/{model_dir.name}",
        "path": str(model_dir),
        "registered_images": registered_images,
        "sparse_points": sparse_points,
        "exists": exists,
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

    largest = max(raw_models, key=lambda row: row.get("registered_images") or 0, default=None)
    selected = aligned_models[0] if aligned_models else largest
    largest_registered = (largest or {}).get("registered_images") or 0
    selected_registered = (selected or {}).get("registered_images") or 0
    return {
        "component_count": len(raw_models),
        "models": raw_models,
        "largest_component": largest,
        "registered_image_ratio": (
            round(largest_registered / input_image_count, 6) if input_image_count > 0 else None
        ),
        "aligned_models": aligned_models,
        "pdr_aligned_exists": bool(aligned_models),
        "selected_model": selected,
        "selected_registered_image_ratio": (
            round(selected_registered / input_image_count, 6) if input_image_count > 0 else None
        ),
    }


def count_ply_element(path: Path, element_name: str) -> Optional[int]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            header = handle.read(65536).decode("ascii", errors="ignore")
    except OSError:
        return None
    for line in header.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == element_name:
            try:
                return int(parts[2])
            except ValueError:
                return None
        if line.strip() == "end_header":
            break
    return None


def count_ply_vertices(path: Path) -> Optional[int]:
    return count_ply_element(path, "vertex")


def count_ply_faces(path: Path) -> Optional[int]:
    return count_ply_element(path, "face")


def summarize_dense(workspace_dir: Path) -> Dict[str, Any]:
    dense_dir = workspace_dir / "dense"
    final = dense_dir / "final.ply"
    fused = dense_dir / "fused.ply"
    filtered = dense_dir / "fused_filtered.ply"
    final_summary = read_json(dense_dir / "final_output_summary.json")
    return {
        "final_ply": str(final),
        "final_exists": final.exists(),
        "final_vertex_count": count_ply_vertices(final),
        "final_face_count": count_ply_faces(final),
        "final_output_summary": final_summary,
        "final_source_kind": final_summary.get("source_kind"),
        "intermediate_ply_cleanup": bool(final_summary) and not bool(final_summary.get("keep_intermediate")),
        "fused_ply": str(fused),
        "fused_exists": fused.exists(),
        "fused_vertex_count": count_ply_vertices(fused),
        "filtered_ply": str(filtered),
        "filtered_exists": filtered.exists(),
        "filtered_vertex_count": count_ply_vertices(filtered),
        "point_filter_summary": read_json(dense_dir / "point_filter_summary.json"),
        "meshed_delaunay_exists": (dense_dir / "meshed-delaunay.ply").exists(),
        "meshed_poisson_exists": (dense_dir / "meshed-poisson.ply").exists(),
    }


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


def database_status(workspace: Path) -> Dict[str, Any]:
    database = workspace / "database.db"
    if not database.exists():
        return {"exists": False, "image_count": 0, "error": ""}
    try:
        connection = sqlite3.connect(database)
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
    image_directory: Dict[str, Any],
    image_similarity: Dict[str, Any],
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
    if database["exists"] and input_image_count and database["image_count"] != input_image_count:
        warnings.append("COLMAP database image count does not match image_list.txt; rebuild sparse from a reset workspace.")

    if sparse["component_count"] == 0:
        warnings.append("No sparse COLMAP model was found yet.")
    elif input_image_count and (sparse["registered_image_ratio"] or 0) < 0.75:
        warnings.append("Sparse registration ratio is below 75%; dense reconstruction will cover only the registered views.")

    if sparse["component_count"] > 1:
        warnings.append("Multiple sparse components exist; more overlap or wider hybrid matching may improve coverage.")

    if workspace_summary.get("pdr_alignment_enabled") and not sparse.get("pdr_aligned_exists"):
        warnings.append("PDR-aligned sparse model is missing; dense reconstruction will fall back to raw sparse/0.")

    cleanup_intermediate_ply = bool(
        workspace_summary.get("cleanup_intermediate_ply") or dense.get("intermediate_ply_cleanup")
    )
    dense_available = bool(dense["final_exists"] or dense["fused_exists"])
    if not dense_available:
        warnings.append("Dense final.ply does not exist yet.")
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
    trajectory_constraint = workspace_summary.get("constrained_match_pairs", {})
    if trajectory_constraint.get("trajectory_constraint_available"):
        notes.append(
            "P2 uses the ordered IMU/PDR route contour as a soft match-pair constraint; "
            "it does not add synthetic geometry to final.ply."
        )
    elif not workspace_summary.get("visual_only") and workspace_summary:
        warnings.append(
            "Trajectory-contour metadata is unavailable, so constrained matching falls back to XY PDR proximity."
        )

    needs_rerun = (
        bool(database.get("error"))
        or (database["exists"] and input_image_count and database["image_count"] != input_image_count)
        or bool(image_directory["missing_listed_count"])
    )
    if needs_rerun:
        status = "needs_colmap_rerun"
    elif sparse["component_count"] == 0:
        status = "missing_sparse"
    elif not dense_available:
        status = "sparse_available_dense_missing"
    else:
        status = "ok"
    return status, warnings, notes


def build_next_actions(report: Dict[str, Any]) -> List[str]:
    actions: List[str] = []
    database = report["database"]
    images = report["image_directory"]
    sparse = report["sparse"]
    dense = report["dense"]
    input_count = report["input_image_count"]
    needs_clean_sparse = (
        (database["exists"] and input_count and database["image_count"] != input_count)
        or bool(images["extra_file_count"])
        or bool(images["missing_listed_count"])
    )

    if needs_clean_sparse:
        actions.append("Run `python run_colmap.py sparse`; the launcher resets stale COLMAP outputs before rebuilding.")
    elif sparse["component_count"] == 0:
        actions.append("Run `python run_colmap.py sparse` to create a sparse model.")
    elif not sparse.get("pdr_aligned_exists") and report["workspace_summary"].get("pdr_alignment_enabled"):
        actions.append("Rerun `python run_colmap.py sparse`; PDR alignment should now create sparse_pdr_aligned/0 first.")
    if sparse["component_count"] > 0 and not dense["final_exists"]:
        actions.append("Run `python run_colmap.py dense` to create `colmap_workspace/dense/final.ply`.")
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
    sparse = collect_sparse_models(workspace_dir, input_image_count)
    dense = summarize_dense(workspace_dir)
    database = database_status(workspace_dir)
    image_directory = image_directory_status(workspace_dir, image_list)
    status, warnings, notes = relative_status(
        workspace_summary=workspace_summary,
        database=database,
        image_directory=image_directory,
        image_similarity=image_similarity,
        sparse=sparse,
        dense=dense,
        input_image_count=input_image_count,
    )
    report = {
        "status": status,
        "root": str(root),
        "processed_dir": str(processed_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "workspace_dir": str(workspace_dir),
        "input_image_count": input_image_count,
        "processing_status": processing_summary.get("status", "available" if processing_summary else "missing"),
        "export_status": export_summary.get("status", "missing"),
        "workspace_status": workspace_summary.get("status", "missing"),
        "workspace_summary": workspace_summary,
        "colmap_reset_recommended": bool(workspace_summary.get("colmap_reset_recommended")),
        "image_directory": image_directory,
        "database": database,
        "match_graph": match_graph,
        "image_similarity": image_similarity,
        "sparse": sparse,
        "dense": dense,
        "warnings": warnings,
        "notes": notes,
    }
    report["next_actions"] = build_next_actions(report)
    return report


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return yes_no(value)
    return str(value)


def markdown_table(rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {format_value(value)} |")
    return "\n".join(lines)


def render_markdown(report: Dict[str, Any]) -> str:
    sparse = report["sparse"]
    dense = report["dense"]
    match_graph = report["match_graph"]
    image_similarity = report["image_similarity"]
    images = report["image_directory"]
    largest = sparse.get("largest_component") or {}
    selected = sparse.get("selected_model") or {}
    rows = [
        ("Status", report["status"]),
        ("Input images", report["input_image_count"]),
        ("Image files in workspace/images", images["file_count"] if images["exists"] else "missing"),
        ("Extra image files not in image_list.txt", images["extra_file_count"]),
        ("Missing image_list.txt files", images["missing_listed_count"]),
        ("Database images", report["database"]["image_count"] if report["database"]["exists"] else "missing"),
        ("Match graph edges", match_graph["edge_count"]),
        ("P1 image similarity status", image_similarity.get("status", "missing")),
        ("P1 image pairs evaluated", image_similarity.get("evaluated_pair_count", 0)),
        ("P1 visually supported pairs", image_similarity.get("visual_supported_pair_count", 0)),
        ("P1 near-duplicate pairs", image_similarity.get("near_duplicate_pair_count", 0)),
        ("Sparse components", sparse["component_count"]),
        ("Largest sparse model", largest.get("label", "-")),
        ("Largest registered images", largest.get("registered_images", 0)),
        ("Largest sparse points", largest.get("sparse_points", 0)),
        ("Registered image ratio", sparse["registered_image_ratio"]),
        ("PDR-aligned sparse exists", sparse["pdr_aligned_exists"]),
        ("Dense input candidate", selected.get("label", "-")),
        ("final.ply exists", dense["final_exists"]),
        ("Final vertex count", dense["final_vertex_count"]),
        ("Final face count", dense["final_face_count"]),
        ("Final source kind", dense["final_source_kind"] or "-"),
        ("Intermediate PLY cleanup", dense["intermediate_ply_cleanup"]),
        ("Intermediate fused.ply kept", dense["fused_exists"]),
        ("Intermediate fused vertex count", dense["fused_vertex_count"]),
        ("Intermediate fused_filtered.ply kept", dense["filtered_exists"]),
        ("Intermediate filtered vertex count", dense["filtered_vertex_count"]),
        ("Intermediate Delaunay mesh kept", dense["meshed_delaunay_exists"]),
        ("Intermediate Poisson mesh kept", dense["meshed_poisson_exists"]),
        (
            "P2 trajectory contour mode",
            report["workspace_summary"].get("constrained_match_pairs", {}).get("trajectory_constraint_mode", "missing"),
        ),
    ]

    reason_lines = ["| Reason | Count |", "| --- | --- |"]
    for reason, count in match_graph.get("reason_counts", {}).items():
        reason_lines.append(f"| {reason} | {count} |")

    similarity_lines = ["| Decision | Count |", "| --- | --- |"]
    for decision, count in image_similarity.get("visual_decision_counts", {}).items():
        similarity_lines.append(f"| {decision} | {count} |")

    warning_lines = "\n".join(f"- {warning}" for warning in report.get("warnings", [])) or "- None"
    note_lines = "\n".join(f"- {note}" for note in report.get("notes", [])) or "- None"
    action_lines = "\n".join(f"- {action}" for action in report.get("next_actions", [])) or "- None"
    return (
        "# Reconstruction Quality Report\n\n"
        f"{markdown_table(rows)}\n\n"
        "## Match Pair Reasons\n\n"
        f"{chr(10).join(reason_lines)}\n\n"
        "## P1 Image Similarity\n\n"
        f"{chr(10).join(similarity_lines)}\n\n"
        "## Warnings\n\n"
        f"{warning_lines}\n\n"
        "## Notes\n\n"
        f"{note_lines}\n\n"
        "## Next Actions\n\n"
        f"{action_lines}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--processed", type=Path, default=Path("processed"))
    parser.add_argument("--reconstruction", type=Path, default=Path("reconstruction_dataset"))
    parser.add_argument("--workspace", type=Path, default=Path("colmap_workspace"))
    parser.add_argument(
        "--summary",
        "--json-output",
        dest="summary",
        type=Path,
        default=Path("colmap_workspace") / "quality_report.json",
    )
    parser.add_argument(
        "--markdown",
        "--markdown-output",
        dest="markdown",
        type=Path,
        default=Path("colmap_workspace") / "QUALITY_REPORT.md",
    )
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    processed_dir = resolve_path(root, args.processed).resolve()
    reconstruction_dir = resolve_path(root, args.reconstruction).resolve()
    workspace_dir = resolve_path(root, args.workspace).resolve()
    report = build_report(root, processed_dir, reconstruction_dir, workspace_dir)

    summary_output = resolve_path(root, args.summary)
    markdown_output = resolve_path(root, args.markdown)
    write_json(summary_output, report)
    write_text(markdown_output, render_markdown(report))
    print(f"[done] wrote {summary_output}")
    print(f"[done] wrote {markdown_output}")
    print(f"[info] status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
