#!/usr/bin/env python3
"""Generate a compact quality report for the prepared COLMAP workspace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def file_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def read_uint64_prefix(path: Path) -> Optional[int]:
    if not path.exists() or path.stat().st_size < 8:
        return None
    with path.open("rb") as handle:
        data = handle.read(8)
    return struct.unpack("<Q", data)[0]


def sparse_model_stats(model_dir: Path) -> Dict[str, Any]:
    registered_images = read_uint64_prefix(model_dir / "images.bin")
    sparse_points = read_uint64_prefix(model_dir / "points3D.bin")
    return {
        "model": model_dir.name,
        "path": str(model_dir),
        "registered_images": registered_images,
        "sparse_points": sparse_points,
        "exists": (model_dir / "images.bin").exists() and (model_dir / "points3D.bin").exists(),
    }


def collect_sparse_models(workspace_dir: Path, input_image_count: int) -> Dict[str, Any]:
    sparse_dir = workspace_dir / "sparse"
    models: List[Dict[str, Any]] = []
    if sparse_dir.exists():
        for model_dir in sorted((path for path in sparse_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            models.append(sparse_model_stats(model_dir))

    largest = max(models, key=lambda row: row.get("registered_images") or 0, default=None)
    largest_registered = (largest or {}).get("registered_images") or 0
    return {
        "component_count": len(models),
        "models": models,
        "largest_component": largest,
        "registered_image_ratio": (
            round(largest_registered / input_image_count, 6)
            if input_image_count > 0 and largest_registered is not None
            else None
        ),
    }


def parse_ply_vertex_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        header = handle.read(65536).decode("ascii", errors="ignore")
    for line in header.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            try:
                return int(parts[2])
            except ValueError:
                return None
        if line.strip() == "end_header":
            break
    return None


def summarize_dense(workspace_dir: Path) -> Dict[str, Any]:
    fused_ply = workspace_dir / "dense" / "fused.ply"
    return {
        "fused_ply": str(fused_ply),
        "exists": fused_ply.exists(),
        "vertex_count": parse_ply_vertex_count(fused_ply),
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

    weight_summary = None
    if weights:
        weight_summary = {
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
        "edge_weight": weight_summary,
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


def dense_status(workspace: Path) -> Dict[str, Any]:
    dense_dir = workspace / "dense"
    fused = dense_dir / "fused.ply"
    filtered = dense_dir / "fused_filtered.ply"
    point_filter_summary = read_json(dense_dir / "point_filter_summary.json")
    floor_summary = read_json(dense_dir / "floor_completion_summary.json")
    dark_surface_summary = read_json(dense_dir / "dark_surface_completion_summary.json")
    return {
        "fused_ply": str(fused.resolve()),
        "fused_exists": fused.exists(),
        "fused_vertex_count": count_ply_vertices(fused),
        "filtered_ply": str(filtered.resolve()),
        "filtered_exists": filtered.exists(),
        "filtered_vertex_count": count_ply_vertices(filtered),
        "point_filter_summary": point_filter_summary,
        "meshed_delaunay_exists": (dense_dir / "meshed-delaunay.ply").exists(),
        "meshed_poisson_exists": (dense_dir / "meshed-poisson.ply").exists(),
        "floor_completed_mesh_exists": (dense_dir / "floor_completed_mesh.ply").exists(),
        "floor_completion_summary": floor_summary,
        "dark_surface_completed_mesh_exists": (dense_dir / "dark_surface_completed_mesh.ply").exists(),
        "dark_surface_completion_summary": dark_surface_summary,
    }


def build_next_actions(report: Dict[str, Any]) -> List[str]:
    actions: List[str] = []
    database = report["database"]
    images = report["image_directory"]
    sparse = report["sparse"]
    dense = report["dense"]
    workspace_summary = report.get("workspace_summary", {})
    input_count = report["input_image_count"]
    needs_database_reset = database["exists"] and input_count and database["image_count"] != input_count

    if needs_database_reset:
        actions.append("Run `python run_colmap.py reset` to remove the stale COLMAP database before rebuilding.")
    if images["extra_file_count"] or images["missing_listed_count"]:
        actions.append(
            "Regenerate the COLMAP workspace so `images/` is copied from the current manifest and stale files are removed."
        )
    if sparse["component_count"] == 0:
        if needs_database_reset:
            actions.append("Then run `python run_colmap.py sparse` to create a sparse model.")
        else:
            actions.append("Run `python run_colmap.py sparse` to create a sparse model.")
    elif sparse["registered_image_ratio"] < 0.75:
        actions.append("Improve sparse coverage with more visual overlap or wider hybrid matching, then rerun sparse.")
    if sparse["component_count"] > 0 and not dense["fused_exists"]:
        actions.append("Run `python run_colmap.py dense` after sparse reconstruction succeeds.")
    if dense["fused_exists"] and workspace_summary.get("run_point_cloud_filter") and not dense["filtered_exists"]:
        actions.append("Rerun `python run_colmap.py dense` to create the filtered dense point cloud.")
    if dense["fused_exists"] and not dense["floor_completed_mesh_exists"]:
        actions.append("Rerun dense with floor completion enabled if the floor inspection layer is required.")
    if dense["fused_exists"] and workspace_summary.get("run_dark_surface_completion") and not dense["dark_surface_completed_mesh_exists"]:
        actions.append("Rerun dense to create the dark low-texture surface completion layer.")
    return actions

def build_report(workspace: Path, sparse_roots: Sequence[str] = SPARSE_ROOTS) -> Dict[str, Any]:
    workspace = workspace.resolve()
    image_list = read_image_list(workspace / "image_list.txt")
    input_image_count = len(image_list)
    workspace_summary = read_json(workspace / "colmap_workspace_summary.json")
    sparse = scan_sparse_models(workspace, input_image_count, sparse_roots=sparse_roots)
    dense = dense_status(workspace)
    database = database_status(workspace)
    image_directory = image_directory_status(workspace, image_list)
    match_graph = count_match_graph(workspace / "match_graph_edges.csv")
    current_colmap_outputs = [
        name
        for name in ("database.db", "sparse", "sparse_pdr_aligned", "sparse_pose_prior", "dense")
        if (workspace / name).exists()
    ]

    warnings: List[str] = []
    notes: List[str] = []
    if not image_directory["exists"]:
        warnings.append("COLMAP images directory is missing.")
    if image_directory["missing_listed_count"]:
        warnings.append("Some image_list.txt entries are missing from the COLMAP images directory.")
    if image_directory["extra_file_count"]:
        warnings.append("COLMAP images directory contains files that are not listed in image_list.txt; regenerate the workspace before rebuilding.")
    missing_source_count = int(workspace_summary.get("missing_image_count") or 0)
    if missing_source_count:
        warnings.append(
            f"Workspace preparation could not copy {missing_source_count} source image file(s); rebuild the reconstruction export before COLMAP."
        )
    if database["exists"] and input_image_count and database["image_count"] != input_image_count:
        warnings.append(
            "COLMAP database image count does not match image_list.txt; run reset_colmap_outputs.bat before rebuilding."
        )
    if database.get("error"):
        warnings.append(f"Could not inspect COLMAP database: {database['error']}")
    if sparse["component_count"] == 0:
        warnings.append("No sparse COLMAP model was found yet.")
    elif input_image_count and sparse["registered_image_ratio"] < 0.75:
        warnings.append(
            "Sparse registration ratio is below 75%; dense reconstruction will only cover the largest registered component."
        )
    if sparse["component_count"] > 1:
        warnings.append(
            "Multiple sparse components exist; hybrid/exhaustive bridge matching or more overlap may improve coverage."
        )
    if not dense["fused_exists"]:
        warnings.append("Dense fused.ply does not exist yet.")
    if dense["fused_exists"] and workspace_summary.get("run_point_cloud_filter") and not dense["filtered_exists"]:
        warnings.append("Filtered dense point cloud is not present yet.")
    if dense["fused_exists"] and not dense["floor_completed_mesh_exists"]:
        warnings.append("Floor completion layer is not present yet.")
    if dense["fused_exists"] and workspace_summary.get("run_dark_surface_completion") and not dense["dark_surface_completed_mesh_exists"]:
        warnings.append("Dark surface completion layer is not present yet.")
    if workspace_summary.get("matcher") == "hybrid":
        notes.append(
            "Hybrid mode explains constrained WiFi/PDR/video-order pairs in match_graph_edges.csv; exhaustive bridge matches are generated inside COLMAP."
        )
    if workspace_summary.get("colmap_reset_recommended") and current_colmap_outputs:
        notes.append(
            "Current COLMAP output paths still exist: "
            + ", ".join(current_colmap_outputs)
            + ". Run reset before rebuilding those outputs."
        )
    elif workspace_summary.get("colmap_reset_recommended"):
        notes.append(
            "Old COLMAP outputs were detected during the last workspace preparation, but no reset-blocking COLMAP output path is present now."
        )

    if workspace_summary.get("colmap_reset_recommended"):
        return "needs_colmap_rerun", warnings
    if sparse.get("component_count", 0) == 0:
        return "missing_sparse", warnings
    if not dense.get("exists"):
        return "sparse_available_dense_missing", warnings
    return "ok", warnings


def markdown_table(rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def render_markdown(report: Dict[str, Any]) -> str:
    sparse = report["sparse"]
    dense = report["dense"]
    match_graph = report["match_graph"]
    largest = sparse.get("largest_component") or {}
    rows = [
        ("Status", report["status"]),
        ("Input images", report["input_image_count"]),
        ("Image files in workspace/images", images["file_count"] if images["exists"] else "missing"),
        ("Extra image files not in image_list.txt", images["extra_file_count"]),
        ("Missing image_list.txt files", images["missing_listed_count"]),
        ("Database images", report["database"]["image_count"] if report["database"]["exists"] else "missing"),
        ("Match graph edges", report["match_graph"]["edge_count"]),
        ("Sparse components", sparse["component_count"]),
        ("Largest registered images", largest.get("registered_images", 0)),
        ("Largest sparse points", largest.get("sparse_points", 0)),
        ("Registered image ratio", sparse["registered_image_ratio"]),
        ("Dense fused.ply exists", yes_no(dense["fused_exists"])),
        ("Dense vertex count", dense["fused_vertex_count"]),
        ("Filtered fused.ply exists", yes_no(dense["filtered_exists"])),
        ("Filtered vertex count", dense["filtered_vertex_count"]),
        ("Point filter removed vertices", dense.get("point_filter_summary", {}).get("removed_vertex_count", "-")),
        ("Delaunay mesh exists", yes_no(dense["meshed_delaunay_exists"])),
        ("Poisson mesh exists", yes_no(dense["meshed_poisson_exists"])),
        ("Floor completion exists", yes_no(dense["floor_completed_mesh_exists"])),
        ("Dark surface completion exists", yes_no(dense["dark_surface_completed_mesh_exists"])),
    ]

    reason_lines = ["| Reason | Count |", "| --- | --- |"]
    for reason, count in match_graph.get("reason_counts", {}).items():
        reason_lines.append(f"| {reason} | {count} |")

    warning_lines = "\n".join(f"- {warning}" for warning in report.get("warnings", [])) or "- None"
    return (
        "# Reconstruction Quality Report\n\n"
        f"{markdown_table(rows)}\n\n"
        "## Match Pair Reasons\n\n"
        f"{chr(10).join(reason_lines)}\n\n"
        "## Warnings\n\n"
        f"{warning_lines}\n"
    )


def build_report(root: Path, processed_dir: Path, reconstruction_dir: Path, workspace_dir: Path) -> Dict[str, Any]:
    processing_summary = read_json(processed_dir / "processing_summary.json")
    export_summary = read_json(reconstruction_dir / "export_summary.json")
    workspace_summary = read_json(workspace_dir / "colmap_workspace_summary.json")

    input_image_count = int(
        workspace_summary.get("image_count")
        or export_summary.get("colmap_image_count")
        or file_line_count(workspace_dir / "image_list.txt")
    )
    match_graph = summarize_match_graph(workspace_dir / "match_graph_edges.csv")
    sparse = collect_sparse_models(workspace_dir, input_image_count)
    dense = summarize_dense(workspace_dir)
    status, warnings = relative_status(workspace_summary, sparse, dense)

    return {
        "status": status,
        "root": str(root),
        "processed_dir": str(processed_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "workspace_dir": str(workspace_dir),
        "input_image_count": input_image_count,
        "processing_status": processing_summary.get("status", "available" if processing_summary else "missing"),
        "export_status": export_summary.get("status", "missing"),
        "workspace_status": workspace_summary.get("status", "missing"),
        "colmap_reset_recommended": bool(workspace_summary.get("colmap_reset_recommended")),
        "match_graph": match_graph,
        "sparse": sparse,
        "dense": dense,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--processed", type=Path, default=Path("processed"))
    parser.add_argument("--reconstruction", type=Path, default=Path("reconstruction_dataset"))
    parser.add_argument("--workspace", type=Path, default=Path("colmap_workspace"))
    parser.add_argument("--json-output", type=Path, default=Path("colmap_workspace") / "quality_report.json")
    parser.add_argument("--markdown-output", type=Path, default=Path("colmap_workspace") / "QUALITY_REPORT.md")
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

    json_output = resolve_path(root, args.json_output)
    markdown_output = resolve_path(root, args.markdown_output)
    write_json(json_output, report)
    write_text(markdown_output, render_markdown(report))
    print(f"[done] wrote {json_output}")
    print(f"[done] wrote {markdown_output}")
    print(f"[info] status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
