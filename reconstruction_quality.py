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


def relative_status(workspace_summary: Dict[str, Any], sparse: Dict[str, Any], dense: Dict[str, Any]) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    if workspace_summary.get("colmap_reset_recommended"):
        warnings.append("Workspace summary reports older COLMAP outputs; run reset and rerun sparse/dense before final reporting.")
    if sparse.get("component_count", 0) == 0:
        warnings.append("No sparse model folders were found.")
    if not dense.get("exists"):
        warnings.append("Dense fused point cloud is not present.")

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
        ("Match graph edges", match_graph.get("edge_count", 0)),
        ("Weighted graph", "yes" if match_graph.get("weighted") else "no"),
        ("Sparse components", sparse.get("component_count", 0)),
        ("Largest registered images", largest.get("registered_images")),
        ("Largest sparse points", largest.get("sparse_points")),
        ("Registered image ratio", sparse.get("registered_image_ratio")),
        ("Dense fused.ply exists", "yes" if dense.get("exists") else "no"),
        ("Dense vertex count", dense.get("vertex_count")),
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
