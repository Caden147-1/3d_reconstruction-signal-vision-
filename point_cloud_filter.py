#!/usr/bin/env python3
"""Filter far and isolated outliers from a COLMAP dense point cloud.

The filter is intentionally conservative by default. It first clips only the
extreme tails of a robust bounding box and radial distance distribution, then
optionally removes isolated points using a voxel-neighborhood density check.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ply_io import PlyRow, read_ply_vertices, row_xyz, write_ply_vertices, xyz_indices


Point3 = Tuple[float, float, float]


def percentile(sorted_values: Sequence[float], ratio: float) -> float:
    if not sorted_values:
        return math.nan
    ratio = max(0.0, min(1.0, ratio))
    pos = ratio * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def axis_limits(points: Sequence[Point3], tail_quantile: float, padding_ratio: float) -> List[Tuple[float, float]]:
    tail_quantile = max(0.0, min(0.2, tail_quantile))
    limits: List[Tuple[float, float]] = []
    for axis in range(3):
        values = sorted(point[axis] for point in points)
        lo = percentile(values, tail_quantile)
        hi = percentile(values, 1.0 - tail_quantile)
        span = max(hi - lo, 1e-9)
        padding = span * max(0.0, padding_ratio)
        limits.append((lo - padding, hi + padding))
    return limits


def robust_center(points: Sequence[Point3]) -> Point3:
    return tuple(
        percentile(sorted(point[axis] for point in points), 0.5)
        for axis in range(3)
    )  # type: ignore[return-value]


def robust_diagonal(limits: Sequence[Tuple[float, float]]) -> float:
    return math.sqrt(sum(max(0.0, hi - lo) ** 2 for lo, hi in limits))


def inside_limits(point: Point3, limits: Sequence[Tuple[float, float]]) -> bool:
    return all(limits[axis][0] <= point[axis] <= limits[axis][1] for axis in range(3))


def voxel_key(point: Point3, origin: Point3, cell_size: float) -> Tuple[int, int, int]:
    return tuple(
        int(math.floor((point[axis] - origin[axis]) / cell_size))
        for axis in range(3)
    )  # type: ignore[return-value]


def density_mask(
    points: Sequence[Point3],
    candidate_indices: Sequence[int],
    limits: Sequence[Tuple[float, float]],
    radius: float,
    min_neighbors: int,
) -> Tuple[set[int], Dict[str, Any]]:
    if radius <= 0 or min_neighbors <= 0 or not candidate_indices:
        return set(candidate_indices), {"status": "skipped", "reason": "disabled_or_empty"}

    origin = (limits[0][0], limits[1][0], limits[2][0])
    grid: Dict[Tuple[int, int, int], int] = defaultdict(int)
    keys: Dict[int, Tuple[int, int, int]] = {}
    for index in candidate_indices:
        key = voxel_key(points[index], origin, radius)
        keys[index] = key
        grid[key] += 1

    kept: set[int] = set()
    for index, key in keys.items():
        neighbor_count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_count += grid.get((key[0] + dx, key[1] + dy, key[2] + dz), 0)
        if max(0, neighbor_count - 1) >= min_neighbors:
            kept.add(index)

    minimum_safe_keep = max(100, int(len(candidate_indices) * 0.2))
    if len(candidate_indices) >= 100 and len(kept) < minimum_safe_keep:
        return set(candidate_indices), {
            "status": "skipped",
            "reason": "would_remove_too_many_points",
            "candidate_count": len(candidate_indices),
            "would_keep_count": len(kept),
            "minimum_safe_keep": minimum_safe_keep,
        }

    return kept, {
        "status": "applied",
        "radius": radius,
        "min_neighbors": min_neighbors,
        "candidate_count": len(candidate_indices),
        "kept_count": len(kept),
    }


def filter_point_cloud(
    input_path: Path,
    output_path: Path,
    summary_path: Optional[Path],
    max_points: int,
    bbox_quantile: float,
    bbox_padding_ratio: float,
    radial_quantile: float,
    radius_ratio: float,
    min_neighbors: int,
    skip_density: bool,
) -> Dict[str, Any]:
    properties, rows, io_summary = read_ply_vertices(input_path, max_vertices=max_points)
    xyz = xyz_indices(properties)
    points = [row_xyz(row, xyz) for row in rows]
    if len(points) < 10:
        raise ValueError("Too few points to filter")

    limits = axis_limits(points, bbox_quantile, bbox_padding_ratio)
    bbox_indices = [index for index, point in enumerate(points) if inside_limits(point, limits)]

    center = robust_center([points[index] for index in bbox_indices] or points)
    distances = sorted(
        math.dist(points[index], center)
        for index in bbox_indices
    )
    radial_limit = percentile(distances, radial_quantile) if distances else math.inf
    radial_indices = [
        index
        for index in bbox_indices
        if math.dist(points[index], center) <= radial_limit
    ]

    diagonal = robust_diagonal(limits)
    radius = diagonal * max(0.0, radius_ratio)
    if skip_density:
        density_indices = set(radial_indices)
        density_summary = {"status": "skipped", "reason": "skip_density"}
    else:
        density_indices, density_summary = density_mask(
            points=points,
            candidate_indices=radial_indices,
            limits=limits,
            radius=radius,
            min_neighbors=min_neighbors,
        )

    filtered_rows: List[PlyRow] = [
        row
        for index, row in enumerate(rows)
        if index in density_indices
    ]
    written_count = write_ply_vertices(
        output_path,
        properties,
        filtered_rows,
        comments=(
            "generated by point_cloud_filter.py",
            "raw COLMAP fused.ply is preserved separately",
        ),
    )

    input_count = len(rows)
    declared_count = int(io_summary.get("declared_vertex_count") or input_count)
    summary = {
        "status": "ok",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "method": "robust_bbox_radial_clip_voxel_density",
        "input_vertex_count": input_count,
        "source_vertex_count": declared_count,
        "max_points": max_points,
        "sample_stride": io_summary.get("sample_stride", 1),
        "output_vertex_count": written_count,
        "removed_vertex_count": input_count - written_count,
        "removed_ratio": round((input_count - written_count) / input_count, 6) if input_count else 0.0,
        "bbox_quantile": bbox_quantile,
        "bbox_padding_ratio": bbox_padding_ratio,
        "bbox_limits": [[round(lo, 6), round(hi, 6)] for lo, hi in limits],
        "bbox_kept_count": len(bbox_indices),
        "radial_quantile": radial_quantile,
        "radial_center": [round(value, 6) for value in center],
        "radial_limit": round(radial_limit, 6) if math.isfinite(radial_limit) else None,
        "radial_kept_count": len(radial_indices),
        "radius_ratio": radius_ratio,
        "density_radius": round(radius, 6),
        "density": density_summary,
        **io_summary,
        "explanation": (
            "The filter removes only extreme spatial tails and isolated voxel "
            "neighborhoods. If the neighborhood check looks too destructive, it "
            "is skipped and recorded in this summary."
        ),
    }
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--max-points", type=int, default=750000, help="Uniformly sample very large PLY files before filtering; use 0 to keep every point.")
    parser.add_argument("--bbox-quantile", type=float, default=0.001)
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.05)
    parser.add_argument("--radial-quantile", type=float, default=0.9995)
    parser.add_argument("--radius-ratio", type=float, default=0.004)
    parser.add_argument("--min-neighbors", type=int, default=3)
    parser.add_argument("--skip-density", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = filter_point_cloud(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        max_points=args.max_points,
        bbox_quantile=args.bbox_quantile,
        bbox_padding_ratio=args.bbox_padding_ratio,
        radial_quantile=args.radial_quantile,
        radius_ratio=args.radius_ratio,
        min_neighbors=args.min_neighbors,
        skip_density=args.skip_density,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
