#!/usr/bin/env python3
"""Create a separate, explainable floor-plane completion mesh from fused.ply.

This does not replace or silently modify COLMAP's dense point cloud. It writes
an inferred inspection layer that can be loaded next to the raw/filtered cloud.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ply_io import read_ply_vertices, row_xyz, xyz_indices


Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


def percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return math.nan
    ratio = max(0.0, min(1.0, ratio))
    pos = ratio * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def convex_hull(points: Sequence[Point2]) -> List[Point2]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return list(unique)

    def cross(origin: Point2, a: Point2, b: Point2) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: List[Point2] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: List[Point2] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def polygon_area(points: Sequence[Point2]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        total += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(total) * 0.5


def expand_polygon(points: Sequence[Point2], padding: float) -> List[Point2]:
    if not points or padding <= 0:
        return list(points)
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    expanded: List[Point2] = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            expanded.append((x, y))
            continue
        scale = (length + padding) / length
        expanded.append((cx + dx * scale, cy + dy * scale))
    return expanded


def sample_points(points: Sequence[Point3], max_points: int) -> Tuple[List[Point3], int]:
    stride = max(1, math.ceil(len(points) / max(1, max_points)))
    return list(points[::stride]), stride


def robust_axis_range(points: Sequence[Point3], axis: int) -> float:
    values = sorted(point[axis] for point in points)
    return percentile(values, 0.98) - percentile(values, 0.02)


def choose_vertical_axis(points: Sequence[Point3]) -> int:
    return min(range(3), key=lambda axis: robust_axis_range(points, axis))


def trim_projected(points: Sequence[Point2]) -> List[Point2]:
    if len(points) < 12:
        return list(points)
    xs = sorted(point[0] for point in points)
    ys = sorted(point[1] for point in points)
    min_x, max_x = percentile(xs, 0.02), percentile(xs, 0.98)
    min_y, max_y = percentile(ys, 0.02), percentile(ys, 0.98)
    trimmed = [
        point
        for point in points
        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y
    ]
    return trimmed or list(points)


def plane_candidate(
    points: Sequence[Point3],
    vertical_axis: int,
    side: str,
    thickness_ratio: float,
) -> Dict[str, Any]:
    values = sorted(point[vertical_axis] for point in points)
    span = max(values[-1] - values[0], 1e-9)
    thickness = max(span * max(0.001, thickness_ratio), 0.02)
    height = percentile(values, 0.05 if side == "lower" else 0.95)
    support = [point for point in points if abs(point[vertical_axis] - height) <= thickness]
    axes = [axis for axis in range(3) if axis != vertical_axis]
    projected = trim_projected([(point[axes[0]], point[axes[1]]) for point in support])
    hull = convex_hull(projected)
    return {
        "side": side,
        "height": height,
        "thickness": thickness,
        "support_count": len(support),
        "support_ratio": len(support) / len(points) if points else 0.0,
        "hull": hull,
        "hull_area": polygon_area(hull),
    }


def build_mesh_vertices(hull: Sequence[Point2], vertical_axis: int, height: float) -> List[Point3]:
    axes = [axis for axis in range(3) if axis != vertical_axis]
    vertices: List[Point3] = []
    for u, v in hull:
        coords = [0.0, 0.0, 0.0]
        coords[vertical_axis] = height
        coords[axes[0]] = u
        coords[axes[1]] = v
        vertices.append((coords[0], coords[1], coords[2]))
    return vertices


def write_floor_mesh(path: Path, vertices: Sequence[Point3]) -> None:
    if len(vertices) < 3:
        raise ValueError("Need at least three vertices to write a floor mesh")
    faces = [(0, index, index + 1) for index in range(1, len(vertices) - 1)]
    lines = [
        "ply",
        "format ascii 1.0",
        "comment generated by floor_plane_completion.py",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for x, y, z in vertices:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} 185 170 135")
    for a, b, c in faces:
        lines.append(f"3 {a} {b} {c}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def complete_floor(
    input_path: Path,
    output_path: Path,
    summary_path: Optional[Path],
    max_points: int,
    floor_side: str,
    thickness_ratio: float,
    hull_padding: float,
) -> Dict[str, Any]:
    properties, rows, io_summary = read_ply_vertices(input_path, max_vertices=max_points)
    xyz = xyz_indices(properties)
    points = [row_xyz(row, xyz) for row in rows]
    sample_stride = int(io_summary.get("sample_stride") or 1)
    if len(points) < 100:
        raise ValueError("Too few sampled points for floor completion")

    vertical_axis = choose_vertical_axis(points)
    candidates = [
        plane_candidate(points, vertical_axis, "lower", thickness_ratio),
        plane_candidate(points, vertical_axis, "upper", thickness_ratio),
    ]
    if floor_side in {"lower", "upper"}:
        chosen = next(candidate for candidate in candidates if candidate["side"] == floor_side)
    else:
        chosen = max(candidates, key=lambda item: (item["hull_area"], item["support_count"]))

    hull = expand_polygon(chosen["hull"], hull_padding)
    vertices = build_mesh_vertices(hull, vertical_axis, chosen["height"])
    write_floor_mesh(output_path, vertices)

    mins = [min(point[axis] for point in points) for axis in range(3)]
    maxs = [max(point[axis] for point in points) for axis in range(3)]
    summary = {
        "status": "ok",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "method": "robust_vertical_axis_floor_band_convex_hull",
        "vertical_axis": ["x", "y", "z"][vertical_axis],
        "floor_side": chosen["side"],
        "floor_height": round(chosen["height"], 6),
        "support_thickness": round(chosen["thickness"], 6),
        "support_count": chosen["support_count"],
        "support_ratio": round(chosen["support_ratio"], 6),
        "hull_vertex_count": len(hull),
        "hull_area": round(polygon_area(hull), 6),
        "hull_padding": hull_padding,
        "bbox_min": [round(value, 6) for value in mins],
        "bbox_max": [round(value, 6) for value in maxs],
        "sample_stride": sample_stride,
        "sampled_count": len(points),
        **io_summary,
        "explanation": (
            "The shortest robust point-cloud axis is treated as vertical. A near-boundary "
            "horizontal support band is converted into a padded floor polygon and written "
            "as a separate inspection mesh for weak-texture or occluded floor regions."
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
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--floor-side", choices=["auto", "lower", "upper"], default="auto")
    parser.add_argument("--thickness-ratio", type=float, default=0.025)
    parser.add_argument("--hull-padding", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = complete_floor(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        max_points=args.max_points,
        floor_side=args.floor_side,
        thickness_ratio=args.thickness_ratio,
        hull_padding=args.hull_padding,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
