"""Build a continuous, uncertainty-aware contour from an ordered IMU/PDR path.

The output is a 2D route corridor for matching, alignment diagnostics, and capture
coverage inspection. It intentionally does not infer walls, floors, or 3D mesh
geometry from the walking path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_pdr_rows(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    rows_by_session: Dict[str, List[Dict[str, Any]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x = safe_float(row.get("pdr_x"))
            y = safe_float(row.get("pdr_y"))
            elapsed_ts = safe_int(row.get("elapsed_ts"), -1)
            if not row.get("session") or elapsed_ts < 0 or not math.isfinite(x) or not math.isfinite(y):
                continue
            rows_by_session.setdefault(row["session"], []).append(
                {
                    "session": row["session"],
                    "point_type": row.get("point_type", ""),
                    "segment_id": row.get("segment_id", ""),
                    "step_index": row.get("step_index", ""),
                    "elapsed_ts": elapsed_ts,
                    "epoch_ts": safe_int(row.get("epoch_ts"), 0),
                    "pdr_x": x,
                    "pdr_y": y,
                }
            )
    for rows in rows_by_session.values():
        rows.sort(
            key=lambda row: (
                row["elapsed_ts"],
                0 if row["point_type"] in {"START", "RESTART"} else 1,
                safe_int(row.get("step_index")),
            )
        )
    return rows_by_session


def point_distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    return math.hypot(float(right["pdr_x"]) - float(left["pdr_x"]), float(right["pdr_y"]) - float(left["pdr_y"]))


def link_status(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    max_gap_ms: int,
    max_link_distance_m: float,
    stationary_tolerance_m: float,
) -> Tuple[bool, str, float, int]:
    distance = point_distance(previous, current)
    gap_ms = max(0, int(current["elapsed_ts"]) - int(previous["elapsed_ts"]))
    if distance <= stationary_tolerance_m:
        return True, "stationary_checkpoint", distance, gap_ms
    if distance > max_link_distance_m:
        return False, "distance_break", distance, gap_ms
    if gap_ms > max_gap_ms:
        return True, "delayed_observed_step_link", distance, gap_ms
    return True, "observed_step_link", distance, gap_ms


def direction_for_vertex(vertices: Sequence[Dict[str, Any]], index: int) -> Tuple[float, float]:
    current = vertices[index]
    before: Optional[Tuple[float, float]] = None
    after: Optional[Tuple[float, float]] = None
    for candidate_index in range(index - 1, -1, -1):
        candidate = vertices[candidate_index]
        dx = float(current["pdr_x"]) - float(candidate["pdr_x"])
        dy = float(current["pdr_y"]) - float(candidate["pdr_y"])
        if math.hypot(dx, dy) > 1e-8:
            before = (dx, dy)
            break
    for candidate_index in range(index + 1, len(vertices)):
        candidate = vertices[candidate_index]
        dx = float(candidate["pdr_x"]) - float(current["pdr_x"])
        dy = float(candidate["pdr_y"]) - float(current["pdr_y"])
        if math.hypot(dx, dy) > 1e-8:
            after = (dx, dy)
            break
    dx = (before[0] if before else 0.0) + (after[0] if after else 0.0)
    dy = (before[1] if before else 0.0) + (after[1] if after else 0.0)
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1e-8:
        return 1.0, 0.0
    return dx / magnitude, dy / magnitude


def add_corridor_offsets(vertices: List[Dict[str, Any]], corridor_half_width_m: float) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for vertex in vertices:
        grouped.setdefault((str(vertex["session"]), str(vertex["contour_id"])), []).append(vertex)
    for group in grouped.values():
        for index, vertex in enumerate(group):
            dx, dy = direction_for_vertex(group, index)
            normal_x, normal_y = -dy, dx
            vertex["left_x"] = round(float(vertex["pdr_x"]) + normal_x * corridor_half_width_m, 6)
            vertex["left_y"] = round(float(vertex["pdr_y"]) + normal_y * corridor_half_width_m, 6)
            vertex["right_x"] = round(float(vertex["pdr_x"]) - normal_x * corridor_half_width_m, 6)
            vertex["right_y"] = round(float(vertex["pdr_y"]) - normal_y * corridor_half_width_m, 6)


def build_trajectory_contour(
    input_path: Path,
    contour_path: Path,
    edges_path: Path,
    summary_path: Path,
    corridor_half_width_m: float = 0.75,
    max_gap_s: float = 10.0,
    max_link_distance_m: float = 1.5,
    stationary_tolerance_m: float = 0.05,
) -> Dict[str, Any]:
    if corridor_half_width_m <= 0:
        raise ValueError("corridor_half_width_m must be positive")
    if max_gap_s <= 0 or max_link_distance_m <= 0 or stationary_tolerance_m < 0:
        raise ValueError("Contour link thresholds must be positive")

    rows_by_session = read_pdr_rows(input_path)
    vertices: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    contour_count = 0
    break_counts: Dict[str, int] = {"time_gap_break": 0, "distance_break": 0}
    delayed_link_count = 0
    observed_length_m = 0.0

    for session, rows in sorted(rows_by_session.items()):
        previous: Optional[Dict[str, Any]] = None
        contour_id = ""
        contour_order = 0
        contour_distance_m = 0.0
        trajectory_distance_m = 0.0
        previous_vertex: Optional[Dict[str, Any]] = None
        for row in rows:
            connected = False
            status = "contour_start"
            distance_m = 0.0
            gap_ms = 0
            if previous is not None:
                connected, status, distance_m, gap_ms = link_status(
                    previous,
                    row,
                    max_gap_ms=int(round(max_gap_s * 1000.0)),
                    max_link_distance_m=max_link_distance_m,
                    stationary_tolerance_m=stationary_tolerance_m,
                )
                if status == "delayed_observed_step_link":
                    delayed_link_count += 1
            if previous is None or not connected:
                if previous is not None:
                    break_counts[status] = break_counts.get(status, 0) + 1
                contour_count += 1
                contour_id = f"{session}__{contour_count:03d}"
                contour_order = 0
                contour_distance_m = 0.0
                status = "contour_start" if previous is None else status
            else:
                contour_distance_m += distance_m
                trajectory_distance_m += distance_m
                if distance_m > stationary_tolerance_m and previous_vertex is not None:
                    edges.append(
                        {
                            "session": session,
                            "contour_id": contour_id,
                            "edge_index": len(
                                [edge for edge in edges if edge["session"] == session and edge["contour_id"] == contour_id]
                            )
                            + 1,
                            "start_vertex_id": previous_vertex["vertex_id"],
                            "end_vertex_id": len(vertices) + 1,
                            "start_elapsed_ts": previous_vertex["elapsed_ts"],
                            "end_elapsed_ts": row["elapsed_ts"],
                            "length_m": round(distance_m, 6),
                            "elapsed_gap_ms": gap_ms,
                            "source": "ordered_imu_pdr_step_chain",
                        }
                    )
                    observed_length_m += distance_m

            contour_order += 1
            vertex = {
                "vertex_id": len(vertices) + 1,
                "session": session,
                "contour_id": contour_id,
                "contour_order": contour_order,
                "elapsed_ts": row["elapsed_ts"],
                "epoch_ts": row["epoch_ts"],
                "source_point_type": row["point_type"],
                "source_segment_id": row["segment_id"],
                "source_step_index": row["step_index"],
                "pdr_x": round(float(row["pdr_x"]), 6),
                "pdr_y": round(float(row["pdr_y"]), 6),
                "contour_distance_m": round(contour_distance_m, 6),
                "trajectory_distance_m": round(trajectory_distance_m, 6),
                "link_distance_m": round(distance_m, 6),
                "link_gap_ms": gap_ms,
                "link_status": status,
                "corridor_half_width_m": corridor_half_width_m,
            }
            vertices.append(vertex)
            previous = row
            previous_vertex = vertex

    add_corridor_offsets(vertices, corridor_half_width_m)
    contour_fields = [
        "vertex_id",
        "session",
        "contour_id",
        "contour_order",
        "elapsed_ts",
        "epoch_ts",
        "source_point_type",
        "source_segment_id",
        "source_step_index",
        "pdr_x",
        "pdr_y",
        "contour_distance_m",
        "trajectory_distance_m",
        "link_distance_m",
        "link_gap_ms",
        "link_status",
        "corridor_half_width_m",
        "left_x",
        "left_y",
        "right_x",
        "right_y",
    ]
    edge_fields = [
        "session",
        "contour_id",
        "edge_index",
        "start_vertex_id",
        "end_vertex_id",
        "start_elapsed_ts",
        "end_elapsed_ts",
        "length_m",
        "elapsed_gap_ms",
        "source",
    ]
    write_csv(contour_path, contour_fields, vertices)
    write_csv(edges_path, edge_fields, edges)
    summary = {
        "status": "ok" if vertices else "empty_path",
        "input": str(input_path.resolve()),
        "contour_output": str(contour_path.resolve()),
        "edges_output": str(edges_path.resolve()),
        "input_session_count": len(rows_by_session),
        "contour_vertex_count": len(vertices),
        "contour_count": contour_count,
        "observed_edge_count": len(edges),
        "observed_route_length_m": round(observed_length_m, 6),
        "break_counts": break_counts,
        "delayed_observed_link_count": delayed_link_count,
        "corridor_half_width_m": corridor_half_width_m,
        "max_gap_s": max_gap_s,
        "max_link_distance_m": max_link_distance_m,
        "stationary_tolerance_m": stationary_tolerance_m,
        "explanation": (
            "The contour follows only time-ordered IMU/PDR samples. Long sampling gaps are "
            "recorded, but links still require consecutive observed PDR samples within the "
            "configured distance bound. WiFi checkpoints can be attached as sparse fingerprint "
            "anchors; neither WiFi nor the corridor infers room walls, unwalked area, or "
            "synthetic 3D surfaces."
        ),
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edges-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--corridor-half-width-m", type=float, default=0.75)
    parser.add_argument("--max-gap-s", type=float, default=10.0)
    parser.add_argument("--max-link-distance-m", type=float, default=1.5)
    parser.add_argument("--stationary-tolerance-m", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_trajectory_contour(
        input_path=args.input,
        contour_path=args.output,
        edges_path=args.edges_output,
        summary_path=args.summary,
        corridor_half_width_m=args.corridor_half_width_m,
        max_gap_s=args.max_gap_s,
        max_link_distance_m=args.max_link_distance_m,
        stationary_tolerance_m=args.stationary_tolerance_m,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
