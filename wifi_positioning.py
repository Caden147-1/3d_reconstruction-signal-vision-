"""WiFi fingerprint localization and signal-boundary modeling."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from wifi_observations import valid_fingerprint


FILL_RSSI = -110.0


def normalize_mac(mac: str) -> str:
    return "".join(ch for ch in mac.upper() if ch.isalnum())


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def parse_vector(row: Dict[str, str], ap_fields: Sequence[str]) -> Dict[str, float]:
    return {field: safe_float(row.get(field), FILL_RSSI) for field in ap_fields}


def visible_fields(vector: Dict[str, float], threshold: float = FILL_RSSI + 0.5) -> List[str]:
    return [field for field, value in vector.items() if math.isfinite(value) and threshold < value < 0]


def load_fingerprints(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    fields, rows = read_csv(path)
    ap_fields = [field for field in fields if field.startswith("ap_")]
    fingerprints: List[Dict[str, Any]] = []
    for row in rows:
        x = safe_float(row.get("pdr_x"))
        y = safe_float(row.get("pdr_y"))
        if not math.isfinite(x) or not math.isfinite(y) or not valid_fingerprint(row):
            continue
        vector = parse_vector(row, ap_fields)
        fingerprints.append(
            {
                "session": row.get("session", ""),
                "point_index": row.get("point_index", ""),
                "global_point_index": row.get("global_point_index", ""),
                "pdr_x": x,
                "pdr_y": y,
                "wifi_scan_index": row.get("wifi_scan_index", ""),
                "wifi_gap_ms": row.get("wifi_gap_ms", ""),
                "vector": vector,
            }
        )
    return ap_fields, fingerprints


def rssi_distance(
    left: Dict[str, float],
    right: Dict[str, float],
    ap_fields: Sequence[str],
    active_fields: Optional[Sequence[str]] = None,
) -> float:
    fields = list(active_fields or ap_fields)
    if not fields:
        fields = list(ap_fields)
    if not fields:
        return math.inf

    total = 0.0
    weight_sum = 0.0
    for field in fields:
        left_value = left.get(field, FILL_RSSI)
        right_value = right.get(field, FILL_RSSI)
        both_missing = left_value <= FILL_RSSI + 0.5 and right_value <= FILL_RSSI + 0.5
        weight = 0.15 if both_missing else 1.0
        diff = left_value - right_value
        total += weight * diff * diff
        weight_sum += weight
    if weight_sum <= 0:
        return math.inf
    return math.sqrt(total / weight_sum)


def estimate_position(
    scan_vector: Dict[str, float],
    ap_fields: Sequence[str],
    fingerprints: Sequence[Dict[str, Any]],
    k: int = 3,
    active_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    visible = set(visible_fields(scan_vector))
    usable_fields = [field for field in (visible if active_fields is None else active_fields) if field in ap_fields and field in visible]
    if not usable_fields:
        return {
            "status": "no_known_aps", "estimated_x": None, "estimated_y": None,
            "known_ap_count": 0, "neighbors": [],
        }

    neighbors: List[Dict[str, Any]] = []
    for fp in fingerprints:
        distance = rssi_distance(scan_vector, fp["vector"], ap_fields, usable_fields)
        if not math.isfinite(distance):
            continue
        neighbors.append(
            {
                "session": fp["session"],
                "point_index": fp["point_index"],
                "global_point_index": fp["global_point_index"],
                "pdr_x": fp["pdr_x"],
                "pdr_y": fp["pdr_y"],
                "rssi_distance": distance,
            }
        )

    neighbors.sort(key=lambda row: row["rssi_distance"])
    top = neighbors[: max(1, min(k, len(neighbors)))]
    if not top:
        return {
            "status": "no_fingerprints",
            "estimated_x": None,
            "estimated_y": None,
            "known_ap_count": len(usable_fields),
            "neighbors": [],
        }

    zero_distance_neighbors = [row for row in neighbors if row["rssi_distance"] <= 1e-9]
    if zero_distance_neighbors:
        exact_x = sum(row["pdr_x"] for row in zero_distance_neighbors) / len(zero_distance_neighbors)
        exact_y = sum(row["pdr_y"] for row in zero_distance_neighbors) / len(zero_distance_neighbors)
        return {
            "status": "ok",
            "estimated_x": exact_x,
            "estimated_y": exact_y,
            "known_ap_count": len(usable_fields),
            "best_distance": 0.0,
            "confidence": round(1.0 / len(zero_distance_neighbors), 6),
            "ambiguity_count": len(zero_distance_neighbors),
            "neighbors": top,
        }

    weights = [1.0 / max(row["rssi_distance"], 1e-6) for row in top]
    weight_sum = sum(weights)
    x = sum(weight * row["pdr_x"] for weight, row in zip(weights, top)) / weight_sum
    y = sum(weight * row["pdr_y"] for weight, row in zip(weights, top)) / weight_sum
    confidence = weights[0] / weight_sum if weight_sum else 0.0
    return {
        "status": "ok",
        "estimated_x": x,
        "estimated_y": y,
        "known_ap_count": len(usable_fields),
        "best_distance": top[0]["rssi_distance"],
        "confidence": confidence,
        "ambiguity_count": 1,
        "neighbors": top,
    }


def scan_payload_to_vector(payload: Any, ap_fields: Sequence[str]) -> Tuple[Dict[str, float], List[str], List[str]]:
    vector = {field: FILL_RSSI for field in ap_fields}
    known_fields: List[str] = []
    unknown_macs: List[str] = []
    if not isinstance(payload, list):
        return vector, known_fields, unknown_macs

    field_set = set(ap_fields)
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        mac = normalize_mac(str(entry.get("mac", "")))
        if not mac:
            continue
        field = f"ap_{mac}"
        rssi = safe_float(entry.get("dbm", entry.get("rssi")), FILL_RSSI)
        if field in field_set:
            vector[field] = max(vector.get(field, FILL_RSSI), rssi)
            if field not in known_fields:
                known_fields.append(field)
        else:
            unknown_macs.append(mac)
    return vector, known_fields, unknown_macs


def parse_scan_json(path: Path, ap_fields: Sequence[str]) -> Tuple[Dict[str, float], List[str], List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return scan_payload_to_vector(payload, ap_fields)


def parse_scan_csv(path: Path, ap_fields: Sequence[str]) -> Tuple[Dict[str, float], List[str], List[str]]:
    fields, rows = read_csv(path)
    payload = []
    for row in rows:
        mac = row.get("mac") or row.get("bssid") or row.get("BSSID")
        rssi = row.get("dbm") or row.get("rssi") or row.get("RSSI")
        payload.append({"mac": mac, "dbm": rssi})
    return scan_payload_to_vector(payload, ap_fields)


def leave_one_out_evaluation(
    ap_fields: Sequence[str],
    fingerprints: Sequence[Dict[str, Any]],
    k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, fp in enumerate(fingerprints):
        train = [item for item_idx, item in enumerate(fingerprints) if item_idx != idx]
        active = visible_fields(fp["vector"])
        estimate = estimate_position(fp["vector"], ap_fields, train, k=k, active_fields=active)
        est_x = estimate.get("estimated_x")
        est_y = estimate.get("estimated_y")
        error = ""
        if est_x is not None and est_y is not None:
            error = math.hypot(float(est_x) - fp["pdr_x"], float(est_y) - fp["pdr_y"])
        rows.append(
            {
                "session": fp["session"],
                "point_index": fp["point_index"],
                "global_point_index": fp["global_point_index"],
                "true_x": round(fp["pdr_x"], 6),
                "true_y": round(fp["pdr_y"], 6),
                "estimated_x": "" if est_x is None else round(float(est_x), 6),
                "estimated_y": "" if est_y is None else round(float(est_y), 6),
                "error_m": "" if error == "" else round(float(error), 6),
                "best_distance": estimate.get("best_distance", ""),
                "confidence": estimate.get("confidence", ""),
                "ambiguity_count": estimate.get("ambiguity_count", ""),
                "known_ap_count": estimate.get("known_ap_count", ""),
                "neighbor_points": ";".join(str(row["global_point_index"]) for row in estimate.get("neighbors", [])),
            }
        )

    errors = [float(row["error_m"]) for row in rows if row["error_m"] != ""]
    summary = {
        "evaluated_points": len(errors),
        "mean_error_m": round(sum(errors) / len(errors), 6) if errors else None,
        "median_error_m": round(statistics.median(errors), 6) if errors else None,
        "max_error_m": round(max(errors), 6) if errors else None,
    }
    return rows, summary


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


def point_in_polygon(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i, point_i in enumerate(polygon):
        xi, yi = point_i
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def distance_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    cx = ax + ratio * dx
    cy = ay + ratio * dy
    return math.hypot(px - cx, py - cy)


def distance_to_polygon(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> float:
    if len(polygon) < 2:
        return math.inf
    distances = []
    for idx, start in enumerate(polygon):
        end = polygon[(idx + 1) % len(polygon)]
        distances.append(distance_to_segment(x, y, start[0], start[1], end[0], end[1]))
    return min(distances)


def idw_interpolate(
    x: float,
    y: float,
    fingerprints: Sequence[Dict[str, Any]],
    ap_fields: Sequence[str],
    power: float = 2.0,
) -> Dict[str, float]:
    exact = next(
        (fp for fp in fingerprints if math.hypot(x - fp["pdr_x"], y - fp["pdr_y"]) <= 1e-9),
        None,
    )
    if exact is not None:
        return dict(exact["vector"])

    weights = []
    for fp in fingerprints:
        distance = max(math.hypot(x - fp["pdr_x"], y - fp["pdr_y"]), 1e-6)
        weights.append(1.0 / (distance**power))
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return {field: FILL_RSSI for field in ap_fields}

    vector: Dict[str, float] = {}
    for field in ap_fields:
        vector[field] = sum(weight * fp["vector"].get(field, FILL_RSSI) for weight, fp in zip(weights, fingerprints)) / weight_sum
    return vector


def signal_metrics(vector: Dict[str, float]) -> Dict[str, Any]:
    visible = [value for value in vector.values() if value > FILL_RSSI + 0.5]
    strong = [value for value in visible if value >= -80.0]
    top = sorted(visible, reverse=True)[:3]
    strongest_field = ""
    strongest_rssi = ""
    if vector:
        strongest_field, strongest_value = max(vector.items(), key=lambda item: item[1])
        strongest_rssi = round(strongest_value, 3)
    mean_top3 = sum(top) / len(top) if top else FILL_RSSI
    score = max(0.0, min(1.0, (mean_top3 - FILL_RSSI) / 50.0))
    return {
        "visible_ap_count": len(visible),
        "strong_ap_count": len(strong),
        "strongest_ap": strongest_field.removeprefix("ap_"),
        "strongest_rssi": strongest_rssi,
        "mean_top3_rssi": round(mean_top3, 3),
        "signal_score": round(score, 6),
    }


def build_signal_grid(
    ap_fields: Sequence[str],
    fingerprints: Sequence[Dict[str, Any]],
    cell_size_m: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    points = [(fp["pdr_x"], fp["pdr_y"]) for fp in fingerprints]
    hull = convex_hull(points)
    if not points or len(hull) < 3 or cell_size_m <= 0:
        return [], [], {"grid_rows": 0, "hull_point_count": len(hull)}

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = math.floor(min(xs) / cell_size_m) * cell_size_m
    max_x = math.ceil(max(xs) / cell_size_m) * cell_size_m
    min_y = math.floor(min(ys) / cell_size_m) * cell_size_m
    max_y = math.ceil(max(ys) / cell_size_m) * cell_size_m

    grid_rows: List[Dict[str, Any]] = []
    y = min_y
    while y <= max_y + 1e-9:
        x = min_x
        while x <= max_x + 1e-9:
            if point_in_polygon(x, y, hull):
                vector = idw_interpolate(x, y, fingerprints, ap_fields)
                nearest_distance = min(math.hypot(x - fp["pdr_x"], y - fp["pdr_y"]) for fp in fingerprints)
                edge_distance = distance_to_polygon(x, y, hull)
                metrics = signal_metrics(vector)
                metrics.update(
                    {
                        "grid_x": round(x, 6),
                        "grid_y": round(y, 6),
                        "nearest_fingerprint_distance_m": round(nearest_distance, 6),
                        "boundary_distance_m": round(edge_distance, 6),
                        "inside_confidence": round(max(0.0, min(1.0, math.exp(-nearest_distance / 4.0))), 6),
                    }
                )
                grid_rows.append(metrics)
            x += cell_size_m
        y += cell_size_m

    boundary_rows: List[Dict[str, Any]] = []
    for order, (x, y) in enumerate(hull, start=1):
        vector = idw_interpolate(x, y, fingerprints, ap_fields)
        metrics = signal_metrics(vector)
        source = min(fingerprints, key=lambda fp: math.hypot(x - fp["pdr_x"], y - fp["pdr_y"]))
        metrics.update(
            {
                "boundary_order": order,
                "pdr_x": round(x, 6),
                "pdr_y": round(y, 6),
                "nearest_point_index": source["point_index"],
                "nearest_global_point_index": source["global_point_index"],
                "source": "wifi_fingerprint_convex_hull_for_rssi_interpolation_not_physical_boundary",
            }
        )
        boundary_rows.append(metrics)

    summary = {
        "grid_rows": len(grid_rows),
        "cell_size_m": cell_size_m,
        "hull_point_count": len(hull),
        "domain_semantics": "wifi_fingerprint_interpolation_envelope_not_route_or_room_boundary",
        "min_x": round(min_x, 6),
        "max_x": round(max_x, 6),
        "min_y": round(min_y, 6),
        "max_y": round(max_y, 6),
    }
    return grid_rows, boundary_rows, summary


def run_wifi_positioning_pipeline(
    fingerprint_path: Path,
    output_dir: Path,
    grid_cell_size_m: float = 1.0,
    localization_k: int = 3,
) -> Dict[str, Any]:
    ap_fields, fingerprints = load_fingerprints(fingerprint_path)
    if not ap_fields or not fingerprints:
        for filename in (
            "wifi_localization_model.json", "wifi_localization_eval.csv",
            "wifi_signal_grid.csv", "wifi_signal_boundary.csv",
        ):
            (output_dir / filename).unlink(missing_ok=True)
        summary = {
            "status": "no_valid_fingerprints" if fingerprint_path.exists() else "missing_input",
            "explanation": "Only fresh, timestamp-validated checkpoint fingerprints are usable; stale generated localization outputs were removed.",
            "fingerprint_path": str(fingerprint_path),
            "ap_count": len(ap_fields),
            "fingerprint_count": len(fingerprints),
        }
        write_json(output_dir / "wifi_positioning_summary.json", summary)
        return summary

    eval_rows, eval_summary = leave_one_out_evaluation(ap_fields, fingerprints, localization_k)
    grid_rows, boundary_rows, grid_summary = build_signal_grid(ap_fields, fingerprints, grid_cell_size_m)

    write_csv(
        output_dir / "wifi_localization_eval.csv",
        [
            "session",
            "point_index",
            "global_point_index",
            "true_x",
            "true_y",
            "estimated_x",
            "estimated_y",
            "error_m",
            "best_distance",
            "confidence",
            "ambiguity_count",
            "known_ap_count",
            "neighbor_points",
        ],
        eval_rows,
    )
    write_csv(
        output_dir / "wifi_signal_grid.csv",
        [
            "grid_x",
            "grid_y",
            "nearest_fingerprint_distance_m",
            "boundary_distance_m",
            "inside_confidence",
            "visible_ap_count",
            "strong_ap_count",
            "strongest_ap",
            "strongest_rssi",
            "mean_top3_rssi",
            "signal_score",
        ],
        grid_rows,
    )
    write_csv(
        output_dir / "wifi_signal_boundary.csv",
        [
            "boundary_order",
            "pdr_x",
            "pdr_y",
            "nearest_point_index",
            "nearest_global_point_index",
            "source",
            "visible_ap_count",
            "strong_ap_count",
            "strongest_ap",
            "strongest_rssi",
            "mean_top3_rssi",
            "signal_score",
        ],
        boundary_rows,
    )

    model = {
        "model_type": "weighted_knn_wifi_fingerprint",
        "fingerprint_path": str(fingerprint_path),
        "missing_rssi_fill": FILL_RSSI,
        "localization_k": localization_k,
        "ap_fields": ap_fields,
        "fingerprints": [
            {
                "session": fp["session"],
                "point_index": fp["point_index"],
                "global_point_index": fp["global_point_index"],
                "pdr_x": fp["pdr_x"],
                "pdr_y": fp["pdr_y"],
                "wifi_scan_index": fp["wifi_scan_index"],
                "vector": fp["vector"],
            }
            for fp in fingerprints
        ],
    }
    write_json(output_dir / "wifi_localization_model.json", model)

    summary = {
        "status": "ok",
        "fingerprint_path": str(fingerprint_path),
        "ap_count": len(ap_fields),
        "fingerprint_count": len(fingerprints),
        "localization_k": localization_k,
        "grid": grid_summary,
        "leave_one_out": eval_summary,
        "outputs": {
            "model": "wifi_localization_model.json",
            "evaluation": "wifi_localization_eval.csv",
            "signal_grid": "wifi_signal_grid.csv",
            "signal_boundary": "wifi_signal_boundary.csv",
        },
    }
    write_json(output_dir / "wifi_positioning_summary.json", summary)
    return summary
