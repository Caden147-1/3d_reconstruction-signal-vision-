"""Bind ordered image captures to WiFi checkpoint rows."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from wifi_observations import (
    MAX_WIFI_AGE_MS, MISSING_RSSI, WIFI_QUALITY_FIELDS, WIFI_ASSOCIATION_FIELDS,
    finite_number, measurement_gap_ms, valid_fingerprint,
    aggregate_checkpoint_fingerprints, fingerprint_confidence,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PDR_BINDING_FIELDS = [
    "pdr_x", "pdr_y", "pdr_interp_status", "pdr_interp_gap_ms",
    "pdr_left_type", "pdr_right_type", "trajectory_contour_id",
    "trajectory_contour_distance_m", "trajectory_distance_m", "trajectory_interp_status",
]


@dataclass
class WifiPoint:
    session: str
    point_index: int
    global_point_index: int
    event: str
    elapsed_ts: int
    epoch_ts: int
    stationary_end_elapsed_ts: Optional[int] = None


def parse_tab_json(path: Path) -> Iterable[Tuple[int, int, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                print(f"[warn] skip malformed line {path}:{line_no}")
                continue
            try:
                elapsed_ts = int(parts[0])
                epoch_ts = int(parts[1])
                payload = json.loads(parts[2])
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"[warn] skip unparsable line {path}:{line_no}: {exc}")
                continue
            yield elapsed_ts, epoch_ts, payload


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


def natural_sort_key(path: Path) -> List[Tuple[int, Any]]:
    parts = re.split(r"(\d+)", path.name.lower())
    key: List[Tuple[int, Any]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def image_number(path: Path) -> str:
    matches = re.findall(r"\d+", path.stem)
    return matches[-1] if matches else ""


def find_image_files(pic_dir: Path) -> List[Path]:
    if not pic_dir.exists() or not pic_dir.is_dir():
        return []
    return sorted(
        (path for path in pic_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_sort_key,
    )


def read_wifi_points(session_dirs: Sequence[Path], point_event: str) -> List[WifiPoint]:
    points: List[WifiPoint] = []
    global_point_index = 0
    for session_dir in session_dirs:
        point_index = 0
        events = list(parse_tab_json(session_dir / "movements.dat"))
        for event_pos, (elapsed_ts, epoch_ts, payload) in enumerate(events):
            event = str(payload)
            if event != point_event:
                continue
            point_index += 1
            global_point_index += 1
            stationary_end = None
            if event == "CHECKPOINT_PAUSE" and event_pos + 1 < len(events):
                next_ts, _, next_event = events[event_pos + 1]
                if next_event == "RESTART" and next_ts > elapsed_ts:
                    stationary_end = next_ts
            points.append(
                WifiPoint(
                    session=session_dir.name,
                    point_index=point_index,
                    global_point_index=global_point_index,
                    event=event,
                    elapsed_ts=elapsed_ts,
                    epoch_ts=epoch_ts,
                    stationary_end_elapsed_ts=stationary_end,
                )
            )
    return points


def read_feature_rows(path: Path) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    if not path.exists():
        return [], {}

    rows_by_session: Dict[str, List[Dict[str, Any]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            try:
                elapsed_ts = int(row["elapsed_ts"])
            except (KeyError, TypeError, ValueError):
                continue
            row["_elapsed_ts_int"] = elapsed_ts
            rows_by_session.setdefault(row.get("session", ""), []).append(row)

    for rows in rows_by_session.values():
        rows.sort(key=lambda row: row["_elapsed_ts_int"])
    return fieldnames, rows_by_session


def group_checkpoint_images(
    images: Sequence[Path], points: Sequence[WifiPoint], images_per_point: int,
    mapping_path: Optional[Path] = None,
    *, mode: str = "fuzzy", tolerance: Optional[int] = None,
) -> Tuple[List[List[Path]], List[Path], Dict[str, Any]]:
    """Bind explicit filenames without shifting later stops when a photo is absent.

    CSV order defines the view order within each stop; omitted images remain
    visual-only. Sensor/photo clock synchronization is deliberately unnecessary.
    """
    if mode not in {"fuzzy", "fixed"} or images_per_point <= 0:
        raise ValueError("Photo grouping requires fuzzy/fixed mode and a positive reference count")
    if tolerance is not None and not 0 <= tolerance < images_per_point:
        raise ValueError("Photo group tolerance must be nonnegative and smaller than the reference count")
    if mapping_path is None and mode == "fuzzy":
        from photo_grouping import fuzzy_checkpoint_groups
        return fuzzy_checkpoint_groups(images, len(points), images_per_point, tolerance)
    if mapping_path is None:
        return ([list(images[i * images_per_point:(i + 1) * images_per_point])
                 for i in range(len(points))], list(images[len(points) * images_per_point:]),
                {"method": "fixed_count", "source": None})
    data = mapping_path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    required = {"session", "point_index", "image_name"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Photo checkpoint map requires columns {sorted(required)}: {mapping_path}")
    point_indices = {(point.session, point.point_index): i for i, point in enumerate(points)}
    image_by_name = {image.name.casefold(): image for image in images}
    if len(point_indices) != len(points) or len(image_by_name) != len(images):
        raise ValueError("Ambiguous checkpoint keys or case-insensitive photo names")
    groups: List[List[Path]] = [[] for _ in points]
    used: set[str] = set()
    for line, row in enumerate(reader, start=2):
        try:
            key = (row["session"].strip(), int(row["point_index"]))
            name = row["image_name"].strip().casefold()
            if key not in point_indices:
                raise ValueError(f"unknown checkpoint {key}")
            if name not in image_by_name:
                raise ValueError(f"photo not found in capture folder: {row['image_name']!r}")
            if name in used:
                raise ValueError(f"photo assigned more than once: {row['image_name']!r}")
        except (TypeError, AttributeError, ValueError) as exc:
            raise ValueError(f"Invalid photo checkpoint map {mapping_path}, row {line}: {exc}") from exc
        groups[point_indices[key]].append(image_by_name[name])
        used.add(name)
    if not used:
        raise ValueError(f"Photo checkpoint map has no assignments: {mapping_path}")
    return groups, [image for image in images if image.name.casefold() not in used], {
        "method": "explicit_photo_map", "source": str(mapping_path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def nearest_fresh_fingerprint(
    rows: Sequence[Dict[str, Any]], target_elapsed_ts: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    best = min(rows, key=lambda row: measurement_gap_ms(row, target_elapsed_ts), default=None)
    gap = float("inf") if best is None else measurement_gap_ms(best, target_elapsed_ts)
    return (best, str(int(gap))) if gap <= MAX_WIFI_AGE_MS else (None, "")


def checkpoint_fingerprint(
    rows: Sequence[Dict[str, Any]], point: WifiPoint,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Prefer measurements inside a documented stay at this checkpoint.

    All AP measurements must fall before RESTART; a cached scan from the
    approach or a scan after departure cannot establish a stationary binding.
    Photo capture times are intentionally not part of this association.
    """
    if point.stationary_end_elapsed_ts is not None:
        stationary = [row for row in rows if valid_fingerprint(row)
                      and point.elapsed_ts <= finite_number(row.get("wifi_measurement_min_elapsed_ts"))
                      and finite_number(row.get("wifi_measurement_max_elapsed_ts")) < point.stationary_end_elapsed_ts]
        if stationary:
            best = min(stationary, key=lambda row: measurement_gap_ms(row, point.elapsed_ts))
            return best, str(int(measurement_gap_ms(best, point.elapsed_ts))), "documented_checkpoint_stay"
    best, gap = nearest_fresh_fingerprint(rows, point.elapsed_ts)
    return best, gap, "near_checkpoint_measurement" if best is not None else "unavailable"


def mapping_status(feature_row: Optional[Dict[str, Any]], missing_slots: int) -> str:
    parts: List[str] = []
    if feature_row is None:
        parts.append("missing_wifi")
    if missing_slots:
        parts.append("missing_images")
    return ";".join(parts) if parts else "ok"


def read_checkpoint_pdr(path: Path) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """Read event positions produced by build_pdr_tables, independent of WiFi scans."""
    positions: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    if not path.exists():
        return positions
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                elapsed_ts = int(row["elapsed_ts"])
                x, y = float(row["pdr_x"]), float(row["pdr_y"])
            except (KeyError, TypeError, ValueError):
                continue
            if not row.get("session") or not all(math.isfinite(value) for value in (x, y)):
                continue
            event = row.get("source_point_type", row.get("point_type", ""))
            positions[(row["session"], elapsed_ts, event)] = {
                "pdr_x": x, "pdr_y": y,
                "pdr_interp_status": "exact_checkpoint", "pdr_interp_gap_ms": 0,
                "pdr_left_type": event, "pdr_right_type": event,
                "trajectory_contour_id": row.get("contour_id", ""),
                "trajectory_contour_distance_m": row.get("contour_distance_m", ""),
                "trajectory_distance_m": row.get("trajectory_distance_m", ""),
                "trajectory_interp_status": "exact_checkpoint" if row.get("contour_id") else "contour_unavailable",
            }
    return positions


def build_image_wifi_tables(
    root: Path,
    session_dirs: Sequence[Path],
    output_dir: Path,
    pic_dir: Path,
    images_per_point: int,
    point_event: str,
    photo_checkpoint_map: Optional[Path] = None,
    photo_grouping: str = "fuzzy",
    photo_group_tolerance: Optional[int] = None,
) -> Dict[str, Any]:
    """Create image-to-WiFi mapping tables from ordered photos and checkpoint events."""
    if images_per_point <= 0:
        raise ValueError("--images-per-wifi-point must be greater than zero")

    resolved_pic_dir = pic_dir if pic_dir.is_absolute() else root / pic_dir
    images = find_image_files(resolved_pic_dir)
    points = read_wifi_points(session_dirs, point_event)
    mapping_path = root / photo_checkpoint_map if photo_checkpoint_map is not None else None
    groups, unassigned_images, grouping = group_checkpoint_images(
        images, points, images_per_point, mapping_path, mode=photo_grouping, tolerance=photo_group_tolerance)
    variable_groups = grouping["method"] != "fixed_count"
    group_sizes = [len(group) if variable_groups else images_per_point for group in groups]

    feature_path = output_dir / "wifi_pdr_features.csv"
    if not feature_path.exists():
        feature_path = output_dir / "wifi_wide_selected.csv"
    feature_fields, features_by_session = read_feature_rows(feature_path)
    aggregated, aggregation_summary = aggregate_checkpoint_fingerprints(output_dir / "wifi_long.csv", points)
    feature_fields += sorted({field for row in aggregated.values() for field in row
                              if field.startswith("ap_") and field not in feature_fields})
    copied_feature_fields = [
        field
        for field in feature_fields
        if field not in {"session", "scan_index", "elapsed_ts", "epoch_ts", "wifi_gap_ms", *WIFI_ASSOCIATION_FIELDS}
    ]
    copied_feature_fields += [field for field in WIFI_QUALITY_FIELDS if field not in copied_feature_fields]
    copied_feature_fields += [field for field in PDR_BINDING_FIELDS if field not in copied_feature_fields]
    contour_path = output_dir / "pdr_trajectory_contour.csv"
    pdr_path = contour_path if contour_path.exists() else output_dir / "pdr_path.csv"
    checkpoint_positions = read_checkpoint_pdr(pdr_path)

    long_base_fields = [
        "session",
        "point_index",
        "global_point_index",
        "point_event",
        "point_elapsed_ts",
        "point_epoch_ts",
        "image_slot",
        "image_name",
        "image_path",
        "image_number",
        "wifi_scan_index",
        "wifi_elapsed_ts",
        "wifi_epoch_ts",
        "wifi_gap_ms",
        "mapping_status",
        "pdr_reference_elapsed_ts",
        "image_association_basis",
        "image_group_size",
        "image_grouping_method",
        *WIFI_ASSOCIATION_FIELDS,
    ]
    point_base_fields = [
        "session",
        "point_index",
        "global_point_index",
        "point_event",
        "point_elapsed_ts",
        "point_epoch_ts",
        "wifi_scan_index",
        "wifi_elapsed_ts",
        "wifi_epoch_ts",
        "wifi_gap_ms",
        "mapping_status",
        "pdr_reference_elapsed_ts",
        "image_association_basis",
        "image_group_size",
        "image_grouping_method",
        *WIFI_ASSOCIATION_FIELDS,
    ]
    image_fields: List[str] = []
    for slot in range(1, max(group_sizes, default=0) + 1):
        image_fields.extend([f"image_{slot}", f"image_{slot}_path", f"image_{slot}_number"])

    long_rows: List[Dict[str, Any]] = []
    point_rows: List[Dict[str, Any]] = []
    mapped_image_count = 0
    complete_point_count = 0
    strict_fresh_checkpoint_count = 0

    for point_pos, point in enumerate(points):
        group_images = groups[point_pos]
        group_size = group_sizes[point_pos]
        missing_slots = group_size - len(group_images)
        if missing_slots == 0 and group_images:
            complete_point_count += 1

        # Every capture event has its own PDR path row. A nearby WiFi logger
        # timestamp can be on a different step, or absent for an entire session.
        pdr_state = checkpoint_positions.get((point.session, point.elapsed_ts, point.event))
        feature_row, wifi_gap_ms, wifi_basis = checkpoint_fingerprint(
            features_by_session.get(point.session, []),
            point,
        )
        strict_fresh_checkpoint_count += feature_row is not None
        aggregate = aggregated.get(point.global_point_index)
        if aggregate is not None and fingerprint_confidence(aggregate) > 0:
            feature_row = aggregate
            wifi_gap_ms = str(aggregate["wifi_gap_ms"])
            wifi_basis = aggregate["wifi_association_basis"]
        status = mapping_status(feature_row, missing_slots or not group_images)

        row_common: Dict[str, Any] = {
            "session": point.session,
            "point_index": point.point_index,
            "global_point_index": point.global_point_index,
            "point_event": point.event,
            "point_elapsed_ts": point.elapsed_ts,
            "point_epoch_ts": point.epoch_ts,
            "wifi_scan_index": "" if feature_row is None else feature_row.get("scan_index", ""),
            "wifi_elapsed_ts": "" if feature_row is None else feature_row.get("elapsed_ts", ""),
            "wifi_epoch_ts": "" if feature_row is None else feature_row.get("epoch_ts", ""),
            "wifi_gap_ms": wifi_gap_ms,
            "mapping_status": status,
            "pdr_reference_elapsed_ts": point.elapsed_ts if pdr_state is not None else "",
            "image_association_basis": "ordered_checkpoint_same_location",
            "image_group_size": group_size,
            "image_grouping_method": grouping["method"],
            "wifi_association_basis": wifi_basis,
            "wifi_stationary_start_elapsed_ts": point.elapsed_ts if wifi_basis in {"documented_checkpoint_stay", "aggregated_stationary"} else "",
            "wifi_stationary_end_elapsed_ts": point.stationary_end_elapsed_ts if wifi_basis in {"documented_checkpoint_stay", "aggregated_stationary"} else "",
        }
        for field in copied_feature_fields:
            row_common[field] = feature_row.get(field, "") if feature_row is not None else ""
        for field in PDR_BINDING_FIELDS:
            row_common[field] = pdr_state.get(field, "") if pdr_state is not None else ""
        if pdr_state is None:
            row_common["pdr_interp_status"] = "checkpoint_pdr_unavailable"
            row_common["trajectory_interp_status"] = "contour_unavailable"
        if feature_row is None:
            for field in copied_feature_fields:
                if field.startswith("ap_"):
                    row_common[field] = MISSING_RSSI
            row_common["wifi_observation_status"] = "no_fresh_checkpoint_observation"
            row_common["wifi_fresh_ap_count"] = 0

        point_row = dict(row_common)
        for slot in range(1, group_size + 1):
            image = group_images[slot - 1] if slot <= len(group_images) else None
            point_row[f"image_{slot}"] = "" if image is None else image.name
            point_row[f"image_{slot}_path"] = "" if image is None else relative_path(image, root)
            point_row[f"image_{slot}_number"] = "" if image is None else image_number(image)

            long_row = dict(row_common)
            long_row.update(
                {
                    "image_slot": slot,
                    "image_name": "" if image is None else image.name,
                    "image_path": "" if image is None else relative_path(image, root),
                    "image_number": "" if image is None else image_number(image),
                }
            )
            long_rows.append(long_row)
            if image is not None:
                mapped_image_count += 1
        point_rows.append(point_row)

    expected_image_count = sum(group_sizes)
    unassigned_image_count = len(unassigned_images)
    missing_image_count = expected_image_count - mapped_image_count
    unassigned_rows = [
        {
            "image_name": image.name,
            "image_path": relative_path(image, root),
            "image_number": image_number(image),
            "reason": "outside_checkpoint_grouping_protocol",
        }
        for image in unassigned_images
    ]

    write_csv(
        output_dir / "image_wifi_long.csv",
        long_base_fields + copied_feature_fields,
        long_rows,
    )
    write_csv(
        output_dir / "image_wifi_points.csv",
        point_base_fields + image_fields + copied_feature_fields,
        point_rows,
    )
    write_csv(
        output_dir / "unassigned_images.csv",
        ["image_name", "image_path", "image_number", "reason"],
        unassigned_rows,
    )

    status = "ok"
    if not resolved_pic_dir.exists():
        status = "missing_pic_dir"
    elif not points:
        status = "no_wifi_points"
    elif missing_image_count:
        status = "count_mismatch"
    elif unassigned_image_count:
        status = "extra_images_ignored"
    elif any(row["mapping_status"] != "ok" for row in point_rows):
        status = "partial"

    summary = {
        "status": status,
        "pic_dir": str(resolved_pic_dir),
        "point_event": point_event,
        "images_per_wifi_point": None if variable_groups else images_per_point,
        "reference_images_per_wifi_point": images_per_point,
        "photo_grouping": grouping,
        "group_image_counts": [len(group) for group in groups],
        "points_without_images": sum(not group for group in groups),
        "wifi_point_count": len(points),
        "complete_point_count": complete_point_count,
        "image_count": len(images),
        "expected_image_count": expected_image_count,
        "mapped_image_count": mapped_image_count,
        "missing_image_count": missing_image_count,
        "unassigned_image_count": unassigned_image_count,
        "feature_source": str(feature_path.name),
        "fresh_wifi_point_count": sum(row.get("wifi_observation_status") == "fresh" for row in point_rows),
        "usable_wifi_point_count": sum(fingerprint_confidence(row) > 0 for row in point_rows),
        "strict_fresh_checkpoint_count": strict_fresh_checkpoint_count,
        "wifi_aggregation": aggregation_summary,
        "image_association_basis": "ordered_checkpoint_same_location",
        "image_capture_time_required": False,
        "wifi_measurement_max_age_ms": MAX_WIFI_AGE_MS,
        "wifi_near_checkpoint_max_gap_ms": MAX_WIFI_AGE_MS,
        "wifi_association_basis": "weighted_checkpoint_aggregation_with_strict_scan_fallback",
        "wifi_stationary_point_count": sum(row.get("wifi_association_basis") == "documented_checkpoint_stay" for row in point_rows),
        "pdr_position_basis": "exact_checkpoint_timestamp_independent_of_wifi_scans",
        "pdr_source": pdr_path.name,
        "pdr_checkpoint_count": sum(row.get("pdr_interp_status") == "exact_checkpoint" for row in point_rows),
        "outputs": {
            "long": "image_wifi_long.csv",
            "points": "image_wifi_points.csv",
            "unassigned_images": "unassigned_images.csv",
        },
    }
    write_json(output_dir / "image_wifi_summary.json", summary)
    return summary
