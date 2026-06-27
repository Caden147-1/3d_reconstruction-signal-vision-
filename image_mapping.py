"""Bind ordered image captures to WiFi checkpoint rows."""

from __future__ import annotations

import bisect
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class WifiPoint:
    session: str
    point_index: int
    global_point_index: int
    event: str
    elapsed_ts: int
    epoch_ts: int


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
        for elapsed_ts, epoch_ts, payload in parse_tab_json(session_dir / "movements.dat"):
            event = str(payload)
            if event != point_event:
                continue
            point_index += 1
            global_point_index += 1
            points.append(
                WifiPoint(
                    session=session_dir.name,
                    point_index=point_index,
                    global_point_index=global_point_index,
                    event=event,
                    elapsed_ts=elapsed_ts,
                    epoch_ts=epoch_ts,
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


def nearest_feature_row(
    rows: Sequence[Dict[str, Any]],
    target_elapsed_ts: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not rows:
        return None, ""

    elapsed_values = [int(row["_elapsed_ts_int"]) for row in rows]
    pos = bisect.bisect_left(elapsed_values, target_elapsed_ts)
    candidate_indices = []
    if pos < len(rows):
        candidate_indices.append(pos)
    if pos > 0:
        candidate_indices.append(pos - 1)
    best_idx = min(candidate_indices, key=lambda idx: abs(elapsed_values[idx] - target_elapsed_ts))
    return rows[best_idx], str(abs(elapsed_values[best_idx] - target_elapsed_ts))


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def mapping_status(feature_row: Optional[Dict[str, Any]], missing_slots: int) -> str:
    parts: List[str] = []
    if feature_row is None:
        parts.append("missing_wifi")
    if missing_slots:
        parts.append("missing_images")
    return ";".join(parts) if parts else "ok"


def build_image_wifi_tables(
    root: Path,
    session_dirs: Sequence[Path],
    output_dir: Path,
    pic_dir: Path,
    images_per_point: int,
    point_event: str,
) -> Dict[str, Any]:
    """Create image-to-WiFi mapping tables from ordered photos and checkpoint events."""
    if images_per_point <= 0:
        raise ValueError("--images-per-wifi-point must be greater than zero")

    resolved_pic_dir = pic_dir if pic_dir.is_absolute() else root / pic_dir
    images = find_image_files(resolved_pic_dir)
    points = read_wifi_points(session_dirs, point_event)

    feature_path = output_dir / "wifi_pdr_features.csv"
    if not feature_path.exists():
        feature_path = output_dir / "wifi_wide_selected.csv"
    feature_fields, features_by_session = read_feature_rows(feature_path)
    copied_feature_fields = [
        field
        for field in feature_fields
        if field not in {"session", "scan_index", "elapsed_ts", "epoch_ts"}
    ]

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
    ]
    image_fields: List[str] = []
    for slot in range(1, images_per_point + 1):
        image_fields.extend([f"image_{slot}", f"image_{slot}_path", f"image_{slot}_number"])

    long_rows: List[Dict[str, Any]] = []
    point_rows: List[Dict[str, Any]] = []
    mapped_image_count = 0
    complete_point_count = 0

    for point_pos, point in enumerate(points):
        start = point_pos * images_per_point
        group_images = images[start : start + images_per_point]
        missing_slots = images_per_point - len(group_images)
        if missing_slots == 0:
            complete_point_count += 1

        feature_row, wifi_gap_ms = nearest_feature_row(
            features_by_session.get(point.session, []),
            point.elapsed_ts,
        )
        status = mapping_status(feature_row, missing_slots)

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
        }
        if feature_row is not None:
            for field in copied_feature_fields:
                row_common[field] = feature_row.get(field, "")

        point_row = dict(row_common)
        for slot in range(1, images_per_point + 1):
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

    expected_image_count = len(points) * images_per_point
    unassigned_images = images[expected_image_count:]
    unassigned_image_count = len(unassigned_images)
    missing_image_count = max(0, expected_image_count - len(images))
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
        "images_per_wifi_point": images_per_point,
        "wifi_point_count": len(points),
        "complete_point_count": complete_point_count,
        "image_count": len(images),
        "expected_image_count": expected_image_count,
        "mapped_image_count": mapped_image_count,
        "missing_image_count": missing_image_count,
        "unassigned_image_count": unassigned_image_count,
        "feature_source": str(feature_path.name),
        "outputs": {
            "long": "image_wifi_long.csv",
            "points": "image_wifi_points.csv",
            "unassigned_images": "unassigned_images.csv",
        },
    }
    write_json(output_dir / "image_wifi_summary.json", summary)
    return summary
