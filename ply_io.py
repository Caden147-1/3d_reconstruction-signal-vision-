"""Small PLY vertex reader/writer used by reconstruction post-processing.

The helpers intentionally support the subset produced by COLMAP dense fusion:
scalar vertex properties in ASCII or binary little-endian PLY files. Outputs
are written as ASCII vertex-only PLY files so downstream scripts can inspect or
consume the filtered point cloud without needing COLMAP internals.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PlyProperty = Tuple[str, str]
PlyRow = Tuple[Any, ...]


PLY_TYPES: Dict[str, Tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}

FLOAT_TYPES = {"float", "float32", "double", "float64"}
INT_RANGES = {
    "char": (-128, 127),
    "int8": (-128, 127),
    "uchar": (0, 255),
    "uint8": (0, 255),
    "short": (-32768, 32767),
    "int16": (-32768, 32767),
    "ushort": (0, 65535),
    "uint16": (0, 65535),
    "int": (-2147483648, 2147483647),
    "int32": (-2147483648, 2147483647),
    "uint": (0, 4294967295),
    "uint32": (0, 4294967295),
}


def parse_ply_header(handle: Any) -> Dict[str, Any]:
    first = handle.readline().decode("ascii", errors="ignore").strip()
    if first != "ply":
        raise ValueError("Input is not a PLY file")

    header: Dict[str, Any] = {
        "format": "",
        "vertex_count": 0,
        "vertex_properties": [],
    }
    current_element = ""
    for raw_line in handle:
        line = raw_line.decode("ascii", errors="ignore").strip()
        if line.startswith("format "):
            parts = line.split()
            if len(parts) >= 2:
                header["format"] = parts[1]
        elif line.startswith("element "):
            parts = line.split()
            current_element = parts[1] if len(parts) >= 2 else ""
            if len(parts) >= 3 and current_element == "vertex":
                header["vertex_count"] = int(parts[2])
        elif line.startswith("property ") and current_element == "vertex":
            parts = line.split()
            if len(parts) == 3:
                kind, name = parts[1], parts[2]
                if kind not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property type: {kind}")
                header["vertex_properties"].append((name, kind))
            elif len(parts) >= 5 and parts[1] == "list":
                raise ValueError("PLY vertex list properties are not supported")
        elif line == "end_header":
            if not header["format"]:
                raise ValueError("PLY header is missing format")
            return header
    raise ValueError("PLY header is missing end_header")


def xyz_indices(properties: Sequence[PlyProperty]) -> Tuple[int, int, int]:
    names = [name for name, _kind in properties]
    try:
        return names.index("x"), names.index("y"), names.index("z")
    except ValueError as exc:
        raise ValueError("PLY vertex properties must include x, y, z") from exc


def color_indices(properties: Sequence[PlyProperty]) -> Tuple[int, int, int] | None:
    names = [name for name, _kind in properties]
    for triplet in (("red", "green", "blue"), ("r", "g", "b")):
        if all(name in names for name in triplet):
            return names.index(triplet[0]), names.index(triplet[1]), names.index(triplet[2])
    return None


def row_xyz(row: PlyRow, indices: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return float(row[indices[0]]), float(row[indices[1]]), float(row[indices[2]])


def is_finite_xyz(row: PlyRow, indices: Tuple[int, int, int]) -> bool:
    return all(math.isfinite(value) for value in row_xyz(row, indices))


def _parse_ascii_value(text: str, kind: str) -> Any:
    if kind in FLOAT_TYPES:
        return float(text)
    return int(float(text))


def _read_ascii_vertices(
    handle: Any,
    vertex_count: int,
    properties: Sequence[PlyProperty],
) -> List[PlyRow]:
    rows: List[PlyRow] = []
    kinds = [kind for _name, kind in properties]
    xyz = xyz_indices(properties)
    for _index in range(vertex_count):
        line = handle.readline().decode("ascii", errors="ignore")
        parts = line.split()
        if len(parts) < len(kinds):
            continue
        row = tuple(_parse_ascii_value(value, kind) for value, kind in zip(parts, kinds))
        if is_finite_xyz(row, xyz):
            rows.append(row)
    return rows


def _read_binary_vertices(
    handle: Any,
    vertex_count: int,
    properties: Sequence[PlyProperty],
    endian: str,
) -> List[PlyRow]:
    fmt = endian + "".join(PLY_TYPES[kind][0] for _name, kind in properties)
    row_size = struct.calcsize(fmt)
    rows: List[PlyRow] = []
    xyz = xyz_indices(properties)
    for _index in range(vertex_count):
        raw = handle.read(row_size)
        if len(raw) != row_size:
            break
        row = tuple(struct.unpack(fmt, raw))
        if is_finite_xyz(row, xyz):
            rows.append(row)
    return rows


def read_ply_vertices(path: Path) -> Tuple[List[PlyProperty], List[PlyRow], Dict[str, Any]]:
    with path.open("rb") as handle:
        header = parse_ply_header(handle)
        properties: List[PlyProperty] = list(header["vertex_properties"])
        vertex_count = int(header["vertex_count"])
        if not properties:
            raise ValueError("PLY file has no vertex properties")
        if header["format"] == "ascii":
            rows = _read_ascii_vertices(handle, vertex_count, properties)
        elif header["format"] == "binary_little_endian":
            rows = _read_binary_vertices(handle, vertex_count, properties, "<")
        elif header["format"] == "binary_big_endian":
            rows = _read_binary_vertices(handle, vertex_count, properties, ">")
        else:
            raise ValueError(f"Unsupported PLY format: {header['format']}")
    return properties, rows, {
        "format": header["format"],
        "declared_vertex_count": vertex_count,
        "loaded_vertex_count": len(rows),
    }


def _format_value(value: Any, kind: str) -> str:
    if kind in FLOAT_TYPES:
        number = float(value)
        return f"{number:.9g}"
    number = int(round(float(value)))
    if kind in INT_RANGES:
        lo, hi = INT_RANGES[kind]
        number = min(max(number, lo), hi)
    return str(number)


def write_ply_vertices(
    path: Path,
    properties: Sequence[PlyProperty],
    rows: Iterable[PlyRow],
    comments: Sequence[str] = (),
) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        *[f"comment {comment}" for comment in comments],
        f"element vertex {len(rows)}",
    ]
    for name, kind in properties:
        lines.append(f"property {kind} {name}")
    lines.append("end_header")
    kinds = [kind for _name, kind in properties]
    for row in rows:
        lines.append(" ".join(_format_value(value, kind) for value, kind in zip(row, kinds)))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return len(rows)
