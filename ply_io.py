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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


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
        "elements": [],
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
            if len(parts) != 3 or int(parts[2]) < 0:
                raise ValueError("Invalid PLY element declaration")
            current_element = parts[1] if len(parts) >= 2 else ""
            if any(element["name"] == current_element for element in header["elements"]):
                raise ValueError("Duplicate PLY element")
            header["elements"].append({"name": current_element, "count": int(parts[2]), "properties": []})
            if len(parts) >= 3 and current_element == "vertex":
                header["vertex_count"] = int(parts[2])
        elif line.startswith("property "):
            parts = line.split()
            if not header["elements"]:
                raise ValueError("PLY property precedes element")
            if len(parts) == 3:
                kind, name = parts[1], parts[2]
                if kind not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property type: {kind}")
                header["elements"][-1]["properties"].append((name, kind, None))
                if current_element == "vertex":
                    header["vertex_properties"].append((name, kind))
            elif len(parts) == 5 and parts[1] == "list":
                count_kind, kind, name = parts[2:]
                if count_kind not in INT_RANGES or kind not in PLY_TYPES:
                    raise ValueError("Unsupported PLY list property")
                if current_element == "vertex":
                    raise ValueError("PLY vertex list properties are not supported")
                header["elements"][-1]["properties"].append((name, kind, count_kind))
            else:
                raise ValueError("Invalid PLY property declaration")
        elif line == "end_header":
            if not header["format"]:
                raise ValueError("PLY header is missing format")
            return header
    raise ValueError("PLY header is missing end_header")


def validate_ply(path: Path) -> Dict[str, int]:
    """Stream and validate the complete vertex/face payload before promotion."""
    with path.open("rb") as handle:
        header = parse_ply_header(handle)
        vertex_count = header["vertex_count"]
        if vertex_count <= 0:
            raise ValueError("PLY has no vertices")
        vertex_xyz = xyz_indices(header["vertex_properties"])
        if not header["elements"] or header["elements"][0]["name"] != "vertex":
            raise ValueError("PLY must begin with vertex data")
        file_size = path.stat().st_size
        endian = {"binary_little_endian": "<", "binary_big_endian": ">", "ascii": ""}.get(header["format"])
        if endian is None:
            raise ValueError("Unsupported PLY format")

        def binary_values(kind: str, count: int) -> Tuple[Any, ...]:
            size = PLY_TYPES[kind][1] * count
            if count < 0 or size > file_size - handle.tell():
                raise ValueError("Truncated PLY data")
            raw = handle.read(size)
            if len(raw) != size:
                raise ValueError("Truncated PLY data")
            return struct.unpack(endian + str(count) + PLY_TYPES[kind][0], raw)

        def ascii_value(token: str, kind: str) -> Any:
            if kind in FLOAT_TYPES:
                return float(token)
            value = int(token)
            lo, hi = INT_RANGES[kind]
            if not lo <= value <= hi:
                raise ValueError("PLY integer is outside its declared type")
            return value

        for element in header["elements"]:
            properties = element["properties"]
            if element["count"] and not properties:
                raise ValueError("PLY element has no properties")
            if element["name"] == "face" and element["count"] and not any(name in {"vertex_index", "vertex_indices"} for name, _kind, _count in properties):
                raise ValueError("PLY faces have no vertex indices")
            if endian and element["name"] == "vertex":
                record = struct.Struct(endian + "".join(PLY_TYPES[kind][0] for _name, kind, _count in properties))
                remaining = element["count"]
                if remaining * record.size > file_size - handle.tell():
                    raise ValueError("Truncated PLY vertex data")
                while remaining:
                    batch = min(8192, remaining)
                    raw = handle.read(batch * record.size)
                    if len(raw) != batch * record.size:
                        raise ValueError("Truncated PLY vertex data")
                    for values in record.iter_unpack(raw):
                        if not all(math.isfinite(values[index]) for index in vertex_xyz):
                            raise ValueError("Non-finite PLY vertex")
                    remaining -= batch
                continue
            for _ in range(element["count"]):
                tokens = [] if endian else handle.readline().decode("ascii").split()
                offset = 0
                xyz: Dict[str, float] = {}
                for name, kind, count_kind in properties:
                    count = 1
                    if count_kind is not None:
                        if endian:
                            count = binary_values(count_kind, 1)[0]
                        else:
                            if offset >= len(tokens):
                                raise ValueError("Truncated PLY list")
                            count = ascii_value(tokens[offset], count_kind)
                            offset += 1
                    if count < 0:
                        raise ValueError("Negative PLY list count")
                    if endian:
                        values = binary_values(kind, count)
                    else:
                        if offset + count > len(tokens):
                            raise ValueError("Truncated PLY row")
                        values = tuple(ascii_value(token, kind) for token in tokens[offset:offset + count])
                        offset += count
                    if element["name"] == "vertex" and name in {"x", "y", "z"}:
                        xyz[name] = float(values[0])
                    if element["name"] == "face" and name in {"vertex_index", "vertex_indices"}:
                        if count_kind is None or kind not in INT_RANGES or count < 3 or any(v < 0 or v >= vertex_count for v in values):
                            raise ValueError("Invalid PLY face indices")
                if not endian and offset != len(tokens):
                    raise ValueError("Unexpected values in PLY row")
                if element["name"] == "vertex" and not all(math.isfinite(xyz[axis]) for axis in ("x", "y", "z")):
                    raise ValueError("Non-finite PLY vertex")
        while extra := handle.read(65536):
            if endian or extra.strip():
                raise ValueError("Unexpected data after declared PLY elements")
    return {element["name"]: element["count"] for element in header["elements"]}


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
    sample_stride: int = 1,
) -> List[PlyRow]:
    rows: List[PlyRow] = []
    kinds = [kind for _name, kind in properties]
    xyz = xyz_indices(properties)
    sample_stride = max(1, sample_stride)
    for index in range(vertex_count):
        line = handle.readline().decode("ascii", errors="ignore")
        if index % sample_stride != 0:
            continue
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
    sample_stride: int = 1,
) -> List[PlyRow]:
    fmt = endian + "".join(PLY_TYPES[kind][0] for _name, kind in properties)
    row_size = struct.calcsize(fmt)
    rows: List[PlyRow] = []
    xyz = xyz_indices(properties)
    sample_stride = max(1, sample_stride)
    for index in range(vertex_count):
        raw = handle.read(row_size)
        if len(raw) != row_size:
            break
        if index % sample_stride != 0:
            continue
        row = tuple(struct.unpack(fmt, raw))
        if is_finite_xyz(row, xyz):
            rows.append(row)
    return rows


def read_ply_vertices(path: Path, max_vertices: Optional[int] = None) -> Tuple[List[PlyProperty], List[PlyRow], Dict[str, Any]]:
    with path.open("rb") as handle:
        header = parse_ply_header(handle)
        properties: List[PlyProperty] = list(header["vertex_properties"])
        vertex_count = int(header["vertex_count"])
        if not properties:
            raise ValueError("PLY file has no vertex properties")
        sample_stride = 1
        if max_vertices is not None and max_vertices > 0:
            sample_stride = max(1, math.ceil(vertex_count / max_vertices))
        if header["format"] == "ascii":
            rows = _read_ascii_vertices(handle, vertex_count, properties, sample_stride)
        elif header["format"] == "binary_little_endian":
            rows = _read_binary_vertices(handle, vertex_count, properties, "<", sample_stride)
        elif header["format"] == "binary_big_endian":
            rows = _read_binary_vertices(handle, vertex_count, properties, ">", sample_stride)
        else:
            raise ValueError(f"Unsupported PLY format: {header['format']}")
    return properties, rows, {
        "format": header["format"],
        "declared_vertex_count": vertex_count,
        "loaded_vertex_count": len(rows),
        "sample_stride": sample_stride,
        "sampling_requested_max_vertices": max_vertices or 0,
    }


def iter_ply_vertices(path: Path, finite_xyz_only: bool = True) -> Iterator[PlyRow]:
    """Yield PLY vertex rows without materializing the full cloud in memory.

    This is used for spatially balanced post-filter sampling of large fused
    clouds. The existing ``read_ply_vertices`` remains useful for compact
    statistics samples and backwards-compatible callers.
    """
    with path.open("rb") as handle:
        header = parse_ply_header(handle)
        properties: List[PlyProperty] = list(header["vertex_properties"])
        vertex_count = int(header["vertex_count"])
        if not properties:
            raise ValueError("PLY file has no vertex properties")
        xyz = xyz_indices(properties)
        kinds = [kind for _name, kind in properties]
        if header["format"] == "ascii":
            for _ in range(vertex_count):
                line = handle.readline().decode("ascii", errors="ignore")
                parts = line.split()
                if len(parts) < len(kinds):
                    continue
                row = tuple(_parse_ascii_value(value, kind) for value, kind in zip(parts, kinds))
                if finite_xyz_only and not is_finite_xyz(row, xyz):
                    continue
                yield row
            return

        endian_by_format = {
            "binary_little_endian": "<",
            "binary_big_endian": ">",
        }
        endian = endian_by_format.get(header["format"])
        if endian is None:
            raise ValueError(f"Unsupported PLY format: {header['format']}")
        fmt = endian + "".join(PLY_TYPES[kind][0] for kind in kinds)
        row_size = struct.calcsize(fmt)
        for _ in range(vertex_count):
            raw = handle.read(row_size)
            if len(raw) != row_size:
                return
            row = tuple(struct.unpack(fmt, raw))
            if finite_xyz_only and not is_finite_xyz(row, xyz):
                continue
            yield row


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
    kinds = [kind for _name, kind in properties]
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        for comment in comments:
            handle.write(f"comment {comment}\n")
        handle.write(f"element vertex {len(rows)}\n")
        for name, kind in properties:
            handle.write(f"property {kind} {name}\n")
        handle.write("end_header\n")
        for row in rows:
            handle.write(" ".join(_format_value(value, kind) for value, kind in zip(row, kinds)))
            handle.write("\n")
    return len(rows)
