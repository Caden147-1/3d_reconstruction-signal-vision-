"""Density-connected selection of observed points, before optional sampling.

The voxel grid is a selection mask, never a replacement point cloud. No point
is moved, averaged or invented. Requires NumPy; memory is bounded by grid size
and the optional point budget, not by millions of Python vertex tuples.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import tempfile

import numpy as np

from ply_io import PLY_TYPES, parse_ply_header, validate_ply


BLOCK_SIZE = 250000
MAX_GRID_CELLS = 12000000
OFFSETS = tuple(offset for offset in itertools.product((-1, 0, 1), repeat=3) if any(offset))


def vertex_chunks(path: Path):
    """Read complete scalar vertex records in either byte order or ASCII."""
    with path.open("rb") as handle:
        header = parse_ply_header(handle)
        if not header["elements"] or header["elements"][0]["name"] != "vertex":
            raise ValueError("Expected vertex-first PLY")
        if any(e["count"] for e in header["elements"][1:]):
            raise ValueError("Connected-density filtering expects a point cloud, not a mesh")
        byteorder = {"ascii": "<", "binary_little_endian": "<", "binary_big_endian": ">"}.get(header["format"])
        if byteorder is None:
            raise ValueError("Unsupported PLY format")
        dtype = np.dtype([(name, byteorder + PLY_TYPES[kind][0]) for name, kind in header["vertex_properties"]])
        if not all(axis in dtype.names for axis in "xyz"):
            raise ValueError("PLY requires x, y and z")
        remaining = header["vertex_count"]
        if byteorder and header["format"] != "ascii" and remaining * dtype.itemsize > path.stat().st_size - handle.tell():
            raise ValueError("Truncated PLY vertex data")
        while remaining:
            count = min(BLOCK_SIZE, remaining)
            if header["format"] == "ascii":
                chunk = np.atleast_1d(np.loadtxt(handle, dtype=dtype, max_rows=count))
            else:
                chunk = np.fromfile(handle, dtype=dtype, count=count)
            if len(chunk) != count:
                raise ValueError("Truncated PLY vertex data")
            yield chunk
            remaining -= count


def positions(vertices):
    return np.column_stack([vertices[axis] for axis in "xyz"]).astype(np.float64)


def grid_keys(xyz, lower, upper, cell, shape):
    good = np.isfinite(xyz).all(axis=1) & (xyz >= lower).all(axis=1) & (xyz <= upper).all(axis=1)
    keys = np.floor((xyz[good] - lower) / cell).astype(np.int64)
    if len(keys) and not ((keys >= 0) & (keys < shape)).all():
        raise ValueError("Point outside allocated voxel grid")
    return good, keys


def connected_core(grid, threshold):
    """26-connected occupied cells, weighted by original point counts."""
    keys = np.argwhere(grid >= threshold)
    remaining = {tuple(map(int, key)) for key in keys}
    components = []
    while remaining:
        seed = remaining.pop()
        stack, members, weight = [seed], [], 0
        while stack:
            key = stack.pop()
            members.append(key)
            weight += int(grid[key])
            for offset in OFFSETS:
                other = tuple(key[a] + offset[a] for a in range(3))
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
        components.append((weight, members))
    if not components:
        raise ValueError("No density-supported surface component")
    # Lexical tie-break is independent of set traversal order.
    components.sort(key=lambda item: (-item[0], min(item[1])))
    weight, members = components[0]
    total = sum(item[0] for item in components)
    share = weight / total
    if share < .75:
        raise ValueError(f"Ambiguous scene: largest density core is only {share:.1%}; refusing to discard other large components")
    core = np.zeros(grid.shape, dtype=bool)
    core[tuple(np.asarray(members).T)] = True
    return core, {
        "component_count": len(components),
        "selected_core_vertex_count": weight,
        "selected_core_cell_count": len(members),
        "largest_core_share": share,
        "discarded_core_vertex_count": total - weight,
    }


def expand_mask(core, halo):
    result = core.copy()
    for _ in range(halo):
        padded = np.pad(result, 1)
        for dx, dy, dz in itertools.product(range(3), repeat=3):
            result |= padded[dx:dx+core.shape[0], dy:dy+core.shape[1], dz:dz+core.shape[2]]
    return result


def filter_connected_cloud(input_path: Path, output_path: Path, summary_path: Path | None,
                           max_points: int = 0, bbox_quantile: float = .001,
                           bbox_padding_ratio: float = .05, cell_ratio: float = .0025,
                           core_quantile: float = .15, support_halo: int = 1):
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input must be preserved separately from filtered output")
    if not 0 <= bbox_quantile < .2 or not 0 <= bbox_padding_ratio <= 1:
        raise ValueError("Invalid robust bounding-box settings")
    if not 0 < cell_ratio < 1 or not 0 <= core_quantile <= .4 or not 0 <= support_halo <= 3:
        raise ValueError("Invalid connected-density settings")
    if max_points < 0:
        raise ValueError("Point budget must be nonnegative; 0 retains all supported points")
    with input_path.open("rb") as handle:
        header = parse_ply_header(handle)
    source_count = header["vertex_count"]
    source_stat = input_path.stat()
    if source_count < 100:
        raise ValueError("At least 100 points are needed for connected-density filtering")

    # Bounded, file-wide statistics only; selection always examines all points.
    stride = max(1, int(np.ceil(source_count / 250000)))
    samples, start, finite_count = [], 0, 0
    for chunk in vertex_chunks(input_path):
        xyz = positions(chunk)
        finite_count += int(np.isfinite(xyz).all(1).sum())
        samples.append(xyz[(-start) % stride::stride])
        start += len(chunk)
    sample = np.concatenate(samples)
    sample = sample[np.isfinite(sample).all(1)]
    if len(sample) < 100:
        raise ValueError("Too few finite statistics points")
    low, high = np.quantile(sample, [bbox_quantile, 1-bbox_quantile], axis=0)
    span = high-low
    diagonal = float(np.linalg.norm(span))
    if diagonal <= 0:
        raise ValueError("Degenerate point cloud")
    padding = np.maximum(span * bbox_padding_ratio, diagonal * 1e-6)
    low, high = low-padding, high+padding
    cell = diagonal * cell_ratio
    shape = np.floor((high-low) / cell).astype(np.int64) + 1
    if int(np.prod(shape, dtype=np.float64)) > MAX_GRID_CELLS:
        raise ValueError("Density grid exceeds memory limit; increase --surface-cell-ratio")
    grid = np.zeros(tuple(shape), dtype=np.int64)
    print(f"[filter] Counting all {source_count:,} points in the density grid...", flush=True)
    for chunk in vertex_chunks(input_path):
        _, keys = grid_keys(positions(chunk), low, high, cell, shape)
        np.add.at(grid, tuple(keys.T), 1)
    bbox_count = int(grid.sum())

    # A point-weighted occupancy percentile avoids treating many empty/sparse
    # background cells as representative of the observed surface density.
    occupancies = np.sort(grid[grid > 0])
    cumulative = np.cumsum(occupancies)
    threshold = max(3, int(occupancies[np.searchsorted(cumulative, bbox_count * core_quantile)]))
    core, component_report = connected_core(grid, threshold)
    support = expand_mask(core, support_halo)
    supported_count = int(grid[support].sum())
    retained_ratio = supported_count / finite_count if finite_count else 0
    if retained_ratio < .5:
        raise ValueError(f"Density selection retains only {retained_ratio:.1%}; inspect parameters before promotion")
    target = min(max_points, supported_count) if max_points else supported_count
    print(f"[filter] Supported {supported_count:,}/{source_count:,}; writing {target:,} observed points...", flush=True)

    def supported_chunks():
        for chunk in vertex_chunks(input_path):
            good, keys = grid_keys(positions(chunk), low, high, cell, shape)
            yield chunk[good][support[tuple(keys.T)]]

    # Random priorities produce a uniform sample *after* geometry selection.
    # Full-resolution mode streams directly without allocating a reservoir.
    selected = None
    if target < supported_count:
        rng = np.random.default_rng(0)
        selected_keys = np.empty(0)
        for chunk in supported_chunks():
            priorities = rng.random(len(chunk))
            if len(selected_keys) == target:
                eligible = priorities < selected_keys.max()
                priorities, chunk = priorities[eligible], chunk[eligible]
            selected_keys = np.concatenate((selected_keys, priorities))
            selected = chunk.copy() if selected is None else np.concatenate((selected, chunk))
            if len(selected_keys) > target:
                take = np.argpartition(selected_keys, target-1)[:target]
                selected_keys, selected = selected_keys[take], selected[take]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix="filtered-", suffix=".tmp", delete=False) as handle:
        staged = Path(handle.name)
    try:
        with staged.open("wb") as handle:
            lines = ["ply", "format binary_little_endian 1.0",
                     "comment density-connected original observations; no new geometry",
                     f"element vertex {target}"]
            lines.extend(f"property {kind} {name}" for name, kind in header["vertex_properties"])
            handle.write(("\n".join(lines) + "\nend_header\n").encode("ascii"))
            written = 0
            for chunk in ([selected] if selected is not None else supported_chunks()):
                chunk.astype(chunk.dtype.newbyteorder("<"), copy=False).tofile(handle)
                written += len(chunk)
        if written != target:
            raise ValueError("Output count changed during streaming")
        current_stat = input_path.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns):
            raise ValueError("Input changed while filtering")
        validate_ply(staged)
        staged.replace(output_path)
    finally:
        staged.unlink(missing_ok=True)

    summary = {
        "status": "ok", "input": str(input_path.resolve()), "output": str(output_path.resolve()),
        "method": "connected_density_before_sampling", "format": "binary_little_endian",
        "source_vertex_count": source_count, "input_vertex_count": source_count,
        "finite_vertex_count": finite_count, "output_vertex_count": written, "max_points": max_points,
        "geometry_candidate_count": supported_count,
        "geometry_removed_vertex_count": finite_count-supported_count,
        "geometry_removed_ratio": 1-retained_ratio,
        "budget_omitted_vertex_count": supported_count-written,
        "removed_vertex_count": source_count-written,
        "removed_ratio": 1-written/source_count,
        "source_retained_ratio": written/source_count,
        "statistics_sample_vertex_count": len(sample), "statistics_sample_stride": stride,
        "bbox_limits": [list(pair) for pair in zip(low.tolist(), high.tolist())],
        "bbox_quantile": bbox_quantile, "bbox_padding_ratio": bbox_padding_ratio,
        "bbox_kept_count": bbox_count,
        "density": {
            "status": "applied", "cell_size_scene_units": cell, "cell_ratio": cell_ratio,
            "grid_shape": shape.tolist(), "core_quantile": core_quantile,
            "minimum_core_cell_points": threshold, "support_halo_cells": support_halo,
            "candidate_count": bbox_count, "kept_count": supported_count,
            **component_report,
        },
        "sampling": {
            "method": "full_resolution" if max_points == 0 or target == supported_count else "uniform_random_priority",
            "target_max_points": max_points, "post_filter_candidate_count": supported_count,
            "sampled_vertex_count": written, "random_seed": 0, "budget_utilization": written/target,
        },
        "quality_warning": "",
        "explanation": (
            "Density is measured on all original points before sampling. Keep the dominant dense "
            "26-connected component and a bounded neighborhood of observed detail. Voxels only select "
            "original vertices; positions, colors and normals are unchanged. No route-boundary crop, "
            "surface interpolation or synthetic completion is applied. This heuristic can remove weak "
            "real surfaces and cannot repair misregistered geometry or missing observations."
        ),
    }
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
