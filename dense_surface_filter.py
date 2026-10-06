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
from progress_output import WorkProgress


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


def sample_original_vertices(input_path: Path, source_count: int, max_candidates: int):
    """Collect a bounded deterministic sample independent of reader chunk size.

    Unsampled points have no geometry verdict. Keeping original indices allows
    a conservative caller to retain them while excluding only measured conflicts.
    """
    if type(max_candidates) is not int or not 0 <= max_candidates <= 2000000:
        raise ValueError('Observation candidate budget must be between 0 and 2000000')
    if source_count <= 0:
        raise ValueError('An observed surface requires original vertices')
    stride = max(1, (source_count + max_candidates - 1) // max_candidates) if max_candidates else None
    samples, indices, offset, dtype = [], [], 0, None
    with WorkProgress('Sampling original observations', source_count, unit='points') as progress:
        for chunk in vertex_chunks(input_path):
            dtype = chunk.dtype
            if not np.isfinite(positions(chunk)).all():
                raise ValueError('Non-finite input geometry')
            if stride is not None:
                local = np.arange((-offset) % stride, len(chunk), stride, dtype=np.int64)
                if len(local):
                    samples.append(chunk[local].copy())
                    indices.append(local + offset)
            offset += len(chunk)
            progress.update(offset)
    if offset != source_count:
        raise ValueError('Original point count changed while sampling observations')
    return (np.concatenate(samples) if samples else np.empty(0, dtype=dtype),
            np.concatenate(indices) if indices else np.empty(0, dtype=np.int64), stride)


def statistics_sample(input_path: Path, source_count: int, max_samples: int = 250000):
    """Copy selected XYZ rows into a bounded buffer; never retain chunk views."""
    if source_count <= 0 or max_samples <= 0:
        raise ValueError("Statistics counts must be positive")
    stride = max(1, (source_count + max_samples - 1) // max_samples)
    sample = np.empty(((source_count + stride - 1) // stride, 3), dtype=np.float64)
    start, count, finite_count = 0, 0, 0
    with WorkProgress('Measuring cloud statistics', source_count, unit='points') as progress:
        for chunk in vertex_chunks(input_path):
            xyz = positions(chunk)
            finite_count += int(np.isfinite(xyz).all(1).sum())
            selected = xyz[(-start) % stride::stride]
            sample[count:count + len(selected)] = selected
            count += len(selected)
            start += len(chunk)
            progress.update(start)
    if start != source_count:
        raise ValueError("Input vertex count changed during statistics sampling")
    return sample[:count][np.isfinite(sample[:count]).all(1)], stride, finite_count


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
    visited = 0
    with WorkProgress('Finding connected density surfaces', len(keys), unit='cells') as progress:
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
                if len(members) % 4096 == 0:
                    progress.update(visited + len(members))
            components.append((weight, members))
            visited += len(members)
            progress.update(visited)
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


def observation_rescue_indices(input_path, density_selection, candidate_count,
                               protect, max_candidates, vertex_keep=None):
    """Assess a bounded, file-wide sample once; retain exact source indices.

    A skipped candidate is not an observed conflict. Sparse indices also avoid
    allocating one evidence record per point in a large fused cloud.
    """
    if not isinstance(max_candidates, int) or not 1 <= max_candidates <= 2000000:
        raise ValueError('Observation candidate budget must be between 1 and 2000000')
    stride = max(1, (candidate_count + max_candidates - 1) // max_candidates)
    rescued = []
    source_offset = candidate_offset = assessed = vetoed = 0
    sample_count = (candidate_count + stride - 1) // stride
    with WorkProgress('Checking density-rejected observations', sample_count, unit='points') as progress:
        for chunk in vertex_chunks(input_path):
            candidates = np.flatnonzero(np.isfinite(positions(chunk)).all(1) & ~density_selection(chunk))
            selected = candidates[(-candidate_offset) % stride::stride]
            candidate_offset += len(candidates)
            for start in range(0, len(selected), 16384):
                indices = selected[start:start+16384]
                subset = chunk[indices]
                supported = np.asarray(protect(subset))
                if supported.dtype != np.bool_ or supported.shape != (len(subset),):
                    raise ValueError('Observation protection must return one boolean per original vertex')
                assessed += len(subset)
                if vertex_keep is not None and supported.any():
                    keep = np.asarray(vertex_keep(subset))
                    if keep.dtype != np.bool_ or keep.shape != supported.shape:
                        raise ValueError('Vertex evidence must return one boolean per original vertex')
                    vetoed += int(np.count_nonzero(supported & ~keep))
                    supported &= keep
                if supported.any():
                    rescued.append(indices[supported] + source_offset)
                progress.update(assessed)
            source_offset += len(chunk)
    if candidate_offset != candidate_count or assessed > max_candidates:
        raise ValueError('Density candidates changed during observation protection')
    indices = np.concatenate(rescued) if rescued else np.empty(0, dtype=np.int64)
    return indices, dict(candidate_count=candidate_count, candidate_budget=max_candidates,
                         sampling_stride=stride, assessed_candidate_count=assessed,
                         unevaluated_candidate_count=candidate_count-assessed,
                         rescued_vertex_count=len(indices), cached_conflict_veto_count=vetoed)


def filter_connected_cloud(input_path: Path, output_path: Path, summary_path: Path | None,
                           max_points: int = 0, bbox_quantile: float = .001,
                           bbox_padding_ratio: float = .05, cell_ratio: float = .0025,
                           core_quantile: float = .15, support_halo: int = 1,
                           *, spatial_support=None, spatial_penalty: float = 1.,
                           vertex_keep=None, vertex_protect=None,
                           observation_max_candidates: int = 250000):
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input must be preserved separately from filtered output")
    if not 0 <= bbox_quantile < .2 or not 0 <= bbox_padding_ratio <= 1:
        raise ValueError("Invalid robust bounding-box settings")
    if not 0 < cell_ratio < 1 or not 0 <= core_quantile <= .4 or not 0 <= support_halo <= 3:
        raise ValueError("Invalid connected-density settings")
    if max_points < 0:
        raise ValueError("Point budget must be nonnegative; 0 retains all supported points")
    if not np.isfinite(spatial_penalty) or not 0 <= spatial_penalty <= 2:
        raise ValueError('Spatial density penalty must be finite and within [0,2]')
    with input_path.open("rb") as handle:
        header = parse_ply_header(handle)
    source_count = header["vertex_count"]
    source_stat = input_path.stat()
    if source_count < 100:
        raise ValueError("At least 100 points are needed for connected-density filtering")

    # Bounded, file-wide statistics only; selection always examines all points.
    sample, stride, finite_count = statistics_sample(input_path, source_count)
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
    def eligible_vertices(chunk):
        if vertex_keep is None:
            return chunk
        mask = np.asarray(vertex_keep(chunk))
        if mask.dtype != np.bool_ or mask.shape != (len(chunk),):
            raise ValueError('Vertex evidence must return one boolean per original vertex')
        return chunk[mask]

    print(f"[filter] Counting all {source_count:,} points in the density grid...", flush=True)
    with WorkProgress('Counting density-grid observations', source_count, unit='points') as progress:
        scanned = 0
        for chunk in vertex_chunks(input_path):
            _, keys = grid_keys(positions(chunk), low, high, cell, shape)
            np.add.at(grid, tuple(keys.T), 1)
            scanned += len(chunk)
            progress.update(scanned)
    bbox_count = int(grid.sum())

    # A point-weighted occupancy percentile avoids treating many empty/sparse
    # background cells as representative of the observed surface density.
    occupancies = np.sort(grid[grid > 0])
    if not len(occupancies):
        raise ValueError('No finite points remain inside the density bounds')
    cumulative = np.cumsum(occupancies)
    threshold = max(3, int(occupancies[np.searchsorted(cumulative, bbox_count * core_quantile)]))
    thresholds, spatial_report = spatial_density_thresholds(
        grid, low, cell, threshold, spatial_support, spatial_penalty)
    core, component_report = connected_core(grid, thresholds)
    del thresholds
    support = expand_mask(core, support_halo)
    density_supported_count = int(grid[support].sum())

    def density_selection(chunk):
        good, keys = grid_keys(positions(chunk), low, high, cell, shape)
        selected = np.zeros(len(chunk), dtype=bool)
        selected[good] = support[tuple(keys.T)]
        return selected

    rescued = np.empty(0, dtype=np.int64)
    protection = dict(status='not_requested', candidate_count=finite_count-density_supported_count,
                      assessed_candidate_count=0, unevaluated_candidate_count=finite_count-density_supported_count,
                      rescued_vertex_count=0)
    if vertex_protect is not None:
        rescued, protection = observation_rescue_indices(
            input_path, density_selection, finite_count-density_supported_count,
            vertex_protect, observation_max_candidates, vertex_keep)
        protection['status'] = 'bounded_observation_assessment'
        print(f"[filter] Observation protection rescued {len(rescued):,} original points; "
              f"assessed {protection['assessed_candidate_count']:,}/{protection['candidate_count']:,} density rejects.", flush=True)

    def supported_chunks(progress=None):
        offset = 0
        for chunk in vertex_chunks(input_path):
            selected = density_selection(chunk)
            lo, hi = np.searchsorted(rescued, [offset, offset+len(chunk)])
            selected[rescued[lo:hi]-offset] = True
            yield eligible_vertices(chunk[selected])
            offset += len(chunk)
            if progress is not None:
                progress.update(offset)

    # Apply observation evidence after density selection: a conflict veto must
    # not shift density thresholds and thereby remove other supported surfaces.
    if vertex_keep is not None:
        with WorkProgress('Checking retained observation evidence', source_count, unit='points') as progress:
            supported_count = sum(len(chunk) for chunk in supported_chunks(progress))
    else:
        supported_count = density_supported_count + len(rescued)
    evidence_excluded = density_supported_count + len(rescued) - supported_count
    retained_ratio = supported_count / finite_count if finite_count else 0
    if retained_ratio < .5:
        raise ValueError(f"Density selection retains only {retained_ratio:.1%}; inspect parameters before promotion")
    target = min(max_points, supported_count) if max_points else supported_count
    protection.update(rescued_before_export_budget_count=len(rescued),
                      density_rejected_unrescued_count=finite_count-density_supported_count-len(rescued),
                      published_rescued_vertex_count=len(rescued) if target == supported_count else None,
                      export_budget_applies_to_rescued_points=target < supported_count)
    print(f"[filter] Supported {supported_count:,}/{source_count:,}; writing {target:,} observed points...", flush=True)

    # Random priorities produce a uniform sample *after* geometry selection.
    # Full-resolution mode streams directly without allocating a reservoir.
    selected = None
    if target < supported_count:
        rng = np.random.default_rng(0)
        selected_keys = np.empty(0)
        with WorkProgress('Sampling retained observations', source_count, unit='points') as progress:
            for chunk in supported_chunks(progress):
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
            with WorkProgress('Writing filtered observations', target, unit='points') as progress:
                for chunk in ([selected] if selected is not None else supported_chunks()):
                    chunk.astype(chunk.dtype.newbyteorder("<"), copy=False).tofile(handle)
                    written += len(chunk)
                    progress.update(written)
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
            "pre_observation_veto_count": density_supported_count + len(rescued),
            "density_only_vertex_count": density_supported_count,
            **component_report,
        },
        "sampling": {
            "method": "full_resolution" if max_points == 0 or target == supported_count else "uniform_random_priority",
            "target_max_points": max_points, "post_filter_candidate_count": supported_count,
            "sampled_vertex_count": written, "random_seed": 0, "budget_utilization": written/target,
        },
        "quality_warning": "",
        "spatial_prior": spatial_report,
        "observation_protection": protection,
        "observation_rescued_vertex_count": len(rescued),
        "observation_excluded_vertex_count": evidence_excluded,
        "explanation": (
            "Density is measured on all original points before sampling. Keep the dominant dense "
            "26-connected component and a bounded neighborhood of observed detail. Voxels only select "
            "original vertices; positions, colors and normals are unchanged. No route-boundary crop, "
            "surface interpolation or synthetic completion is applied. When requested, a bounded assessment "
            "rescues original rejected vertices supported by calibrated observations. Unevaluated/unknown "
            "points retain the density decision and are not labelled observed conflicts. This heuristic can remove weak "
            "real surfaces and cannot repair misregistered geometry or missing observations."
        ),
    }
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def spatial_density_thresholds(grid, lower, cell, base, support, penalty):
    """Apply a confidence-gated density penalty, without a hard boundary crop.

    A field may return support alone (legacy, confidence=1), or a mapping with
    separate support/confidence arrays. Zero confidence is exactly the visual
    baseline. Strong connected surfaces can survive low support. These values
    are heuristic scores, not occupancy probabilities or depth observations.
    """
    report = dict(status='disabled', base_threshold=int(base), penalty=penalty,
                  tightened_core_cell_count=0, excluded_core_vertex_count=0,
                  zero_confidence_core_cell_count=0,
                  formula='ceil(base * (1 + penalty * confidence * (1 - support)))')
    if support is None or penalty == 0:
        return base, report
    thresholds = np.full(grid.shape, base, dtype=np.float32)
    keys = np.argwhere(grid >= base)
    for start in range(0, len(keys), BLOCK_SIZE):
        block = keys[start:start + BLOCK_SIZE]
        evidence = support(lower + (block + .5) * cell)
        if isinstance(evidence, dict):
            values = np.asarray(evidence['support'], dtype=float)
            confidence = np.asarray(evidence['confidence'], dtype=float)
        else:
            values = np.asarray(evidence, dtype=float)
            confidence = np.ones(len(block))
        for label, array in (('support', values), ('confidence', confidence)):
            if (array.shape != (len(block),) or not np.isfinite(array).all()
                    or np.any((array < 0) | (array > 1))):
                raise ValueError(f'Spatial {label} must return one finite [0,1] value per voxel')
        required = np.ceil(base * (1 + penalty * confidence * (1 - values)))
        index = tuple(block.T)
        thresholds[index] = required
        excluded = grid[index] < required
        report['tightened_core_cell_count'] += int(excluded.sum())
        report['excluded_core_vertex_count'] += int(grid[index][excluded].sum())
        report['zero_confidence_core_cell_count'] += int((confidence == 0).sum())
    report['status'] = 'confidence_gated_density_constraint'
    return thresholds, report
