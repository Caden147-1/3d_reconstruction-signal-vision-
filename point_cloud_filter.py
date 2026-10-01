#!/usr/bin/env python3
"""Filter a COLMAP cloud before optional sampling; preserve raw fused.ply.

The default observation-support mode uses calibrated real depth observations.
Unknown geometry is retained; density-only methods remain explicit opt-ins.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ply_io import iter_ply_vertices, PlyRow, read_ply_vertices, row_xyz, write_ply_vertices, xyz_indices


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


def reservoir_sample_rows(
    input_path: Path,
    properties: Sequence[Tuple[str, str]],
    limits: Sequence[Tuple[float, float]],
    center: Point3,
    radial_limit: float,
    max_points: int,
) -> Tuple[List[PlyRow], Dict[str, Any]]:
    """Uniform reservoir sampling preserves observed density with bounded memory."""
    if max_points <= 0:
        raise ValueError("Reservoir sampling requires a positive point budget")
    xyz = xyz_indices(properties)
    rng = random.Random(0)
    reservoir: List[Tuple[int, PlyRow]] = []
    finite_count = 0
    bbox_count = 0
    radial_count = 0
    for source_index, row in enumerate(iter_ply_vertices(input_path)):
        finite_count += 1
        point = row_xyz(row, xyz)
        if not inside_limits(point, limits):
            continue
        bbox_count += 1
        if math.dist(point, center) > radial_limit:
            continue
        radial_count += 1
        # Every eligible source point has the same probability of retention.
        if len(reservoir) < max_points:
            reservoir.append((source_index, row))
        else:
            slot = rng.randrange(radial_count)
            if slot < max_points:
                reservoir[slot] = (source_index, row)
    selected = [row for _index, row in sorted(reservoir, key=lambda item: item[0])]
    return selected, {
        "method": "uniform_reservoir",
        "target_max_points": max_points,
        "sampled_vertex_count": len(selected),
        "random_seed": 0,
        "budget_utilization": len(selected) / min(max_points, radial_count) if radial_count else 0.0,
        "finite_vertex_count": finite_count,
        "bbox_candidate_count": bbox_count,
        "radial_candidate_count": radial_count,
    }


def sampling_quality_warning(summary: Dict[str, Any]) -> str:
    sampling = summary.get("sampling", {})
    budget = int(sampling.get("target_max_points") or summary.get("max_points") or 0)
    candidates = int(sampling.get("radial_candidate_count") or summary.get("source_vertex_count") or 0)
    retained = int(summary.get("output_vertex_count") or 0)
    if "post_filter_candidate_count" in sampling:
        candidates = int(sampling["post_filter_candidate_count"])
        if candidates <= 0:
            return "Dense filter has no supported geometry candidates; do not promote this output."
    expected = min(budget, candidates)
    if budget == 0 and "post_filter_candidate_count" in sampling:
        expected = candidates
    if expected > 0 and retained < expected * 0.5:
        return (
            f"Dense post-filter retained only {retained}/{expected} available budgeted points. "
            "Inspect the raw fused cloud and sampling before accepting final.ply."
        )
    return ""


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
    statistics_max_points = None if max_points <= 0 else min(max_points, 250000)
    properties, statistics_rows, statistics_io = read_ply_vertices(
        input_path,
        max_vertices=statistics_max_points,
    )
    xyz = xyz_indices(properties)
    statistics_points = [row_xyz(row, xyz) for row in statistics_rows]
    if len(statistics_points) < 10:
        raise ValueError("Too few points to filter")

    limits = axis_limits(statistics_points, bbox_quantile, bbox_padding_ratio)
    bbox_indices = [index for index, point in enumerate(statistics_points) if inside_limits(point, limits)]

    center = robust_center([statistics_points[index] for index in bbox_indices] or statistics_points)
    distances = sorted(
        math.dist(statistics_points[index], center)
        for index in bbox_indices
    )
    radial_limit = percentile(distances, radial_quantile) if distances else math.inf

    diagonal = robust_diagonal(limits)
    radius = diagonal * max(0.0, radius_ratio)
    declared_count = int(statistics_io.get("declared_vertex_count") or len(statistics_rows))
    if max_points <= 0 or declared_count <= max_points:
        properties, rows, io_summary = read_ply_vertices(input_path, max_vertices=None)
        sampling_summary: Dict[str, Any] = {
            "method": "full_resolution",
            "target_max_points": max_points,
            "finite_vertex_count": len(rows),
            "bbox_candidate_count": None,
            "radial_candidate_count": None,
        }
    else:
        rows, sampling_summary = reservoir_sample_rows(
            input_path=input_path,
            properties=properties,
            limits=limits,
            center=center,
            radial_limit=radial_limit,
            max_points=max_points,
        )
        io_summary = {
            "format": statistics_io.get("format", ""),
            "declared_vertex_count": declared_count,
            "loaded_vertex_count": len(rows),
            "sample_stride": 1,
            "sampling_requested_max_vertices": max_points,
        }
    points = [row_xyz(row, xyz) for row in rows]
    bbox_indices = [index for index, point in enumerate(points) if inside_limits(point, limits)]
    radial_indices = [
        index
        for index in bbox_indices
        if math.dist(points[index], center) <= radial_limit
    ]
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
    summary = {
        "status": "ok",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "method": "robust_bbox_radial_clip_uniform_reservoir_density",
        "input_vertex_count": input_count,
        "source_vertex_count": declared_count,
        "max_points": max_points,
        "sample_stride": io_summary.get("sample_stride", 1),
        "statistics_sample_vertex_count": len(statistics_rows),
        "statistics_sample_stride": statistics_io.get("sample_stride", 1),
        "sampling": sampling_summary,
        "output_vertex_count": written_count,
        # Keep the original summary keys for callers that read prior reports.
        "removed_vertex_count": input_count - written_count,
        "removed_ratio": round((input_count - written_count) / input_count, 6) if input_count else 0.0,
        "post_sample_removed_vertex_count": input_count - written_count,
        "post_sample_removed_ratio": round((input_count - written_count) / input_count, 6) if input_count else 0.0,
        "source_retained_ratio": round(written_count / declared_count, 6) if declared_count else 0.0,
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
            "Robust limits are estimated from a bounded statistics sample. Large "
            "clouds are streamed in full and uniformly reservoir-sampled with a "
            "fixed seed, preserving source density without a file-order stride. The filter "
            "removes only extreme spatial tails and isolated neighborhoods after "
            "that representation is chosen."
        ),
    }
    warning = sampling_quality_warning(summary)
    summary["quality_warning"] = warning
    if warning:
        summary["status"] = "needs_review"
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def cached_observation_layers(workspace, snapshot, binding, source_identity):
    """Carry original observation evidence across filtering of the same fusion.

    These layers describe the source, not the filtered output. In particular,
    retaining supported evidence keeps later wall fitting independent of the
    unknown points that may survive density selection.
    """
    receipt = snapshot.read_json(workspace / 'dense/point_filter_summary.json')
    if (receipt.get('reconstruction_binding') != binding
            or receipt.get('input_identity') != source_identity):
        return {}
    layers = (receipt.get('layers', {}) if receipt.get('method') == 'observation_support'
              else receipt.get('observation_layers', {}))
    verified = {}
    for name in ('supported', 'conflict'):
        layer = layers.get(name, {})
        if not layer:
            continue
        path = Path(layer['path']).resolve()
        if not path.is_relative_to((workspace / 'dense').resolve()):
            raise ValueError('Observation evidence must be inside the dense workspace')
        if snapshot.identity(path) is None or snapshot.identity(path) != layer.get('identity'):
            raise ValueError(f'Observation {name} layer changed since assessment')
        verified[name] = layer
    return verified


def cached_observation_veto(workspace, snapshot, binding, source_identity, *, layers=None):
    """Reuse exact-source conflict evidence; never match by proximity or voxel.

    Unknown observations remain eligible. Full scalar vertex records must agree
    byte-for-byte with a certified conflict layer from this same fusion/model.
    The bounded cache can be absent on a fresh reconstruction.
    """
    import numpy as np
    from dense_surface_filter import vertex_chunks

    if layers is None:
        layers = cached_observation_layers(workspace, snapshot, binding, source_identity)
    layer = layers.get('conflict', {})
    if not layer:
        return None, dict(status='unavailable', reason='no_matching_source_observation_evidence')
    count = int(layer.get('vertex_count', 0))
    if count <= 0 or count > 2000000:
        return None, dict(status='unavailable', reason='empty_or_oversized_conflict_cache')
    path = Path(layer['path']).resolve()
    chunks = list(vertex_chunks(path))
    if sum(len(chunk) for chunk in chunks) != count:
        raise ValueError('Observation conflict count differs from receipt')
    dtype = chunks[0].dtype.newbyteorder('<')

    def records(chunk):
        if chunk.dtype.newbyteorder('<') != dtype:
            raise ValueError('Conflict records do not match fusion vertex properties')
        return np.ascontiguousarray(chunk.astype(dtype, copy=False)).view(
            np.dtype((np.void, dtype.itemsize))).reshape(-1)

    keys = np.unique(np.concatenate([records(chunk) for chunk in chunks]))

    def keep(chunk):
        query = records(chunk)
        indices = np.searchsorted(keys, query)
        return (indices == len(keys)) | (keys[np.minimum(indices, len(keys)-1)] != query)

    return keep, dict(status='applied', layer=layer, source_identity=source_identity,
                      matching='exact_full_vertex_record',
                      interpretation='Veto previously observed conflicts; unknown points remain eligible.')


def filter_wifi_density(args):
    """Apply the Git baseline density filter with authenticated, soft RF support."""
    from colmap_workspace_check import file_identity, write_provenance
    from dense_surface_filter import filter_connected_cloud
    from reconstruction_evaluation import Snapshot
    from reconstruction_stages import verified_context
    from wifi_boundary import density_boundary_prior
    from workspace_lock import workspace_lock

    if args.workspace is None or args.skip_density:
        raise ValueError('wifi-density requires --workspace and does not accept --skip-density')
    workspace = args.workspace.resolve()
    dense = workspace / 'dense'
    source, output = args.input.resolve(), args.output.resolve()
    summary_path = (args.summary or output.with_suffix('.json')).resolve()
    if len({source, output, summary_path}) != 3:
        raise ValueError('Source, candidate and summary paths must be distinct')
    for path in (output, summary_path):
        if (path in {dense/'final.ply', dense/'fused.ply'}
                or any(path.is_relative_to(dense/folder) for folder in ('images', 'sparse', 'stereo'))):
            raise ValueError('Write a separate candidate; preserve final, fusion and calibrated inputs')
    with workspace_lock(workspace):
        binding = verified_context(workspace)
        snapshot = Snapshot()
        source_identity = snapshot.identity(source)
        fusion = snapshot.read_json(source.parent / 'fusion_provenance.json')
        if (source_identity is None or fusion.get('reconstruction_binding') != binding
                or fusion.get('output_identity') != source_identity):
            raise ValueError('WiFi density filtering requires an authenticated fusion source')
        settings = snapshot.read_json(workspace/'colmap_workspace_summary.json')
        if settings.get('visual_only'):
            field, prior = None, dict(status='disabled', reason='visual_only_workspace')
        else:
            field, prior = density_boundary_prior(
                workspace, snapshot, mode=args.wifi_prior_mode, seed=args.wifi_permutation_seed)
        observation_layers = cached_observation_layers(workspace, snapshot, binding, source_identity)
        if args.no_observation_veto:
            vertex_keep, observation_veto = None, dict(status='disabled', reason='explicit_ablation')
        else:
            vertex_keep, observation_veto = cached_observation_veto(
                workspace, snapshot, binding, source_identity, layers=observation_layers)
        code_names = ('point_cloud_filter.py', 'dense_surface_filter.py', 'wifi_boundary.py',
                      'wifi_observations.py', 'reconstruction_diagnostics.py', 'mvs_observations.py')
        codes = {name: snapshot.identity(Path(__file__).with_name(name)) for name in code_names}
        report = filter_connected_cloud(
            source, output, None, max_points=args.max_points,
            bbox_quantile=args.bbox_quantile, bbox_padding_ratio=args.bbox_padding_ratio,
            cell_ratio=args.surface_cell_ratio, core_quantile=args.core_quantile,
            support_halo=args.support_halo, spatial_support=field, spatial_penalty=args.wifi_density_weight,
            vertex_keep=vertex_keep)
        snapshot.verify_unchanged()
        if verified_context(workspace) != binding:
            raise ValueError('Reconstruction changed; density candidate is not certified for promotion')
        report.update(method='wifi_connected_density', input_identity=source_identity,
                      output_identity=file_identity(output), reconstruction_binding=binding,
                      wifi_boundary=prior, observation_conflict_veto=observation_veto,
                      observation_layers=observation_layers, code_identity=codes,
                      inputs={str(p): identity for p, (identity, _) in snapshot.files.items()})
        write_provenance(summary_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--method", choices=("wifi-density", "observation-support", "connected-density", "legacy"), default="wifi-density")
    parser.add_argument("--workspace", type=Path, help="Authenticated COLMAP workspace, required for wifi-density or observation-support.")
    parser.add_argument("--wifi-density-weight", type=float, default=1., help="Confidence-gated density penalty (0-2); 0 disables RF. Add --no-observation-veto for the plain density baseline.")
    parser.add_argument("--no-observation-veto", action="store_true", help="Disable reuse of exact-source observation conflicts for a controlled ablation.")
    parser.add_argument("--wifi-prior-mode", choices=("radio", "positions-only", "permuted"), default="radio", help="Radio prior or an ablation control; controls keep capture positions and per-stop quality fixed.")
    parser.add_argument("--wifi-permutation-seed", type=int, default=0, help="Reproducible within-session fingerprint shuffle for --wifi-prior-mode permuted.")
    parser.add_argument("--observation-max-views", type=int, default=24, help="Per-block observation evaluation budget; unevaluated evidence remains unknown.")
    parser.add_argument("--max-points", type=int, help="Export point budget; 0 retains all selected original points (default: 0, legacy: 750000).")
    parser.add_argument("--surface-cell-ratio", type=float, default=0.0025, help="Grid cell width relative to the robust scene diagonal.")
    parser.add_argument("--core-quantile", type=float, default=0.15, help="Point-weighted occupancy quantile for the dense core.")
    parser.add_argument("--support-halo", type=int, default=1, help="Observed-detail neighborhood around the core, in voxel cells (0-3).")
    parser.add_argument("--bbox-quantile", type=float, default=0.001)
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.05)
    parser.add_argument("--radial-quantile", type=float, default=0.9995)
    parser.add_argument("--radius-ratio", type=float, default=0.004)
    parser.add_argument("--min-neighbors", type=int, default=3)
    parser.add_argument("--skip-density", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_points is None:
        args.max_points = 750000 if args.method == "legacy" else 0
    if args.method == 'wifi-density':
        summary = filter_wifi_density(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.method == "observation-support":
        if args.workspace is None or args.skip_density:
            raise ValueError("observation-support requires --workspace and does not accept --skip-density")
        from dense_cache import CACHE_RECORD, read_record
        from colmap_workspace_check import file_identity
        from observation_filter import filter_observed_cloud
        from reconstruction_stages import verified_context
        from workspace_lock import workspace_lock
        workspace = args.workspace.resolve()
        with workspace_lock(workspace):
            binding = verified_context(workspace)
            dense = workspace / "dense"
            fusion = read_record(args.input.resolve().parent / 'fusion_provenance.json')
            if (fusion.get('reconstruction_binding') != binding or
                    fusion.get('output_identity') != file_identity(args.input)):
                raise ValueError('Observation filtering requires an authenticated fusion source; use run_colmap.py filter to re-fuse verified maps if necessary')
            record = read_record(dense / CACHE_RECORD)
            summary = filter_observed_cloud(
                args.input, args.output, args.summary or args.output.with_suffix('.json'), dense,
                max_points=args.max_points, policy={'max_views': args.observation_max_views},
                expected=record['outputs']['files'], binding=binding)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.method == "connected-density":
        if args.skip_density:
            raise ValueError("--skip-density requires --method legacy")
        from dense_surface_filter import filter_connected_cloud
        summary = filter_connected_cloud(
            args.input, args.output, args.summary, max_points=args.max_points,
            bbox_quantile=args.bbox_quantile, bbox_padding_ratio=args.bbox_padding_ratio,
            cell_ratio=args.surface_cell_ratio, core_quantile=args.core_quantile,
            support_halo=args.support_halo,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
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
