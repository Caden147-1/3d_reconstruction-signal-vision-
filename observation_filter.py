"""Conservative, auditable filtering against real calibrated depth observations.

Unknown points are retained. Evidence is a consistency measurement, not a
probability of correctness or independent ground truth. No density crop is used.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from colmap_workspace_check import file_identity, write_provenance
from dense_surface_filter import positions, vertex_chunks
from mvs_observations import DepthStore, project, read_model
from ply_io import parse_ply_header, validate_ply
from reconstruction_evaluation import Snapshot


UNKNOWN, SUPPORTED, CONFLICT = 0, 1, 2
EVIDENCE_DTYPE = np.dtype([('status', 'u1'), ('tested', '<u2'), ('support', '<u2'),
                         ('conflict', '<u2'), ('support_a', '<i4'), ('support_b', '<i4'),
                         ('conflict_a', '<i4'), ('conflict_b', '<i4'), ('best_relative_error', '<f4')])
DEFAULT_POLICY = dict(max_views=24, min_angle_degrees=1.5, sigma_multiplier=3.,
                      max_relative_uncertainty=.05, max_normal_degrees=30.)


def checked_policy(overrides=None):
    policy = {**DEFAULT_POLICY, **(overrides or {})}
    if (not isinstance(policy['max_views'], int) or not 2 <= policy['max_views'] <= 1024
            or not 0 < policy['min_angle_degrees'] < 90
            or not 0 < policy['sigma_multiplier'] <= 10
            or not 0 < policy['max_relative_uncertainty'] < 1
            or not 0 < policy['max_normal_degrees'] < 90
            or not all(np.isfinite(v) for v in policy.values())):
        raise ValueError('Invalid observation policy')
    return policy


def conditioning(views, points):
    """Estimate depth conditioning from actual sparse tracks, not scene extent."""
    by_id = {v['id']: v for v in views}
    result = {}
    for view in views:
        values = []
        ids = view['point_ids']
        for pid in ids[::max(1, (len(ids)+255)//256)]:
            xyz, _, track = points[int(pid)]
            a = xyz-view['C']
            an = np.linalg.norm(a)
            if an == 0:
                continue
            angles = []
            for iid in track:
                if int(iid) == view['id'] or int(iid) not in by_id:
                    continue
                b = xyz-by_id[int(iid)]['C']
                bn = np.linalg.norm(b)
                if bn:
                    cosine = float(a @ b / (an*bn))
                    # Opposite-facing/near-collinear rays are poorly conditioned.
                    angles.append(np.sqrt(max(0., 1-np.clip(cosine, -1, 1)**2)))
            if angles:
                values.append(max(angles))
        result[view['id']] = float(np.median(values)) if values else 0.
    return result


def candidate_views(xyz, views, max_views):
    """Bound work using projected samples; missed evidence remains unknown."""
    sample = xyz[::max(1, (len(xyz)+127)//128)]
    scores = []
    for view in views:
        _, z, valid = project(sample, view)
        if valid.any():
            footprint = np.sqrt(view['K'][0, 0]*view['K'][1, 1]) / np.median(z[valid])
            scores.append((int(valid.sum()), float(footprint), view['name'], view))
    return [v for _, _, _, v in sorted(scores, key=lambda x: (-x[0], -x[1], x[2]))[:max_views]]


def assess_points(xyz, views, store, quality, policy=None, normals=None):
    policy = checked_policy(policy)
    evidence = np.zeros(len(xyz), dtype=EVIDENCE_DTYPE)
    for field in ('support_a', 'support_b', 'conflict_a', 'conflict_b'):
        evidence[field] = -1
    evidence['best_relative_error'] = np.nan
    directions = {kind: np.zeros((len(xyz), 3)) for kind in ('support', 'conflict')}
    cos_angle = np.cos(np.deg2rad(policy['min_angle_degrees']))
    cos_normal = np.cos(np.deg2rad(policy['max_normal_degrees']))
    tested_views = []
    # Eligibility precedes the view cap. Missing or poorly conditioned maps must
    # not consume slots that could supply useful evidence farther down the list.
    eligible_count = 0
    for view in candidate_views(xyz, views, len(views)):
        pair = store.get(view)
        if pair is None:
            continue
        depth, normal = pair
        h, w = depth.shape
        f = min(view['K'][0, 0]*w/view['size'][0], view['K'][1, 1]*h/view['size'][1])
        sine = quality.get(view['id'], 0.)
        if sine <= 0:
            continue
        relative_sigma = policy['sigma_multiplier'] * view['noise_px'] / (f*sine)
        if relative_sigma > policy['max_relative_uncertainty']:
            continue
        if eligible_count >= policy['max_views']:
            break
        eligible_count += 1
        uv, z, good = project(xyz, view, (w, h))
        idx = np.flatnonzero(good)
        if not len(idx):
            continue
        # Require a valid, coherent 2x2 footprint. Missing depth, discontinuities,
        # and silhouettes do not establish empty space.
        pixel = np.floor(uv[idx]-.5).astype(int)
        x, y = pixel.T
        samples = np.column_stack([depth[y+dy, x+dx] for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))])
        d = np.mean(samples, axis=1)
        tolerance = d * relative_sigma
        valid = np.isfinite(samples).all(1) & (samples > 0).all(1)
        valid &= (np.ptp(samples, axis=1) <= tolerance)
        n = normal[y, x].astype(float)
        length = np.linalg.norm(n, axis=1)
        valid &= np.isfinite(n).all(1) & (length > .5)
        idx, d, tolerance, n, length = idx[valid], d[valid], tolerance[valid], n[valid], length[valid]
        if not len(idx):
            continue
        n = n / length[:, None]
        error = np.abs(z[idx]-d)
        near = error <= tolerance
        if normals is not None:
            original = normals[idx]
            original_length = np.linalg.norm(original, axis=1)
            normal_ok = np.isfinite(original).all(1) & (original_length > 1e-8)
            world_normals = n @ view['R']
            alignment = np.abs(np.einsum('ij,ij->i', original, world_normals)) / np.maximum(original_length, 1e-8)
            near &= normal_ok & (alignment >= cos_normal)
        # Behind a measured surface means occlusion, never a conflict.
        conflict = z[idx] < d-tolerance
        evidence['tested'][idx] += 1
        rel = (error / d).astype(np.float32)
        evidence['best_relative_error'][idx] = np.fmin(evidence['best_relative_error'][idx], rel)
        tested_views.append(view['id'])
        for kind, votes in [('support', near), ('conflict', conflict)]:
            selected = idx[votes]
            if not len(selected):
                continue
            evidence[kind][selected] += 1
            ray = xyz[selected]-view['C']
            ray /= np.maximum(np.linalg.norm(ray, axis=1)[:, None], 1e-12)
            first = evidence[kind+'_a'][selected] < 0
            fresh = selected[first]
            evidence[kind+'_a'][fresh] = view['id']
            directions[kind][fresh] = ray[first]
            cosine = np.einsum('ij,ij->i', ray, directions[kind][selected])
            diverse = (~first) & (np.abs(cosine) < cos_angle)
            evidence[kind+'_b'][selected[diverse]] = view['id']
    # Contradictory support and conflict are ambiguous, not a majority vote.
    evidence['status'][(evidence['support_b'] >= 0) & (evidence['conflict'] == 0)] = SUPPORTED
    evidence['status'][(evidence['conflict_b'] >= 0) & (evidence['support'] == 0)] = CONFLICT
    return evidence, tested_views


def write_cloud(path, raw_path, properties, count):
    with path.open('wb') as out, raw_path.open('rb') as src:
        lines = ['ply', 'format binary_little_endian 1.0',
                 'comment original observations; evidence is consistency, not ground truth', f'element vertex {count}']
        lines += [f'property {kind} {name}' for name, kind in properties]
        out.write(('\n'.join(lines)+'\nend_header\n').encode('ascii'))
        shutil.copyfileobj(src, out)
    if count:
        validate_ply(path)


def filter_observed_cloud(input_path, output_path, summary_path, dense, *, max_points=0,
                          policy=None, expected=None, binding=None, deadline=None):
    """Stream original vertices and sidecars; publish only after validation.

    The caller authenticates its model/cache context. `expected` additionally
    authenticates each accessed map. Independent adaptive runs pass their own
    exact output manifest instead of modifying production cache receipts.
    """
    policy = checked_policy(policy)
    input_path, output_path, dense = map(lambda p: Path(p).resolve(), (input_path, output_path, dense))
    summary_path = Path(summary_path).resolve()
    if len({input_path, output_path, summary_path}) != 3 or max_points < 0:
        raise ValueError('Distinct input/output/summary paths and a nonnegative point budget are required')
    for destination in (output_path, summary_path):
        if any(destination.is_relative_to(dense/folder) for folder in ('images','sparse','stereo')):
            raise ValueError('Filter outputs must not overwrite calibrated observations')
    start = time.monotonic()
    snapshot = Snapshot()
    input_identity = snapshot.identity(input_path)
    with input_path.open('rb') as handle:
        header = parse_ply_header(handle)
    count = header['vertex_count']
    if count <= 0:
        raise ValueError('Cannot filter an empty point cloud')
    views, points = read_model(dense/'sparse', snapshot)
    available = [v for v in views if (
        'stereo/depth_maps/'+v['name']+'.geometric.bin' in expected if expected is not None
        else (dense/'stereo/depth_maps'/(v['name']+'.geometric.bin')).is_file())]
    quality = conditioning(available, points)
    store = DepthStore(dense, snapshot, expected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Immutable sidecar directory makes a failed run unable to corrupt the
    # evidence belonging to an earlier output/summary.
    artifacts = output_path.parent / ('observation_evidence_'+uuid.uuid4().hex)
    artifacts.mkdir()
    evidence_path = artifacts/'points.npy'
    evidence_map = np.lib.format.open_memmap(evidence_path, mode='w+', dtype=EVIDENCE_DTYPE, shape=(count,))
    totals = np.zeros(3, dtype=np.int64)
    names = ('unknown', 'supported', 'conflict')
    seen_views, offset, last_update = set(), 0, start
    raw_paths = {name: artifacts/(name+'.raw') for name in (*names, 'retained')}
    try:
        with ExitStack() as stack:
            handles = {name: stack.enter_context(path.open('wb')) for name, path in raw_paths.items()}
            for source_chunk in vertex_chunks(input_path):
                for lo in range(0, len(source_chunk), 16384):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError('Observation assessment exhausted the runtime budget')
                    chunk = source_chunk[lo:lo+16384]
                    xyz = positions(chunk)
                    if not np.isfinite(xyz).all():
                        raise ValueError('Non-finite input geometry')
                    normals = np.column_stack([chunk[n] for n in ('nx', 'ny', 'nz')]) if {'nx','ny','nz'} <= set(chunk.dtype.names) else None
                    ev, used = assess_points(xyz, available, store, quality, policy, normals)
                    seen_views.update(used)
                    evidence_map[offset:offset+len(chunk)] = ev
                    little = chunk.astype(chunk.dtype.newbyteorder('<'), copy=False)
                    for status, name in enumerate(names):
                        mask = ev['status'] == status
                        little[mask].tofile(handles[name])
                        totals[status] += mask.sum()
                    little[ev['status'] != CONFLICT].tofile(handles['retained'])
                    offset += len(chunk)
                    if time.monotonic()-last_update >= 20:
                        print(f'[observations] {offset:,}/{count:,}; supported={totals[1]:,}, unknown={totals[0]:,}, conflict={totals[2]:,}', flush=True)
                        last_update = time.monotonic()
        evidence_map.flush()
    finally:
        del evidence_map
        store.close()
    if offset != count:
        raise ValueError('Point count changed during assessment')
    retained = int(totals[0]+totals[1])
    if retained == 0:
        raise ValueError('All points conflict; candidate withheld, evidence retained for inspection')
    properties = header['vertex_properties']
    outputs = {}
    for status, name in enumerate(names):
        path = artifacts/(name+'.ply')
        write_cloud(path, raw_paths[name], properties, int(totals[status]))
        outputs[name] = dict(path=str(path), vertex_count=int(totals[status]), identity=file_identity(path))
    # A point budget is a display/export decision, never a geometry verdict.
    target = min(max_points, retained) if max_points else retained
    if target < retained:
        dtype = next(vertex_chunks(input_path)).dtype.newbyteorder('<')
        raw = np.memmap(raw_paths['retained'], dtype=dtype, mode='r', shape=(retained,))
        indices = np.sort(np.random.default_rng(0).choice(retained, target, replace=False))
        sampled = artifacts/'sampled.raw'
        with sampled.open('wb') as handle:
            for lo in range(0, target, 250000):
                raw[indices[lo:lo+250000]].tofile(handle)
        del raw
        raw_paths['sampled'] = sampled
        selected = sampled
    else:
        selected = raw_paths['retained']
    staged = artifacts/'candidate.ply'
    write_cloud(staged, selected, properties, target)
    snapshot.verify_unchanged()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Observation budget exhausted before candidate publication')
    receipt = dict(status='ok', method='observation_support', schema_version=1,
                   input=str(input_path), input_identity=input_identity, output=str(output_path),
                   output_identity=file_identity(staged), reconstruction_binding=binding,
                   source_vertex_count=count, input_vertex_count=count, output_vertex_count=target,
                   max_points=max_points, geometry_candidate_count=retained,
                   geometry_removed_vertex_count=int(totals[2]), budget_omitted_vertex_count=retained-target,
                   removed_vertex_count=count-target, removed_ratio=1-target/count,
                   source_retained_ratio=target/count, policy=policy,
                   classification={name:int(totals[i]) for i,name in enumerate(names)},
                   evidence=dict(path=str(evidence_path), identity=file_identity(evidence_path),
                                 index='zero-based original PLY vertex order', classes={'0':'unknown','1':'supported','2':'conflict'},
                                 views=[dict(id=v['id'], name=v['name'], noise_px=v['noise_px'], conditioning_sine=quality.get(v['id'], 0.)) for v in views],
                                 interpretation='Reprojected support witnesses, not recovered original fusion visibility'),
                   layers=outputs, used_view_count=len(seen_views),
                   observation_inputs={str(p): identity for p,(identity,_) in snapshot.files.items()},
                   code_identity={name:file_identity(Path(__file__).parent/name) for name in ('observation_filter.py','mvs_observations.py')},
                   elapsed_seconds=time.monotonic()-start,
                   sampling=dict(target_max_points=max_points, post_filter_candidate_count=retained, sampled_vertex_count=target),
                   quality_warning='Unknown observations remain in the main cloud and in a separate layer.' if totals[0] else '',
                   explanation='Only repeated, angularly diverse free-space conflicts without supporting views are excluded. Occlusion, missing/unstable depths, poor conditioning and conflicting evidence remain unknown. Scores do not establish real-world accuracy. Candidate views are bounded; no untested region is certified empty.')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_provenance(artifacts/'receipt.json', receipt)
    staged.replace(output_path)
    write_provenance(summary_path, receipt)
    for path in raw_paths.values():
        path.unlink(missing_ok=True)
    return receipt
