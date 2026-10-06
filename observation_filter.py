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
from reconstruction_metrics import Snapshot
from progress_output import WorkProgress


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
    with WorkProgress('Checking observation-view conditioning', len(views), unit='views') as progress:
        for index, view in enumerate(views, 1):
            values = []
            ids = view['point_ids']
            for pid in ids[::max(1, (len(ids)+255)//256)]:
                xyz, _, track = points[int(pid)]
                a = xyz-view['C']
                an = np.linalg.norm(a)
                if an == 0:
                    continue
                centers = [by_id[int(iid)]['C'] for iid in track
                           if int(iid) != view['id'] and int(iid) in by_id]
                if centers:
                    b = xyz-np.asarray(centers)
                    lengths = np.linalg.norm(b, axis=1)
                    valid = lengths > 0
                    if valid.any():
                        cosines = (b[valid] @ a)/(an*lengths[valid])
                        # Opposite-facing/near-collinear rays are poorly conditioned.
                        values.append(float(np.sqrt(np.maximum(0., 1-np.clip(cosines, -1, 1)**2)).max()))
            result[view['id']] = float(np.median(values)) if values else 0.
            progress.update(index)
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


def view_measurements(xyz, view, pair, relative_sigma, normals, cos_normal):
    """Return only coherent observed footprints; holes never vote empty space."""
    depth, normal = pair
    h, w = depth.shape
    uv, z, good = project(xyz, view, (w, h))
    idx = np.flatnonzero(good)
    if not len(idx):
        return idx, np.empty(0, bool), np.empty(0, bool), np.empty(0, np.float32)
    pixel = np.floor(uv[idx]-.5).astype(int)
    x, y = pixel.T
    samples = np.column_stack([depth[y+dy, x+dx] for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))])
    d = np.mean(samples, axis=1)
    tolerance = d * relative_sigma
    valid = np.isfinite(samples).all(1) & (samples > 0).all(1)
    valid &= np.ptp(samples, axis=1) <= tolerance
    n = normal[y, x].astype(float)
    length = np.linalg.norm(n, axis=1)
    valid &= np.isfinite(n).all(1) & (length > .5)
    idx, d, tolerance, n, length = idx[valid], d[valid], tolerance[valid], n[valid], length[valid]
    if not len(idx):
        return idx, np.empty(0, bool), np.empty(0, bool), np.empty(0, np.float32)
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
    return idx, near, z[idx] < d-tolerance, (error/d).astype(np.float32)


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
        if eligible_count >= policy['max_views']:
            break
        sine = quality.get(view['id'], 0.)
        if sine <= 0:
            continue
        pair = store.get(view)
        if pair is None:
            continue
        depth, normal = pair
        h, w = depth.shape
        f = min(view['K'][0, 0]*w/view['size'][0], view['K'][1, 1]*h/view['size'][1])
        relative_sigma = policy['sigma_multiplier'] * view['noise_px'] / (f*sine)
        if relative_sigma > policy['max_relative_uncertainty']:
            continue
        eligible_count += 1
        idx, near, conflict, rel = view_measurements(xyz, view, pair, relative_sigma, normals, cos_normal)
        if not len(idx):
            continue
        evidence['tested'][idx] += 1
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


def confirm_free_space_conflicts(xyz, evidence, views, store, quality, policy=None,
                                 normals=None, deadline=None):
    """Veto bounded-search conflicts using every eligible projectable view.

    One credible support observation is enough to make a conflict ambiguous.
    A deadline or unavailable expected map leaves unconfirmed points unknown.
    Only points already carrying two diverse free-space votes can be removed;
    this pass never promotes an unknown or changes an existing supported point.
    """
    policy = checked_policy(policy)
    active = np.flatnonzero(evidence['status'] == CONFLICT)
    candidate_count = len(active)
    rescued = unavailable = checked_views = 0
    tested_views = []
    timed_out = False
    cos_normal = np.cos(np.deg2rad(policy['max_normal_degrees']))
    if candidate_count:
        with WorkProgress('Confirming free-space conflicts', len(views), unit='views') as progress:
            for view_number, view in enumerate(views, 1):
                if len(active) and deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break
                checked_views += 1
                sine = quality.get(view['id'], 0.)
                if not len(active) or sine <= 0:
                    progress.update(view_number)
                    continue
                # Cheap full-candidate projection precedes loading or hashing
                # maps. A 128-point view-ranking sample is insufficient here.
                _, _, projects = project(xyz[active], view)
                if not projects.any():
                    progress.update(view_number)
                    continue
                pair = store.get(view)
                if pair is None:
                    missing = active[projects]
                    evidence['status'][missing] = UNKNOWN
                    unavailable += len(missing)
                    active = active[~projects]
                    progress.update(view_number)
                    continue
                h, w = pair[0].shape
                f = min(view['K'][0, 0]*w/view['size'][0], view['K'][1, 1]*h/view['size'][1])
                sigma = policy['sigma_multiplier'] * view['noise_px'] / (f*sine)
                if sigma <= policy['max_relative_uncertainty']:
                    selected = active[projects]
                    idx, near, _, relative_error = view_measurements(
                        xyz[selected], view, pair, sigma,
                        normals[selected] if normals is not None else None, cos_normal)
                    if len(idx):
                        tested_views.append(view['id'])
                    supported = selected[idx[near]]
                    if len(supported):
                        evidence['status'][supported] = UNKNOWN
                        evidence['support'][supported] += 1
                        evidence['tested'][supported] += 1
                        evidence['support_a'][supported] = view['id']
                        evidence['best_relative_error'][supported] = np.fmin(
                            evidence['best_relative_error'][supported], relative_error[near])
                        rescued += len(supported)
                        active = active[evidence['status'][active] == CONFLICT]
                progress.update(view_number)
    unresolved = len(active) if timed_out else 0
    if unresolved:
        evidence['status'][active] = UNKNOWN
    confirmed = candidate_count-rescued-unavailable-unresolved
    return evidence, tested_views, dict(
        policy='all_eligible_projectable_views_checked_or_conflict_retained_as_unknown',
        candidate_count=candidate_count, confirmed_conflict_count=confirmed,
        rescued_by_support_count=rescued, unresolved_budget_count=unresolved,
        unavailable_observation_count=unavailable, checked_view_count=checked_views,
        budget_exhausted=timed_out,
        interpretation='Any credible support vetoes deletion. Missing or incoherent depth casts no free-space vote; '
                       'the original pair of diverse free-space observations remains required. Vote counters retain '
                       'initial local-search evidence plus the first rescue support, not duplicated verification votes.')


class DensityObservationProtector:
    """Rescue assessed original vertices using authenticated MVS evidence only.

    The caller bounds the candidate count. Missing unrecorded observations,
    occlusion and contradictory votes remain unknown; none causes deletion.
    """

    def __init__(self, dense, snapshot, expected, binding, *, max_views=24):
        self.policy = checked_policy({'max_views': max_views})
        self.snapshot = snapshot
        self.store = None
        self.views = []
        self.quality = {}
        self.counts = np.zeros(3, dtype=np.int64)
        self.seen_views = set()
        self.tested_count = 0
        self.reason = ''
        self.confirmation = {}
        dense = Path(dense).resolve()
        if not isinstance(binding, dict):
            raise ValueError('Observation protection requires a completed MVS identity receipt')
        if not isinstance(expected, dict) or not expected:
            self.reason = 'missing_authenticated_observation_receipt'
            return
        for name, identity in binding.get('dense_model', {}).items():
            if (identity is None or expected.get('sparse/'+name) != identity
                    or snapshot.identity(dense/'sparse'/name) != identity):
                raise ValueError('Observation protection model differs from MVS receipt')
        if not binding.get('dense_model'):
            raise ValueError('Observation protection requires a bound calibrated model')
        views, points = read_model(dense/'sparse', snapshot)
        self.views = [v for v in views if all(
            expected.get('stereo/'+folder+'/'+v['name']+'.geometric.bin') is not None
            for folder in ('depth_maps', 'normal_maps'))]
        if len(self.views) < 2:
            self.reason = 'fewer_than_two_authenticated_depth_normal_pairs'
            return
        self.quality = conditioning(self.views, points)
        if sum(value > 0 for value in self.quality.values()) < 2:
            self.reason = 'insufficient_sparse_track_conditioning'
            return
        self.store = DepthStore(dense, snapshot, expected, max_open=8)

    @property
    def available(self):
        return self.store is not None

    def __call__(self, chunk):
        return self.assess(chunk)['status'] == SUPPORTED

    def assess(self, chunk):
        """Return measured classes without assigning any class to unseen points."""
        if not self.available:
            raise ValueError('Unavailable observation protection must not assess candidates')
        normals = (np.column_stack([chunk[n] for n in ('nx', 'ny', 'nz')])
                   if {'nx', 'ny', 'nz'} <= set(chunk.dtype.names) else None)
        evidence, used = assess_points(positions(chunk), self.views, self.store,
                                       self.quality, self.policy, normals)
        self.counts += np.bincount(evidence['status'], minlength=3)
        self.tested_count += int(np.count_nonzero(evidence['tested']))
        self.seen_views.update(used)
        return evidence

    def summary(self):
        assessed = int(self.counts.sum())
        return dict(status=('unavailable' if not self.available else
                            'assessed_candidates' if assessed else 'no_candidates_assessed'),
                    reason=self.reason, observations_checked=self.tested_count > 0,
                    assessed_candidate_count=assessed, tested_candidate_count=self.tested_count,
                    unknown_candidate_count=int(self.counts[UNKNOWN]),
                    supported_candidate_count=int(self.counts[SUPPORTED]),
                    conflict_candidate_count=int(self.counts[CONFLICT]),
                    tested_view_ids=sorted(self.seen_views), policy=self.policy,
                    available_view_count=len(self.views),
                    conditioned_view_count=sum(value > 0 for value in self.quality.values()),
                    tested_view_fraction=len(self.seen_views)/len(self.views) if self.views else 0.,
                    tested_candidate_fraction=self.tested_count/assessed if assessed else 0.,
                    conflict_confirmation=self.confirmation,
                    authentication='completed_MVS_model_and_each_accessed_geometric_depth_normal_hash',
                    interpretation=('Only multi-view supported original vertices are rescued. '
                        'Unknown and unevaluated candidates retain the density decision; '
                        'missing depth is not conflicting evidence.'))

    def confirm_conflicts(self, chunk, evidence, deadline=None):
        normals = (np.column_stack([chunk[n] for n in ('nx', 'ny', 'nz')])
                   if {'nx', 'ny', 'nz'} <= set(chunk.dtype.names) else None)
        before = np.bincount(evidence['status'], minlength=3)
        evidence, used, self.confirmation = confirm_free_space_conflicts(
            positions(chunk), evidence, self.views, self.store, self.quality,
            self.policy, normals, deadline)
        self.counts += np.bincount(evidence['status'], minlength=3)-before
        self.seen_views.update(used)
        return evidence

    def close(self):
        if self.store is not None:
            self.store.close()


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
    seen_views, offset = set(), 0
    raw_paths = {name: artifacts/(name+'.raw') for name in (*names, 'retained')}
    try:
        with ExitStack() as stack:
            handles = {name: stack.enter_context(path.open('wb')) for name, path in raw_paths.items()}
            progress = stack.enter_context(WorkProgress('Assessing original observations', count, unit='points'))
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
                    progress.update(offset)
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


def spatial_assessment_batches(xyz, batch_size=4096):
    """Partition a bounded point sample into local, evenly scheduled batches.

    A batch spread across the entire scene lets the most broadly visible cameras
    consume every view slot. Median splits give each small region its own view
    search. Bit-reversed leaf order alternates distant branches, so an elapsed
    budget does not assess only one end of the room. The returned values are
    sample indices, never generated coordinates or propagated classifications.
    """
    xyz = np.asarray(xyz)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Observation batch size must be a positive integer')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Observation batches require finite XYZ positions')
    if not len(xyz):
        return []
    leaves = []
    pending = [np.arange(len(xyz), dtype=np.int64)]
    while pending:
        members = pending.pop()
        if len(members) <= batch_size:
            leaves.append(np.sort(members))
            continue
        extents = [np.ptp(xyz[members, axis]) for axis in range(3)]
        axis = int(np.argmax(extents))
        middle = len(members)//2
        order = np.argpartition(xyz[members, axis], middle)
        # Stack traversal keeps leaves ordered along their spatial branches.
        pending.extend((members[order[middle:]], members[order[:middle]]))
    bits = max(1, (len(leaves)-1).bit_length())
    order = sorted(range(len(leaves)), key=lambda i: int(f'{i:0{bits}b}'[::-1], 2))
    return [leaves[index] for index in order]


def filter_bounded_observed_cloud(input_path, output_path, summary_path, dense, *,
                                  max_points=0, policy=None, expected=None, binding=None,
                                  max_candidates=250000, budget_seconds=120.,
                                  assessment_batch_size=4096):
    """Keep unknown/unsampled original geometry; remove only measured conflicts.

    Source-index sampling and spatial assessment batches are independent of I/O
    chunk boundaries. The time budget stops new assessments; authentication and
    streaming publication still finish. It never converts unfinished work into
    rejection, and it never infers empty space from density or missing maps.
    """
    from dense_surface_filter import sample_original_vertices

    policy = checked_policy(policy)
    if type(max_points) is not int or max_points < 0:
        raise ValueError('Export point budget must be a nonnegative integer')
    if not np.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError('Observation runtime budget must be finite and positive')
    if type(assessment_batch_size) is not int or assessment_batch_size < 1:
        raise ValueError('Observation batch size must be a positive integer')
    input_path, output_path, summary_path, dense = map(lambda p: Path(p).resolve(),
                                                     (input_path, output_path, summary_path, dense))
    if len({input_path, output_path, summary_path}) != 3:
        raise ValueError('Input, output and summary paths must be distinct')
    for destination in (output_path, summary_path):
        if (destination in {dense/'final.ply', dense/'fused.ply'} or
                any(destination.is_relative_to(dense/folder) for folder in ('images', 'sparse', 'stereo'))):
            raise ValueError('Preserve original fusion, final and calibrated observations; write a separate candidate')
    started = time.monotonic()
    snapshot = Snapshot()
    input_identity = snapshot.identity(input_path)
    codes = {name: snapshot.identity(Path(__file__).with_name(name)) for name in
             ('observation_filter.py', 'dense_surface_filter.py', 'point_cloud_filter.py', 'mvs_observations.py')}
    with input_path.open('rb') as handle:
        header = parse_ply_header(handle)
    count = header['vertex_count']
    sample, indices, stride = sample_original_vertices(input_path, count, max_candidates)
    assessor = None
    reason = 'missing_authenticated_observations'
    assessment_available = False
    evidence_batches, assessed_index_batches, assessed, exhausted = [], [], 0, False
    evidence = np.empty(0, dtype=EVIDENCE_DTYPE)
    assessment_batches = 0
    observation_summary = {}
    assessment_started, assessment_elapsed = None, 0.
    try:
        if isinstance(binding, dict) and isinstance(expected, dict) and expected:
            assessor = DensityObservationProtector(dense, snapshot, expected, binding, max_views=policy['max_views'])
            assessor.policy = policy
            assessment_available = assessor.available
            reason = assessor.reason
        if assessment_available:
            assessment_started = time.monotonic()
            deadline = assessment_started + budget_seconds
            # Keep time for a global support check before any bounded-search
            # conflict can remove an original point.
            confirmation_reserve = min(30., budget_seconds*.25)
            search_deadline = deadline-confirmation_reserve
            with WorkProgress('Assessing sampled observations', len(sample), unit='points') as progress:
                for selected in spatial_assessment_batches(positions(sample), assessment_batch_size):
                    if time.monotonic() >= search_deadline:
                        exhausted = True
                        break
                    evidence = assessor.assess(sample[selected])
                    evidence_batches.append(evidence)
                    assessed_index_batches.append(indices[selected])
                    assessed += len(evidence)
                    assessment_batches += 1
                    progress.update(assessed)
            if evidence_batches:
                evidence = np.concatenate(evidence_batches)
                sample_order = np.searchsorted(indices, np.concatenate(assessed_index_batches))
                evidence = assessor.confirm_conflicts(sample[sample_order], evidence, deadline=deadline)
        if assessor is not None:
            observation_summary = assessor.summary()
            exhausted |= observation_summary.get('conflict_confirmation', {}).get('budget_exhausted', False)
    finally:
        if assessment_started is not None:
            assessment_elapsed = time.monotonic() - assessment_started
        if assessor is not None:
            assessor.close()
    assessed_indices = (np.concatenate(assessed_index_batches) if assessed_index_batches
                        else np.empty(0, dtype=np.int64))
    # Downstream streaming uses searchsorted in original PLY order. Preserve
    # the exact index/evidence association after spatially reordered assessment.
    source_order = np.argsort(assessed_indices)
    assessed_indices, evidence = assessed_indices[source_order], evidence[source_order]
    supported = assessed_indices[evidence['status'] == SUPPORTED]
    conflicts = assessed_indices[evidence['status'] == CONFLICT]
    totals = {'unknown': count - len(supported) - len(conflicts),
              'supported': len(supported), 'conflict': len(conflicts)}
    retained = count - len(conflicts)
    if retained == 0:
        raise ValueError('All points conflict; candidate withheld for review')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = output_path.parent / ('observation_evidence_' + uuid.uuid4().hex)
    artifacts.mkdir()
    evidence_path = artifacts / 'sampled_points.npz'
    np.savez(evidence_path, source_indices=assessed_indices, evidence=evidence)
    names = ('unknown', 'supported', 'conflict')
    raw_paths = {name: artifacts/(name+'.raw') for name in (*names, 'retained')}
    offset = 0
    with ExitStack() as stack:
        handles = {name: stack.enter_context(path.open('wb')) for name, path in raw_paths.items()}
        progress = stack.enter_context(WorkProgress('Writing observation classification layers', count, unit='points'))
        for chunk in vertex_chunks(input_path):
            if not np.isfinite(positions(chunk)).all():
                raise ValueError('Non-finite input geometry')
            status = np.zeros(len(chunk), dtype=np.uint8)
            for value, source_indices in ((SUPPORTED, supported), (CONFLICT, conflicts)):
                left, right = np.searchsorted(source_indices, (offset, offset + len(chunk)))
                status[source_indices[left:right] - offset] = value
            little = chunk.astype(chunk.dtype.newbyteorder('<'), copy=False)
            for value, name in enumerate(names):
                little[status == value].tofile(handles[name])
            little[status != CONFLICT].tofile(handles['retained'])
            offset += len(chunk)
            progress.update(offset)
    if offset != count:
        raise ValueError('Original point count changed during filtering')
    outputs = {}
    properties = header['vertex_properties']
    for name in names:
        path = artifacts/(name+'.ply')
        write_cloud(path, raw_paths[name], properties, totals[name])
        outputs[name] = {'path': str(path), 'vertex_count': totals[name], 'identity': file_identity(path)}
    target = min(max_points, retained) if max_points else retained
    selected = raw_paths['retained']
    if target < retained:
        raw = np.memmap(selected, dtype=sample.dtype.newbyteorder('<'), mode='r', shape=(retained,))
        chosen = np.sort(np.random.default_rng(0).choice(retained, target, replace=False))
        selected = artifacts/'sampled.raw'
        raw_paths['sampled'] = selected
        try:
            with selected.open('wb') as handle:
                for offset in range(0, target, 250000):
                    raw[chosen[offset:offset+250000]].tofile(handle)
        finally:
            del raw
    staged = artifacts/'candidate.ply'
    write_cloud(staged, selected, properties, target)
    snapshot.verify_unchanged()
    unknown_warning = (f"Unknown observations retained: {totals['unknown']:,} points, including "
                       f"{count-assessed:,} unassessed points. These are not verified errors or verified surfaces.")
    report = dict(
        status='ok', method='observation_support', assessment_mode='bounded_observed', schema_version=3,
        input=str(input_path), input_identity=input_identity, output=str(output_path),
        output_identity=file_identity(staged), reconstruction_binding=binding, source_vertex_count=count,
        input_vertex_count=count, output_vertex_count=target, max_points=max_points,
        geometry_candidate_count=retained, geometry_removed_vertex_count=len(conflicts),
        geometry_removed_ratio=len(conflicts)/count, budget_omitted_vertex_count=retained-target,
        removed_vertex_count=count-target, removed_ratio=1-target/count, source_retained_ratio=target/count,
        classification=totals, policy=policy,
        assessment=dict(status='assessed' if assessed else 'unavailable' if not assessment_available
                        else 'disabled' if not len(sample) else 'budget_exhausted',
                        reason=reason, candidate_budget=max_candidates, selected_candidate_count=len(sample),
                        assessed_candidate_count=assessed, unevaluated_candidate_count=count-assessed,
                        assessed_source_fraction=assessed/count,
                        tested_candidate_count=observation_summary.get('tested_candidate_count', 0),
                        tested_source_fraction=observation_summary.get('tested_candidate_count', 0)/count,
                        classified_source_fraction=(len(supported)+len(conflicts))/count,
                        assessed_unknown_count=int(np.count_nonzero(evidence['status'] == UNKNOWN)),
                        sampling_stride=stride, budget_seconds=budget_seconds, budget_exhausted=exhausted,
                        assessment_order='balanced_spatial_batches_interleaved_across_regions',
                        assessment_batch_size=assessment_batch_size, assessment_batch_count=assessment_batches,
                        candidate_selection='uniform_original_source_index_stride',
                        confirmation_reserved_seconds=min(30., budget_seconds*.25),
                        elapsed_seconds=assessment_elapsed,
                        budget_scope='Spatial partitioning and assessment batches including accessed-map authentication; initial source/model checks and output streaming excluded.',
                        observations_checked=observation_summary.get('observations_checked', False)),
        evidence=dict(path=str(evidence_path), identity=file_identity(evidence_path),
                      index='zero-based original PLY vertex order; source_indices identifies assessed vertices only',
                      default_class='unknown', classes={'0': 'unknown', '1': 'supported', '2': 'conflict'},
                      interpretation='Unlisted source indices were not assessed and are retained; no classification is propagated to neighbors.'),
        layers=outputs, used_view_count=len(observation_summary.get('tested_view_ids', [])),
        available_view_count=observation_summary.get('available_view_count', 0),
        conditioned_view_count=observation_summary.get('conditioned_view_count', 0),
        tested_view_fraction=observation_summary.get('tested_view_fraction', 0.),
        tested_view_ids=observation_summary.get('tested_view_ids', []),
        conflict_confirmation=observation_summary.get('conflict_confirmation', {}),
        observation_inputs={str(path): identity for path, (identity, _) in snapshot.files.items()},
        code_identity=codes, elapsed_seconds=time.monotonic()-started, assessment_elapsed_seconds=assessment_elapsed,
        sampling=dict(method='full_resolution' if target == retained else 'uniform_export_sample',
                      target_max_points=max_points, post_filter_candidate_count=retained, sampled_vertex_count=target),
        quality_warning=unknown_warning if totals['unknown'] else '',
        explanation='Only assessed angularly diverse free-space conflicts are removed after checking all eligible '
                    'projectable views for supporting evidence; any support or unfinished verification retains the point. '
                    'Low density, disconnected geometry, missing or ambiguous depths, occlusion and budget-limited points '
                    'are preserved as unknown. All retained positions, colors and normals are original observations; '
                    'no holes are filled and no sparse neighborhood inherits a sampled verdict.')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_provenance(artifacts/'receipt.json', report)
    staged.replace(output_path)
    write_provenance(summary_path, report)
    for path in raw_paths.values():
        path.unlink(missing_ok=True)
    return report
