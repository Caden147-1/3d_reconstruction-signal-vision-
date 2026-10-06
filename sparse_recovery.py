"""Bounded late intrinsic refinement and registration, preserving the baseline.

Only a pending, locked sparse run may use this stage. Trial models live outside
the selectable sparse directory until independently recomputed geometry passes
the retention checks. Sparse proxy improvements do not prove dense completeness.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import threading
import time
import uuid

import numpy as np

from colmap_workspace_check import (
    CAMERA_MODELS, PENDING_NAME, PROVENANCE_NAME, inspect_sparse_model,
    primary_sparse_model, read_exact, sparse_model_identity, write_provenance,
)
from compute_resources import worker_count
from generated_input_cleanup import checked_path
from process_control import managed_process
from sparse_optimization import find_model, read_metadata, video_time
from sparse_quality import assess_sparse_quality, summarize_coverage
from workspace_lock import inherited_lock_held


REPORT_NAME = 'sparse_recovery_summary.json'
SUPPORTED_MODELS = {'SIMPLE_PINHOLE', 'PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'OPENCV'}


class PromotionRollbackError(RuntimeError):
    """The baseline needs manual restoration; never certify this workspace."""


class BaselineChangedError(ValueError):
    """The locked baseline was altered independently of this recovery attempt."""


def native_exit_status(code, log_text):
    """A normal bounded mapper export differs from killing its process tree."""
    if code != 0:
        return {'status': 'failed', 'exit_code': code, 'export_eligible': False}
    if 'Reached maximum runtime' in log_text:
        return {'status': 'budget_limited', 'exit_code': code, 'export_eligible': True,
                'budget_limited': True, 'final_global_refinement_may_be_skipped': True}
    if 'Keeping reconstruction due to interrupt' in log_text:
        return {'status': 'interrupted', 'exit_code': code, 'export_eligible': False}
    return {'status': 'complete', 'exit_code': code, 'export_eligible': True}


def read_cameras(model):
    cameras = {}
    with (model / 'cameras.bin').open('rb') as stream:
        for _ in range(struct.unpack('<Q', read_exact(stream, 8))[0]):
            cid, kind, width, height = struct.unpack('<iiQQ', read_exact(stream, 24))
            name, count = CAMERA_MODELS[kind]
            cameras[cid] = {'model': name, 'width': width, 'height': height,
                            'params': list(struct.unpack('<' + 'd' * count, read_exact(stream, 8 * count)))}
    return cameras


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm < 1e-12:
        raise ValueError('Invalid sparse camera rotation')
    w, x, y, z = q / norm
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y-z*w), 2 * (x*z+y*w)],
                     [2 * (x*y+z*w), 1 - 2 * (x*x+z*z), 2 * (y*z-x*w)],
                     [2 * (x*z-y*w), 2 * (y*z+x*w), 1 - 2 * (x*x+y*y)]])


def project_camera(xyz, camera):
    """Project supported original camera models, including nonzero distortion."""
    kind, p = camera['model'], camera['params']
    if kind not in SUPPORTED_MODELS:
        raise ValueError(f'Unsupported recovery projection: {kind}')
    if kind in ('PINHOLE', 'OPENCV'):
        fx, fy, cx, cy = p[:4]
    else:
        fx, cx, cy = p[:3]
        fy = fx
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        x, y = xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2]
        r2 = x*x + y*y
        radial = np.ones(len(x))
        if kind in ('SIMPLE_RADIAL', 'RADIAL'):
            radial += p[3] * r2
            if kind == 'RADIAL':
                radial += p[4] * r2*r2
        elif kind == 'OPENCV':
            radial += p[4] * r2 + p[5] * r2*r2
        xd, yd = x * radial, y * radial
        if kind == 'OPENCV':
            p1, p2 = p[6:8]
            xd += 2 * p1*x*y + p2 * (r2 + 2*x*x)
            yd += p1 * (r2 + 2*y*y) + 2 * p2*x*y
        return np.column_stack((fx * xd + cx, fy * yd + cy))


def evaluate_model(model, metadata):
    """Recompute observation errors; never trust stale points3D.error after BA."""
    model = Path(model)
    checked = inspect_sparse_model(model)
    cameras = read_cameras(model)
    if any(camera['model'] not in SUPPORTED_MODELS for camera in cameras.values()):
        raise ValueError('Sparse recovery supports perspective pinhole/radial/OpenCV cameras only')
    count = checked['point_count']
    ids, points = np.empty(count, dtype=np.uint64), np.empty((count, 3), dtype=float)
    with (model / 'points3D.bin').open('rb') as stream:
        if struct.unpack('<Q', read_exact(stream, 8))[0] != count:
            raise ValueError('Sparse point count changed during recovery evaluation')
        for index in range(count):
            point = struct.unpack('<Q3d3BdQ', read_exact(stream, 51))
            ids[index], points[index] = point[0], point[1:4]
            stream.seek(point[-1] * 8, 1)
    order = np.argsort(ids)
    ids, points = ids[order], points[order]
    if len(ids) > 1 and np.any(ids[1:] == ids[:-1]):
        raise ValueError('Duplicate sparse point identifiers')
    observations = {}
    total_error = squared_error = 0.
    observed = invalid_projection = 0
    dtype = [('x', '<f8'), ('y', '<f8'), ('id', '<i8')]
    with (model / 'images.bin').open('rb') as stream:
        for _ in range(struct.unpack('<Q', read_exact(stream, 8))[0]):
            pose = struct.unpack('<i7di', read_exact(stream, 64))
            name_bytes = bytearray()
            while (character := read_exact(stream, 1)) != b'\0':
                name_bytes.extend(character)
                if len(name_bytes) > 65536:
                    raise ValueError('Invalid sparse image name')
            name = name_bytes.decode('utf-8').replace('\\', '/')
            size = struct.unpack('<Q', read_exact(stream, 8))[0]
            data = np.frombuffer(read_exact(stream, size * 24), dtype=dtype)
            chosen = np.flatnonzero(data['id'] >= 0)
            observations[name] = chosen.astype(np.uint32)
            if not len(chosen):
                continue
            point_ids = data['id'][chosen].astype(np.uint64)
            offsets = np.searchsorted(ids, point_ids)
            if np.any(offsets >= len(ids)) or np.any(ids[offsets] != point_ids):
                raise ValueError('Image observation references a missing sparse point')
            camera_points = points[offsets] @ rotation(pose[1:5]).T + np.array(pose[5:8])
            projected = project_camera(camera_points, cameras[pose[-1]])
            measured = np.column_stack((data['x'][chosen], data['y'][chosen]))
            if not np.isfinite(measured).all():
                raise ValueError('Non-finite sparse image observations')
            valid = np.isfinite(projected).all(axis=1) & (camera_points[:, 2] > 1e-9)
            errors = np.linalg.norm(projected[valid] - measured[valid], axis=1)
            invalid_projection += int(np.count_nonzero(~valid))
            total_error += float(errors.sum())
            squared_error += float(errors @ errors)
            observed += len(chosen)
    coverage = {}
    for name, row in metadata.items():
        stamp = video_time(row)
        clip = row.get('video_name')
        if stamp is None or not clip:
            continue
        entry = coverage.setdefault(clip, {'input': [], 'registered': []})
        entry['input'].append(stamp)
        if name in observations:
            entry['registered'].append(stamp)
    for clip, entry in list(coverage.items()):
        available, registered = sorted(entry['input']), sorted(entry['registered'])
        anchors = [available[0], *registered, available[-1]]
        coverage[clip] = {'input_frames': len(available), 'registered_frames': len(registered),
                          'max_uncovered_gap_s': max((b-a for a, b in zip(anchors, anchors[1:])), default=0.)}
    valid_observations = observed - invalid_projection
    summary = {'registered_images': len(observations), 'points3D': count, 'observations': observed,
               'invalid_projection_count': invalid_projection,
               'mean_reprojection_error_px': total_error / valid_observations if valid_observations else None,
               'rms_reprojection_error_px': math.sqrt(squared_error / valid_observations) if valid_observations else None,
               'reprojection_error_source': 'recomputed from current poses, points and distortion; observation weighted',
               'video_coverage': coverage,
               'unexpected_images': sorted(set(observations) - set(metadata)) if metadata else []}
    return {'summary': summary, 'cameras': cameras, 'observations': observations}


def intrinsic_checks(before, after):
    if set(before) != set(after):
        return False
    for cid, original in before.items():
        current = after[cid]
        if any(current[key] != original[key] for key in ('model', 'width', 'height')):
            return False
        p, previous = np.asarray(current['params']), np.asarray(original['params'])
        if not np.isfinite(p).all() or p.shape != previous.shape:
            return False
        split = 4 if current['model'] in ('PINHOLE', 'OPENCV') else 3
        focal_count = split - 2
        if np.any(previous[:focal_count] <= 0) or np.any(p[:focal_count] / previous[:focal_count] < .7) or np.any(p[:focal_count] / previous[:focal_count] > 1.3):
            return False
        if not np.allclose(p[focal_count:split], previous[focal_count:split], atol=1e-8, rtol=0):
            return False
        fx, fy = p[0], p[1] if focal_count == 2 else p[0]
        cx, cy = p[focal_count:split]
        radius2 = max(cx, current['width']-cx)**2/fx**2 + max(cy, current['height']-cy)**2/fy**2
        r2 = np.linspace(0., radius2, 64)
        extra = p[split:]
        if len(extra):
            k1, k2 = extra[0], extra[1] if len(extra) > 1 else 0.
            radial = 1 + k1*r2 + k2*r2*r2
            derivative = 1 + 3*k1*r2 + 5*k2*r2*r2
            if np.max(np.abs(extra)) > .5 or np.any(radial <= .5) or np.any(radial >= 1.5) or np.any(derivative <= .1):
                return False
            if current['model'] == 'OPENCV' and np.any(np.abs(extra[2:]) > .05):
                return False
    return True


def candidate_gate(baseline, candidate, *, require_improvement=True):
    before, after = baseline['summary'], candidate['summary']
    original, proposed = baseline['observations'], candidate['observations']
    missing = sorted(set(original) - set(proposed))
    retained = sum(np.intersect1d(indices, proposed.get(name, []), assume_unique=True).size
                   for name, indices in original.items())
    retention = retained / max(before['observations'], 1)
    initial_error, final_error = before['mean_reprojection_error_px'], after['mean_reprojection_error_px']
    checks = {
        'retains_all_registered_images': not missing,
        'retains_95_percent_original_observations': retention >= .95,
        'retains_95_percent_points': after['points3D'] >= .95 * before['points3D'],
        'valid_projections': after['invalid_projection_count'] == 0,
        'known_input_images': not after.get('unexpected_images'),
        'reprojection_error': initial_error is not None and final_error is not None and
                              final_error <= initial_error * 1.05 + .05,
        'intrinsics': intrinsic_checks(baseline['cameras'], candidate['cameras']),
        'temporal_coverage': all(clip in after['video_coverage'] and
            after['video_coverage'][clip]['max_uncovered_gap_s'] <= entry['max_uncovered_gap_s'] + 1e-6
            for clip, entry in before['video_coverage'].items()),
    }
    improved = (after['registered_images'] > before['registered_images'] or
                (initial_error is not None and final_error is not None and
                 final_error < initial_error - max(.01, .01 * initial_error)))
    if require_improvement:
        checks['objective_improved'] = improved
    return {'accepted': all(checks.values()), 'checks': checks,
            'baseline_observation_retention': retention, 'lost_registered_images': missing,
            'additional_registered_images': sorted(set(proposed) - set(original)),
            'objective_improved': improved, 'dense_completeness_verified': False}


def require_pending_run(workspace):
    if not inherited_lock_held(workspace):
        raise ValueError('Sparse recovery requires the inherited workspace lock')
    if (workspace / PROVENANCE_NAME).exists() or (workspace / 'dense').exists():
        raise ValueError('Sparse recovery cannot modify a certified or dense workspace')
    pending = json.loads((workspace / PENDING_NAME).read_text(encoding='utf-8'))
    if pending.get('schema_version') != 1 or not isinstance(pending.get('inputs'), dict):
        raise ValueError('Sparse recovery requires a pending sparse input record')


def promote_candidate(workspace, baseline_path, candidate_path, backup_path):
    """Two directory renames under the run lock, with rollback on failure."""
    require_pending_run(workspace)
    for path in (baseline_path, candidate_path, backup_path):
        checked_path(workspace, path)
    if not baseline_path.is_relative_to(workspace / 'sparse') or backup_path.exists():
        raise ValueError('Invalid sparse recovery promotion paths')
    baseline_path.rename(backup_path)
    try:
        candidate_path.rename(baseline_path)
    except BaseException as original_error:
        try:
            backup_path.rename(baseline_path)
        except OSError as rollback_error:
            raise PromotionRollbackError(f'Promotion rollback failed; restore {backup_path} to {baseline_path}: '
                                         f'{rollback_error}') from original_error
        raise


def run_recovery(workspace, executable, *, ba_seconds=300., mapper_seconds=900.,
                 threads=0, launch=None, clock=time.monotonic):
    workspace, executable = Path(workspace).resolve(), Path(executable).resolve()
    require_pending_run(workspace)
    if any(not math.isfinite(value) or value <= 0 for value in (ba_seconds, mapper_seconds)):
        raise ValueError('Sparse recovery time budgets must be finite and positive')
    report = {'schema_version': 1, 'status': 'running', 'accepted': False, 'stages': [],
              'ba_budget_seconds': ba_seconds, 'mapper_budget_seconds': mapper_seconds,
              'interpretation': 'Bounded sparse quality recovery; does not establish dense surface completeness.'}
    report_path = checked_path(workspace, workspace / REPORT_NAME)
    baseline_path = primary_sparse_model(workspace)
    if baseline_path is None:
        raise ValueError('Sparse recovery requires a complete baseline model')
    checked_path(workspace, baseline_path)
    calibration = json.loads((workspace / 'video_intrinsics_summary.json').read_text(encoding='utf-8'))
    if not calibration.get('intrinsics_held_during_mapping'):
        report.update(status='skipped', reason='intrinsics_were_not_held')
        write_provenance(report_path, report)
        return report
    cameras = read_cameras(baseline_path)
    if len(cameras) != 1 or next(iter(cameras.values()))['model'] not in SUPPORTED_MODELS:
        report.update(status='skipped', reason='requires_supported_single_video_camera')
        write_provenance(report_path, report)
        return report
    metadata = read_metadata(workspace / 'metadata/images_manifest.csv')
    if not metadata:
        raise ValueError('Sparse recovery requires image metadata for coverage validation')
    baseline_identity = sparse_model_identity(baseline_path)
    baseline = evaluate_model(baseline_path, metadata)
    report.update(baseline_model=str(baseline_path.relative_to(workspace)),
                  baseline=baseline['summary'], baseline_cameras=baseline['cameras'],
                  baseline_identity=baseline_identity)
    attempt = checked_path(workspace, workspace / 'sparse_recovery' / uuid.uuid4().hex)
    attempt.mkdir(parents=True, exist_ok=False)
    report['attempt_directory'] = str(attempt.relative_to(workspace))
    resources_path = workspace / 'compute_resources.json'
    resources = json.loads(resources_path.read_text(encoding='utf-8')) if resources_path.is_file() else {}
    gpu = bool(resources.get('ba_gpu_probe', {}).get('supported'))
    threads = worker_count(threads)
    deadline = clock() + min(1200., ba_seconds + mapper_seconds)

    def execute(label, command, budget):
        remaining = min(budget, deadline - clock())
        record = {'stage': label, 'command': command, 'timeout_seconds': max(0., remaining)}
        report['stages'].append(record)
        write_provenance(report_path, report)
        if remaining <= 0:
            record['status'] = 'budget_exhausted'
            return False
        started = clock()
        print(f'[sparse recovery] {label}; budget={remaining:.0f}s; CPU threads={threads}; GPU BA={gpu}.', flush=True)
        try:
            if launch is None:
                log_path = attempt / (label + '.log')
                record['log'] = str(log_path.relative_to(workspace))
                with log_path.open('w', encoding='utf-8') as log:
                    reader = None
                    reader_errors = []
                    try:
                        with managed_process(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding='utf-8', errors='replace') as process:
                            def relay():
                                try:
                                    for line in process.stdout:
                                        log.write(line)
                                        log.flush()
                                        print(line, end='', flush=True)
                                except (OSError, ValueError) as exc:
                                    reader_errors.append(exc)
                            reader = threading.Thread(target=relay, name='sparse-recovery-output', daemon=True)
                            reader.start()
                            while True:
                                left = remaining - (clock() - started)
                                if left <= 0:
                                    raise subprocess.TimeoutExpired(command, remaining)
                                try:
                                    code = process.wait(timeout=min(30., left))
                                    break
                                except subprocess.TimeoutExpired:
                                    if clock() - started >= remaining:
                                        raise
                                    print(f'[sparse recovery] {label}: elapsed={clock()-started:.0f}s; '
                                          f'budget remaining={remaining-(clock()-started):.0f}s; ETA unavailable.', flush=True)
                            reader.join()
                            if reader_errors:
                                raise OSError(f'Sparse recovery log relay failed: {reader_errors[0]}')
                    finally:
                        if reader is not None:
                            reader.join(timeout=5.)
                native_status = native_exit_status(code, log_path.read_text(encoding='utf-8', errors='replace'))
            else:
                code = launch(command, label, remaining)
                native_status = native_exit_status(code, '')
            record.update(native_status)
            return native_status['export_eligible']
        except subprocess.TimeoutExpired:
            record['status'] = 'budget_exhausted'
            return False
        except (OSError, ValueError) as exc:
            record.update(status='failed', error=str(exc))
            raise
        finally:
            record['elapsed_seconds'] = clock() - started
            write_provenance(report_path, report)

    refined_path, expanded_path = attempt / 'intrinsics', attempt / 'registered'
    refined_path.mkdir()
    try:
        command = [str(executable), 'bundle_adjuster', '--input_path', str(baseline_path),
                   '--output_path', str(refined_path), '--log_level', '2',
                   '--BundleAdjustment.max_num_iterations', '15',
                   '--BundleAdjustment.refine_focal_length', '1', '--BundleAdjustment.refine_extra_params', '1',
                   '--BundleAdjustment.refine_principal_point', '0', '--BundleAdjustment.use_gpu', str(int(gpu))]
        if not execute('late_intrinsics', command, ba_seconds):
            report.update(status='baseline_retained', reason='intrinsic_refinement_did_not_complete')
            return report
        refined_model = find_model(refined_path)
        if refined_model is None:
            raise ValueError('Intrinsic refinement produced no complete sparse model')
        refined = evaluate_model(refined_model, metadata)
        safety = candidate_gate(baseline, refined, require_improvement=False)
        report.update(refined=refined['summary'], refined_cameras=refined['cameras'], refined_safety=safety)
        if not safety['accepted']:
            report.update(status='baseline_retained', reason='intrinsic_refinement_failed_geometry_gate')
            return report
        selected_path, selected = refined_model, refined
        coverage = summarize_coverage(metadata, baseline['observations'], metadata)
        quality = assess_sparse_quality(coverage)
        report['baseline_quality'] = quality
        poor_coverage = bool(set(metadata) - set(baseline['observations'])) or not quality['ready_for_dense']
        if poor_coverage:
            expanded_path.mkdir()
            command = [str(executable), 'mapper', '--database_path', str(workspace / 'database.db'),
                       '--image_path', str(workspace / 'images'), '--input_path', str(refined_model),
                       '--output_path', str(expanded_path), '--Mapper.multiple_models', '0',
                       '--Mapper.num_threads', str(threads), '--Mapper.ba_use_gpu', str(int(gpu)),
                       '--Mapper.ba_refine_focal_length', '0', '--Mapper.ba_refine_extra_params', '0',
                       '--Mapper.ba_refine_principal_point', '0', '--Mapper.max_reg_trials', '2',
                       '--Mapper.ba_global_max_num_iterations', '20', '--Mapper.ba_global_max_refinements', '2',
                       '--Mapper.max_runtime_seconds', str(max(1, int(.75 * min(mapper_seconds, deadline - clock()))))]
            if execute('register_remaining', command, mapper_seconds):
                expanded_model = find_model(expanded_path)
                if expanded_model is None:
                    report['registration_retry_warning'] = 'no_complete_model'
                else:
                    expanded = evaluate_model(expanded_model, metadata)
                    expanded_gate = candidate_gate(baseline, expanded)
                    report.update(expanded=expanded['summary'], expanded_gate=expanded_gate)
                    before_key = (selected['summary']['registered_images'],
                                  -selected['summary']['mean_reprojection_error_px'],
                                  selected['summary']['observations'])
                    expanded_error = expanded['summary']['mean_reprojection_error_px']
                    after_key = (expanded['summary']['registered_images'],
                                 -expanded_error if expanded_error is not None else -float('inf'),
                                 expanded['summary']['observations'])
                    if expanded_gate['accepted'] and after_key > before_key:
                        selected_path, selected = expanded_model, expanded
            else:
                report['registration_retry_warning'] = 'did_not_complete; only completed intrinsic refinement remains eligible'
        gate = candidate_gate(baseline, selected)
        report['selection_gate'] = gate
        if not gate['accepted']:
            report.update(status='baseline_retained', reason='no_valid_objective_improvement')
            return report
        if sparse_model_identity(baseline_path) != baseline_identity:
            raise BaselineChangedError('Baseline changed during sparse recovery; refusing promotion')
        backup_path = attempt / 'baseline_backup'
        promote_candidate(workspace, baseline_path, selected_path, backup_path)
        report.update(status='accepted', accepted=True, result=selected['summary'],
                      result_cameras=selected['cameras'], baseline_backup=str(backup_path.relative_to(workspace)))
        return report
    except PromotionRollbackError as exc:
        report.update(status='promotion_failed', reason='baseline_rollback_failed', error=str(exc))
        raise
    except BaselineChangedError as exc:
        report.update(status='failed', reason='baseline_changed', error=str(exc))
        raise
    except (OSError, ValueError, RuntimeError, struct.error) as exc:
        report.update(status='baseline_retained', reason='recovery_failed', error=str(exc))
        return report
    except BaseException:
        report.update(status='interrupted', reason='interrupted_before_promotion')
        raise
    finally:
        write_provenance(report_path, report)
        print(f"[sparse recovery] {report['status']}; dense completeness remains unverified.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--colmap-exe', type=Path, required=True)
    parser.add_argument('--num-threads', type=int, default=0)
    parser.add_argument('--ba-seconds', type=float, default=300.)
    parser.add_argument('--mapper-seconds', type=float, default=900.)
    args = parser.parse_args(argv)
    try:
        run_recovery(args.workspace, args.colmap_exe, ba_seconds=args.ba_seconds,
                     mapper_seconds=args.mapper_seconds, threads=args.num_threads)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'[error] Sparse recovery: {exc}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
