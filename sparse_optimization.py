"""Conservative BA scheduling and optional video backbone mapping.

All input frames remain in the database and are offered to the final mapper.
The legacy resource adapter is kept unchanged for already running recipes.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sqlite3
import struct
import time

from colmap_workspace_check import read_exact, write_provenance
from compute_resources import available_cpus, worker_count, probe_gpu_ba
from process_control import managed_process
from video_sparse_mapper import mapper_command

PAIR_BASE = 2147483647


def read_metadata(path):
    if not path.is_file():
        return {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {row['image_file']: row for row in csv.DictReader(stream)}


def video_time(row):
    try:
        value = float(row['video_time_s'])
        return value if math.isfinite(value) and value >= 0 else None
    except (KeyError, ValueError, TypeError):
        return None


def select_backbone(database, metadata, gap_s=1.0):
    """Time-stratify, then conservatively restore weakly connected segments.

    A frame is omitted only if at least two selected neighbours have verified
    matches. Original graph component connectivity is restored by adding paths.
    Graph support is a screening heuristic, not proof of triangulatable geometry.
    Unknown timestamps are always retained.
    """
    if not math.isfinite(gap_s) or gap_s <= 0:
        raise ValueError('Sparse keyframe interval must be finite and positive')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        names = dict(db.execute('SELECT image_id,name FROM images'))
        edges = list(db.execute('SELECT pair_id,rows FROM two_view_geometries WHERE rows>=30 AND config IN (2,3)'))
    graph = {key: set() for key in names}
    support = {key: 0 for key in names}
    for pair, count in edges:
        a, b = divmod(pair, PAIR_BASE)
        if a in names and b in names:
            graph[a].add(b); graph[b].add(a)
            support[a] += count; support[b] += count
    groups, selected = {}, set()
    for key, name in names.items():
        row = metadata.get(name, {})
        stamp = video_time(row)
        clip = row.get('video_name')
        if stamp is None or not clip:
            selected.add(key)
        else:
            groups.setdefault(clip, []).append((stamp, key))
    for frames in groups.values():
        frames.sort()
        selected.update((frames[0][1], frames[-1][1]))
        bins = {}
        for stamp, key in frames:
            bins.setdefault(int((stamp - frames[0][0]) / gap_s), []).append(key)
        for candidates in bins.values():
            selected.add(max(candidates, key=lambda key: (support[key], -key)))
    initial_count = len(selected)
    # Preserve all weak segments; do not force a target count at their expense.
    for key in sorted(names):
        if key not in selected and len(graph[key] & selected) < 2:
            selected.add(key)
    # Contract selected components. A maximum-support spanning forest of the
    # full graph supplies missing bridge vertices without arbitrary pair caps.
    parent = {key: key for key in names}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def join(a, b):
        a, b = find(a), find(b)
        if a == b:
            return False
        parent[b] = a
        return True
    for a in sorted(selected):
        for b in graph[a] & selected:
            join(a, b)
    forest = {key: set() for key in names}
    for pair, count in sorted(edges, key=lambda item: (-item[1], item[0])):
        a, b = divmod(pair, PAIR_BASE)
        if a in names and b in names and join(a, b):
            forest[a].add(b); forest[b].add(a)
    # Prune non-selected leaves; internal vertices connect selected components.
    leaves = [key for key in names if key not in selected and len(forest[key]) <= 1]
    while leaves:
        key = leaves.pop()
        for other in list(forest[key]):
            forest[other].remove(key)
            if other not in selected and len(forest[other]) == 1:
                leaves.append(other)
        forest[key].clear()
    selected.update(key for key in names if forest[key])
    # Bin selection bounds spacing by <2*gap; preserve a strict temporal bound.
    for frames in groups.values():
        previous = frames[0][0]
        for index, (stamp, key) in enumerate(frames):
            if stamp - previous > gap_s and index:
                previous, added = frames[index - 1]
                selected.add(added)
            if key in selected:
                previous = stamp
    return sorted(names[key] for key in selected), {
        'input_frames': len(names), 'temporal_frames': initial_count,
        'selected_frames': len(selected), 'max_target_gap_s': gap_s,
        'graph_min_inliers': 30,
    }


def model_observations(model, *, include_indices=True):
    """Read stable feature-index observations, independent of point IDs/cameras."""
    images, errors, lengths = {}, [], []
    with (model / 'images.bin').open('rb') as stream:
        for _ in range(struct.unpack('<Q', read_exact(stream, 8))[0]):
            read_exact(stream, 64)  # Image id, pose and camera id.
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError('Truncated COLMAP image name')
                if char == b'\0':
                    break
                name.extend(char)
            size = struct.unpack('<Q', read_exact(stream, 8))[0]
            data = read_exact(stream, size * 24)
            indices = (index for index, (_, _, point) in enumerate(struct.iter_unpack('<ddq', data)) if point >= 0)
            images[name.decode('utf-8')] = set(indices) if include_indices else sum(1 for _ in indices)
    with (model / 'points3D.bin').open('rb') as stream:
        for _ in range(struct.unpack('<Q', read_exact(stream, 8))[0]):
            values = struct.unpack('<Q3d3BdQ', read_exact(stream, 51))
            if not math.isfinite(values[-2]):
                raise ValueError('Non-finite sparse reprojection error')
            errors.append(values[-2]); lengths.append(values[-1])
            read_exact(stream, values[-1] * 8)
    return images, errors, lengths


def model_summary(model, metadata):
    images, errors, lengths = model_observations(model, include_indices=False)
    clips = {}
    for name, row in metadata.items():
        stamp = video_time(row)
        if stamp is not None and row.get('video_name'):
            entry = clips.setdefault(row['video_name'], {'input_times': [], 'registered_times': []})
            entry['input_times'].append(stamp)
            if name in images:
                entry['registered_times'].append(stamp)
    coverage = {}
    for clip, times in clips.items():
        available, registered = sorted(times['input_times']), sorted(times['registered_times'])
        # Include lost start/end coverage in the gap measure.
        anchors = [available[0], *registered, available[-1]]
        coverage[clip] = {'input_frames': len(available), 'registered_frames': len(registered),
                          'max_uncovered_gap_s': max((b-a for a,b in zip(anchors, anchors[1:])), default=0)}
    return {'registered_images': len(images), 'points3D': len(errors),
            'observations': sum(images.values()),
            'mean_reprojection_error_px': sum(errors)/len(errors) if errors else None,
            'mean_track_length': sum(lengths)/len(lengths) if lengths else 0,
            'video_coverage': coverage}


def find_model(root):
    candidates = ([root] if (root/'images.bin').is_file() else []) + sorted(path.parent for path in root.glob('*/images.bin'))
    valid = [path for path in candidates if (path/'points3D.bin').is_file()]
    def image_count(path):
        with (path/'images.bin').open('rb') as stream:
            return struct.unpack('<Q', read_exact(stream, 8))[0]
    return max(valid, key=image_count) if valid else None


def mapping_command(workspace, executable, snapshots, calibration, threads, gpu, ratio,
                    output, image_list=None, input_model=None):
    if not math.isfinite(ratio) or ratio <= 1:
        raise ValueError('Global BA growth ratio must be finite and greater than 1')
    command = mapper_command(workspace, executable, snapshots, calibration, worker_count(threads))
    command[command.index('--output_path') + 1] = str(output)
    command += ['--Mapper.ba_use_gpu', str(int(gpu)),
                '--Mapper.ba_global_frames_ratio', str(ratio),
                '--Mapper.ba_global_points_ratio', str(ratio)]
    if image_list is not None:
        command += ['--Mapper.image_list_path', str(image_list)]
    if input_model is not None:
        command += ['--input_path', str(input_model)]
    # Keep default iteration limits, refinement counts, fixed safety intervals,
    # triangulation thresholds and the mapper's final global refinement.
    return command


def map_workspace(workspace, executable, snapshots, calibration, *, threads=0,
                  gpu=False, strategy='all', ratio=1.1, gap_s=1.0, launch=None):
    """Map under the caller's workspace lock; never silently drop non-keyframes."""
    if strategy not in ('all', 'backbone'):
        raise ValueError('Unknown sparse strategy')
    metadata = read_metadata(workspace/'metadata/images_manifest.csv')
    audit = {'status': 'running', 'strategy': strategy, 'ba_growth_ratio': ratio, 'commands': [], 'stages': []}
    audit_path = workspace/'sparse_optimization.json'
    def execute(label, output, image_list=None, input_model=None):
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError(f'Refusing to mix sparse outputs: {output}')
        stage_snapshots = snapshots/label
        stage_snapshots.mkdir(parents=True, exist_ok=True)
        command = mapping_command(workspace, executable, stage_snapshots, calibration, threads,
                                  gpu, ratio, output, image_list, input_model)
        audit['commands'].append(command)
        write_provenance(audit_path, audit)
        print(f'[sparse] {label}: BA growth ratio={ratio}; CPU threads={worker_count(threads)}', flush=True)
        start = time.monotonic()
        try:
            if launch is None:
                with managed_process(command) as process:
                    code = process.wait()
            else:
                code = launch(command, label)
        except BaseException as exc:
            audit.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            audit['stages'].append({'stage': label, 'elapsed_s': time.monotonic()-start, 'completed': False})
            write_provenance(audit_path, audit)
            raise
        model = find_model(output) if code == 0 else None
        audit['stages'].append({'stage': label, 'exit_code': code, 'elapsed_s': time.monotonic()-start,
                                'summary': model_summary(model, metadata) if model else None})
        write_provenance(audit_path, audit)
        if code:
            audit['status'] = 'failed'
            write_provenance(audit_path, audit)
            raise RuntimeError(f'Sparse {label} failed with exit code {code}')
        return model
    seed = None
    if strategy == 'backbone':
        selected, audit['selection'] = select_backbone(workspace/'database.db', metadata, gap_s)
        count = audit['selection']['input_frames']
        if len(selected) >= 20 and len(selected) < count * .9:
            selected_path = workspace/'sparse_backbone_images.txt'
            selected_path.write_text('\n'.join(selected)+'\n', encoding='utf-8')
            seed = execute('backbone', workspace/'sparse_backbone', selected_path)
            if seed is None or audit['stages'][-1]['summary']['registered_images'] < .9*len(selected):
                audit['fallback_reason'] = 'backbone_registered_less_than_90_percent'
                seed = None
                print('[sparse] Backbone coverage insufficient; mapping all frames from scratch.', flush=True)
        else:
            audit['fallback_reason'] = 'too_few_frames_or_insufficient_reduction'
    model = execute('expand_all_frames' if seed else 'all_frames', workspace/'sparse', input_model=seed)
    if model is None:
        raise RuntimeError('Mapper produced no valid sparse model')
    audit['result'] = audit['stages'][-1]['summary']
    audit['status'] = 'complete'
    write_provenance(audit_path, audit)
    return model


def main():
    from video_intrinsics import estimate_video_intrinsics, apply_video_intrinsics
    from workspace_lock import inherited_lock_held
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--colmap-exe', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--num-threads', type=int, default=0)
    parser.add_argument('--ba-gpu', choices=('auto', 'on', 'off'), default='auto')
    parser.add_argument('--strategy', choices=('all', 'backbone'), default='all')
    parser.add_argument('--ba-growth-ratio', type=float, default=1.1)
    parser.add_argument('--keyframe-gap-s', type=float, default=1.)
    args = parser.parse_args()
    try:
        workspace = args.workspace.resolve()
        if not inherited_lock_held(workspace):
            raise ValueError('Mapper requires the sparse runner workspace lock')
        gpu = probe_gpu_ba(args.colmap_exe) if args.ba_gpu != 'off' else {'supported': False, 'reason': 'disabled_by_user'}
        threads = worker_count(args.num_threads)
        write_provenance(workspace/'compute_resources.json', {'available_cpus': available_cpus(),
                         'mapper_threads': threads, 'ba_gpu_policy': args.ba_gpu, 'ba_gpu_probe': gpu})
        if args.ba_gpu == 'on' and not gpu['supported']:
            raise ValueError('GPU BA requested but unsupported; see compute_resources.json')
        print(f"[resources] Mapper CPU threads={threads}; GPU BA={gpu['supported']} ({gpu['reason']}).", flush=True)
        calibration = estimate_video_intrinsics(workspace/'database.db', workspace/'metadata/images_manifest.csv')
        apply_video_intrinsics(workspace/'database.db', calibration)
        calibration['resource_policy'] = {'threads': threads, 'gpu_ba': gpu['supported']}
        calibration['mapper_commands_report'] = 'sparse_optimization.json'
        calibration['intrinsics_held_during_mapping'] = len(calibration.get('cameras', [])) == 1 and calibration['cameras'][0].get('status') == 'estimated'
        write_provenance(workspace/'video_intrinsics_summary.json', calibration)
        map_workspace(workspace, args.colmap_exe, args.snapshots, calibration, threads=threads,
                      gpu=gpu['supported'], strategy=args.strategy, ratio=args.ba_growth_ratio, gap_s=args.keyframe_gap_s)
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f'[sparse error] {exc}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
