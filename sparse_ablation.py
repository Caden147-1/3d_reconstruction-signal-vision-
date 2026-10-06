"""Isolated sparse A/B experiment using a read-only subset of an existing DB.

This measures sparse coverage proxies, not dense surface completeness. Timings
on a shared machine are observational and must not be treated as a benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess

from colmap_workspace_check import file_identity, write_provenance
from process_control import managed_process
from sparse_optimization import (PAIR_BASE, find_model, map_workspace,
                                 model_observations, read_metadata, video_time)


def coverage_gate(baseline, candidate, comparison):
    """Conservative screening thresholds, deliberately not a surface guarantee."""
    checks = {
        'retains_registered_images': not comparison['lost_registered_images'],
        'retains_95_percent_observations': (comparison['baseline_observation_retention'] or 0) >= .95,
        'retains_95_percent_points': candidate['points3D'] >= .95 * baseline['points3D'],
        'reprojection_error': baseline['mean_reprojection_error_px'] is not None and
            candidate['mean_reprojection_error_px'] is not None and
            candidate['mean_reprojection_error_px'] <= baseline['mean_reprojection_error_px'] * 1.1 + .05,
        'temporal_coverage': all(clip in candidate['video_coverage'] and
            candidate['video_coverage'][clip]['max_uncovered_gap_s'] <= value['max_uncovered_gap_s'] + .001
            for clip, value in baseline['video_coverage'].items()),
    }
    return {'checks': checks, 'passed_sparse_proxy_gate': all(checks.values()),
            'dense_surface_completeness_verified': False}


def subset_database(source, destination, names):
    """Copy only selected rows/blobs, retaining original feature indices."""
    if destination.exists():
        raise ValueError('Experiment database must be new')
    with sqlite3.connect(destination, uri=True) as db:
        db.execute('ATTACH DATABASE ? AS original', (source.resolve().as_uri()+'?mode=ro',))
        db.execute('BEGIN')
        db.execute('CREATE TEMP TABLE selected_names(name TEXT PRIMARY KEY)')
        db.executemany('INSERT INTO selected_names VALUES (?)', ((name,) for name in names))
        db.execute('CREATE TEMP TABLE selected AS SELECT image_id FROM original.images WHERE name IN (SELECT name FROM selected_names)')
        tables = list(db.execute("SELECT name,sql FROM original.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
        for table, schema in tables:
            if not table.replace('_', '').isalnum():
                raise ValueError('Unexpected database table')
            db.execute(schema)
            columns = {row[1] for row in db.execute(f'PRAGMA original.table_info("{table}")')}
            where = ''
            if 'image_id' in columns:
                where = ' WHERE image_id IN (SELECT image_id FROM selected)'
            elif 'pair_id' in columns:
                where = f' WHERE pair_id/{PAIR_BASE} IN (SELECT image_id FROM selected) AND pair_id%{PAIR_BASE} IN (SELECT image_id FROM selected)'
            elif table == 'frames':
                where = ' WHERE frame_id IN (SELECT frame_id FROM original.frame_data WHERE data_id IN (SELECT image_id FROM selected) AND sensor_type=0)'
            elif table == 'frame_data':
                where = ' WHERE data_id IN (SELECT image_id FROM selected) AND sensor_type=0'
            db.execute(f'INSERT INTO "{table}" SELECT * FROM original."{table}"{where}')


def compare_models(baseline, candidate):
    before, _, _ = model_observations(baseline)
    after, _, _ = model_observations(candidate)
    common = sum(len(indices & after.get(name, set())) for name, indices in before.items())
    total = sum(map(len, before.values()))
    missing = sorted(set(before)-set(after))
    return {'lost_registered_images': missing, 'additional_registered_images': sorted(set(after)-set(before)),
            'baseline_observation_retention': common/total if total else None,
            'note': 'Observations use (image name, feature index), not unstable 3D point IDs. This is not dense surface coverage.'}


def cleanup_trial_inputs(workspace):
    """Remove completed trial inputs; retain models, logs, metadata and reports."""
    workspace = workspace.resolve()
    report = json.loads((workspace/'sparse_optimization.json').read_text())
    if report.get('status') != 'complete':
        return []
    names = ('images', 'snapshots', 'database.db', 'database.db-wal', 'database.db-shm', 'database.db-journal')
    targets = [workspace/name for name in names if (workspace/name).exists() or (workspace/name).is_symlink()]
    for target in targets:
        if target.is_symlink() or target.resolve() != workspace/target.name:
            raise ValueError(f'Refusing redirected experiment cleanup: {target}')
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    return [target.name for target in targets]


def run(source, output, executable, names, *, threads=4, timeout=300, gap_s=1., keep_intermediates=False):
    output.mkdir(parents=True, exist_ok=False)
    metadata = read_metadata(source/'metadata/images_manifest.csv')
    calibration = json.loads((source/'video_intrinsics_summary.json').read_text(encoding='utf-8'))
    report = {'source': str(source), 'sample_images': names, 'threads': threads,
              'executable_identity': file_identity(executable),
              'calibration_identity': file_identity(source/'video_intrinsics_summary.json'),
              'scope': 'Sparse subset only; concurrent workloads affect timing; full dense completeness remains unverified.',
              'variants': {}}
    for variant, strategy, ratio in (('baseline', 'all', 1.1), ('schedule', 'all', 1.2), ('backbone', 'backbone', 1.2)):
        workspace = output/variant
        workspace.mkdir()
        subset_database(source/'database.db', workspace/'database.db', names)
        input_identity = file_identity(workspace/'database.db')
        for name in names:
            target = workspace/'images'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            # The experiment requires a same-volume output; avoid copying 4K JPEGs.
            os.link(source/'images'/name, target)
        (workspace/'metadata').mkdir()
        rows = [metadata[name] for name in names]
        with (workspace/'metadata/images_manifest.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        def launch(command, label):
            with (workspace/(label+'.log')).open('wb') as stream:
                with managed_process(command, stdout=stream, stderr=subprocess.STDOUT) as process:
                    return process.wait(timeout=timeout)
        try:
            model = map_workspace(workspace, executable, workspace/'snapshots', calibration,
                                  threads=threads, ratio=ratio, strategy=strategy, gap_s=gap_s, launch=launch)
            result = json.loads((workspace/'sparse_optimization.json').read_text())
            result['input_database_identity'] = input_identity
            logs = '\n'.join(path.read_text(errors='replace') for path in workspace.glob('*.log'))
            result['global_refinement_events'] = logs.count('Retriangulation and Global bundle adjustment')
            result['total_mapping_s'] = sum(stage['elapsed_s'] for stage in result['stages'])
            if variant != 'baseline':
                result['comparison'] = compare_models(find_model(output/'baseline/sparse'), model)
                result['coverage_gate'] = coverage_gate(report['variants']['baseline']['result'], result['result'], result['comparison'])
            report['variants'][variant] = result
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            report['variants'][variant] = {'status': 'failed', 'error': str(exc)}
            write_provenance(output/'comparison.json', report)
            raise
        write_provenance(output/'comparison.json', report)
        print(f"[ablation] {variant}: {result['total_mapping_s']:.1f}s, {result['result']}", flush=True)
    if not keep_intermediates:
        for variant, result in report['variants'].items():
            result['cleaned_intermediates'] = cleanup_trial_inputs(output/variant)
        write_provenance(output/'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--colmap-exe', type=Path, required=True)
    parser.add_argument('--start-s', type=float, default=0.)
    parser.add_argument('--clip', help='Video name for multi-clip sources')
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--timeout-s', type=float, default=300.)
    parser.add_argument('--keyframe-gap-s', type=float, default=1.)
    parser.add_argument('--keep-intermediates', action='store_true', help='Keep trial databases, linked input images and snapshots after success.')
    args = parser.parse_args()
    if args.frames < 20 or args.threads < 1 or args.timeout_s <= 0:
        parser.error('Use at least 20 frames, one thread and a positive timeout')
    metadata = read_metadata(args.source/'metadata/images_manifest.csv')
    clips = {row.get('video_name') for row in metadata.values() if row.get('video_name')}
    if not args.clip and len(clips) != 1:
        parser.error('--clip is required for multi-clip sources')
    clip = args.clip or next(iter(clips))
    candidates = sorted((video_time(row), name) for name, row in metadata.items()
                        if row.get('video_name') == clip and video_time(row) is not None and video_time(row) >= args.start_s)
    names = [name for _, name in candidates[:args.frames]]
    if len(names) < args.frames:
        parser.error('Not enough frames in the requested interval')
    try:
        run(args.source.resolve(), args.output.resolve(), args.colmap_exe.resolve(), names,
            threads=args.threads, timeout=args.timeout_s, gap_s=args.keyframe_gap_s, keep_intermediates=args.keep_intermediates)
    except (OSError, RuntimeError, ValueError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        print(f'[ablation error] {exc}', flush=True)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
