"""Hardware-derived worker policy and resource-aware video mapping.

The legacy video mapper stays byte-stable for in-flight reconstruction receipts.
New runners use this adapter and always pass an explicit, uncapped CPU budget.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def available_cpus():
    """Honor OS affinity/container allocations instead of a fixed model cap."""
    counts = [os.cpu_count() or 1]
    if hasattr(os, 'process_cpu_count'):
        counts.append(os.process_cpu_count() or 1)
    if hasattr(os, 'sched_getaffinity'):
        counts.append(len(os.sched_getaffinity(0)) or 1)
    if os.name == 'nt':
        import ctypes
        mask, system = ctypes.c_size_t(), ctypes.c_size_t()
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.GetCurrentProcess.restype = ctypes.c_void_p
        api.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        if api.GetProcessAffinityMask(api.GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(system)):
            # An unrestricted mask describes one processor group, not the
            # machine's total allocation on systems with more than 64 CPUs.
            if mask.value != system.value or counts[0] <= 64:
                counts.append(mask.value.bit_count() or 1)
    # Linux cgroup quotas can be lower than the visible affinity mask.
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            counts.append(max(1, (int(quota) + int(period) - 1) // int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return max(1, min(counts))


def worker_count(requested=0, tasks=None):
    if type(requested) is not int or requested < 0:
        raise ValueError('Worker count must be a nonnegative integer (0=hardware automatic)')
    count = requested or available_cpus()
    return min(count, max(1, tasks)) if tasks is not None else count


def pipeline_resources(args):
    """Resolve one default CPU budget, retaining explicit stage overrides."""
    available = available_cpus()
    budget = worker_count(args.cpu_workers or available)
    return {
        'available_cpus': available,
        'cpu_workers': budget,
        'colmap_threads': worker_count(args.colmap_num_threads or budget),
        'image_similarity_workers': worker_count(args.image_similarity_workers or budget),
        'video_decode_threads': worker_count(args.video_decode_threads or budget),
        # There are exactly two independent writers, not one job per CPU.
        'sensor_workers': args.preprocess_workers or min(2, budget),
    }


def _capture(command, timeout=30):
    from process_control import managed_process
    with managed_process(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        output, _ = process.communicate(timeout=timeout)
        return process.returncode, output.decode('utf-8', errors='replace')


def probe_gpu_ba(executable):
    """Run a tiny isolated BA, forcing the GPU sparse solver path when exposed.

    CUDA SIFT support alone is insufficient. Unsupported/ambiguous builds stay
    on CPU in auto mode; a mandatory GPU request fails before real mapping.
    """
    result = {'supported': False, 'reason': 'probe_not_completed'}
    try:
        code, help_text = _capture([str(executable), 'bundle_adjuster', '-h'])
        prefix = next((name for name in ('BundleAdjustmentCeres', 'BundleAdjustment')
                       if '--' + name + '.use_gpu' in help_text), None)
        if not prefix:
            return dict(result, reason='gpu_ba_option_unavailable')
        with tempfile.TemporaryDirectory(prefix='colmap_ba_probe_') as folder:
            root = Path(folder)
            source, output = root / 'input', root / 'output'
            source.mkdir(); output.mkdir()
            (source / 'cameras.txt').write_text('1 PINHOLE 640 480 500 500 320 240\n')
            points = [(i + 1, (i % 5 - 2) * .2, (i // 5 - 2) * .2, 4. + (i % 3) * .3) for i in range(25)]
            images, tracks = [], []
            for image_id, center in enumerate((0., .3, .6), 1):
                images.append(f'{image_id} 1 0 0 0 {-center} 0 0 1 probe_{image_id}.jpg')
                images.append(' '.join(f'{500*(x-center)/z+320+.1} {500*y/z+240} {i}' for i, x, y, z in points))
            for i, x, y, z in points:
                tracks.append(f'{i} {x} {y} {z} 128 128 128 0 1 {i-1} 2 {i-1} 3 {i-1}')
            (source / 'images.txt').write_text('\n'.join(images) + '\n')
            (source / 'points3D.txt').write_text('\n'.join(tracks) + '\n')
            options = {'use_gpu': 1, 'min_num_images_gpu_solver': 0,
                       'max_num_images_direct_dense_gpu_solver': 0, 'max_num_iterations': 1}
            if any('--' + prefix + '.' + key not in help_text for key in options):
                return dict(result, reason='gpu_solver_probe_options_unavailable')
            command = [str(executable), 'bundle_adjuster', '--input_path', str(source), '--output_path', str(output),
                       '--log_level', '3']
            for key, value in options.items():
                command += ['--' + prefix + '.' + key, str(value)]
            code, log = _capture(command)
            lower = log.lower()
            failed = any(token in lower for token in ('without cuda', 'without cudss', 'falling back',
                         'not available', 'no cuda', 'cuda error', 'failed', 'failure'))
            # Require evidence that GPU configuration reached Ceres, not merely
            # acceptance of a flag followed by a silent CPU solve.
            gpu_evidence = bool(re.search(r'sparse linear algebra library\s+(cuda_sparse|cudss)\b', lower))
            supported = code == 0 and not failed and gpu_evidence
            reason = ('gpu_sparse_solver_verified' if supported else
                      'ceres_missing_cuda_or_cudss' if 'compiled without' in lower else
                      'gpu_solver_unavailable_or_unverified')
            return dict(supported=supported, reason='gpu_sparse_solver_verified' if supported else
                        reason, exit_code=code, output=log[-12000:])
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return dict(result, reason='gpu_probe_failed', error=str(exc))


def run_mapper(workspace, executable, snapshots, num_threads=0, ba_gpu='auto'):
    from colmap_workspace_check import write_provenance
    from process_control import managed_process
    from video_intrinsics import estimate_video_intrinsics, apply_video_intrinsics
    from video_sparse_mapper import mapper_command
    from workspace_lock import inherited_lock_held
    if not inherited_lock_held(workspace):
        raise ValueError('Mapper requires the sparse runner workspace lock')
    threads = worker_count(num_threads)
    gpu = probe_gpu_ba(executable) if ba_gpu != 'off' else {'supported': False, 'reason': 'disabled_by_user'}
    write_provenance(workspace / 'compute_resources.json', {
        'available_cpus': available_cpus(), 'mapper_threads': threads,
        'ba_gpu_policy': ba_gpu, 'ba_gpu_probe': gpu})
    if ba_gpu == 'on' and not gpu['supported']:
        raise ValueError('GPU bundle adjustment requested but could not be verified; see compute_resources.json')
    print(f"[resources] Mapper CPU threads={threads}; GPU BA={'enabled' if gpu['supported'] else 'disabled'} ({gpu['reason']}).", flush=True)
    report = estimate_video_intrinsics(workspace / 'database.db', workspace / 'metadata/images_manifest.csv')
    apply_video_intrinsics(workspace / 'database.db', report)
    command = mapper_command(workspace, executable, snapshots, report, threads)
    command += ['--Mapper.ba_use_gpu', '1' if gpu['supported'] else '0']
    report.update(mapper_command=command,
                  intrinsics_held_during_mapping='--Mapper.ba_refine_focal_length' in command,
                  resource_policy={'threads': threads, 'gpu_ba': gpu['supported']})
    write_provenance(workspace / 'video_intrinsics_summary.json', report)
    print(f"[calibration] Visual intrinsics status={report.get('status')}; held={report['intrinsics_held_during_mapping']}.", flush=True)
    with managed_process(command) as process:
        return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--colmap-exe', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--num-threads', type=int, default=0)
    parser.add_argument('--ba-gpu', choices=('auto', 'on', 'off'), default='auto')
    args = parser.parse_args()
    try:
        return run_mapper(args.workspace.resolve(), args.colmap_exe.resolve(), args.snapshots.resolve(), args.num_threads, args.ba_gpu)
    except (OSError, ValueError) as exc:
        print(f'[resources error] {exc}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
