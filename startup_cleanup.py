"""Prune recognized reconstruction caches; preserve captures and model outputs."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

from generated_input_cleanup import checked_path
from colmap_workspace_check import write_provenance
from workspace_lock import WorkspaceBusyError, workspace_lock

IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MODEL_FILES = {name + suffix for name in ('cameras', 'images', 'points3D', 'frames', 'rigs')
               for suffix in ('.bin', '.txt')}
SKIP = {'.git', '.venv', '.agents', '.codex', 'video', 'pic', 'checkpoints', 'node_modules', '__pycache__'}


def intersects(path, protected):
    return any(path == other or path.is_relative_to(other) or other.is_relative_to(path) for other in protected)


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def format_bytes(value):
    """Keep small cleanups visible instead of rounding them to 0.00 GiB."""
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(value) < 1024 or unit == 'TiB':
            return f'{value:,.0f} B' if unit == 'B' else f'{value:,.2f} {unit}'
        value /= 1024


def files_in(root, folder):
    """Validate every descendant before any recursive deletion is permitted."""
    pending, files = [checked_path(root, folder)], []
    while pending:
        for path in pending.pop().iterdir():
            checked_path(root, path)
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                files.append(path)
    return files


def discover(root, protected):
    scopes = [root/'generated', root/'experiments', root/'processed']
    scopes += [path for path in root.glob('colmap_workspace*') if path.is_dir()]
    scopes += [path for path in root.glob('processed_*') if path.is_dir()]
    workspaces, processed = [], []
    pending = [path for path in scopes if path.is_dir() and not intersects(path, protected)]
    # The generated root contains protected current outputs; visit its siblings.
    for scope in (root/'generated', root/'experiments'):
        if scope.is_dir() and scope not in pending and not any(scope == p or scope.is_relative_to(p) for p in protected):
            checked_path(root, scope)
            pending.extend(p for p in scope.iterdir() if p.is_dir() and not intersects(p, protected))
    seen = set()
    while pending:
        folder = pending.pop()
        if folder in seen or intersects(folder, protected) or folder.name in SKIP or folder.name.startswith('.') or folder.suffix == '.sess':
            continue
        seen.add(folder)
        try:
            checked_path(root, folder)
        except ValueError:
            continue
        if ((folder/'colmap_workspace_summary.json').is_file() or
                ((folder/'database.db').is_file() and (folder/'images').is_dir()) or
                (folder/'sparse_optimization.json').is_file()):
            workspaces.append(folder)
            continue
        if (folder/'video_frames_summary.json').is_file():
            processed.append(folder)
            continue
        pending.extend(p for p in folder.iterdir() if p.is_dir())
    return sorted(workspaces), sorted(processed)


def cleanup_plan(root, protected):
    workspaces, processed = discover(root, protected)
    targets, skipped = [], []
    def add(folder, owner, kind, allowed):
        if not folder.is_dir() or intersects(folder, protected):
            return
        try:
            files = files_in(root, folder)
            if not files or not all(allowed(path) for path in files):
                return
            targets.append({'path': str(folder.relative_to(root)), 'owner': str(owner.relative_to(root)),
                            'kind': kind, 'logical_bytes': sum(p.stat().st_size for p in files), 'files': len(files)})
        except (OSError, ValueError) as exc:
            skipped.append({'path': str(folder), 'reason': str(exc)})
    for workspace in workspaces:
        for name in ('depth_maps', 'normal_maps', 'consistency_graphs'):
            add(workspace/'dense/stereo'/name, workspace, 'recomputable_stereo', lambda p: p.suffix == '.bin')
        # Original workspace images are deliberately retained, including old photos
        # whose original capture may no longer exist elsewhere.
        if (workspace/'images').is_dir() and (workspace/'dense/sparse').is_dir():
            add(workspace/'dense/images', workspace, 'undistorted_image_copies', lambda p: p.suffix.lower() in IMAGES)
        snapshots = workspace/'sparse_snapshots'
        if snapshots.is_dir():
            try:
                files = files_in(root, snapshots)
                models = {p.parent for p in files if p.name == 'images.bin' and
                          all((p.parent/name).is_file() for name in ('cameras.bin', 'points3D.bin'))}
                ordered = sorted(models, key=lambda p: ((p/'images.bin').stat().st_mtime_ns, str(p)), reverse=True)
                for model in ordered[2:]:
                    add(model, workspace, 'old_sparse_snapshot', lambda p: p.name in MODEL_FILES)
            except (OSError, ValueError) as exc:
                skipped.append({'path': str(snapshots), 'reason': str(exc)})
    for folder in processed:
        summary = read_json(folder/'video_frames_summary.json')
        captures = [root/row['video'] for row in summary.get('videos', []) if isinstance(row, dict) and row.get('video')]
        if not captures or not all(path.is_file() for path in captures):
            continue
        for name in ('video_frames', 'video_refinement', 'video_restoration'):
            # Reports/configs must survive; remove only directories containing
            # pure frame data, recursing past mixed report/frame directories.
            base = folder/name
            if not base.is_dir():
                continue
            try:
                files = files_in(root, base)
                image_parents = {p.parent for p in files if p.suffix.lower() in IMAGES}
                for parent in sorted(image_parents):
                    if not any(parent.is_relative_to(root/t['path']) for t in targets):
                        add(parent, folder, 'reextractable_video_frames', lambda p: p.suffix.lower() in IMAGES)
            except (OSError, ValueError) as exc:
                skipped.append({'path': str(base), 'reason': str(exc)})
    return targets, skipped


def cleanup_project(root, *, protected=(), apply=True):
    root = Path(root).resolve()
    protected = [Path(p).absolute() for p in protected]
    # Keep the most recent/default workspace resumable, including failed runs.
    protected += [root/'generated'/name for name in ('colmap_workspace', 'processed', 'reconstruction_dataset', 'results')]
    protected += [root/name for name in ('.git', '.venv', 'VideoScene', 'video', 'pic', 'colmap-x64-windows-cuda')]
    before = shutil.disk_usage(root).free
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'planned',
              'scope': 'historical_caches_only',
              'note': 'Selected run outputs are reset separately and are not counted in this report. Result archives are retained.',
              'removed_logical_bytes': 0, 'removed_files': 0, 'removed': [], 'protected': list(map(str, protected))}
    targets, skipped = cleanup_plan(root, protected)
    result.update(targets=targets, skipped=skipped, candidate_logical_bytes=sum(t['logical_bytes'] for t in targets))
    if apply:
        try:
            with workspace_lock(root/'generated/.storage_cleanup'), ExitStack() as locks:
                for owner in sorted({t['owner'] for t in targets}):
                    locks.enter_context(workspace_lock(checked_path(root, root/owner)))
                # Rebuild under all owner locks; do not delete a stale plan.
                refreshed, _ = cleanup_plan(root, protected)
                if refreshed != targets:
                    raise ValueError('Cleanup candidates changed; retry after writers stop')
                for item in targets:
                    target = checked_path(root, root/item['path'])
                    if intersects(target, protected):
                        raise ValueError('Cleanup target overlaps protected data')
                    files_in(root, target)
                    shutil.rmtree(target)
                    result['removed'].append(item['path'])
                    result['removed_logical_bytes'] += item['logical_bytes']
                    result['removed_files'] += item['files']
                result['status'] = 'completed'
        except (OSError, ValueError, WorkspaceBusyError) as exc:
            result.update(status='deferred' if not result['removed'] else 'partial', error=str(exc))
    result['free_bytes_after'] = shutil.disk_usage(root).free
    result['observed_free_bytes_increase'] = result['free_bytes_after']-before
    if not targets:
        print('[cleanup] Historical caches: no eligible files found.', flush=True)
    else:
        size = result['removed_logical_bytes'] if apply else result['candidate_logical_bytes']
        count = result['removed_files'] if apply else sum(item['files'] for item in targets)
        print(f"[cleanup] Historical caches: {'removed' if apply else 'eligible'} {count:,} files "
              f"({format_bytes(size)} logical size); status={result['status']}.", flush=True)
    print(f"[cleanup] This pass excludes selected run outputs and saved results. "
          f"Disk free space: {format_bytes(result['free_bytes_after'])}.", flush=True)
    if result.get('error'):
        print(f"[cleanup] {result['error']}", flush=True)
    if apply:
        try:
            write_provenance(root/'generated/startup_cleanup_report.json', result)
        except OSError as exc:
            print(f'[cleanup] Cannot write cleanup report: {exc}', flush=True)
    return result


def archive_results(root, workspace):
    """Keep PLY outputs and their interpretation before an authorized fresh run."""
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    outputs = [p for p in files_in(root, workspace) if p.suffix.lower() == '.ply']
    if not outputs:
        return None
    destination = checked_path(root, root/'generated/results'/
                               (workspace.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]))
    # Hard links preserve large final clouds on this same volume without a second
    # allocation. Retain logs, metrics, metadata and camera models as context.
    context = [p for p in files_in(root, workspace) if p.suffix.lower() in {'.json', '.csv', '.log', '.md', '.vis'}
               or (p.name in MODEL_FILES and 'sparse_snapshots' not in p.parts)]
    destination.mkdir(parents=True)
    for source in sorted(set(outputs+context)):
        target = checked_path(root, destination/source.relative_to(workspace))
        target.parent.mkdir(parents=True, exist_ok=True)
        if source in outputs or source.name in MODEL_FILES or source.suffix == '.vis':
            try:
                target.hardlink_to(source)
                continue
            except OSError:
                pass
        # Logs and JSON records may be rewritten in place on the next startup.
        # They need independent bytes rather than a shared hard-link inode.
        if not target.exists():
            shutil.copy2(source, target)
    write_provenance(destination/'archive.json', {'source_workspace': str(workspace),
                     'kind': 'preserved_results_before_fresh_run',
                     'note': 'Results and metadata archive; not a resumable MVS workspace.'})
    print(f'[cleanup] Preserved previous point clouds and reports: {destination}', flush=True)
    return destination
