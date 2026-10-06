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
MODEL_FILES = {name + suffix for name in ('cameras', 'images', 'points3d', 'frames', 'rigs')
               for suffix in ('.bin', '.txt')}
SKIP = {'.git', '.venv', '.agents', '.codex', 'checkpoints', 'weights', 'node_modules'}
FINAL_NAMES = {'final.ply', 'meshed-poisson.ply', 'meshed-delaunay.ply'}
CODE_SUFFIXES = {'.py', '.bat', '.ps1', '.sh', '.exe', '.dll', '.so'}
TEMP_SUFFIXES = {'.bin', '.npy', '.npz', '.raw', '.pyc', '.tmp', '.db', '.sqlite', '.sqlite3', '.vis'}


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


def cleanup_plan(root, protected):
    """Identify generated data, including stopped partial runs and old archives.

    Scripts, reports, captures and final exports are not disposable intermediates.
    Lock inodes are retained, including inside an otherwise empty old directory.
    """
    targets, skipped = [], []
    scopes = [root/name for name in ('generated', 'experiments', 'processed', 'reconstruction_dataset', '__pycache__')]
    for pattern in ('colmap_workspace*', 'processed_*', 'reconstruction_dataset_*'):
        scopes.extend(root.glob(pattern))
    # Discover custom main outputs by their ownership record, without traversing
    # arbitrary source directories or treating every .bin as a generated file.
    scopes.extend(path for path in root.iterdir() if path.is_dir() and (path/'.main_output_owner.json').is_file())
    seen = set()

    def walk(folder, owner, kind, archive=False):
        if folder in seen or any(folder == p or folder.is_relative_to(p) for p in protected):
            return
        if folder.name in SKIP or folder.suffix.lower() == '.sess' or (folder.name.startswith('.') and folder.name != '.storage_cleanup'):
            return
        try:
            checked_path(root, folder)
            if folder.parent == root/'generated/results' and not (folder/'archive.json').is_file():
                return  # A new result archive is still being published.
            seen.add(folder)
            archive = archive or (folder/'archive.json').is_file()
            if folder.parent in (root/'generated', root/'experiments'):
                owner, kind = folder, 'diagnostic'
            ownership = read_json(folder/'.main_output_owner.json')
            if (folder/'colmap_workspace_summary.json').is_file() or (folder/'sparse_optimization.json').is_file():
                owner, kind = folder, 'workspace'
            elif (folder/'processing_summary.json').is_file() or (folder/'video_frames_summary.json').is_file():
                owner, kind = folder, 'processed'
            elif (folder/'export_summary.json').is_file():
                owner, kind = folder, 'dataset'
            elif ownership.get('root') == str(root) and ownership.get('role') in (0, 1, 2):
                owner, kind = folder, ('processed', 'dataset', 'workspace')[ownership['role']]
            # Partially generated frame directories may predate the summaries.
            relative = folder.relative_to(root)
            if len(relative.parts) == 1 and folder.name.startswith(('processed', 'colmap_workspace', 'reconstruction_dataset')):
                owner = folder
                kind = 'workspace' if folder.name.startswith('colmap_workspace') else 'processed' if folder.name.startswith('processed') else 'dataset'
            for path in folder.iterdir():
                if intersects(path, protected) and not path.is_dir():
                    continue
                checked_path(root, path)
                if path.is_dir():
                    walk(path, owner, kind, archive)
                    continue
                suffix, name = path.suffix.lower(), path.name.lower()
                if name.startswith('.') or suffix in CODE_SUFFIXES or name in FINAL_NAMES or suffix in {'.mov', '.mp4', '.mkv', '.avi', '.m4v', '.webm', '.sess', '.dat', '.pt', '.pth', '.onnx', '.safetensors'}:
                    continue
                rel = path.relative_to(owner)
                model_text = name in MODEL_FILES and any(part.lower().startswith('sparse') or part.lower() in ('model', 'models') for part in rel.parts[:-1])
                temporary = suffix in TEMP_SUFFIXES or name.startswith('database.db-') or model_text
                # All non-final PLY layers in generated runs are intermediate.
                temporary |= suffix == '.ply'
                if not archive:
                    temporary |= suffix in IMAGES or suffix == '.csv'
                    temporary |= name in {'image_list.txt', 'match_pairs.txt', 'match_pairs_route.txt', 'match_pairs_unfiltered.txt'}
                    temporary |= kind == 'workspace' and suffix == '.cfg'
                if temporary:
                    info = path.stat()
                    targets.append({'path': str(path.relative_to(root)), 'owner': str(owner.relative_to(root)),
                                    'kind': 'generated_intermediate', 'logical_bytes': info.st_size,
                                    'mtime_ns': info.st_mtime_ns, 'files': 1})
        except (OSError, ValueError) as exc:
            skipped.append({'path': str(folder), 'reason': str(exc)})

    for scope in sorted(set(scopes)):
        if not scope.is_dir():
            continue
        walk(scope, scope, 'diagnostic')
    return targets, skipped


def cleanup_project(root, *, protected=(), apply=True):
    root = Path(root).resolve()
    protected = [Path(p).absolute() for p in protected]
    protected += [root/name for name in ('.git', '.venv', 'VideoScene', 'video', 'pic', 'colmap-x64-windows-cuda')]
    before = shutil.disk_usage(root).free
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'planned',
              'scope': 'other_generated_intermediates',
              'note': 'Selected run outputs are reset separately. Final PLY/meshes, reports, code, captures and lock inodes are retained.',
              'removed_logical_bytes': 0, 'removed_files': 0, 'removed': [], 'protected': list(map(str, protected))}
    print('[cleanup] Scanning generated intermediate data...', flush=True)
    targets, skipped = cleanup_plan(root, protected)
    result.update(targets=targets, skipped=skipped, candidate_logical_bytes=sum(t['logical_bytes'] for t in targets))
    if apply:
        print(f"[cleanup] Found {len(targets):,} candidate files "
              f"({format_bytes(result['candidate_logical_bytes'])} logical size); active owners will be skipped.", flush=True)
        try:
            with workspace_lock(root/'generated/.storage_cleanup'):
                grouped = {}
                for item in targets:
                    grouped.setdefault(item['owner'], []).append(item)
                for relative_owner, items in sorted(grouped.items()):
                    owner = root/relative_owner
                    try:
                        checked_path(root, owner)
                        with ExitStack() as locks:
                            dependencies = {owner}
                            ownership = read_json(owner/'.main_output_owner.json')
                            if ownership.get('root') == str(root) and ownership.get('workspace'):
                                dependencies.add(checked_path(root, Path(ownership['workspace'])))
                            for item in items:
                                target = checked_path(root, root/item['path'])
                                for parent in target.parents:
                                    if parent == root:
                                        break
                                    if (parent/'.pipeline.lock').exists():
                                        dependencies.add(parent)
                            for dependency in sorted(dependencies):
                                locks.enter_context(workspace_lock(dependency))
                            # Validate the complete owner group after acquiring
                            # every relevant lock, before deleting its first file.
                            for item in items:
                                target = checked_path(root, root/item['path'])
                                info = target.stat()
                                if intersects(target, protected) or not target.is_file() or (info.st_size, info.st_mtime_ns) != (item['logical_bytes'], item['mtime_ns']):
                                    raise ValueError(f'Intermediate changed during cleanup planning: {target}')
                            empty = set()
                            for item in items:
                                target = checked_path(root, root/item['path'])
                                target.unlink()
                                result['removed'].append(item['path'])
                                result['removed_logical_bytes'] += item['logical_bytes']
                                result['removed_files'] += 1
                                for parent in target.parents:
                                    if parent == owner:
                                        break
                                    empty.add(parent)
                            for folder in sorted(empty, key=lambda p: len(p.parts), reverse=True):
                                try:
                                    checked_path(root, folder).rmdir()  # Empty only; never remove lock inodes.
                                except OSError:
                                    pass
                    except (OSError, ValueError, WorkspaceBusyError) as exc:
                        result['skipped'].append({'path': str(owner), 'reason': str(exc)})
                result['status'] = 'completed' if not result['skipped'] else 'partial' if result['removed'] else 'deferred'
        except (OSError, ValueError, WorkspaceBusyError) as exc:
            result.update(status='deferred' if not result['removed'] else 'partial', error=str(exc))
    result['free_bytes_after'] = shutil.disk_usage(root).free
    result['observed_free_bytes_increase'] = result['free_bytes_after']-before
    if not targets:
        print('[cleanup] Other generated intermediates: no eligible files found.', flush=True)
    else:
        size = result['removed_logical_bytes'] if apply else result['candidate_logical_bytes']
        count = result['removed_files'] if apply else sum(item['files'] for item in targets)
        print(f"[cleanup] Other generated intermediates: {'removed' if apply else 'eligible'} {count:,} files "
              f"({format_bytes(size)} logical size); status={result['status']}.", flush=True)
    print(f"[cleanup] Selected outputs are reset next; final results and reports are retained. "
          f"Disk free space: {format_bytes(result['free_bytes_after'])}.", flush=True)
    if result['skipped']:
        print(f"[cleanup] Preserved {len(result['skipped'])} busy, changed or inaccessible path(s); see cleanup report.", flush=True)
    if result.get('error'):
        print(f"[cleanup] {result['error']}", flush=True)
    if apply:
        try:
            write_provenance(root/'generated/startup_cleanup_report.json', result)
        except OSError as exc:
            print(f'[cleanup] Cannot write cleanup report: {exc}', flush=True)
    return result


def record_startup_cleanup(root, other, selected):
    """Report both cleanup phases together, without equating logical and disk bytes."""
    other = other or {'status': 'disabled', 'removed_files': 0, 'removed_logical_bytes': 0}
    result = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'startup_cleanup',
        'status': 'completed' if other['status'] in ('completed', 'disabled') else 'partial',
        'other_outputs': other, 'selected_outputs': selected,
        'removed_files': other['removed_files'] + selected['removed_files'],
        'removed_logical_bytes': other['removed_logical_bytes'] + selected['removed_logical_bytes'],
        'free_bytes_after': selected['free_bytes_after'],
        'note': 'Logical sizes count deleted paths, not physical space reclaimed. Final results, reports and active runs are retained.',
    }
    print(f"[cleanup] Startup total: removed {result['removed_files']:,} files "
          f"({format_bytes(result['removed_logical_bytes'])} logical size); "
          f"free now: {format_bytes(result['free_bytes_after'])}; status={result['status']}.", flush=True)
    try:
        write_provenance(Path(root)/'generated/startup_cleanup_report.json', result)
    except OSError as exc:
        print(f'[cleanup] Cannot write combined cleanup report: {exc}', flush=True)
    return result


def archive_results(root, workspace):
    """Keep PLY outputs and their interpretation before an authorized fresh run."""
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    files = files_in(root, workspace)
    outputs = [p for p in files if p.name.lower() in FINAL_NAMES]
    if not outputs:
        return None
    destination = checked_path(root, root/'generated/results'/
                               (workspace.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]))
    # Retain final deliverables and readable context, not raw fusion, evidence
    # layers, visibility arrays or sparse trial models that keep caches alive.
    context = [p for p in files if p.suffix.lower() in {'.json', '.csv', '.log', '.md'}]
    destination.mkdir(parents=True)
    for source in sorted(set(outputs+context)):
        target = checked_path(root, destination/source.relative_to(workspace))
        target.parent.mkdir(parents=True, exist_ok=True)
        if source in outputs:
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
