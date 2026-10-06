"""Clean only recorded, regenerable image copies after a verified final result."""
from __future__ import annotations

import json
import stat
from contextlib import ExitStack
from pathlib import Path

from colmap_workspace_check import file_identity, write_provenance
from workspace_lock import WorkspaceBusyError, locked_workspace, workspace_lock

PLAN = "generated_input_cleanup.json"
RECEIPT = "generated_input_cleanup_result.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def checked_path(root: Path, path: Path) -> Path:
    """Reject directory junctions, symlinks and paths outside the project."""
    root = root.resolve()
    path = path.absolute()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Cleanup path is outside the project or is its root: {path}")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"Cleanup refuses redirected path: {part}")
    resolved = path.resolve()
    if resolved != path or not resolved.is_relative_to(root):
        raise ValueError(f"Cleanup refuses redirected path: {path}")
    return resolved


def image_files(root: Path, folders: list[Path]) -> set[Path]:
    files = set()
    for folder in folders:
        checked_path(root, folder)
        if not folder.exists():
            continue
        # Do not follow junctions while walking an output tree.
        pending = [folder]
        while pending:
            for path in pending.pop().iterdir():
                checked_path(root, path)
                if path.is_dir():
                    pending.append(path)
                elif path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                    files.add(path)
    return files


def cleanup_output_roots(root: Path, workspace: Path, folders: list[Path]) -> list[Path]:
    """Locate the same directory locks used by preprocessing, including old plans."""
    if (len(folders) != 3 or [path.name for path in folders] !=
            ["video_frames", "video_refinement", "images"] or
            folders[0].parent != folders[1].parent):
        raise ValueError("Cleanup requires the recorded video_frames, video_refinement and images layout")
    outputs = [checked_path(root, folders[0].parent), checked_path(root, folders[2].parent)]
    for index, output in enumerate(outputs):
        for other in [workspace, *outputs[index + 1:]]:
            if output == other or output.is_relative_to(other) or other.is_relative_to(output):
                raise ValueError("Generated input cleanup output roots must not overlap each other or the COLMAP workspace")
    return outputs


def prepare_tracking(root: Path, workspace: Path, processed: Path, dataset: Path) -> dict:
    root, workspace = root.resolve(), workspace.resolve()
    folders = [processed.absolute() / "video_frames", processed.absolute() / "video_refinement",
               dataset.absolute() / "images"]
    checked_path(root, workspace)
    folders = [checked_path(root, folder) for folder in folders]
    outputs = cleanup_output_roots(root, workspace, folders)
    return {"root": root, "workspace": workspace, "folders": folders,
            "output_roots": outputs, "before": image_files(root, folders)}


def record_generated_inputs(tracking: dict) -> None:
    root, workspace, folders = (tracking[key] for key in ("root", "workspace", "folders"))
    # Existing files are not owned by this run, even if preprocessing touched them.
    created = image_files(root, folders) - tracking["before"]
    entries = {str(path.relative_to(root)): file_identity(path) for path in sorted(created)}
    write_provenance(workspace / PLAN, {"schema_version": 1, "root": str(root),
        "workspace": str(workspace), "folders": [str(p.relative_to(root)) for p in folders],
        "output_roots": [str(p.relative_to(root)) for p in tracking["output_roots"]],
        "files": entries})
    print(f"[cleanup] Recorded {len(entries)} generated image copies; cleanup follows verified completion.", flush=True)


def validate_completed_result(workspace: Path) -> None:
    from reconstruction_stages import verified_context, verified_final
    from reconstruction_quality import quality_report_status
    from colmap_workspace_check import dense_readiness_failures
    final = verified_final(workspace, verified_context(workspace))
    if not final or final.get("vertex_count", 0) <= 0:
        raise ValueError("No verified, nonempty final point cloud")
    if quality_report_status(workspace).get("status") != "current":
        raise ValueError("The quality report is missing or stale")
    report = json.loads((workspace / "quality_report.json").read_text(encoding="utf-8"))
    if dense_readiness_failures(report.get("input_coverage", {})):
        raise ValueError("Sparse coverage failed the dense continuation check")
    if report.get("status") not in {"ok", "dense_needs_review"}:
        raise ValueError("The quality report does not describe a completed reconstruction")


@locked_workspace("workspace")
def cleanup_generated_inputs(workspace: Path) -> dict:
    workspace = workspace.resolve()
    plan_path = workspace / PLAN
    if not plan_path.exists():
        return {"status": "not_registered"}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = Path(plan["root"])
    if plan.get("schema_version") != 1 or Path(plan["workspace"]) != workspace:
        raise ValueError("Cleanup plan does not belong to this workspace")
    checked_path(root, workspace)
    folders = [checked_path(root, root / name) for name in plan["folders"]]
    outputs = cleanup_output_roots(root, workspace, folders)
    # Schema-1 plans predating output_roots use the exact known folder layout.
    # New plans must agree with that layout; never lock arbitrary substituted roots.
    if "output_roots" in plan and [checked_path(root, root / name)
                                    for name in plan["output_roots"]] != outputs:
        raise ValueError("Cleanup output roots do not match its recorded folders")
    with ExitStack() as locks:
        for output in sorted(outputs):
            locks.enter_context(workspace_lock(output))
        return cleanup_locked_inputs(root, workspace, folders, plan)


def cleanup_locked_inputs(root: Path, workspace: Path, folders: list[Path], plan: dict) -> dict:
    """Caller holds the workspace and both input-owner locks until all writes finish."""
    # Validate the entire plan before removing anything.
    targets = []
    for name, identity in plan["files"].items():
        path = checked_path(root, root / name)
        if path.suffix.lower() not in IMAGE_SUFFIXES or not any(path.is_relative_to(p) for p in folders):
            raise ValueError(f"Unrecognized generated image target: {path}")
        targets.append((path, identity))
    validate_completed_result(workspace)
    result = {"status": "completed", "removed_count": 0, "removed_bytes": 0,
              "already_missing_count": 0, "preserved": []}
    empty_candidates = set()
    for path, identity in targets:
        try:
            checked_path(root, path)
            current = file_identity(path)
            if current is None:
                result["already_missing_count"] += 1
            elif current != identity:
                result["preserved"].append({"path": str(path), "reason": "changed since generation"})
                continue
            else:
                path.unlink()
                result["removed_count"] += 1
                result["removed_bytes"] += current["size"]
            for parent in path.parents:
                if not any(parent == folder or parent.is_relative_to(folder) for folder in folders):
                    break
                empty_candidates.add(parent)
        except (OSError, ValueError) as exc:
            result["preserved"].append({"path": str(path), "reason": str(exc)})
    for folder in sorted(empty_candidates, key=lambda p: len(p.parts), reverse=True):
        try:
            checked_path(root, folder).rmdir()  # Only empty directories; never recursive deletion.
        except OSError:
            pass
    if result["preserved"]:
        result["status"] = "partial"
    write_provenance(workspace / RECEIPT, result)
    print(f"[cleanup] Removed {result['removed_count']} generated image copies "
          f"({result['removed_bytes'] / 1024**3:.2f} GiB logical size); "
          f"preserved {len(result['preserved'])} changed or unavailable files.", flush=True)
    return result


def cleanup_after_success(workspace: Path) -> None:
    """Cleanup trouble must not turn a successfully saved final into a failure."""
    try:
        cleanup_generated_inputs(workspace)
    except (OSError, ValueError, KeyError, TypeError, WorkspaceBusyError) as exc:
        print(f"[cleanup] Deferred; generated inputs retained: {exc}", flush=True)
