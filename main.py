"""Reconstruct from video with WiFi/IMU assistance."""

from __future__ import annotations

import codecs
import json
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from preprocess_pipeline import main as run_pipeline
from preprocess_pipeline import parse_args as parse_pipeline_args, find_session_dirs
from run_colmap import run_colmap_script
from colmap_bridge import find_colmap_exe
from workspace_lock import WorkspaceBusyError, workspace_lock
from colmap_workspace_check import write_provenance
from generated_input_cleanup import checked_path
from pipeline_output import CONSOLE, STATE, console_run
from workspace_lock import workspace_activity


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_ARGS = [
    "--root",
    str(PROJECT_ROOT),
    "--output",
    "generated/processed",
    "--reconstruction-output",
    "generated/reconstruction_dataset",
    "--colmap-workspace",
    "generated/colmap_workspace",
    "--reconstruction-profile",
    "final",
    "--video-dir",
    "video",
    "--video-refine-window-s",
    "0.25",
    "--scene-completion",
    "off",
]


KEEP = {".pipeline.lock", CONSOLE, STATE}
LEGACY_OUTPUT_SUFFIXES = ("", "_video_main", "_video_only", "_video_sensor_main", "_video_sensors")


def read_state(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def output_paths(args):
    root = args.root.resolve()
    paths = [root / args.output, root / args.reconstruction_output, root / args.colmap_workspace]
    paths = [checked_path(root, path) for path in paths]
    protected = [root / name for name in (
        ".git", ".agents", ".codex", ".venv", "tests", "video", "pic", "paper", "VideoScene",
        "colmap-x64-windows-cuda")]
    protected.append(root / 'generated/results')
    protected.extend(root.glob("*.sess"))
    # All explicit input paths, including external tools/masks/sync CSVs, are protected.
    for key, value in vars(args).items():
        if isinstance(value, Path) and key not in {"root", "output", "reconstruction_output", "colmap_workspace"}:
            protected.append((root / value).resolve())
    # PATH lookup can select a tool that lives inside an output tree, even
    # without an explicit --*-exe argument. Never remove the selected binary.
    from video_frames import resolve_ffmpeg
    discovered_tools = [resolve_ffmpeg(root, args.ffmpeg_exe)]
    if not (args.skip_colmap_prepare or args.skip_reconstruction_export):
        discovered_tools.append(find_colmap_exe(root, args.colmap_dir, args.colmap_exe))
    protected.extend(path.resolve() for path in discovered_tools if path is not None and path.is_file())
    protected.extend(root.glob("*.py"))
    for target in paths:
        for name in KEEP:
            checked_path(root, target / name)
        if any(target == p or target.is_relative_to(p) or p.is_relative_to(target) for p in protected):
            raise ValueError(f"Output directory overlaps original inputs or project code: {target}")
        if target.exists() and not target.is_dir():
            raise ValueError(f"Output path is not a directory: {target}")
    for index, target in enumerate(paths):
        if any(target == other or target.is_relative_to(other) or other.is_relative_to(target)
               for other in paths[index + 1:]):
            raise ValueError("Processed, dataset and COLMAP output directories must not overlap")
    return paths


def reset_stopped_outputs(args):
    """Delete only the three selected output trees, retaining their lock inodes."""
    root = args.root.resolve()
    paths = output_paths(args)
    with ExitStack() as stack:
        for path in sorted(paths):
            stack.enter_context(workspace_lock(path))
        targets = []
        for folder in paths:
            # A custom nonempty directory needs evidence that it is generated output.
            children = [p for p in folder.iterdir() if p.name not in KEEP]
            index = paths.index(folder)
            prefix = ("processed", "reconstruction_dataset", "colmap_workspace")[index]
            markers = (("processing_summary.json", "video_frames_manifest.csv"),
                       ("export_summary.json", "images_manifest.csv"),
                       ("colmap_workspace_summary.json", "database.db"))[index]
            ownership = read_state(folder / ".main_output_owner.json")
            owned = ownership == {"root": str(root), "workspace": str(paths[2]), "role": index}
            legacy_name = folder.name in {prefix + suffix for suffix in LEGACY_OUTPUT_SUFFIXES}
            if children and not owned and not legacy_name and not any((folder / m).is_file() for m in markers):
                raise ValueError(f"Refusing to clear an unrecognized output directory: {folder}")
            for child in children:
                checked_path(root, child)
                # Validate every descendant before the first recursive deletion.
                pending = [child] if child.is_dir() else []
                while pending:
                    for entry in pending.pop().iterdir():
                        checked_path(root, entry)
                        if entry.is_dir():
                            pending.append(entry)
                targets.append(child)
        if targets:
            print("[restart] Clearing stopped-run outputs before rebuilding:", flush=True)
            for path in paths:
                print(f"[restart] {path}", flush=True)
        if not getattr(args, '_results_archived', False):
            from startup_cleanup import archive_results
            archive_results(root, paths[2])
        for target in targets:
            checked_path(root, target)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        for index, folder in enumerate(paths):
            write_provenance(folder / ".main_output_owner.json",
                             {"root": str(root), "workspace": str(paths[2]), "role": index})


def follow_active_run(workspace, *, poll_seconds=0.5):
    """Tail an existing owner until it releases the lock; never restart or signal it."""
    activity = workspace_activity(workspace)
    if not activity["active"]:
        state = read_state(workspace / STATE)
        if (activity["owner"].get("token") and state.get("owner_token") == activity["owner"]["token"]
                and state.get("exit_code") is not None):
            print(f"[follow] Previous run already stopped with exit code {state['exit_code']}.", flush=True)
            return int(state["exit_code"])
        print("[follow] Previous run stopped without a recorded exit code; success is unverified.", flush=True)
        return 2
    token = activity["owner"].get("token")
    print(f"[follow] Existing run PID {activity['owner'].get('pid', 'unknown')}; following its output. "
          "Ctrl+C detaches this viewer only.", flush=True)
    offsets, decoders = {}, {}
    last_output = time.monotonic()

    def drain(*, complete=False):
        nonlocal last_output
        state = read_state(workspace / STATE)
        if token and state.get("owner_token") == token:
            logs = [workspace / CONSOLE]
        else:
            # Compatibility with runs started by older main/run_colmap launchers.
            candidates = list((workspace / "logs").glob("*.log"))
            logs = sorted(candidates, key=lambda p: p.stat().st_mtime_ns)[-1:]
        for path in logs:
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                size = path.stat().st_size
                if path not in offsets:
                    offsets[path] = max(0, size - 8192)  # Brief recent context, then live output.
                    decoders[path] = codecs.getincrementaldecoder("utf-8")(errors="replace")
                    print(f"[follow] {path}", flush=True)
                if size < offsets[path]:
                    offsets[path] = 0
                    decoders[path].reset()
                handle.seek(offsets[path])
                # While running, cap each poll so ownership changes remain
                # responsive. On exit, drain the whole observed file snapshot
                # before returning its status; final diagnostics may be large.
                remaining = max(0, size - offsets[path])
                while remaining:
                    data = handle.read(min(256 * 1024, remaining))
                    if not data:
                        break
                    offsets[path] = handle.tell()
                    remaining -= len(data)
                    print(decoders[path].decode(data), end="", flush=True)
                    last_output = time.monotonic()
                    if not complete:
                        break
        return state

    try:
        while True:
            state = drain()
            activity = workspace_activity(workspace)
            if not activity["active"] or activity["owner"].get("token") != token:
                state = drain(complete=True)
                if token and state.get("owner_token") == token and state.get("exit_code") is not None:
                    return int(state["exit_code"])
                print("[follow] Original run stopped; no final exit code was recorded.", flush=True)
                return 2
            if time.monotonic() - last_output >= 30:
                print("[follow] Original process still holds the workspace; waiting for output...", flush=True)
                last_output = time.monotonic()
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\n[follow] Detached. The original run continues.", flush=True)
        return 130


def resolve_colmap_workspace(args) -> Path:
    return args.colmap_workspace if args.colmap_workspace.is_absolute() else args.root.resolve() / args.colmap_workspace


def validate_capture_inputs(args):
    """Catch missing captures before the explicitly requested destructive restart."""
    from video_frames import find_video_files
    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"Capture root is missing: {root}")
    if args.sensor_processing_enabled and not find_session_dirs(root):
        raise ValueError(f"No .sess sensor directories found under {root}")
    if not find_video_files(root / args.video_dir):
        raise ValueError(f"No supported video files found under {root / args.video_dir}")
    if args.video_sensor_sync is not None and not (root / args.video_sensor_sync).is_file():
        raise ValueError(f"Video/sensor synchronization CSV is missing: {root / args.video_sensor_sync}")
    if (args.colmap_mask_dir is not None and not (args.skip_colmap_prepare or args.skip_reconstruction_export)
            and not (root / args.colmap_mask_dir).is_dir()):
        raise ValueError(f"COLMAP mask directory is missing: {root / args.colmap_mask_dir}")


def main() -> int:
    argv = [*DEFAULT_ARGS, *sys.argv[1:]]
    args = parse_pipeline_args(argv)
    try:
        # Keep following a pre-layout-change run instead of starting a duplicate.
        if args.colmap_workspace == Path("generated/colmap_workspace"):
            legacy = args.root.resolve() / "colmap_workspace_video_sensor_main"
            if workspace_activity(legacy)["active"]:
                return follow_active_run(legacy)
        paths = output_paths(args)
        workspace = resolve_colmap_workspace(args).resolve()
        if workspace_activity(workspace)['active']:
            return follow_active_run(workspace)
        if not args.skip_startup_cleanup:
            from startup_cleanup import cleanup_project
            inputs = [(args.root.resolve() / value).absolute() for key, value in vars(args).items()
                      if isinstance(value, Path) and key not in {'root', 'output', 'reconstruction_output', 'colmap_workspace'}]
            try:
                cleanup_project(args.root, protected=[*paths, *inputs])
            except (OSError, ValueError) as exc:
                print(f'[cleanup] Startup cleanup deferred: {exc}', flush=True)
        with workspace_lock(workspace), ExitStack() as stack:
            for path in sorted(paths[:2]):
                stack.enter_context(workspace_lock(path))
            # Save old logs before console_run truncates the current console file.
            from startup_cleanup import archive_results
            archive_results(args.root, workspace)
            args._results_archived = True
            with console_run(workspace) as state:
                state["exit_code"] = run_locked_pipeline(argv, args)
                return state["exit_code"]
    except WorkspaceBusyError as exc:
        if workspace_activity(resolve_colmap_workspace(args))["active"]:
            return follow_active_run(resolve_colmap_workspace(args).resolve())
        print(f"[error] {exc}")
        return 2
    except (OSError, ValueError) as exc:
        print(f"[error] {exc}")
        return 2
    except KeyboardInterrupt:
        return 130


def run_locked_pipeline(argv, args) -> int:
    workspace = resolve_colmap_workspace(args)
    try:
        validate_capture_inputs(args)
    except (OSError, ValueError) as exc:
        print(f"[error] {exc}. Existing generated results were not cleared.", flush=True)
        return 2
    run_after_preprocess = not (
        args.skip_colmap_prepare or args.skip_reconstruction_export
    )
    if run_after_preprocess:
        filter_label = {"observed": "observation-conflict filtering", "wifi-density": "WiFi-guided density filtering",
                        "connected-density": "connected-density filtering"}[args.point_filter_method]
        stages = ["Video + WiFi/IMU preprocessing" if args.sensor_processing_enabled else "Video preprocessing", "COLMAP"]
        if not args.skip_point_cloud_filter:
            stages.append(filter_label)
        if args.scene_completion == "videoscene":
            stages.append("VideoScene completion")
        stages.extend(["final.ply", "quality report"])
        print("[pipeline] " + " -> ".join(stages), flush=True)
        if args.scene_completion == "videoscene":
            from videoscene_runner import environment_preflight
            repo = PROJECT_ROOT / "VideoScene"
            if not environment_preflight(repo):
                return 2
        executable = find_colmap_exe(args.root.resolve(), args.colmap_dir, args.colmap_exe)
        if not executable.is_file():
            print(f"[error] COLMAP executable is missing: {executable}. Existing results were not reset.", flush=True)
            return 2
        from colmap_capabilities import validate_colmap_capabilities
        try:
            capabilities = validate_colmap_capabilities(executable, matcher=args.colmap_matcher or "constrained")
        except ValueError as exc:
            print(f"[error] {exc}. Existing results were not reset.", flush=True)
            return 2
        print(f"[preflight] COLMAP {capabilities['version']}: CUDA build and required command options verified.", flush=True)
    reset_stopped_outputs(args)
    tracking = None
    if run_after_preprocess and not args.keep_generated_inputs:
        from generated_input_cleanup import prepare_tracking
        try:
            tracking = prepare_tracking(args.root.resolve(), workspace,
                                        args.root.resolve() / args.output,
                                        args.root.resolve() / args.reconstruction_output)
        except (OSError, ValueError) as exc:
            print(f"[cleanup] Automatic cleanup disabled for these paths: {exc}", flush=True)
    pipeline_result = run_pipeline(argv)
    if pipeline_result != 0:
        return pipeline_result
    if run_after_preprocess:
        if tracking is not None:
            from generated_input_cleanup import record_generated_inputs
            try:
                record_generated_inputs(tracking)
            except (OSError, ValueError) as exc:
                print(f"[cleanup] Could not record generated inputs; they will be retained: {exc}", flush=True)
        return run_colmap_script(workspace, "full", runtime_estimate=True, log_level=args.colmap_log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
