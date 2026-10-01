"""Launch signal/vision reconstruction with optional gated scene completion."""

from __future__ import annotations

import sys
from pathlib import Path

from preprocess_pipeline import main as run_pipeline
from preprocess_pipeline import parse_args as parse_pipeline_args
from run_colmap import confirm_full_colmap_run, run_colmap_script, reset_existing_colmap_outputs
from colmap_bridge import find_colmap_exe
from workspace_lock import WorkspaceBusyError, workspace_lock
from workspace_inputs import existing_reconstruction_outputs


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_ARGS = [
    "--root",
    str(PROJECT_ROOT),
    "--output",
    "processed",
    "--reconstruction-profile",
    "final",
    "--pic-dir",
    "pic",
    "--images-per-wifi-point",
    "20",
    "--video-dir",
    "video",
    "--video-refine-window-s",
    "0.25",
    "--scene-completion",
    "off",
    "--pdr-capture-mode",
    "stationary",
    "--point-event",
    "CHECKPOINT_PAUSE",
]


def resolve_colmap_workspace(args) -> Path:
    return args.colmap_workspace if args.colmap_workspace.is_absolute() else args.root.resolve() / args.colmap_workspace


def can_auto_run_colmap(args) -> bool:
    return not (args.skip_colmap_prepare or args.skip_reconstruction_export or args.skip_image_wifi)


def ask_run_colmap(args) -> bool:
    if not can_auto_run_colmap(args) or not sys.stdin.isatty():
        return False
    return confirm_full_colmap_run(resolve_colmap_workspace(args))


def run_colmap(args) -> int:
    workspace = resolve_colmap_workspace(args)
    script = workspace / "run_colmap_full.bat"
    if not script.exists():
        print(f"[warn] COLMAP script not found: {script}")
        print("[info] run `python main.py` again after COLMAP workspace preparation succeeds.")
        return 1

    print("[info] running COLMAP full pipeline...")
    return run_colmap_script(workspace, "full", runtime_estimate=True, log_level=args.colmap_log_level)


def main() -> int:
    argv = [*DEFAULT_ARGS, *sys.argv[1:]]
    args = parse_pipeline_args(argv)
    try:
        with workspace_lock(resolve_colmap_workspace(args)):
            return run_locked_pipeline(argv, args)
    except WorkspaceBusyError as exc:
        print(f"[error] {exc}")
        return 2


def run_locked_pipeline(argv, args) -> int:
    run_after_preprocess = ask_run_colmap(args)
    workspace = resolve_colmap_workspace(args)
    existing = existing_reconstruction_outputs(workspace)
    if existing and not run_after_preprocess:
        print("[error] Existing reconstruction inputs are protected; preprocessing was not started.", flush=True)
        print("[info] Use `python run_colmap.py resume` to continue, select a new --colmap-workspace, or confirm a full rebuild in main.py.", flush=True)
        return 2
    if run_after_preprocess:
        if args.scene_completion == "videoscene":
            from videoscene_runner import environment_preflight
            repo = PROJECT_ROOT / "VideoScene"
            print("[pipeline] Preprocess -> COLMAP -> WiFi-guided density filtering -> observed final -> separate VideoScene hypotheses -> quality report.", flush=True)
            if not environment_preflight(repo):
                return 2
        executable = find_colmap_exe(args.root.resolve(), args.colmap_dir, args.colmap_exe)
        if not executable.is_file():
            print(f"[error] COLMAP executable is missing: {executable}. Existing results were not reset.", flush=True)
            return 2
        if existing:
            print("[recovery] Confirmed full rebuild: resetting existing COLMAP outputs before preprocessing. This restarts sparse reconstruction, not a resume.", flush=True)
            reset_result = reset_existing_colmap_outputs(workspace, respect_allow_stale=False)
            if reset_result != 0:
                return reset_result
    pipeline_result = run_pipeline(argv)
    if pipeline_result != 0:
        return pipeline_result
    if run_after_preprocess:
        return run_colmap(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
