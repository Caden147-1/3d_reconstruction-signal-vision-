"""Convenient launcher for the classroom raw data preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from preprocess_pipeline import main as run_pipeline
from preprocess_pipeline import parse_args as parse_pipeline_args
from run_colmap import confirm_full_colmap_run, print_upfront_runtime_estimate, run_colmap_script


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_ARGS = [
    "--root",
    str(PROJECT_ROOT),
    "--output",
    "processed",
    "--reconstruction-profile",
    "rtx4080",
    "--pic-dir",
    "pic",
    "--images-per-wifi-point",
    "20",
    "--video-dir",
    "video",
    "--point-event",
    "CHECKPOINT_PAUSE",
]


def resolve_colmap_workspace(args) -> Path:
    return args.colmap_workspace if args.colmap_workspace.is_absolute() else args.root.resolve() / args.colmap_workspace


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass


def print_existing_colmap_estimate(args) -> None:
    workspace = resolve_colmap_workspace(args)
    has_workspace_metadata = (workspace / "colmap_workspace_summary.json").exists() or (workspace / "image_list.txt").exists()
    if not has_workspace_metadata:
        print("[estimate] No prepared COLMAP workspace yet; preprocessing will create one before COLMAP ETA is available.", flush=True)
        return
    print("[estimate] Existing COLMAP workspace estimate; preprocessing may refresh this before the run:", flush=True)
    print_upfront_runtime_estimate(workspace)


def can_auto_run_colmap(args) -> bool:
    return not (args.skip_colmap_prepare or args.skip_reconstruction_export or args.skip_image_wifi)


def ask_run_colmap(args) -> bool:
    if not can_auto_run_colmap(args):
        return False
    if not sys.stdin.isatty():
        print("[info] COLMAP full run is not auto-started because stdin is not interactive.", flush=True)
        print("[info] Run `python run_colmap.py full --yes` after preprocessing if you want the full reconstruction.", flush=True)
        return False
    return confirm_full_colmap_run(resolve_colmap_workspace(args))

def run_colmap(args) -> int:
    workspace = resolve_colmap_workspace(args)
    script = workspace / "run_colmap_full.bat"
    if not script.exists():
        print(f"[warn] COLMAP script not found: {script}")
        print("[info] run `python main.py` again after COLMAP workspace preparation succeeds.")
        return 0

    print("[info] running COLMAP full pipeline...")
    return run_colmap_script(workspace, "full")


def main() -> int:
    configure_stdout()
    argv = [*DEFAULT_ARGS, *sys.argv[1:]]
    args = parse_pipeline_args(argv)
    print("[info] main launcher started; preprocessing runs first, then COLMAP can run after workspace preparation.", flush=True)
    print_existing_colmap_estimate(args)
    pipeline_result = run_pipeline(argv)
    if pipeline_result != 0:
        return pipeline_result
    if ask_run_colmap(args):
        return run_colmap(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
