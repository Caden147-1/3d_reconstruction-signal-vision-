"""Convenient launcher for the classroom raw data preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from preprocess_pipeline import main as run_pipeline
from preprocess_pipeline import parse_args as parse_pipeline_args
from run_colmap import confirm_full_colmap_run, run_colmap_script


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
        return 0

    print("[info] running COLMAP full pipeline...")
    return run_colmap_script(workspace, "full")


def main() -> int:
    argv = [*DEFAULT_ARGS, *sys.argv[1:]]
    args = parse_pipeline_args(argv)
    pipeline_result = run_pipeline(argv)
    if pipeline_result != 0:
        return pipeline_result
    if ask_run_colmap(args):
        return run_colmap(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
