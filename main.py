"""Convenient launcher for the classroom raw data preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from preprocess_pipeline import main as run_pipeline
from preprocess_pipeline import parse_args as parse_pipeline_args
from preprocess_pipeline import prepare_colmap_workspace_from_args
from run_colmap import confirm_full_colmap_run, run_colmap_script


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_ARGS = [
    "--root",
    str(PROJECT_ROOT),
    "--output",
    "processed",
    "--reconstruction-profile",
    "final",
    "--colmap-load-profile",
    "low-peak",
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


def ask_low_peak_colmap_mode() -> bool:
    answer = input(
        "Start quality-preserving low-peak mode "
        "(lower peak load, longer runtime)? [Y/n]: "
    ).strip().lower()
    return answer not in {"n", "no"}


def prepare_selected_colmap_mode(args, use_low_peak: bool) -> None:
    if use_low_peak:
        print("[info] enabling quality-preserving low-peak COLMAP mode...")
        prepare_colmap_workspace_from_args(args, load_profile_override="low-peak")
        return

    print("[warn] low-peak mode disabled; COLMAP may use higher instantaneous CPU/GPU load.")
    prepare_colmap_workspace_from_args(
        args,
        load_profile_override="max",
        colmap_num_threads_override=0,
        stage_cooldown_s_override=0,
    )


def ask_colmap_run_plan(args) -> tuple[bool, bool]:
    run_after_preprocess = ask_run_colmap(args)
    if not run_after_preprocess:
        return False, True
    return True, ask_low_peak_colmap_mode()


def run_colmap(args) -> int:
    workspace = resolve_colmap_workspace(args)
    script = workspace / "run_colmap_full.bat"
    if not script.exists():
        print(f"[warn] COLMAP script not found: {script}")
        print("[info] run `python main.py` again after COLMAP workspace preparation succeeds.")
        return 1

    print("[info] running COLMAP full pipeline...")
    return run_colmap_script(workspace, "full", runtime_estimate=True)


def main() -> int:
    argv = [*DEFAULT_ARGS, *sys.argv[1:]]
    args = parse_pipeline_args(argv)
    run_after_preprocess, use_low_peak_colmap = ask_colmap_run_plan(args)
    pipeline_result = run_pipeline(argv)
    if pipeline_result != 0:
        return pipeline_result
    if run_after_preprocess:
        prepare_selected_colmap_mode(args, use_low_peak_colmap)
        return run_colmap(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
