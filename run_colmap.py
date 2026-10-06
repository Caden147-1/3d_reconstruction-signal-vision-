#!/usr/bin/env python3
"""Run the generated COLMAP workspace scripts from the project root."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

from colmap_workspace_check import inspect_workspace, print_summary as print_workspace_coverage, validation_failures, PENDING_NAME
from workspace_lock import WorkspaceBusyError, locked_workspace, workspace_lock
from process_control import managed_process
from pipeline_output import console_run, duration, stream_command
from workspace_inputs import RECONSTRUCTION_OUTPUTS


SCRIPT_BY_MODE = {
    "sparse": "run_colmap_sparse.bat",
    "dense": "run_colmap_dense.bat",
    "full": "run_colmap_full.bat",
    "automatic": "run_colmap_automatic.bat",
    "gui": "open_colmap_gui.bat",
    "reset": "reset_colmap_outputs.bat",
}
STAGE_MODES = ("resume", "filter", "completion", "report")


CLEAN_REBUILD_MODES = {"sparse", "full", "automatic"}
DEFAULT_RUNTIME_SAMPLE_VIEWS = 2

STALE_COLMAP_OUTPUTS = (
    *RECONSTRUCTION_OUTPUTS,
    "quality_report.json",
    "QUALITY_REPORT.md",
)
ROOT_REPORT_OUTPUTS = ("quality_report.json", "QUALITY_REPORT.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(*SCRIPT_BY_MODE, *STAGE_MODES, "boundary"), nargs="?", default="sparse")
    parser.add_argument("--workspace", type=Path, default=Path("generated/colmap_workspace"))
    parser.add_argument("--log-level", choices=("compact", "verbose"), default="compact",
                        help="Console output for COLMAP stages; full logs are always saved under workspace/logs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show a resume/filter/completion/report plan without running stages.")
    parser.add_argument("--max-clips", type=int, help="Completion clip budget; defaults to the workspace setting.")
    parser.add_argument("--budget-hours", type=float, help="Completion runtime budget in hours.")
    parser.add_argument("--with-videoscene", action="store_true",
                        help="Resume observed reconstruction and then run gated VideoScene completion.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run full reconstruction without the upfront confirmation prompt.",
    )
    parser.add_argument(
        "--runtime-estimate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate bounded subtasks from measured throughput; PatchMatch estimates cover the current pass only.",
    )
    parser.add_argument(
        "--runtime-estimate-sample-views",
        type=int,
        default=DEFAULT_RUNTIME_SAMPLE_VIEWS,
        help="Completed PatchMatch views to sample before the first live estimate.",
    )
    args = parser.parse_args()
    if args.runtime_estimate_sample_views < 1:
        parser.error("--runtime-estimate-sample-views must be positive")
    return args


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def allow_stale_colmap_outputs() -> bool:
    return os.environ.get("ALLOW_STALE_COLMAP_OUTPUTS", "").strip() == "1"


def existing_colmap_outputs(workspace: Path) -> list[str]:
    return [name for name in STALE_COLMAP_OUTPUTS
            if (workspace / name).exists() or (workspace / name).is_symlink()]


def validate_sparse_runner_provenance(script: Path) -> int:
    """Reject pre-provenance generated scripts before any rebuild cleanup."""
    try:
        commands = [line.strip() for line in script.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError) as exc:
        print(f"[error] Cannot inspect sparse runner: {exc}")
        return 2
    actions = ("--record-sparse-start", "--record-sparse-complete")
    if not all(any(line.startswith("python ") and "colmap_workspace_check.py" in line and action in line
                   for line in commands) for action in actions):
        print(f"[error] {script.name} predates run-time sparse provenance recording.")
    elif script.name == "run_colmap_sparse.bat" and not any(
            line.startswith("python ") and "pdr_alignment.py" in line for line in commands):
        print("[error] Sparse runner predates the PDR reliability gate.")
    else:
        return 0
    print("[info] Prepare a new workspace or confirm a full rebuild in main.py to regenerate scripts. Existing outputs were not cleared.")
    return 2


def validate_dense_input_coverage(workspace: Path) -> int:
    """Stop before dense cleanup for stale inputs or collapsed sparse models."""
    coverage = inspect_workspace(workspace)
    print_workspace_coverage(coverage)
    failures = validation_failures(
        coverage,
        require_database_coverage=True,
        require_sparse_model=True,
    )
    if not failures:
        return 0
    for failure in failures:
        print(f"[error] Dense preflight: {failure}")
    print("[info] Dense reconstruction was not started. Preserve existing results and diagnose the sparse input before rebuilding.")
    return 2


@locked_workspace("workspace")
def reset_existing_colmap_outputs(
    workspace: Path,
    *,
    respect_allow_stale: bool = True,
    report_dir: Optional[Path] = None,
) -> int:
    report_dir = report_dir or (workspace.parent if workspace.resolve() == Path(__file__).resolve().parent / "colmap_workspace" else workspace)
    existing = existing_colmap_outputs(workspace)
    root_reports = [report_dir / name for name in ROOT_REPORT_OUTPUTS if (report_dir / name).exists()]
    if not existing and not root_reports:
        print("[info] No old COLMAP outputs to reset.")
        return 0

    if respect_allow_stale and allow_stale_colmap_outputs():
        print("[warn] Keeping existing COLMAP outputs because ALLOW_STALE_COLMAP_OUTPUTS=1.")
        return 0

    targets = [(workspace / name, workspace.resolve()) for name in existing if name != PENDING_NAME]
    existing_targets = {path.resolve() for path, _ in targets}
    targets.extend((path, report_dir.resolve()) for path in root_reports if path.resolve() not in existing_targets)
    # Keep the interruption guard until every old output has been removed.
    if PENDING_NAME in existing:
        targets.append((workspace / PENDING_NAME, workspace.resolve()))
    for path, parent in targets:
        if path.is_symlink() or path.resolve() != parent / path.name:
            print(f"[error] Refusing redirected cleanup target: {path}")
            return 2
    printable = existing + [str(path) for path in root_reports]
    print("[info] Removing old COLMAP outputs: " + ", ".join(printable))
    for path, _ in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            print(f"[error] Failed to remove {path}: {exc}")
            return 2
    print("[done] Old COLMAP outputs removed.")
    return 0


def confirm_full_colmap_run(workspace: Path) -> bool:
    if not sys.stdin.isatty():
        return False
    if existing_colmap_outputs(workspace):
        print("[rebuild] Full reconstruction replaces existing sparse/dense outputs. Use `python run_colmap.py resume` to continue them instead.")
    answer = input("Run full COLMAP now (sparse + dense)? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("[info] COLMAP will be skipped. Later you can run: python run_colmap.py full")
        return False
    return True


def require_script(workspace: Path, mode: str) -> Path:
    script = workspace / SCRIPT_BY_MODE[mode]
    if not script.exists():
        raise SystemExit(f"COLMAP script not found: {script}. Run `python main.py` first.")
    return script


def batch_environment() -> dict[str, str]:
    env = os.environ.copy()
    # Generated scripts invoke python; keep them on the launcher's interpreter.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    # Python children now write through a pipe, so publish stage messages promptly.
    env["PYTHONUNBUFFERED"] = "1"
    return env


def stage_command(script: Path) -> list[str]:
    """Validate a recipe and select its launcher before resetting any outputs."""
    from portable_stage import MARKER, parse_recipe

    content = script.read_text(encoding="utf-8")
    if MARKER in content.splitlines():
        parse_recipe(content)
        return [sys.executable, str(Path(__file__).parent / "portable_stage.py"), "--script", str(script)]
    if os.name == "nt":
        return ["cmd", "/c", "call", str(script)]
    raise ValueError("Legacy Windows workspace: prepare a new workspace on Linux before running stages")


def call_batch(
    script: Path,
    workspace: Path,
    *,
    log_level: str = "compact",
    runtime_estimate: bool = False,
    sample_views: int = DEFAULT_RUNTIME_SAMPLE_VIEWS,
) -> int:
    command = stage_command(script)
    if script.name == SCRIPT_BY_MODE["gui"]:
        with managed_process(command, cwd=str(workspace), env=batch_environment()) as process:
            return process.wait()
    return stream_command(command, workspace=workspace, env=batch_environment(),
                          verbose=log_level == "verbose", estimate=runtime_estimate,
                          sample_views=sample_views, label=script.stem)


def run_full_pipeline(
    workspace: Path,
    *,
    runtime_estimate: bool,
    sample_views: int,
    log_level: str = "compact",
) -> int:
    sparse_script = require_script(workspace, "sparse")
    dense_script = require_script(workspace, "dense")
    full_start = time.monotonic()

    if runtime_estimate:
        print("[estimate] ETA covers the displayed subtask, not the whole run. "
              "Sparse mapping/global BA has no reliable fixed workload; its ETA stays unavailable.")

    print("[info] running COLMAP sparse step...")
    sparse_result = call_batch(sparse_script, workspace, log_level=log_level,
                              runtime_estimate=runtime_estimate, sample_views=sample_views)
    if sparse_result != 0:
        return sparse_result

    coverage_result = validate_dense_input_coverage(workspace)
    if coverage_result != 0:
        return coverage_result

    print("[info] running COLMAP dense step...")
    dense_result = call_batch(
        dense_script,
        workspace,
        runtime_estimate=runtime_estimate,
        sample_views=sample_views,
        log_level=log_level,
    )
    if dense_result == 0:
        print(f"[timing] Full COLMAP elapsed: {duration(time.monotonic() - full_start)}")
        from generated_input_cleanup import cleanup_after_success
        cleanup_after_success(workspace)
    return dense_result


def validate_execution_inputs(workspace: Path, mode: str) -> int:
    # Full runs must validate both recipes before clearing an old reconstruction
    # or spending hours rebuilding sparse output. Resume reaches the same dense
    # preflight through run_colmap_script without changing its reuse policy.
    planned_modes = ("sparse", "dense") if mode == "full" else (mode,)
    for planned_mode in planned_modes:
        script = require_script(workspace, planned_mode)
        try:
            stage_command(script)
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"[error] Cannot run {script.name}: {exc}. Existing results were not reset.")
            return 2
    summary = read_json(workspace / "colmap_workspace_summary.json")
    if mode in ("full", "dense") and summary.get("scene_completion") == "videoscene":
        if not summary.get("run_point_cloud_filter"):
            print("[error] VideoScene requires filtering. Existing results were not reset.")
            return 2
        from videoscene_runner import environment_preflight
        repo = Path(__file__).resolve().parent / "VideoScene"
        if not environment_preflight(repo):
            return 2
    executable = summary.get("colmap_exe")
    if not executable or not Path(executable).is_file():
        print("[error] COLMAP executable is missing; existing results were not reset.")
        return 2
    if mode in ("full", "dense"):
        script = require_script(workspace, "dense")
        lines = script.read_text(encoding="utf-8").splitlines()
        if not all(any(line.startswith("python ") and "dense_cache.py" in line and
                       f' {action} --workspace ' in line for line in lines)
                   for action in ("prepare", "complete")):
            print("[error] Dense runner predates cache validation. Regenerate workspace scripts after the active run finishes.")
            return 2
        if summary.get("scene_completion") == "videoscene":
            stages = {}
            for name in ("point_cloud_filter.py", "finalize_dense_outputs.py", "videoscene_completion.py", "reconstruction_quality.py"):
                stages[name] = [index for index, line in enumerate(lines) if line.startswith("python ") and name in line]
            if any(len(indices) != 1 for indices in stages.values()):
                print("[error] Dense runner does not contain the configured VideoScene pipeline exactly once. Regenerate workspace scripts before reconstruction.")
                return 2
            order = [indices[0] for indices in stages.values()]
            completion_index = stages["videoscene_completion.py"][0]
            following = next((line.strip().lower() for line in lines[completion_index+1:] if line.strip()), "")
            if (order != sorted(order) or "--prepare-only" in lines[completion_index] or
                    following != 'if not "%errorlevel%"=="0" exit /b %errorlevel%'):
                print("[error] Dense runner must execute VideoScene after the filtered final, before quality reporting, and propagate completion failures.")
                return 2
    return 0


@locked_workspace("workspace")
def run_colmap_script(
    workspace: Path,
    mode: str,
    *,
    runtime_estimate: bool = False,
    sample_views: int = DEFAULT_RUNTIME_SAMPLE_VIEWS,
    report_dir: Optional[Path] = None,
    dry_run: bool = False,
    max_clips: Optional[int] = None,
    budget_hours: Optional[float] = None,
    log_level: str = "compact",
) -> int:
    if log_level not in {"compact", "verbose"}:
        raise ValueError("Unknown console log level")
    if mode == "boundary":
        if max_clips is not None or budget_hours is not None:
            print("[error] Completion budgets do not apply to boundary estimation.")
            return 2
        if dry_run:
            print("[boundary] Estimate fuzzy WiFi extent and visual wall candidates from the authenticated final; write workspace/wifi_boundary outputs.")
            return 0
        from wifi_boundary import main as estimate_wifi_boundary
        return estimate_wifi_boundary(["--workspace", str(workspace)])
    report_dir = report_dir or (workspace.parent if workspace.resolve() == Path(__file__).resolve().parent / "colmap_workspace" else workspace)
    if mode in STAGE_MODES:
        from reconstruction_stages import run_stage
        try:
            return run_stage(workspace, mode, dry_run=dry_run, max_clips=max_clips, budget_hours=budget_hours)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"[stages error] {exc}", flush=True)
            return 2
    if dry_run or max_clips is not None or budget_hours is not None:
        print("[error] --dry-run/--max-clips/--budget-hours apply to resume/filter/completion/report modes only.")
        return 2
    if mode == "reset":
        return reset_existing_colmap_outputs(workspace, respect_allow_stale=False, report_dir=report_dir)

    if mode == "full":
        require_script(workspace, "sparse")
        require_script(workspace, "dense")
    else:
        script = require_script(workspace, mode)

    if mode in CLEAN_REBUILD_MODES:
        sparse_script = require_script(workspace, "sparse" if mode == "full" else mode)
        if validate_sparse_runner_provenance(sparse_script):
            return 2
    if validate_execution_inputs(workspace, mode):
        return 2
    if mode in CLEAN_REBUILD_MODES:
        reset_result = reset_existing_colmap_outputs(workspace, report_dir=report_dir)
        if reset_result != 0:
            return reset_result

    if mode == "full":
        return run_full_pipeline(
            workspace,
            runtime_estimate=runtime_estimate,
            sample_views=sample_views,
            log_level=log_level,
        )

    if mode == "dense":
        coverage_result = validate_dense_input_coverage(workspace)
        if coverage_result != 0:
            return coverage_result
        result = call_batch(
            script,
            workspace,
            runtime_estimate=runtime_estimate,
            sample_views=sample_views,
            log_level=log_level,
        )
        if result == 0:
            from generated_input_cleanup import cleanup_after_success
            cleanup_after_success(workspace)
        return result
    return call_batch(script, workspace, log_level=log_level)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    workspace = (args.workspace if args.workspace.is_absolute() else root / args.workspace).resolve()
    try:
        from workspace_lock import inherited_lock_held
        if args.dry_run or inherited_lock_held(workspace):
            return run_cli(args)
        # Direct resume/filter/dense launches publish a token-bound console too.
        with workspace_lock(workspace), console_run(workspace) as state:
            state["exit_code"] = run_cli(args)
            return state["exit_code"]
    except WorkspaceBusyError as exc:
        print(f"[error] {exc}", flush=True)
        return 2
    except KeyboardInterrupt:
        return 130


def run_cli(args=None) -> int:
    args = args or parse_args()
    root = Path(__file__).resolve().parent
    workspace = args.workspace if args.workspace.is_absolute() else root / args.workspace
    if args.with_videoscene:
        if args.mode != 'resume':
            print('[error] --with-videoscene requires resume mode.')
            return 2
        from reconstruction_stages import run_stage
        try:
            return run_stage(workspace, 'resume', dry_run=args.dry_run,
                             max_clips=args.max_clips, budget_hours=args.budget_hours, with_videoscene=True)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f'[stages error] {exc}', flush=True)
            return 2
    if args.mode == "full":
        require_script(workspace, "sparse")
        require_script(workspace, "dense")
        if not args.yes and sys.stdin.isatty() and not confirm_full_colmap_run(workspace):
            return 0
    return run_colmap_script(
        workspace,
        args.mode,
        runtime_estimate=args.runtime_estimate,
        sample_views=args.runtime_estimate_sample_views,
        report_dir=root if workspace.resolve() == root / "colmap_workspace" else workspace,
        dry_run=args.dry_run,
        max_clips=args.max_clips,
        budget_hours=args.budget_hours,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    raise SystemExit(main())
