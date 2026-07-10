#!/usr/bin/env python3
"""Run the generated COLMAP workspace scripts from the project root."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


SCRIPT_BY_MODE = {
    "sparse": "run_colmap_sparse.bat",
    "dense": "run_colmap_dense.bat",
    "full": "run_colmap_full.bat",
    "automatic": "run_colmap_automatic.bat",
    "gui": "open_colmap_gui.bat",
    "reset": "reset_colmap_outputs.bat",
}


CLEAN_REBUILD_MODES = {"sparse", "full", "automatic"}
PATCHMATCH_VIEW_RE = re.compile(r"Processing view\s+(\d+)\s*/\s*(\d+)")
DEFAULT_RUNTIME_SAMPLE_VIEWS = 2

STALE_COLMAP_OUTPUTS = (
    "database.db",
    "sparse",
    "sparse_pdr_aligned",
    "sparse_pose_prior",
    "dense",
    "stereo",
    "meshed-poisson.ply",
    "meshed-delaunay.ply",
    "quality_report.json",
    "QUALITY_REPORT.md",
)
ROOT_REPORT_OUTPUTS = ("quality_report.json", "QUALITY_REPORT.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=SCRIPT_BY_MODE, nargs="?", default="sparse")
    parser.add_argument("--workspace", type=Path, default=Path("colmap_workspace"))
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run full reconstruction without the upfront confirmation prompt.",
    )
    parser.add_argument(
        "--no-upfront-estimate",
        action="store_true",
        help="Skip the upfront full-pipeline runtime estimate.",
    )
    parser.add_argument(
        "--no-runtime-estimate",
        action="store_true",
        help="Disable live PatchMatch runtime estimation for full/dense runs.",
    )
    parser.add_argument(
        "--runtime-estimate-sample-views",
        type=int,
        default=DEFAULT_RUNTIME_SAMPLE_VIEWS,
        help="Number of completed PatchMatch views to time before printing the first live estimate.",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_duration_range(low_seconds: float, high_seconds: float) -> str:
    return f"{format_duration(low_seconds)} - {format_duration(high_seconds)}"


def estimate_range(center_seconds: float, spread: float) -> tuple[float, float]:
    center_seconds = max(0.0, center_seconds)
    spread = min(max(spread, 0.0), 0.9)
    return center_seconds * (1.0 - spread), center_seconds * (1.0 + spread)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def count_nonempty_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def positive_number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def allow_stale_colmap_outputs() -> bool:
    return os.environ.get("ALLOW_STALE_COLMAP_OUTPUTS", "").strip() == "1"


def existing_colmap_outputs(workspace: Path) -> list[str]:
    return [name for name in STALE_COLMAP_OUTPUTS if (workspace / name).exists()]


def reset_existing_colmap_outputs(
    workspace: Path,
    *,
    respect_allow_stale: bool = True,
    report_dir: Optional[Path] = None,
) -> int:
    report_dir = report_dir or workspace.parent
    existing = existing_colmap_outputs(workspace)
    root_reports = [report_dir / name for name in ROOT_REPORT_OUTPUTS if (report_dir / name).exists()]
    if not existing and not root_reports:
        print("[info] No old COLMAP outputs to reset.")
        return 0

    if respect_allow_stale and allow_stale_colmap_outputs():
        print("[warn] Keeping existing COLMAP outputs because ALLOW_STALE_COLMAP_OUTPUTS=1.")
        return 0

    printable = existing + [str(path) for path in root_reports]
    print("[info] Removing old COLMAP outputs: " + ", ".join(printable))
    for name in existing:
        path = workspace / name
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            print(f"[error] Failed to remove {path}: {exc}")
            return 2
    for path in root_reports:
        try:
            path.unlink()
        except OSError as exc:
            print(f"[error] Failed to remove {path}: {exc}")
            return 2
    print("[done] Old COLMAP outputs removed.")
    return 0


def estimate_dense_view_count(summary: dict[str, Any], image_count: int) -> int:
    constrained = summary.get("constrained_match_pairs") if isinstance(summary.get("constrained_match_pairs"), dict) else {}
    pdr_refs = int(positive_number(summary.get("pdr_alignment_ref_count"), 0))
    video_frames = int(positive_number(constrained.get("video_frame_count"), 0))
    matcher = str(summary.get("matcher") or "").lower()

    if pdr_refs and video_frames:
        estimate = int((pdr_refs * 0.9) + (video_frames * 0.03))
    elif pdr_refs:
        estimate = int(pdr_refs * 0.9)
    else:
        ratio = 0.45 if matcher in {"hybrid", "exhaustive"} else 0.35
        estimate = int(image_count * ratio)

    return max(1, min(image_count, estimate or image_count))


def build_upfront_runtime_estimate(workspace: Path) -> dict[str, Any]:
    summary = read_json(workspace / "colmap_workspace_summary.json")
    constrained = summary.get("constrained_match_pairs") if isinstance(summary.get("constrained_match_pairs"), dict) else {}
    dense_profile_config = summary.get("dense_profile_config") if isinstance(summary.get("dense_profile_config"), dict) else {}
    patch_options = dense_profile_config.get("patch_match") if isinstance(dense_profile_config.get("patch_match"), dict) else {}

    image_count = int(positive_number(summary.get("image_count"), count_nonempty_lines(workspace / "image_list.txt")))
    image_count = max(1, image_count)
    constrained_pairs = int(positive_number(constrained.get("pair_count"), count_nonempty_lines(workspace / "match_pairs.txt")))
    matcher = str(summary.get("matcher") or "unknown").lower()
    feature_max_image_size = positive_number(summary.get("feature_max_image_size"), 1600.0)
    max_num_features = positive_number(summary.get("max_num_features"), 6000.0)
    dense_max_image_size = positive_number(summary.get("dense_max_image_size"), 1200.0)
    dense_profile = str(summary.get("dense_profile") or "unknown")
    dense_views = estimate_dense_view_count(summary, image_count)

    exhaustive_pairs = image_count * max(0, image_count - 1) // 2
    if matcher == "constrained":
        matching_pairs = constrained_pairs
        matcher_cost_factor = 0.0025
    elif matcher == "sequential":
        matching_pairs = max(constrained_pairs, image_count * 12)
        matcher_cost_factor = 0.0035
    else:
        matching_pairs = exhaustive_pairs
        matcher_cost_factor = 0.006

    feature_seconds = (
        image_count
        * 0.75
        * (feature_max_image_size / 1600.0) ** 1.35
        * (max_num_features / 6000.0) ** 0.25
    )
    matching_seconds = (
        matching_pairs
        * matcher_cost_factor
        * (max_num_features / 6000.0) ** 0.55
        * (feature_max_image_size / 1600.0) ** 0.30
    )
    if matcher == "hybrid":
        matching_seconds += constrained_pairs * 0.0015

    mapper_seconds = (
        image_count
        * 1.6
        * (max_num_features / 6000.0) ** 0.35
        * (1.25 if matcher in {"hybrid", "exhaustive"} else 1.0)
    )
    mapper_seconds += min(matching_pairs, 200000) * 0.002

    dense_undistort_seconds = dense_views * 1.2 * (dense_max_image_size / 1600.0) ** 1.6

    window_radius = positive_number(patch_options.get("window_radius"), 5.0)
    num_samples = positive_number(patch_options.get("num_samples"), 15.0)
    num_iterations = positive_number(patch_options.get("num_iterations"), 5.0)
    patch_per_view_seconds = (
        5.5
        * (dense_max_image_size / 1200.0) ** 2.0
        * (num_iterations / 5.0)
        * (num_samples / 15.0)
        * (window_radius / 5.0) ** 0.6
    )
    if int(patch_options.get("geom_consistency", 0) or 0):
        patch_per_view_seconds *= 1.15
    if int(patch_options.get("filter", 0) or 0):
        patch_per_view_seconds *= 1.05
    patchmatch_seconds = dense_views * patch_per_view_seconds

    post_dense_factor = 0.05
    if summary.get("run_meshing"):
        post_dense_factor += 0.16
    if summary.get("run_floor_completion"):
        post_dense_factor += 0.02
    if summary.get("run_point_cloud_filter"):
        post_dense_factor += 0.03
    if summary.get("run_dark_surface_completion"):
        post_dense_factor += 0.01
    post_dense_factor += 0.01
    post_dense_seconds = patchmatch_seconds * post_dense_factor

    stages = [
        ("Reset old COLMAP outputs", *estimate_range(5.0 if existing_colmap_outputs(workspace) else 1.0, 0.5)),
        ("Sparse: SIFT feature extraction", *estimate_range(feature_seconds, 0.45)),
        (f"Sparse: matching ({matcher})", *estimate_range(matching_seconds, 0.55)),
        ("Sparse: mapper + PDR alignment + report", *estimate_range(mapper_seconds, 0.55)),
        ("Dense: image undistortion", *estimate_range(dense_undistort_seconds, 0.45)),
        ("Dense: PatchMatch stereo", *estimate_range(patchmatch_seconds, 0.50)),
        ("Dense: fusion + filtering + meshing + completion/report", *estimate_range(post_dense_seconds, 0.60)),
    ]
    low_total = sum(stage[1] for stage in stages)
    high_total = sum(stage[2] for stage in stages)

    return {
        "summary": summary,
        "image_count": image_count,
        "dense_views": dense_views,
        "constrained_pairs": constrained_pairs,
        "matching_pairs": matching_pairs,
        "matcher": matcher,
        "dense_profile": dense_profile,
        "dense_max_image_size": int(dense_max_image_size),
        "feature_max_image_size": int(feature_max_image_size),
        "max_num_features": int(max_num_features),
        "patch_per_view_seconds": patch_per_view_seconds,
        "stages": stages,
        "total_range": (low_total, high_total),
    }


def print_upfront_runtime_estimate(workspace: Path) -> None:
    estimate = build_upfront_runtime_estimate(workspace)
    total_low, total_high = estimate["total_range"]
    print("[estimate] Upfront COLMAP full-run estimate (rough, before running COLMAP):")
    print(
        "[estimate]   "
        f"images={estimate['image_count']}, estimated_dense_views={estimate['dense_views']}, "
        f"matcher={estimate['matcher']}, dense_profile={estimate['dense_profile']}, "
        f"dense_max_image_size={estimate['dense_max_image_size']}"
    )
    print(
        "[estimate]   "
        f"constrained_pairs={estimate['constrained_pairs']}, matching_pairs~={estimate['matching_pairs']}, "
        f"patchmatch_per_view~={format_duration(estimate['patch_per_view_seconds'])}"
    )
    for label, low_seconds, high_seconds in estimate["stages"]:
        print(f"[estimate]   {label}: {format_duration_range(low_seconds, high_seconds)}")
    print(f"[estimate]   Total full COLMAP: {format_duration_range(total_low, total_high)}")
    print("[estimate]   This is a planning estimate; the live PatchMatch estimator will refine it once dense starts.")


def confirm_full_colmap_run(workspace: Path) -> bool:
    print_upfront_runtime_estimate(workspace)
    if not sys.stdin.isatty():
        return False
    answer = input("Run full COLMAP now (sparse + dense)? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("[info] COLMAP will be skipped. Later you can run: python run_colmap.py full")
        return False
    return True


class PatchMatchRuntimeEstimator:
    def __init__(self, *, sample_views: int, full_start: Optional[float], tail_factor: float) -> None:
        self.sample_views = max(1, sample_views)
        self.full_start = full_start
        self.tail_factor = max(0.0, tail_factor)
        self.patch_start: Optional[float] = None
        self.current_view: Optional[int] = None
        self.current_start: Optional[float] = None
        self.total_views: Optional[int] = None
        self.completed_durations: list[float] = []
        self.first_report_done = False
        self.next_progress_report = 10

    def observe(self, line: str) -> None:
        match = PATCHMATCH_VIEW_RE.search(line)
        if not match:
            return

        now = time.monotonic()
        view_index = int(match.group(1))
        total_views = int(match.group(2))

        if self.patch_start is None:
            self.patch_start = now
            print(
                f"[estimate] PatchMatch live estimator enabled; "
                f"refining after {self.sample_views} completed view(s).",
                flush=True,
            )

        if self.current_start is not None and self.current_view is not None:
            duration = now - self.current_start
            if duration > 0:
                self.completed_durations.append(duration)
                completed = len(self.completed_durations)
                should_report_first = not self.first_report_done and completed >= self.sample_views
                should_report_progress = self.first_report_done and completed >= self.next_progress_report
                if should_report_first or should_report_progress:
                    self._report(now)
                    self.first_report_done = True
                    while self.next_progress_report <= completed:
                        self.next_progress_report += 10

        self.current_view = view_index
        self.current_start = now
        self.total_views = total_views

    def _report(self, now: float) -> None:
        if not self.completed_durations or not self.total_views:
            return

        completed = len(self.completed_durations)
        avg_view_seconds = sum(self.completed_durations) / completed
        patch_total_seconds = avg_view_seconds * self.total_views
        patch_remaining_seconds = avg_view_seconds * max(0, self.total_views - completed)
        tail_seconds = patch_total_seconds * self.tail_factor
        rough_remaining_seconds = patch_remaining_seconds + tail_seconds

        print("[estimate] Live runtime update from real PatchMatch sample:", flush=True)
        print(
            f"[estimate]   sampled_views={completed}/{self.total_views}, "
            f"avg_per_view={format_duration(avg_view_seconds)}",
            flush=True,
        )
        print(
            f"[estimate]   PatchMatch total ~= {format_duration(patch_total_seconds)}, "
            f"remaining ~= {format_duration(patch_remaining_seconds)}",
            flush=True,
        )
        if self.full_start is not None:
            rough_total_seconds = (now - self.full_start) + rough_remaining_seconds
            print(
                f"[estimate]   Rough full remaining ~= {format_duration(rough_remaining_seconds)} "
                f"including fusion/meshing tail",
                flush=True,
            )
            print(f"[estimate]   Rough full total ~= {format_duration(rough_total_seconds)}", flush=True)
        else:
            print(
                f"[estimate]   Rough dense remaining ~= {format_duration(rough_remaining_seconds)} "
                f"including fusion/meshing tail",
                flush=True,
            )


def post_patch_tail_factor(script: Path) -> float:
    try:
        text = script.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0.15

    factor = 0.05 if "stereo_fusion" in text else 0.0
    if "delaunay_mesher" in text:
        factor += 0.08
    if "poisson_mesher" in text:
        factor += 0.08
    if "floor_plane_completion.py" in text:
        factor += 0.02
    if "point_cloud_filter.py" in text:
        factor += 0.03
    if "dark_surface_completion.py" in text:
        factor += 0.01
    if "reconstruction_quality.py" in text:
        factor += 0.01
    return factor


def require_script(workspace: Path, mode: str) -> Path:
    script = workspace / SCRIPT_BY_MODE[mode]
    if not script.exists():
        raise SystemExit(f"COLMAP script not found: {script}. Run `python main.py` first.")
    return script


def call_batch(script: Path, workspace: Path) -> int:
    return subprocess.call(["cmd", "/c", "call", str(script)], cwd=str(workspace))


def call_batch_with_runtime_estimate(
    script: Path,
    workspace: Path,
    *,
    runtime_estimate: bool,
    sample_views: int,
    full_start: Optional[float],
) -> int:
    if not runtime_estimate:
        return call_batch(script, workspace)

    estimator = PatchMatchRuntimeEstimator(
        sample_views=sample_views,
        full_start=full_start,
        tail_factor=post_patch_tail_factor(script),
    )
    process = subprocess.Popen(
        ["cmd", "/c", "call", str(script)],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        estimator.observe(line)
    return process.wait()


def run_full_pipeline(workspace: Path, *, runtime_estimate: bool, sample_views: int) -> int:
    sparse_script = require_script(workspace, "sparse")
    dense_script = require_script(workspace, "dense")
    full_start = time.monotonic()

    if runtime_estimate:
        print(
            f"[estimate] Live runtime estimator is on; dense PatchMatch will be refined "
            f"after {max(1, sample_views)} completed view(s)."
        )

    print("[info] running COLMAP sparse step...")
    sparse_result = call_batch(sparse_script, workspace)
    if sparse_result != 0:
        return sparse_result
    print(f"[timing] Sparse step elapsed: {format_duration(time.monotonic() - full_start)}")

    print("[info] running COLMAP dense step...")
    dense_result = call_batch_with_runtime_estimate(
        dense_script,
        workspace,
        runtime_estimate=runtime_estimate,
        sample_views=sample_views,
        full_start=full_start,
    )
    if dense_result == 0:
        print(f"[timing] Full COLMAP elapsed: {format_duration(time.monotonic() - full_start)}")
    return dense_result


def run_colmap_script(
    workspace: Path,
    mode: str,
    *,
    runtime_estimate: bool = True,
    sample_views: int = DEFAULT_RUNTIME_SAMPLE_VIEWS,
    report_dir: Optional[Path] = None,
) -> int:
    report_dir = report_dir or workspace.parent
    if mode == "reset":
        return reset_existing_colmap_outputs(workspace, respect_allow_stale=False, report_dir=report_dir)

    if mode == "full":
        require_script(workspace, "sparse")
        require_script(workspace, "dense")
    else:
        script = require_script(workspace, mode)

    if mode in CLEAN_REBUILD_MODES:
        reset_result = reset_existing_colmap_outputs(workspace, report_dir=report_dir)
        if reset_result != 0:
            return reset_result

    if mode == "full":
        return run_full_pipeline(workspace, runtime_estimate=runtime_estimate, sample_views=sample_views)

    if mode == "dense":
        return call_batch_with_runtime_estimate(
            script,
            workspace,
            runtime_estimate=runtime_estimate,
            sample_views=sample_views,
            full_start=None,
        )
    return call_batch(script, workspace)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    workspace = args.workspace if args.workspace.is_absolute() else root / args.workspace
    if args.mode == "full" and not args.no_upfront_estimate:
        require_script(workspace, "sparse")
        require_script(workspace, "dense")
        if args.yes or not sys.stdin.isatty():
            print_upfront_runtime_estimate(workspace)
        elif not confirm_full_colmap_run(workspace):
            return 0
    return run_colmap_script(
        workspace,
        args.mode,
        runtime_estimate=not args.no_runtime_estimate,
        sample_views=args.runtime_estimate_sample_views,
        report_dir=root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
