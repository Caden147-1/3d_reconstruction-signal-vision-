"""Initialize missing video intrinsics before mapping, without rematching frames."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from colmap_workspace_check import write_provenance
from process_control import managed_process
from video_intrinsics import estimate_video_intrinsics, apply_video_intrinsics
from workspace_lock import inherited_lock_held


def mapper_command(workspace, executable, snapshots, report, num_threads=0):
    cameras = report.get("cameras", [])
    # Global mapper options must never freeze unknown photo/other-camera parameters.
    hold_video = len(cameras) == 1 and cameras[0].get("status") == "estimated"
    command = [str(executable), "mapper", "--database_path", str(workspace / "database.db"),
               "--image_path", str(workspace / "images"), "--output_path", str(workspace / "sparse"),
               "--Mapper.multiple_models", "0", "--Mapper.snapshot_path", str(snapshots),
               "--Mapper.snapshot_frames_freq", "50"]
    if num_threads > 0 or hold_video:
        command += ["--Mapper.num_threads", str(num_threads if num_threads > 0 else min(8, os.cpu_count() or 1))]
    if hold_video:
        command += ["--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0"]
    return command


def run_mapper(workspace, executable, snapshots, num_threads=0):
    if not inherited_lock_held(workspace):
        raise ValueError("Video mapper requires the sparse runner's workspace lock")
    report = estimate_video_intrinsics(workspace / "database.db", workspace / "metadata/images_manifest.csv")
    apply_video_intrinsics(workspace / "database.db", report)
    command = mapper_command(workspace, executable, snapshots, report, num_threads)
    report["mapper_command"] = command
    report["intrinsics_held_during_mapping"] = "--Mapper.ba_refine_focal_length" in command
    write_provenance(workspace / "video_intrinsics_summary.json", report)
    for camera in report["cameras"]:
        if camera["status"] == "estimated":
            print(f"[calibration] Video camera {camera['camera_id']}: default focal "
                  f"{camera['original_params'][0]:.1f} -> visual estimate {camera['focal_px']:.1f} px; "
                  f"{camera['consensus_pairs']} pair estimates agree across {camera['time_coverage']:.1%} of clip.", flush=True)
    if report["intrinsics_held_during_mapping"]:
        print("[calibration] Holding shared video focal/distortion during mapping; visual estimate is recorded for review.", flush=True)
    else:
        print("[calibration] Keeping standard intrinsic refinement; no eligible single-video focal consensus.", flush=True)
    with managed_process(command) as process:
        return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--colmap-exe", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=0)
    args = parser.parse_args()
    try:
        return run_mapper(args.workspace.resolve(), args.colmap_exe, args.snapshots, args.num_threads)
    except (OSError, ValueError) as exc:
        print(f"[error] Video mapper: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
