"""Resume verified reconstruction stages while preserving existing PLY results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys
import uuid

from colmap_workspace_check import file_identity, write_provenance
from dense_cache import (CACHE_RECORD, cache_identity, checked_dense_path,
                         read_record, reconstruction_binding,
                         validate_cache_outputs)
from finalize_dense_outputs import commit_final
from ply_io import validate_ply
from process_control import managed_process
from workspace_lock import WorkspaceBusyError, locked_workspace, inherited_lock_held


MODES = ("resume", "filter", "completion", "report")
ROOT = Path(__file__).resolve().parent
FUSION_RECEIPT = "fusion_provenance.json"


def verified_context(workspace: Path, *, verify_maps: bool = False) -> dict:
    """Require current sparse inputs, cache settings and undistorted model hashes."""
    dense = checked_dense_path(workspace)
    current = cache_identity(workspace)
    receipt = read_record(dense / CACHE_RECORD)
    if receipt.get("status") != "complete" or receipt.get("identity") != current:
        raise ValueError("No completed MVS cache matches the current input hashes/settings; use the explicit dense stage to rebuild it")
    if not isinstance(receipt.get("outputs", {}).get("files"), dict):
        raise ValueError("MVS cache has no output integrity receipt")
    binding = reconstruction_binding(dense)
    if not binding:
        raise ValueError("MVS cache has no reconstruction binding")
    for name, identity in binding["dense_model"].items():
        if identity is None or receipt["outputs"]["files"].get("sparse/"+name) != identity:
            raise ValueError("The undistorted model changed since MVS completion")
    if verify_maps:
        print("[stages] Checking MVS image/depth/normal hashes before fusion reuse...", flush=True)
        if validate_cache_outputs(workspace) != receipt["outputs"]:
            raise ValueError("MVS outputs differ from their completed integrity receipt")
    return binding


def verified_final(workspace: Path, binding: dict) -> dict | None:
    dense = checked_dense_path(workspace)
    final = dense / "final.ply"
    receipt = read_record(dense / "final_output_summary.json")
    if not final.exists():
        return None
    if not receipt.get("final_identity") or file_identity(final) != receipt["final_identity"]:
        raise ValueError("Existing final.ply has no matching identity receipt; it was preserved")
    if receipt.get("reconstruction_binding") != binding:
        raise ValueError("Existing final.ply belongs to different reconstruction inputs; it was preserved")
    if validate_ply(final).get("vertex") != receipt.get("vertex_count"):
        raise ValueError("Existing final.ply vertex count differs from its receipt")
    if receipt.get("source_kind") == "videoscene_completed_point_cloud":
        observed = Path(receipt.get("observed_source", "")).resolve()
        if (not observed.is_relative_to(dense) or not receipt.get("observed_identity")
                or file_identity(observed) != receipt["observed_identity"]):
            raise ValueError("Completed final has no verified retained observed point cloud")
    return receipt


def checked_fusion_source(workspace: Path, binding: dict) -> Path | None:
    dense = checked_dense_path(workspace)
    source = dense / "fused.ply"
    receipt = read_record(dense / FUSION_RECEIPT)
    if (receipt.get("reconstruction_binding") == binding and receipt.get("output_identity")
            and file_identity(source) == receipt["output_identity"]):
        validate_ply(source)
        return source
    return None


@locked_workspace("workspace")
def record_fusion(workspace: Path) -> dict:
    """Called only after the generated batch's successful stereo_fusion stage."""
    binding = verified_context(workspace)
    dense = checked_dense_path(workspace)
    source = dense / "fused.ply"
    vertices = validate_ply(source).get("vertex", 0)
    if vertices <= 0:
        raise ValueError("Cannot certify an empty fused point cloud")
    receipt = {"status": "complete", "output": str(source),
               "output_identity": file_identity(source), "vertex_count": vertices,
               "visibility_identity": file_identity(source.with_suffix('.ply.vis')),
               "reconstruction_binding": binding}
    write_provenance(dense / FUSION_RECEIPT, receipt)
    return receipt


def run_command(command: list[str]) -> None:
    with managed_process(command, cwd=str(ROOT)) as process:
        result = process.wait()
    if result:
        raise RuntimeError(f"Stage command exited with code {result}: {Path(command[0]).name}")


def refresh_report(workspace: Path) -> None:
    from reconstruction_quality import refresh_quality_report
    refresh_quality_report(workspace)


def completed_empty_attempt(workspace: Path, final: dict | None, binding: dict, effective: dict) -> bool:
    """A verified zero-addition attempt is completed work, not a missing stage."""
    if final is None:
        return False
    receipt = read_record(workspace / "dense/videoscene_summary.json")
    expected_code = {name: file_identity(ROOT / name) for name in
                     ("videoscene_completion.py", "videoscene_geometry.py", "videoscene_conditioning.py", "videoscene_runner.py", "mvs_observations.py")}
    config = receipt.get("effective_config", {})
    return (receipt.get("status") in {"no_endpoint_pairs", "no_eligible_pairs", "no_observed_overlap",
                                      "no_supported_holes", "no_geometry_passed", "no_observed_geometry_passed"}
            and receipt.get("inference_requested") is True
            and receipt.get("input_final_identity") == final.get("final_identity")
            and receipt.get("reconstruction_binding") == binding
            and receipt.get("completion_code_identity") == expected_code
            and config.get("max_clips") == effective["max_clips"]
            and config.get("budget_hours") == effective["budget_hours"]
            and config.get("merge_generated") is True
            and config.get("geometry_policy") == "two_original_depth_maps_v1"
            and config.get("prior_conditioning") == "endpoints" and config.get("seed") == 20260915)


def filter_stage(workspace: Path, binding: dict, run_dir: Path, settings: dict) -> dict:
    """Validate cached MVS; filter a bound source and atomically publish a candidate."""
    from runner_migration import effective_execution_settings
    settings = effective_execution_settings(workspace, settings)
    if verified_context(workspace, verify_maps=True) != binding:
        raise ValueError("Reconstruction binding changed before filtering")
    dense = checked_dense_path(workspace)
    source = checked_fusion_source(workspace, binding)
    if source is None:
        # A valid PLY header cannot prove which MVS run produced an old fused
        # file. Re-fuse verified maps into this run's isolated directory.
        source = run_dir / "fused.ply"
        executable = Path(settings["colmap_exe"])
        if not executable.is_absolute():
            executable = workspace / executable
        command = [str(executable), "stereo_fusion", "--workspace_path", str(dense),
                   "--workspace_format", "COLMAP", "--input_type", "geometric"]
        # Match the original dense runner's resolution cap. Zero-distortion
        # COLMAP cameras may still carry native dimensions after undistortion,
        # so omitting this cap when re-fusing cached maps changes calibration.
        fusion_settings = {**settings.get("dense_profile_config", {}).get("fusion", {}),
                           "max_image_size": settings.get("dense_max_image_size", 2000)}
        for key, value in fusion_settings.items():
            command.extend(["--StereoFusion."+key, str(value)])
        if settings.get("colmap_num_threads", 0) > 0:
            command.extend(["--StereoFusion.num_threads", str(settings["colmap_num_threads"])])
        mask = settings.get("colmap_mask", {})
        if mask.get("enabled") and mask.get("path"):
            command.extend(["--StereoFusion.mask_path", str(mask["path"])])
        command.extend(["--output_path", str(source)])
        print("[stages] Fusing verified MVS maps into an isolated candidate...", flush=True)
        run_command(command)
        write_provenance(run_dir / FUSION_RECEIPT,
                         {'status':'complete', 'output':str(source), 'output_identity':file_identity(source),
                          'visibility_identity':file_identity(source.with_suffix('.ply.vis')),
                          'reconstruction_binding':binding})
    if validate_ply(source).get("vertex", 0) <= 0:
        raise ValueError("Fusion did not produce a nonempty point cloud")
    source_identity = file_identity(source)
    filtered = run_dir / "fused_filtered.ply"
    filter_summary = run_dir / "point_filter_summary.json"
    max_points = int(settings.get("dense_profile_config", {}).get("point_filter_max_points", 0))
    filter_config = settings.get("point_filter_config", {})
    filter_method = filter_config.get("method", "wifi-density")
    if filter_method == "observed":
        max_points = 0
    run_command([sys.executable, str(ROOT / "point_cloud_filter.py"), "--input", str(source),
                 "--output", str(filtered), "--summary", str(filter_summary),
                 "--method", filter_method, "--workspace", str(workspace),
                 "--observation-max-candidates", str(filter_config.get("max_candidates", 250000)),
                 "--observation-budget-seconds", str(filter_config.get("budget_seconds", 120.)),
                 "--max-points", str(max_points)])
    count = validate_ply(filtered).get("vertex", 0)
    if count <= 0 or file_identity(source) != source_identity:
        raise ValueError("Filter source changed or produced an empty point cloud")
    if verified_context(workspace) != binding:
        raise ValueError("Reconstruction inputs changed while filtering; candidate retained without promotion")
    filtered_identity = file_identity(filtered)
    previous = verified_final(workspace, binding)
    filter_receipt = {**read_record(filter_summary), "input_identity": source_identity,
                      "output_identity": filtered_identity, "reconstruction_binding": binding}
    write_provenance(filter_summary, filter_receipt)
    if previous and (filtered_identity == previous.get("final_identity") or
                     filtered_identity == previous.get("observed_identity")):
        return {"status": "unchanged", "candidate": str(filtered), "retained_final": str(dense / "final.ply")}
    if previous:
        backup = run_dir / "previous_final.ply"
        shutil.copy2(dense / "final.ply", backup)
        if file_identity(backup) != previous["final_identity"]:
            raise ValueError("Could not verify the retained final backup")
        write_provenance(run_dir / "previous_final_output_summary.json", previous)
    payload = {"status": "ok", "action": "copied_and_replaced", "source_kind": "filtered_point_cloud",
               "source": str(filtered), "output": str(dense / "final.ply"), "vertex_count": count,
               "face_count": None, "raw_fused_preserved": True, "keep_intermediate": True,
               "reconstruction_binding": binding, "filter_receipt": str(filter_summary),
               "explanation": "Filtered observed geometry; staged inputs and any previous final were retained."}
    commit_final(filtered, dense / "final.ply", dense / "final_output_summary.json", payload)
    write_provenance(dense / "point_filter_summary.json", filter_receipt)
    return {"status": "completed", "candidate": str(filtered), "vertex_count": count}


@locked_workspace("workspace")
def run_stage(workspace: Path, mode: str, *, dry_run: bool = False,
              max_clips: int | None = None, budget_hours: float | None = None,
              with_videoscene: bool = False) -> int:
    workspace = workspace.resolve()
    if mode not in MODES:
        raise ValueError("Unknown reconstruction stage")
    if mode == "report":
        if dry_run:
            print("[stages] Planned: report", flush=True)
        else:
            refresh_report(workspace)
        return 0
    settings = read_record(workspace / "colmap_workspace_summary.json")
    if with_videoscene:
        if mode != 'resume':
            raise ValueError('VideoScene resume override requires resume mode')
        settings = {**settings, 'scene_completion': 'videoscene'}
    try:
        binding = verified_context(workspace)
    except ValueError:
        # A completed sparse stage can continue through an interrupted MVS
        # stage using the canonical dense runner and its cache checks. Never
        # run its cleanup over an existing PLY while resolving invalid identity.
        from colmap_workspace_check import inspect_provenance
        dense = checked_dense_path(workspace)
        if mode != "resume" or any(dense.glob("*.ply")) or inspect_provenance(workspace)["status"] != "verified":
            raise
        print("[stages] Planned: dense (verified sparse inputs) -> configured post-processing -> report", flush=True)
        if dry_run:
            return 0
        from run_colmap import run_colmap_script
        code = run_colmap_script(workspace, "dense", runtime_estimate=True)
        if code == 0 and with_videoscene:
            return run_stage(workspace, 'resume', max_clips=max_clips, budget_hours=budget_hours,
                             with_videoscene=True)
        return code
    final = verified_final(workspace, binding)
    if mode == "completion" and final is None:
        raise ValueError("Completion requires a verified filtered final; use resume to finish filtering first")
    stages = []
    if mode == "filter" or (mode == "resume" and final is None):
        stages.append("filter")
    if mode == "completion" or (mode == "resume" and settings.get("scene_completion") == "videoscene"
                                and (final is None or final.get("source_kind") != "videoscene_completed_point_cloud"
                                     or final.get("completion_policy") != "two_original_depth_maps_v1")):
        stages.append("completion")
    stages.append("report")
    effective = {"max_clips": int(max_clips if max_clips is not None else settings.get("videoscene_max_clips", 3)),
                 "budget_hours": float(budget_hours if budget_hours is not None else settings.get("videoscene_budget_hours", 4.))}
    if (not 1 <= effective["max_clips"] <= 8 or not math.isfinite(effective["budget_hours"])
            or not 0 < effective["budget_hours"] <= 20):
        raise ValueError("Completion requires 1..8 clips and a finite time budget in (0,20] hours")
    if mode == "resume" and "completion" in stages and completed_empty_attempt(workspace, final, binding, effective):
        stages.remove("completion")
    print("[stages] Planned: " + " -> ".join(stages), flush=True)
    if dry_run:
        return 0
    dense = checked_dense_path(workspace)
    run_dir = dense / "stage_runs" / uuid.uuid4().hex
    if not run_dir.resolve().is_relative_to(dense):
        raise ValueError("Stage output directory redirects outside dense")
    run_dir.mkdir(parents=True)
    receipt = {"status": "running", "mode": mode, "stages": stages,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "reconstruction_binding": binding, "effective_completion_config": effective,
               "completion_explicitly_requested": mode == "completion", "completed_stages": []}
    receipt_path = run_dir / "stage_receipt.json"
    write_provenance(receipt_path, receipt)
    try:
        for stage in stages:
            if stage == "filter":
                receipt["filter"] = filter_stage(workspace, binding, run_dir, settings)
            elif stage == "completion":
                from videoscene_completion import main as complete
                code = complete(["--workspace", str(workspace), "--max-clips", str(effective["max_clips"]),
                                 "--budget-hours", str(effective["budget_hours"])], refresh_report=False)
                if code:
                    raise RuntimeError(f"VideoScene completion exited with code {code}")
            else:
                refresh_report(workspace)
            receipt["completed_stages"].append(stage)
            write_provenance(receipt_path, receipt)
        receipt.update(status="completed", final_identity=file_identity(dense / "final.ply"))
        write_provenance(receipt_path, receipt)
        from generated_input_cleanup import cleanup_after_success
        cleanup_after_success(workspace)
        return 0
    except BaseException as error:
        receipt.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_provenance(receipt_path, receipt)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(*MODES, "record-fusion"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--budget-hours", type=float)
    args = parser.parse_args(argv)
    try:
        if args.mode == "record-fusion":
            if not inherited_lock_held(args.workspace.resolve()):
                raise ValueError("Fusion certification is internal to the locked dense runner; old fused clouds cannot be certified retrospectively")
            record_fusion(args.workspace.resolve())
            return 0
        return run_stage(args.workspace, args.mode, dry_run=args.dry_run,
                         max_clips=args.max_clips, budget_hours=args.budget_hours)
    except (OSError, ValueError, KeyError, RuntimeError, WorkspaceBusyError) as error:
        print(f"[stages error] {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
