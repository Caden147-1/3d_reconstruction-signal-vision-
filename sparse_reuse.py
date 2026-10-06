"""Prepare a separate dense run from a byte-verified existing sparse solve.

Original inputs/results are never rewritten. The imported sparse runner is a
non-executable stub, and its receipt explicitly records import rather than a
new SfM run. Dense maps and dense receipts are deliberately not imported.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil

from colmap_workspace_check import (PROVENANCE_NAME, contained_path, file_identity,
                                    inspect_provenance, write_provenance, RECOVERY_PIPELINE_CODE)
from workspace_lock import workspace_lock

ROOT = Path(__file__).resolve().parent
IMPORT_KIND = "verified_sparse_import_v1"
HISTORY = "sparse_import/source_sparse_run_provenance.json"
STUB = ("@echo off\r\necho [error] This workspace imports a completed sparse reconstruction.\r\n"
        "echo Run the dense stage only; create a fresh workspace to repeat sparse mapping.\r\nexit /b 2\r\n").encode("ascii")


def source_artifact(workspace, relative):
    if relative in {"pipeline_code/" + name for name in RECOVERY_PIPELINE_CODE}:
        return ROOT / Path(relative).name
    return contained_path(workspace, relative)


def verify_import_record(workspace, record):
    """Validate the exact allowed difference from the archived sparse receipt."""
    if record.get("provenance_kind") != IMPORT_KIND:
        raise ValueError("Unknown sparse import provenance kind")
    imported = record.get("sparse_import", {})
    if imported.get("source_receipt") != HISTORY:
        raise ValueError("Unexpected sparse import history path")
    history = workspace / HISTORY
    if not imported.get("source_receipt_identity") or file_identity(history) != imported["source_receipt_identity"]:
        raise ValueError("Sparse import history changed")
    original = json.loads(history.read_text(encoding="utf-8"))
    if original.get("provenance_kind") or original.get("schema_version") != 1 or original.get("status") != "complete":
        raise ValueError("Import requires an original completed sparse-run receipt")
    inputs = json.loads(json.dumps(original["inputs"]))
    runner = inputs["runner"]
    if runner not in {"run_colmap_sparse.bat", "run_colmap_automatic.bat"}:
        raise ValueError("Unexpected imported sparse runner")
    if (workspace / runner).read_bytes() != STUB:
        raise ValueError("Imported sparse runner must remain a non-executable stub")
    inputs["files"][runner] = file_identity(workspace / runner)
    if inputs != record.get("inputs") or original["outputs"] != record.get("outputs"):
        raise ValueError("Sparse import changed original inputs or model/database identities")


def coverage_policy(max_images=1200, initial_images=480, single=.9, three=.8, gap=4.):
    return {"mode": "adaptive", "initial_images": min(initial_images, max_images) if max_images else initial_images,
            "min_single_view_coverage": single, "min_three_view_coverage": three, "max_video_gap_s": gap}


def prepare_dense_reuse(source, destination, *, max_images=1200, max_sources=12,
                        policy=None, dense_profile="efficient", image_size=None,
                        link_images=True, filter_max_candidates=2000000, filter_budget_seconds=120.):
    from colmap_bridge import dense_profile_config, write_dense_batch
    from dense_keyframes import selection_config

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must be separate, non-nested workspaces")
    if destination.exists():
        raise ValueError("Dense reuse requires a new destination; existing paths are preserved")
    config = {"enabled": True, "max_images": max_images, "max_source_images": max_sources}
    if policy is not None:
        config["coverage_policy"] = dict(policy)
    selection_config({"dense_selection_config": config})
    profile = dense_profile_config(dense_profile)
    if image_size is not None and (type(image_size) is not int or image_size < 100):
        raise ValueError("Dense image size must be an integer of at least 100 pixels")
    if type(filter_max_candidates) is not int or not 1 <= filter_max_candidates <= 2000000 or not math.isfinite(filter_budget_seconds) or filter_budget_seconds <= 0:
        raise ValueError("Invalid observation filter budget")

    with workspace_lock(source):
        proof = inspect_provenance(source)
        if proof.get("status") != "verified":
            raise ValueError("Source sparse reconstruction is not verified: " + proof.get("status", "unknown"))
        original = json.loads((source / PROVENANCE_NAME).read_text(encoding="utf-8"))
        if original.get("provenance_kind"):
            raise ValueError("Use the original source workspace, not another imported workspace")
        settings = json.loads((source / "colmap_workspace_summary.json").read_text(encoding="utf-8"))
        # Masks remain external read-only inputs and are covered by sparse provenance.
        mask = settings.get("colmap_mask", {})
        if original["inputs"].get("mask_root") != (str(Path(mask["path"]).resolve()) if mask.get("enabled") and mask.get("path") else None):
            raise ValueError("Source mask settings disagree with the sparse receipt")
        executable = Path(settings["colmap_exe"])
        executable = executable if executable.is_absolute() else source / executable
        if not executable.is_file():
            raise ValueError("COLMAP executable is missing")
        paths = {**original["inputs"]["files"], **original["outputs"]["files"]}
        # Validate every path before creating the destination or copying any bytes.
        for relative, identity in paths.items():
            contained_path(source, relative)
            contained_path(destination, relative)
            if identity is not None and not source_artifact(source, relative).is_file():
                raise ValueError("Missing source artifact: " + relative)
        destination.mkdir(parents=True)
        pending = destination / "sparse_import_pending.json"
        write_provenance(pending, {"source": str(source), "status": "copying"})
        image_links = image_copies = 0
        for relative, identity in paths.items():
            if identity is None or relative == original["inputs"]["runner"]:
                continue
            before, after = source_artifact(source, relative), contained_path(destination, relative)
            after.parent.mkdir(parents=True, exist_ok=True)
            if link_images and relative.startswith("images/"):
                try:
                    os.link(before, after)
                    image_links += 1
                    continue
                except OSError:
                    pass
            shutil.copy2(before, after)
            image_copies += int(relative.startswith("images/"))
        # Additional metadata aids diagnostics but never replaces authenticated files.
        if (source / "metadata").is_dir():
            for before in sorted((source / "metadata").iterdir()):
                after = destination / "metadata" / before.name
                if before.is_file() and not after.exists():
                    after.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(before, after)
        history = destination / HISTORY
        history.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / PROVENANCE_NAME, history)
        runner = original["inputs"]["runner"]
        (destination / runner).write_bytes(STUB)
        imported = json.loads(json.dumps(original))
        imported.update(provenance_kind=IMPORT_KIND, sparse_import={
            "source_workspace": str(source), "source_receipt": HISTORY,
            "source_receipt_identity": file_identity(history),
            "interpretation": "Verified sparse artifact import; no new feature matching or camera solve was performed."})
        imported["inputs"]["files"][runner] = file_identity(destination / runner)
        write_provenance(destination / PROVENANCE_NAME, imported)
        settings.update(workspace_dir=str(destination), colmap_exe=str(executable.resolve()),
                        dense_selection_config=config, dense_profile=dense_profile, dense_profile_config=profile,
                        dense_max_image_size=image_size or settings.get("dense_max_image_size", 2000),
                        point_filter_config={"method": "observed", "max_candidates": filter_max_candidates,
                                             "budget_seconds": filter_budget_seconds},
                        run_point_cloud_filter=True, run_meshing=False, requested_run_meshing=False,
                        scene_completion="off", cleanup_intermediate_ply=bool(profile.get("cleanup_intermediate_ply", True)),
                        sparse_reuse={"source_workspace": str(source), "kind": IMPORT_KIND,
                                      "image_hardlinks": image_links, "image_copies": image_copies,
                                      "note": "Image links share original bytes; pipeline treats capture images as immutable. Use --copy-images for independent copies."},
                        scripts=[runner, "run_colmap_dense.bat"])
        write_provenance(destination / "colmap_workspace_summary.json", settings)
        write_dense_batch(ROOT, destination, executable.resolve(), Path(mask["path"]) if mask.get("enabled") else None,
                          settings["dense_max_image_size"], bool(settings.get("pdr_alignment_enabled")),
                          dense_profile, False, True, settings.get("colmap_num_threads", 0),
                          dense_selection_enabled=True, point_filter_method="observed",
                          point_filter_max_candidates=filter_max_candidates,
                          point_filter_budget_seconds=filter_budget_seconds)
        # Both copies must still match their receipts after all file operations.
        checked = inspect_provenance(source)
        if checked.get("status") != "verified":
            raise ValueError(f"Source changed during sparse import: {checked}")
        pending.unlink()
        checked = inspect_provenance(destination)
        if checked.get("status") != "verified":
            write_provenance(pending, {"source": str(source), "status": "verification_failed", "error": checked})
            raise ValueError(f"Imported sparse verification failed: {checked}")
        return {"status": "prepared", "source_workspace": str(source), "workspace": str(destination),
                "dense_input": proof["dense_input"], "dense_selection_config": config,
                "image_hardlinks": image_links, "image_copies": image_copies,
                "next_command": f'python run_colmap.py dense --workspace "{destination}"'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Verified original workspace to reuse.")
    parser.add_argument("--output", type=Path, required=True, help="New independent workspace; must not exist.")
    parser.add_argument("--dense-max-images", type=int, default=1200)
    parser.add_argument("--dense-max-source-images", type=int, default=12)
    parser.add_argument("--dense-initial-images", type=int, default=480)
    parser.add_argument("--dense-min-single-coverage", type=float, default=.9)
    parser.add_argument("--dense-min-three-coverage", type=float, default=.8)
    parser.add_argument("--dense-max-video-gap-s", type=float, default=4.)
    parser.add_argument("--dense-profile", choices=("efficient", "balanced", "rtx4080", "complete"), default="efficient")
    parser.add_argument("--dense-max-image-size", type=int)
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = prepare_dense_reuse(args.workspace, args.output, max_images=args.dense_max_images,
            max_sources=args.dense_max_source_images,
            policy=coverage_policy(args.dense_max_images, args.dense_initial_images,
                                   args.dense_min_single_coverage, args.dense_min_three_coverage,
                                   args.dense_max_video_gap_s),
            dense_profile=args.dense_profile, image_size=args.dense_max_image_size, link_images=not args.copy_images)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[error] {exc}", flush=True)
        return 2
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
