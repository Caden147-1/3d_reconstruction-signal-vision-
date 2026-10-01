"""Gate PDR alignment after mapping and publish only a validated complete model."""

import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np

from colmap_workspace_check import (
    ALIGNMENT_RECORD, PENDING_NAME, PROVENANCE_NAME, file_identity,
    inspect_sparse_model, primary_sparse_model, sparse_input_identity, sparse_model_identity, write_provenance,
)
from reconstruction_diagnostics import (
    PHOTO_GROUPING_REVIEW_FIELDS, fit_similarity, pdr_diagnostics,
    photo_grouping_review_rows, transform,
)
from reconstruction_evaluation import Snapshot, sparse_metrics
from process_control import managed_process
from workspace_lock import WorkspaceBusyError, locked_workspace


REVIEWED_REFS = "pdr_alignment_reviewed_refs.txt"
POLICY = {"method": "checkpoint_consensus_and_held_out_validation", "max_residual": 2.0,
          "max_checkpoint_scatter": 2.0, "minimum_checkpoints": 4,
          "weighting": "robust_stationary_centers_and_relative_scatter",
          "session_policy": "single_best_reliable_session"}


def write_similarity_transform(path, fit):
    """Write COLMAP Sim3 text: scale qw qx qy qz tx ty tz (not a matrix)."""
    scale, r, translation = fit
    # Largest eigenvector gives xyzw, including rotations close to 180 degrees.
    k = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
        [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
        [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ])
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    if q[0] < 0:
        q = -q
    values = np.r_[scale, q, translation]
    Path(path).write_text(" ".join(f"{value:.17g}" for value in values) + "\n", encoding="utf-8")


def inspect_alignment(model, manifest_path, report_path, *, capture_mode="auto", exported_manifest=False,
                      grouping_review_csv=None):
    """Inspect existing reconstruction inputs without alignment or workspace writes."""
    model, manifest_path, report_path = (Path(p).resolve() for p in (model, manifest_path, report_path))
    if report_path.suffix.lower() != ".json" or report_path == manifest_path or report_path.is_relative_to(model):
        raise ValueError("Inspection report must be a separate JSON file outside the input model")
    if grouping_review_csv is not None:
        grouping_review_csv = Path(grouping_review_csv).resolve()
        if (grouping_review_csv.suffix.lower() != ".csv" or grouping_review_csv in {manifest_path, report_path}
                or grouping_review_csv.is_relative_to(model)
                or (grouping_review_csv.is_file() and manifest_path.is_file() and grouping_review_csv.samefile(manifest_path))):
            raise ValueError("Grouping review must be a separate CSV outside the input model and manifest")
    snapshot = Snapshot()
    manifest_identity = snapshot.identity(manifest_path)
    if manifest_identity is None:
        raise ValueError("Inspection manifest is missing")
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if exported_manifest:
        from colmap_bridge import build_workspace_manifest
        manifest = build_workspace_manifest(manifest)
    metrics = sparse_metrics(model, snapshot)
    anchors = [row for row in manifest if row.get("source_type") == "wifi_anchor_photo"]
    matched = sum(row.get("image_file") in metrics["camera_centers"] for row in anchors)
    if anchors and not matched:
        raise ValueError("No checkpoint image names match the model; check --exported-manifest and input dataset")
    diagnostics, refs = pdr_diagnostics(manifest, metrics["camera_centers"],
                                      POLICY["max_residual"], POLICY["max_checkpoint_scatter"], capture_mode)
    report = {"schema_version": 1, "inspection_only": True, "model": str(model),
              "model_identity": metrics["model_identity"], "manifest": str(manifest_path),
              "manifest_identity": manifest_identity, "exported_manifest": exported_manifest,
              "registered_checkpoint_images": matched, "reviewable_reference_count": len(refs),
              "diagnostics": diagnostics,
              "code_identities": {name: snapshot.identity(Path(__file__).with_name(name))
                                  for name in ("pdr_alignment.py", "reconstruction_diagnostics.py")}}
    snapshot.verify_unchanged()
    if grouping_review_csv is not None:
        review_rows = photo_grouping_review_rows(manifest, metrics["camera_centers"],
                                                diagnostics["photo_grouping"], metrics["camera_support"])
        grouping_review_csv.parent.mkdir(parents=True, exist_ok=True)
        with grouping_review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PHOTO_GROUPING_REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(review_rows)
        report["grouping_review_csv"] = {"path": str(grouping_review_csv),
            "identity": file_identity(grouping_review_csv), "rows": len(review_rows),
            "unregistered_photos": sum(not row["registered"] for row in review_rows),
            "interpretation": "Current assignments and geometric review evidence only; no_flag does not mean verified. Nearer checkpoint is not a proposed replacement."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[PDR inspection] {matched} registered checkpoint images; {len(refs)} reviewable references; report {report_path}", flush=True)
    return report


@locked_workspace("workspace")
def align_workspace(workspace, colmap_exe, *, disabled=False, capture_mode="auto"):
    if capture_mode not in {"auto", "stationary", "off"}:
        raise ValueError("Unknown PDR capture mode")
    workspace = Path(workspace).resolve()
    if (workspace / PROVENANCE_NAME).exists():
        raise ValueError("Existing certified geometry must not be realigned; reset for a new reconstruction")
    pending = json.loads((workspace / PENDING_NAME).read_text(encoding="utf-8"))
    if pending.get("schema_version") != 1 or sparse_input_identity(workspace, pending["inputs"]["runner"]) != pending["inputs"]:
        raise ValueError("Sparse inputs changed before PDR validation")
    destination = workspace / "sparse_pdr_aligned"
    staging = workspace / "sparse_pdr_aligned.pending"
    if any(path.exists() for path in (destination, staging, workspace / ALIGNMENT_RECORD, workspace / REVIEWED_REFS)):
        raise ValueError("Previous alignment artifacts exist; reset before starting a new sparse run")
    snapshot = Snapshot()
    raw = primary_sparse_model(workspace)
    if raw is None:
        raise ValueError("No valid sparse component is available for PDR alignment")
    raw_name = raw.relative_to(workspace).as_posix()
    source = sparse_metrics(raw, snapshot)
    manifest_path = workspace / "metadata/images_manifest.csv"
    decision = dict(schema_version=1, status="withheld", reason="visual_only" if disabled else "capture_mode_off" if capture_mode == "off" else "no_reliable_checkpoints",
                    selected_model=raw_name, source_model=raw_name, policy={**POLICY, "capture_mode": capture_mode},
                    source_model_identity=sparse_model_identity(raw), manifest_identity=snapshot.identity(manifest_path))
    if not disabled:
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        diagnostics, _ = pdr_diagnostics(manifest, source["camera_centers"], POLICY["max_residual"], POLICY["max_checkpoint_scatter"], capture_mode)
        decision["diagnostics"] = diagnostics
        # Independent capture sessions may use different PDR origins. Never pool
        # their references into one transformation of the visual reconstruction.
        sessions = [session for session in diagnostics["sessions"] if session["status"] == "reviewable_reference_subset"]
        sessions.sort(key=lambda item: (-item["accepted_checkpoint_count"], item["session"]))
        if sessions:
            session = sessions[0]
            entries = [entry for entry in session["checkpoints"] if entry["accepted_reference"]]
            refs = [entry["representative"] + " " + " ".join(map(str, entry["pdr"])) for entry in entries]
            reference_centers = np.array([source["camera_centers"][entry["representative"]]["center_scene_units"] for entry in entries])
            targets = np.array([entry["pdr"] for entry in entries])
            group_centers = np.array([entry["center"] for entry in entries])
            group_weights = np.array([entry["group_soft_weight"] for entry in entries])
            try:
                fit = fit_similarity(group_centers, targets, group_weights)
                if np.max(np.linalg.norm(transform(group_centers, fit) - targets, axis=1)) > POLICY["max_residual"]:
                    raise ValueError("Weighted checkpoint centers exceed the accepted residual threshold")
                if max(entry["scatter_scene_units"] for entry in entries) * fit[0] > POLICY["max_checkpoint_scatter"]:
                    raise ValueError("Weighted checkpoint scatter exceeds the accepted threshold")
                if np.max(np.linalg.norm(transform(reference_centers, fit) - targets, axis=1)) > POLICY["max_residual"]:
                    raise ValueError("Representative camera centers exceed the accepted residual threshold")
            except ValueError as exc:
                decision["reason"] = str(exc)
            else:
                refs_path = workspace / REVIEWED_REFS
                refs_path.write_text("\n".join(refs) + "\n", encoding="utf-8")
                decision.update(selected_session=session["session"], accepted_reference_count=len(refs),
                                fit_reference_kind="robust_checkpoint_centers",
                                checkpoint_weights={e["checkpoint"]: e["group_soft_weight"] for e in entries},
                                reviewed_refs_identity=file_identity(refs_path), status="aligning")
                write_provenance(workspace / ALIGNMENT_RECORD, decision)
                staged_model = staging / "0"
                staged_model.mkdir(parents=True)
                try:
                    snapshot.verify_unchanged()
                    matrix = np.eye(4)
                    matrix[:3, :3], matrix[:3, 3] = fit[0] * fit[1], fit[2]
                    transform_path = staging / "transform.txt"
                    write_similarity_transform(transform_path, fit)
                    transform_identity = file_identity(transform_path)
                    decision["similarity_transform"] = matrix.tolist()
                    command = [str(colmap_exe), "model_transformer", "--input_path", str(raw),
                        "--output_path", str(staged_model), "--transform_path", str(transform_path), "--is_inverse", "0"]
                    with managed_process(command, cwd=workspace) as process:
                        returncode = process.wait()
                    if returncode:
                        raise ValueError(f"COLMAP model_transformer exited with code {returncode}")
                    checked = inspect_sparse_model(staged_model)
                    if checked["names"] != set(source["camera_centers"]):
                        raise ValueError("Alignment changed the registered image set")
                    aligned = sparse_metrics(staged_model, Snapshot())
                    names = sorted(source["camera_centers"])
                    expected = transform([source["camera_centers"][name]["center_scene_units"] for name in names], fit)
                    actual = np.array([aligned["camera_centers"][name]["center_scene_units"] for name in names])
                    if not np.allclose(actual, expected, rtol=1e-7, atol=1e-7):
                        raise ValueError("Native camera transform differs from weighted checkpoint fit")
                    if aligned["point_count"] != source["point_count"] or aligned["observation_count"] != source["observation_count"]:
                        raise ValueError("Alignment changed points or observation tracks")
                    residuals = [float(np.linalg.norm(np.array(aligned["camera_centers"][entry["representative"]]["center_scene_units"]) - entry["pdr"])) for entry in entries]
                    if max(residuals) > POLICY["max_residual"]:
                        raise ValueError("Aligned model exceeds the accepted PDR residual threshold")
                    snapshot.verify_unchanged()
                    if file_identity(refs_path) != decision["reviewed_refs_identity"]:
                        raise ValueError("Reviewed PDR references changed during alignment")
                    if file_identity(transform_path) != transform_identity:
                        raise ValueError("Weighted transform changed during alignment")
                    model_identity = sparse_model_identity(staged_model)
                    staging.rename(destination)
                    decision.update(status="aligned", reason="reliable_checkpoints_and_validated_model",
                                    selected_model="sparse_pdr_aligned/0", aligned_model_identity=model_identity,
                                    max_aligned_residual=max(residuals))
                except (OSError, ValueError, struct.error) as exc:
                    decision.update(status="failed", reason=str(exc), selected_model=raw_name)
    snapshot.verify_unchanged()
    write_provenance(workspace / ALIGNMENT_RECORD, decision)
    print(f"[PDR] {decision['status']}: {decision['reason']}; selected {decision['selected_model']}", flush=True)
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--colmap-exe", type=Path)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--capture-mode", choices=("auto", "stationary", "off"), default="auto")
    parser.add_argument("--inspect-only", action="store_true", help="Read an existing model and write diagnostics without alignment or reconstruction.")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--grouping-review-csv", type=Path,
                        help="With --inspect-only, export all checkpoint photos and review evidence, including unregistered photos.")
    parser.add_argument("--exported-manifest", action="store_true", help="Map exported dataset image names to COLMAP workspace names.")
    args = parser.parse_args(argv)
    if args.inspect_only:
        if not all((args.model, args.manifest, args.report)) or args.workspace or args.colmap_exe or args.disabled:
            parser.error("--inspect-only requires --model, --manifest and --report; omit alignment options")
    elif not args.workspace or not args.colmap_exe or any((args.model, args.manifest, args.report, args.exported_manifest, args.grouping_review_csv)):
        parser.error("Alignment requires --workspace and --colmap-exe; inspection inputs require --inspect-only")
    try:
        if args.inspect_only:
            inspect_alignment(args.model, args.manifest, args.report, capture_mode=args.capture_mode,
                              exported_manifest=args.exported_manifest, grouping_review_csv=args.grouping_review_csv)
            return 0
        align_workspace(args.workspace, args.colmap_exe, disabled=args.disabled, capture_mode=args.capture_mode)
        return 0
    except (OSError, ValueError, KeyError, TypeError, WorkspaceBusyError) as exc:
        print(f"[PDR error] {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
