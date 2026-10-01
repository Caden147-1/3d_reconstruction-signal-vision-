"""Create immutable, input-bound reconstruction evaluations and comparisons.

Internal reprojection and completion consistency are not surface accuracy.
Independent measurements are optional; unavailable accuracy remains null.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import uuid

import numpy as np

from colmap_workspace_check import (
    CAMERA_MODELS, contained_path, dense_input_model, file_identity,
    inspect_provenance, inspect_sparse_model, read_exact, read_image_list,
    read_manifest_sources, source_coverage, sparse_model_identity,
)
from dense_cache import dense_input_identity, reconstruction_binding
from ply_io import validate_ply
from workspace_lock import WorkspaceBusyError, workspace_lock


SCHEMA_VERSION = 1
CONFIG_KEYS = (
    "camera_model", "single_camera", "single_camera_per_folder", "matcher",
    "visual_only", "pdr_alignment_enabled", "feature_max_image_size",
    "max_num_features", "max_num_matches", "quality", "dense_max_image_size",
    "dense_profile", "dense_profile_config", "run_meshing",
    "run_point_cloud_filter", "scene_completion", "videoscene_max_clips",
    "videoscene_budget_hours", "load_profile", "colmap_num_threads",
    "wifi_recovery_enabled", "visual_pairing_mode", "image_similarity_config",
    "constrained_pairing_config",
    "pdr_alignment_policy", "pdr_alignment_runtime_status", "pdr_capture_mode",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


class Snapshot:
    """Hash each source once and refuse sources modified while it is evaluated."""

    def __init__(self):
        self.files = {}

    def identity(self, path):
        path = Path(path).resolve()
        if path not in self.files:
            identity = file_identity(path)
            stat = path.stat() if identity is not None else None
            self.files[path] = (identity, (stat.st_size, stat.st_mtime_ns) if stat else None)
        return self.files[path][0]

    def read_json(self, path, *, required=False):
        if self.identity(path) is None:
            if required:
                raise ValueError(f"Missing evaluation input: {path}")
            return {}
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path}")
        return value

    def verify_unchanged(self):
        for path, (_, before) in self.files.items():
            stat = path.stat() if path.exists() else None
            after = (stat.st_size, stat.st_mtime_ns) if stat else None
            if after != before:
                raise ValueError(f"Evaluation source changed during snapshot: {path}")


def number_summary(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite metric input")
    return {"count": len(values), **({key: None for key in ("mean", "median", "p95", "max")}
            if not len(values) else {"mean": float(values.mean()), "median": float(np.median(values)),
                                    "p95": float(np.quantile(values, .95)), "max": float(values.max())})}


def model_identities(model: Path, snapshot: Snapshot) -> dict:
    """Include all model files in the evaluation's change-detection snapshot."""
    return sparse_model_identity(model, identify=snapshot.identity)


def camera_center(q, translation):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid COLMAP quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return (-rotation.T @ np.asarray(translation)).tolist()


def image_observation_support(xy, point_ids, width, height, grid_size=8):
    """Describe measured feature coverage, independently of camera model type.

    Only observations linked to 3D points count. Off-image coordinates are
    reported separately; coverage is image support, not reconstructed area.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    point_ids = np.asarray(point_ids, dtype=np.int64)
    if point_ids.shape != (len(xy),) or width <= 0 or height <= 0 or grid_size <= 0:
        raise ValueError("Invalid image observation dimensions")
    linked = xy[point_ids >= 0]
    if not np.isfinite(linked).all():
        raise ValueError("Non-finite linked image observation")
    inside = (linked[:, 0] >= 0) & (linked[:, 0] < width) & (linked[:, 1] >= 0) & (linked[:, 1] < height)
    normalized = linked[inside] / [width, height]
    cells = np.floor(normalized * grid_size).astype(int)
    occupied = len(np.unique(cells, axis=0))
    eigenvalues = None
    if len(normalized) >= 2:
        centered = normalized - normalized.mean(0)
        eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(centered)), 0)
    return {"linked_observations": len(linked), "in_frame_observations": int(inside.sum()),
            "out_of_frame_observations": int((~inside).sum()),
            "grid_size": grid_size, "occupied_grid_cells": occupied,
            "occupied_grid_fraction": occupied / grid_size**2,
            "bounding_box_fraction": float(np.prod(np.ptp(normalized, axis=0))) if len(normalized) else 0.0,
            "spatial_eigenvalue_ratio": float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues is not None and eigenvalues[-1] > 0 else None}


def camera_support_summary(views):
    return {"status": "available", "views": views,
            "linked_observation_counts": number_summary([v["linked_observations"] for v in views.values()]),
            "occupied_grid_cell_counts": number_summary([v["occupied_grid_cells"] for v in views.values()]),
            "zero_observation_views": sorted(n for n, v in views.items() if not v["linked_observations"]),
            "views_with_at_most_two_occupied_cells": sorted(n for n, v in views.items() if v["occupied_grid_cells"] <= 2),
            "interpretation": "8x8 occupancy and spatial spread describe 2D feature support, not pose accuracy or 3D surface coverage. Concentrated support requests review; no camera is removed."}


def compare_camera_support(before, after):
    """Expose local count/coverage regressions hidden by aggregate point gains."""
    left, right = before.get("camera_support", {}), after.get("camera_support", {})
    if left.get("status") != "available" or right.get("status") != "available":
        return {"status": "unavailable", "reason": "per_camera_support_missing"}
    a, b = left["views"], right["views"]
    changes = [{"image": name, "before_observations": a[name]["linked_observations"],
                "after_observations": b[name]["linked_observations"],
                "observation_delta": b[name]["linked_observations"]-a[name]["linked_observations"],
                "before_occupied_cells": a[name]["occupied_grid_cells"],
                "after_occupied_cells": b[name]["occupied_grid_cells"],
                "occupied_cell_delta": b[name]["occupied_grid_cells"]-a[name]["occupied_grid_cells"]}
               for name in sorted(a.keys() & b.keys())]
    losses = sorted((r for r in changes if r["observation_delta"] < 0), key=lambda r: (r["observation_delta"], r["image"]))
    return {"status": "available", "common_images": len(changes),
            "added_images": sorted(b.keys()-a.keys()), "lost_images": sorted(a.keys()-b.keys()),
            "views_gaining_observations": sum(r["observation_delta"] > 0 for r in changes),
            "views_losing_observations": len(losses),
            "views_gaining_occupied_cells": sum(r["occupied_cell_delta"] > 0 for r in changes),
            "views_losing_occupied_cells": sum(r["occupied_cell_delta"] < 0 for r in changes),
            "largest_observation_losses": losses[:10], "per_image": changes,
            "interpretation": "Count and image-grid changes are support diagnostics, not accuracy. They do not identify shared feature correspondences or distinguish track replacement from new geometry."}


def sparse_metrics(model, snapshot):
    identities = model_identities(model, snapshot)
    checked = inspect_sparse_model(model)
    errors, tracks, cameras, centers = [], [], [], {}
    support, required_points, point_ids_seen = {}, set(), set()
    if (model / "cameras.bin").is_file():
        with (model / "cameras.bin").open("rb") as handle:
            for _ in range(struct.unpack("<Q", read_exact(handle, 8))[0]):
                cid, kind, width, height = struct.unpack("<iiQQ", read_exact(handle, 24))
                name, count = CAMERA_MODELS[kind]
                params = struct.unpack("<" + "d" * count, read_exact(handle, 8 * count))
                cameras.append(dict(id=cid, model=name, width=width, height=height, params=list(params)))
        camera_sizes = {c["id"]: (c["width"], c["height"]) for c in cameras}
        with (model / "images.bin").open("rb") as handle:
            for _ in range(struct.unpack("<Q", read_exact(handle, 8))[0]):
                record = struct.unpack("<idddddddi", read_exact(handle, 64))
                name = bytearray()
                while True:
                    byte = read_exact(handle, 1)
                    if byte == b"\0":
                        break
                    name.extend(byte)
                decoded = name.decode("utf-8").replace("\\", "/")
                centers[decoded] = {"camera_id": record[-1], "center_scene_units": camera_center(record[1:5], record[5:8])}
                count = struct.unpack("<Q", read_exact(handle, 8))[0]
                observations = np.frombuffer(read_exact(handle, count * 24), dtype=[("x", "<f8"), ("y", "<f8"), ("point", "<i8")])
                support[decoded] = image_observation_support(np.column_stack((observations["x"], observations["y"])), observations["point"], *camera_sizes[record[-1]])
                required_points.update(map(int, observations["point"][observations["point"] >= 0]))
        with (model / "points3D.bin").open("rb") as handle:
            for _ in range(struct.unpack("<Q", read_exact(handle, 8))[0]):
                record = struct.unpack("<QdddBBBd", read_exact(handle, 43))
                if record[0] in point_ids_seen:
                    raise ValueError("Duplicate sparse point ID")
                point_ids_seen.add(record[0])
                count = struct.unpack("<Q", read_exact(handle, 8))[0]
                errors.append(record[-1])
                tracks.append(count)
                handle.seek(count * 8, 1)
    else:
        with (model / "cameras.txt").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                p = line.split()
                cameras.append(dict(id=int(p[0]), model=p[1], width=int(p[2]), height=int(p[3]), params=list(map(float, p[4:]))))
        camera_sizes = {c["id"]: (c["width"], c["height"]) for c in cameras}
        with (model / "images.txt").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                p = line.split(maxsplit=9)
                centers[p[9].strip().replace("\\", "/")] = {
                    "camera_id": int(p[8]), "center_scene_units": camera_center(list(map(float, p[1:5])), list(map(float, p[5:8])))}
                observations = handle.readline().split()
                xy = np.array([(float(observations[i]), float(observations[i+1])) for i in range(0, len(observations), 3)]).reshape(-1, 2)
                point_ids = np.array([int(observations[i+2]) for i in range(0, len(observations), 3)], dtype=np.int64)
                support[p[9].strip().replace("\\", "/")] = image_observation_support(xy, point_ids, *camera_sizes[int(p[8])])
                required_points.update(map(int, point_ids[point_ids >= 0]))
        with (model / "points3D.txt").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                p = line.split()
                if int(p[0]) in point_ids_seen:
                    raise ValueError("Duplicate sparse point ID")
                point_ids_seen.add(int(p[0]))
                errors.append(float(p[7]))
                tracks.append((len(p)-8)//2)
    if any(value < 0 for value in errors):
        raise ValueError("Negative sparse reprojection error")
    if not required_points <= point_ids_seen:
        raise ValueError("Image observations reference missing sparse points")
    if sum(v["linked_observations"] for v in support.values()) != sum(tracks):
        raise ValueError("Linked image observation count differs from point track count")
    return {"path": str(model), "files": identities, "model_identity": digest(identities),
            "registered_images": len(checked["names"]), "point_count": checked["point_count"],
            "observation_count": int(sum(tracks)), "reprojection_error_px": number_summary(errors),
            "track_length": number_summary(tracks), "cameras": cameras, "camera_centers": centers,
            "camera_support": camera_support_summary(support),
            "interpretation": "Stored COLMAP reprojection errors and tracks describe internal fit, not ground-truth accuracy."}


def cloud_record(path, snapshot):
    identity = snapshot.identity(path)
    if identity is None:
        return None
    elements = validate_ply(path)
    return {"path": str(Path(path).resolve()), "identity": identity,
            "vertex_count": elements.get("vertex", 0), "face_count": elements.get("face", 0)}


def completion_metrics(dense, summary, snapshot, warnings):
    result = {"status": "unavailable", "target_projection_coverage": None,
              "interpretation": "Projected accepted point pixels in supported target masks; neither completed surface area nor real-world correctness."}
    generated = summary.get("generated_source")
    if not generated:
        return result
    run = Path(generated).resolve().parent
    status = snapshot.read_json(dense / "videoscene_summary.json")
    if not run.is_relative_to(dense.resolve()) or Path(status.get("run", "")).resolve() != run or status.get("status") not in {"completed", "hypotheses_only"}:
        warnings.append("Completion status does not bind to the current generated cloud; coverage unavailable.")
        return result
    clips = []
    for job in sorted(run.glob("clip_*")):
        report = snapshot.read_json(job / "geometry_validation.json")
        if not report:
            continue
        if report.get("eligible_for_merge") is False:
            continue
        target = report.get("target_pixels_by_view", [])
        covered = report.get("covered_pixels_by_view", [])
        if not target or len(target) != len(covered) or any(
            not isinstance(t, int) or not isinstance(c, int) or not 0 <= c <= t for t, c in zip(target, covered)
        ):
            warnings.append(f"Invalid completion coverage counts in {job.name}.")
            return result
        clips.append({"clip": job.name, "target_pixels": sum(target), "covered_pixels": sum(covered),
                      "accepted_points": report.get("accepted_points"), "policy": report.get("policy"),
                      "report_identity": snapshot.identity(job / "geometry_validation.json")})
    if not clips or sum(int(c.get("accepted_points") or 0) for c in clips) != summary.get("generated_vertex_count"):
        warnings.append("Completion clip counts do not match the current final summary; coverage unavailable.")
        return result
    target, covered = sum(c["target_pixels"] for c in clips), sum(c["covered_pixels"] for c in clips)
    result.update(status="available_from_bound_clip_reports", clips=clips,
                  target_projection_coverage=covered / target if target else None,
                  target_pixels=target, covered_pixels=covered)
    return result


def measured_distances(reference, model_identity, snapshot):
    unavailable = {"status": "unavailable", "mean_absolute_error_m": None, "rmse_m": None,
                   "reason": "No independent control-distance measurements supplied."}
    if reference is None:
        return unavailable
    data = snapshot.read_json(reference, required=True)
    if data.get("coordinate_frame_identity") != model_identity:
        raise ValueError("Control coordinates must name the exact evaluated model_identity")
    scale = data.get("scene_units_to_m")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Control distances require explicit positive scene_units_to_m; no scale is inferred")
    entries = data.get("distances")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Control-distance reference requires a nonempty distances list")
    rows = []
    for entry in entries:
        a, b = np.asarray(entry["point_a"], dtype=float), np.asarray(entry["point_b"], dtype=float)
        truth = float(entry["reference_distance_m"])
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all() or not math.isfinite(truth) or truth <= 0:
            raise ValueError("Invalid independent distance measurement")
        predicted = float(np.linalg.norm(a-b)) * scale
        rows.append({"name": str(entry.get("name", len(rows)+1)), "predicted_distance_m": predicted,
                     "reference_distance_m": truth, "signed_error_m": predicted-truth,
                     "relative_error": (predicted-truth)/truth})
    errors = np.asarray([row["signed_error_m"] for row in rows])
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Independent control-distance names must be unique")
    return {"status": "available", "reference_identity": snapshot.identity(reference),
            "scene_units_to_m": scale, "distances": rows,
            "control_set_identity": digest([{key: row[key] for key in ("name", "reference_distance_m")} for row in rows]),
            "mean_absolute_error_m": float(np.abs(errors).mean()), "rmse_m": float(np.sqrt((errors**2).mean())),
            "interpretation": "User-declared independent control distances, evaluated without fitting scale to those controls; not a surface-accuracy score."}


def runtime_metrics(receipt_path, snapshot, report):
    result = {"elapsed_seconds": None, "stage_durations_seconds": None,
              "status": "unavailable", "reason": "No matching measured runtime receipt; file timestamps and estimates are not runtime."}
    if receipt_path is None:
        return result
    receipt = snapshot.read_json(receipt_path, required=True)
    final_identity = (report["dense"].get("final") or {}).get("identity")
    if (receipt.get("status") != "completed" or
        receipt.get("input_image_set_identity") != report["inputs"]["image_set_identity"] or
        receipt.get("model_identity") != (report.get("sparse") or {}).get("model_identity") or
        receipt.get("final_identity") != final_identity):
        result["reason"] = "Runtime receipt is incomplete or does not bind to these inputs and outputs."
        return result
    elapsed = receipt.get("elapsed_seconds")
    stages = receipt.get("stage_durations_seconds", {})
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0 or
        not isinstance(stages, dict) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in stages.values())):
        raise ValueError("Invalid measured runtime receipt")
    return {"status": "available", "elapsed_seconds": elapsed, "stage_durations_seconds": stages,
            "receipt_identity": snapshot.identity(receipt_path), "scope": receipt.get("scope", "declared_execution")}


def build_report(workspace, label=None, reference_measurements=None, experiment_receipt=None):
    workspace = Path(workspace).resolve()
    snapshot, warnings = Snapshot(), []
    settings = snapshot.read_json(workspace / "colmap_workspace_summary.json", required=True)
    snapshot.identity(workspace / "image_list.txt")
    names = read_image_list(workspace / "image_list.txt")
    if not names or len(set(names)) != len(names):
        raise ValueError("Evaluation requires a nonempty unique current image list")
    images = {}
    for name in sorted(names):
        identity = snapshot.identity(contained_path(workspace / "images", name))
        if identity is None:
            raise ValueError(f"Missing input image: {name}")
        images[name] = identity
    manifest_identity = snapshot.identity(workspace / "metadata/images_manifest.csv")
    sources = read_manifest_sources(workspace / "metadata/images_manifest.csv")
    config = {key: settings.get(key) for key in CONFIG_KEYS}
    alignment = snapshot.read_json(workspace / "pdr_alignment_decision.json")
    config["pdr_alignment_runtime_status"] = alignment.get("status", "legacy_unrecorded")
    config["mask_enabled"] = bool(settings.get("colmap_mask", {}).get("enabled"))
    mask = settings.get("colmap_mask", {})
    config["mask_identities"] = {name: snapshot.identity(contained_path(Path(mask["path"]), name + ".png"))
                                 for name in sorted(names)} if mask.get("enabled") and mask.get("path") else {}
    executable = Path(settings["colmap_exe"]) if settings.get("colmap_exe") else None
    if executable is not None and not executable.is_absolute():
        executable = workspace / executable
    config["colmap_executable_identity"] = snapshot.identity(executable) if executable else None
    provenance = inspect_provenance(workspace)
    snapshot.identity(workspace / "sparse_run_provenance.json")
    for name in ("run_colmap_sparse.bat", "run_colmap_dense.bat", "match_pairs.txt", "pdr_alignment_refs.txt"):
        snapshot.identity(workspace / name)
    if provenance["status"] != "verified":
        warnings.append("Current sparse model is not authenticated to current inputs: " + provenance["status"])
    components = []
    for folder in sorted((workspace / "sparse").glob("*")):
        if not folder.is_dir():
            continue
        try:
            identities = model_identities(folder, snapshot)
            check = inspect_sparse_model(folder)
        except (OSError, ValueError, struct.error) as exc:
            warnings.append(f"Invalid sparse component {folder.relative_to(workspace).as_posix()}: {exc}")
            continue
        components.append({"model": folder.name, "registered_images": len(check["names"]),
                           "point_count": check["point_count"], "model_identity": digest(identities)})
    selected = dense_input_model(workspace)
    sparse = sparse_metrics(selected, snapshot) if selected else None
    registered = set(sparse["camera_centers"]) if sparse else set()
    if sparse:
        sparse.update(registration_ratio=len(registered & set(names))/len(names),
                      missing_images=sorted(set(names)-registered), unexpected_images=sorted(registered-set(names)),
                      source_registration=source_coverage(names, sources, registered), components=components,
                      component_count=len(components))
    dense = workspace / "dense"
    final = cloud_record(dense / "final.ply", snapshot)
    summary = snapshot.read_json(dense / "final_output_summary.json")
    observed, generated, binding_status = None, None, "unavailable"
    if final and summary:
        if summary.get("final_identity") != final["identity"] or summary.get("vertex_count") != final["vertex_count"]:
            warnings.append("Final PLY differs from final_output_summary; derived dense metrics withheld.")
            binding_status = "mismatch"
        else:
            binding_status = "identity_verified"
            if summary.get("source_kind") == "videoscene_completed_point_cloud":
                observed_path, generated_path = Path(summary["observed_source"]).resolve(), Path(summary["generated_source"]).resolve()
                if not observed_path.is_relative_to(dense.resolve()) or not generated_path.is_relative_to(dense.resolve()):
                    raise ValueError("Dense provenance points outside the workspace dense directory")
                observed, generated = cloud_record(observed_path, snapshot), cloud_record(generated_path, snapshot)
                if (not observed or not generated or observed["identity"] != summary.get("observed_identity") or
                    (summary.get("generated_identity") is not None and generated["identity"] != summary["generated_identity"]) or
                    observed["vertex_count"] != summary.get("observed_vertex_count") or
                    generated["vertex_count"] != summary.get("generated_vertex_count") or
                    observed["vertex_count"] + generated["vertex_count"] != final["vertex_count"]):
                    raise ValueError("Observed/generated clouds do not match the final export provenance")
            elif summary.get("source_kind") == "filtered_point_cloud":
                observed = final
            if summary.get("reconstruction_binding"):
                snapshot.identity(dense / "mvs_cache_identity.json")
                model_identities(dense / "sparse", snapshot)
                current_cache_inputs = dense_input_identity(
                    workspace, selected, executable, identify=snapshot.identity,
                )
                if (provenance["status"] != "verified" or selected is None or
                    reconstruction_binding(dense) != summary["reconstruction_binding"] or
                    summary["reconstruction_binding"].get("cache_identity") != current_cache_inputs):
                    warnings.append("Final PLY reconstruction binding differs from the current dense model.")
                    binding_status = "mismatch"
                else:
                    binding_status = "verified"
            else:
                warnings.append("Final PLY identity is known but has no authenticated reconstruction binding.")
    requested_config = dict(config)
    # Completion can be explicitly invoked after an off-configured dense run.
    # Derive its actual treatment from the bound final artifact, preserving the
    # requested configuration separately rather than editing cache inputs.
    config["scene_completion"] = "videoscene" if generated and binding_status == "verified" else "off"
    complete = completion_metrics(dense, summary, snapshot, warnings) if generated and binding_status != "mismatch" else {
        "status": "unavailable", "target_projection_coverage": None}
    attempt = snapshot.read_json(dense / "videoscene_summary.json")
    attempt_bound = bool(final and binding_status == "verified" and
                         attempt.get("reconstruction_binding") == summary.get("reconstruction_binding") and
                         attempt.get("input_final_identity") == (summary.get("observed_identity") or final["identity"]))
    hypotheses = None
    if attempt_bound and attempt.get('status') == 'hypotheses_only':
        path = Path(attempt.get('generated_source', '')).resolve()
        if not path.is_relative_to(dense.resolve()):
            raise ValueError('Hypothesis source points outside the dense workspace')
        hypotheses = cloud_record(path, snapshot)
        if (not hypotheses or hypotheses['identity'] != attempt.get('generated_identity') or
                hypotheses['vertex_count'] != attempt.get('generated_points')):
            raise ValueError('Separate generated hypotheses do not match their receipt')
        complete = completion_metrics(dense, {**summary, 'generated_source':str(path),
                                              'generated_vertex_count':hypotheses['vertex_count']}, snapshot, warnings)
        config['scene_completion'] = 'videoscene_hypotheses'
    complete.update(attempt_status=attempt.get("status") if attempt_bound else None,
                    attempt_configuration=attempt.get("effective_config") if attempt_bound else None,
                    attempt_receipt_identity=snapshot.identity(dense / "videoscene_summary.json") if attempt_bound else None)
    report = {"schema_version": SCHEMA_VERSION, "report_type": "reconstruction_evaluation",
              "evaluation_id": uuid.uuid4().hex, "created_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_code_identity": snapshot.identity(Path(__file__)),
              "workspace": str(workspace), "label": label or workspace.name,
              "inputs": {"image_count": len(names), "images": images, "image_set_identity": digest(images),
                         "manifest_identity": manifest_identity},
              "configuration": config, "configuration_identity": digest(config), "requested_configuration": requested_config,
              "pdr_alignment_decision": alignment,
              "sparse_provenance": provenance, "sparse": sparse,
              "dense": {"final": final, "observed": observed, "generated": generated, "separate_hypotheses": hypotheses,
                        "binding_status": binding_status, "completion": complete},
              "accuracy": {"surface_area_coverage": None, "false_fill_rate": None, "true_scale_error": None,
                           "unavailable_reason": "No independent reference surface/empty-space labels or scene-scale ground truth supplied.",
                           "control_distances": measured_distances(reference_measurements, sparse["model_identity"] if sparse else None, snapshot)},
              "warnings": warnings, "comparable": provenance["status"] == "verified" and (final is None or binding_status == "verified") and sparse is not None}
    report["runtime"] = runtime_metrics(experiment_receipt, snapshot, report)
    report["source_files"] = {str(path): identity for path, (identity, _) in snapshot.files.items()}
    snapshot.verify_unchanged()
    report["report_identity"] = digest(report)
    return report


def write_exclusive(output_dir, filename, payload):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / filename).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def evaluate(workspace, output_dir, label=None, reference_measurements=None, experiment_receipt=None):
    if Path(output_dir).exists():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    with workspace_lock(Path(workspace).resolve()):
        report = build_report(workspace, label, reference_measurements, experiment_receipt)
        write_exclusive(output_dir, "report.json", report)
    return report


def load_evaluation(path):
    path = Path(path)
    path = path / "report.json" if path.is_dir() else path
    report = json.loads(path.read_text(encoding="utf-8"))
    identity = report.pop("report_identity", None)
    if report.get("schema_version") != SCHEMA_VERSION or report.get("report_type") != "reconstruction_evaluation" or digest(report) != identity:
        raise ValueError(f"Invalid or changed immutable evaluation report: {path}")
    report["report_identity"] = identity
    return report


def compare(reports, output_dir, allowed_config_keys=()):
    if Path(output_dir).exists():
        raise FileExistsError(f"Comparison output already exists: {output_dir}")
    if len(reports) < 2:
        raise ValueError("Comparison requires at least two evaluation reports")
    allowed = set(allowed_config_keys)
    if not allowed <= set(CONFIG_KEYS) | {"mask_enabled"}:
        raise ValueError("Unknown intended configuration difference")
    loaded = [load_evaluation(path) for path in reports]
    base = loaded[0]
    failures = []
    if len({report["evaluation_id"] for report in loaded}) != len(loaded):
        failures.append("The same evaluation snapshot was supplied more than once.")
    for report in loaded:
        if not report.get("comparable"):
            failures.append(f"{report['label']}: input/output provenance is not verified")
        if report["inputs"]["image_set_identity"] != base["inputs"]["image_set_identity"]:
            failures.append(f"{report['label']}: different source image bytes or names")
        if report.get("evaluation_code_identity") != base.get("evaluation_code_identity"):
            failures.append(f"{report['label']}: different evaluation implementation")
        differences = {key for key in set(base["configuration"]) | set(report["configuration"])
                       if base["configuration"].get(key) != report["configuration"].get(key)}
        if differences-allowed:
            failures.append(f"{report['label']}: undeclared configuration differences: {sorted(differences-allowed)}")
    rows = []
    for report in loaded:
        sparse, dense = report.get("sparse") or {}, report["dense"]
        rows.append({"label": report["label"], "evaluation_id": report["evaluation_id"], "report_identity": report["report_identity"],
                     "registration_ratio": sparse.get("registration_ratio"), "registered_images": sparse.get("registered_images"),
                     "component_count": sparse.get("component_count"), "reprojection_error_px": sparse.get("reprojection_error_px"),
                     "track_length": sparse.get("track_length"),
                     "camera_support_vs_first": compare_camera_support(base.get("sparse") or {}, sparse),
                     "observed_points": (dense.get("observed") or {}).get("vertex_count"),
                     "generated_points": (dense.get("generated") or {}).get("vertex_count", 0 if dense.get("binding_status") == "verified" and dense.get("observed") else None),
                     "completion_attempt_status": dense.get("completion", {}).get("attempt_status"),
                     "target_projection_coverage": dense.get("completion", {}).get("target_projection_coverage"),
                     "runtime_seconds": report["runtime"].get("elapsed_seconds"), "accuracy": report["accuracy"]})
    result = {"schema_version": SCHEMA_VERSION, "report_type": "reconstruction_comparison",
              "created_utc": datetime.now(timezone.utc).isoformat(), "comparable": not failures,
              "incomparability_reasons": failures, "declared_variant_keys": sorted(allowed), "results": rows,
              "interpretation": "No composite accuracy ranking. Point count, internal fit, and target-mask projection coverage are distinct diagnostics; null means unavailable."}
    controls = [report["accuracy"]["control_distances"] for report in loaded]
    result["control_distance_errors_comparable"] = (all(c["status"] == "available" for c in controls) and
                                                   len({c.get("control_set_identity") for c in controls}) == 1)
    write_exclusive(output_dir, "comparison.json", result)
    with (Path(output_dir) / "comparison.md").open("x", encoding="utf-8") as handle:
        handle.write("Comparable: " + str(result["comparable"]) + "\n\n")
        for reason in failures:
            handle.write("- " + reason + "\n")
        handle.write("\n| Variant | Registered images | Components | Reprojection mean px | Observed points | Generated points | Runtime s |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            values = [row["label"], row["registered_images"], row["component_count"], (row["reprojection_error_px"] or {}).get("mean"), row["observed_points"], row["generated_points"], row["runtime_seconds"]]
            handle.write("| " + " | ".join("unavailable" if value is None else str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |\n")
        handle.write("\n" + result["interpretation"] + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--workspace", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--label")
    evaluation.add_argument("--reference-measurements", type=Path)
    evaluation.add_argument("--experiment-receipt", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--reports", type=Path, nargs="+", required=True)
    comparison.add_argument("--output-dir", type=Path, required=True)
    comparison.add_argument("--allow-config-key", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            result = evaluate(args.workspace, args.output_dir, args.label, args.reference_measurements, args.experiment_receipt)
        else:
            result = compare(args.reports, args.output_dir, args.allow_config_key)
        print(json.dumps({"output": str(args.output_dir), "comparable": result["comparable"]}))
        return 0 if result["comparable"] else 2
    except (OSError, ValueError, KeyError, TypeError, struct.error, WorkspaceBusyError) as exc:
        print(f"[evaluation error] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
