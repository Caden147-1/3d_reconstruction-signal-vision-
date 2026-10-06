"""Validate and resume completed PatchMatch views for identical reconstruction inputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

from colmap_workspace_check import (
    FileIdentityReader,
    PROVENANCE_NAME,
    contained_path,
    dense_input_model,
    file_identity,
    inspect_provenance,
    inspect_sparse_model,
    sparse_model_identity,
    write_provenance,
)
from workspace_lock import WorkspaceBusyError, locked_workspace
from runner_migration import compatible_runner_identity


CACHE_RECORD = "mvs_cache_identity.json"
CACHE_IDENTITY_VERSION = 2
CACHE_FOLDERS = ("images", "sparse", "stereo")


def dense_selection_config(workspace: Path) -> dict[str, Any] | None:
    """Opt in explicitly; absent/disabled selection retains the legacy identity."""
    config = read_record(workspace / "colmap_workspace_summary.json").get("dense_selection_config")
    if config is None:
        return None
    if not isinstance(config, dict) or type(config.get("enabled", False)) is not bool:
        raise ValueError("Dense selection configuration must use a boolean enabled flag")
    if not config.get("enabled", False):
        return None
    required = {"enabled", "max_images", "max_source_images"}
    if not required <= set(config) <= required | {"coverage_policy"}:
        raise ValueError("Dense selection configuration has missing or unknown fields")
    maximum, sources = config.get("max_images"), config.get("max_source_images")
    if type(maximum) is not int or (maximum != 0 and maximum < 3):
        raise ValueError("Dense selection max_images must be zero (unlimited) or at least three")
    if type(sources) is not int or not 2 <= sources <= 64:
        raise ValueError("Dense selection max_source_images must be between two and 64")
    if "coverage_policy" in config:
        policy = config["coverage_policy"]
        fields = {"mode", "initial_images", "min_single_view_coverage", "min_three_view_coverage", "max_video_gap_s"}
        if not isinstance(policy, dict) or set(policy) != fields or policy.get("mode") != "adaptive":
            raise ValueError("Dense coverage_policy requires the complete adaptive policy")
        initial = policy["initial_images"]
        if type(initial) is not int or initial < 3 or (maximum and initial > maximum):
            raise ValueError("Dense coverage initial_images must be at least three and within the hard image budget")
        for key in ("min_single_view_coverage", "min_three_view_coverage", "max_video_gap_s"):
            value = policy[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Dense coverage {key} must be finite")
            if value <= 0 or (key != "max_video_gap_s" and value > 1):
                raise ValueError(f"Dense coverage {key} is outside its valid range")
    return config


def dense_selection_identity(workspace: Path, *, identify: FileIdentityReader = file_identity):
    config = dense_selection_config(workspace)
    if config is None:
        return None
    identify(workspace / "colmap_workspace_summary.json")
    code = {name: identify(Path(__file__).with_name(name))
            for name in ("dense_keyframes.py", "mvs_observations.py", "dense_cache.py")}
    if any(value is None for value in code.values()):
        raise ValueError("Dense selection source code is missing")
    return {"schema_version": 1, "config": config, "source_code": code}


def dense_input_identity(
    workspace: Path,
    selected_model: Path | None,
    executable: Path | None,
    *,
    identify: FileIdentityReader = file_identity,
) -> dict[str, Any]:
    """Describe cache inputs; callers verify provenance before accepting reuse.

    Evaluation supplies its snapshot reader so it uses the same record format
    while still detecting files changed during a report.
    """
    model_identity = None
    if selected_model is not None:
        model_identity = {
            "path": selected_model.resolve().relative_to(workspace.resolve()).as_posix(),
            "model": sparse_model_identity(selected_model, identify=identify),
        }
    # Snapshot readers must observe the receipt used to select a historical
    # compatible runner identity, as well as the migration proof and scripts.
    identify(workspace / "dense" / CACHE_RECORD)
    identity = {
        "schema_version": CACHE_IDENTITY_VERSION,
        "dense_input": model_identity,
        "sparse_provenance": identify(workspace / PROVENANCE_NAME),
        "dense_runner": compatible_runner_identity(
            workspace, "run_colmap_dense.bat",
            read_record(workspace / "dense" / CACHE_RECORD).get("identity", {}).get("dense_runner"),
            identify=identify),
        "colmap_executable": identify(executable) if executable is not None else None,
    }
    selection = dense_selection_identity(workspace, identify=identify)
    stage_recipe = workspace / 'run_colmap_dense.bat'
    if stage_recipe.is_file() and 'rem portable-stage-v1' in stage_recipe.read_text(encoding='utf-8').splitlines():
        identity['execution_code'] = {name: identify(Path(__file__).resolve().parent / name)
                                      for name in ('portable_stage.py', 'process_control.py')}
    if selection is not None:
        identity["dense_selection"] = selection
    return identity


def cache_identity(workspace: Path) -> dict[str, Any]:
    """Return reusable cache inputs only after sparse provenance is verified."""
    proof = inspect_provenance(workspace)
    if proof["status"] != "verified":
        raise ValueError("Cannot use dense caches without verified sparse provenance: " + proof["status"])
    if not proof["dense_input"]:
        raise ValueError("No valid sparse component is available for dense reconstruction")
    model = contained_path(workspace, proof["dense_input"])
    summary = json.loads((workspace / "colmap_workspace_summary.json").read_text(encoding="utf-8"))
    executable = Path(summary["colmap_exe"])
    executable = executable if executable.is_absolute() else workspace / executable
    identity = dense_input_identity(workspace, model, executable)
    if any(identity[field] is None for field in ("sparse_provenance", "dense_runner", "colmap_executable")):
        raise ValueError("Dense runner, COLMAP executable or sparse provenance is missing")
    return identity


def read_record(path: Path) -> dict[str, Any]:
    """Read an optional JSON object, treating missing or invalid records as absent."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def reconstruction_binding(dense):
    """Bind a final export to the completed MVS run AND its undistorted model."""
    record = read_record(dense / CACHE_RECORD)
    if record.get("status") != "complete" or not record.get("identity"):
        return None
    model = dense / "sparse"
    for suffix in (".bin", ".txt"):
        names = [name + suffix for name in ("cameras", "images", "points3D")]
        if all((model / name).is_file() for name in names):
            names += [name + suffix for name in ("frames", "rigs") if (model / (name + suffix)).is_file()]
            binding = {"cache_identity": record["identity"],
                       "dense_model": {name: file_identity(model / name) for name in names}}
            if record["identity"].get("dense_selection") is not None:
                if record["identity"]["dense_selection"] != dense_selection_identity(dense.parent):
                    raise ValueError("Completed dense selection configuration or source code changed")
                expected = record.get("inputs", {}).get("files", {})
                files = {name: file_identity(dense / name) for name in (
                    "depth_source_summary.json", "stereo/patch-match.cfg", "stereo/fusion.cfg")}
                if any(value is None or expected.get(name) != value for name, value in files.items()):
                    raise ValueError("Completed dense selection report or stereo configuration changed")
                binding["dense_selection_files"] = files
            return binding
    raise ValueError("Completed MVS receipt has no complete undistorted model")


def checked_dense_path(workspace):
    root = workspace.resolve()
    dense = root / "dense"
    if dense.is_symlink() or dense.resolve() != dense:
        raise ValueError("Dense directory must not redirect outside its workspace")
    return dense


def check_stereo_map(path, channels, image_size, max_image_size=-1):
    """Validate COLMAP's width&height&channels& header and bounded float payload."""
    import numpy as np

    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        header = bytearray()
        while header.count(b"&") < 3 and len(header) < 80:
            value = handle.read(1)
            if not value:
                break
            header.extend(value)
        try:
            width, height, count = map(int, bytes(header).rstrip(b"&").split(b"&"))
        except ValueError as exc:
            raise ValueError(f"Invalid stereo header: {path}") from exc
        if header.count(b"&") != 3 or min(width, height) <= 0 or count != channels:
            raise ValueError(f"Invalid stereo dimensions/channels: {path}")
        native_w, native_h = image_size
        factor = min(1., max_image_size / max(image_size)) if max_image_size > 0 else 1.
        if (width > native_w or height > native_h or
            abs(width - native_w * factor) > 1 or abs(height - native_h * factor) > 1):
            raise ValueError(f"Stereo dimensions do not match the undistorted image: {path}")
        size = width * height * channels * 4
        if before.st_size != len(header) + size:
            raise ValueError(f"Missing/truncated stereo payload: {path}")
        digest.update(header)
        positive = 0
        remaining = size
        while remaining:
            raw = handle.read(min(4 * 1024**2, remaining))
            if not raw or len(raw) % 4:
                raise ValueError(f"Truncated stereo payload: {path}")
            remaining -= len(raw)
            values = np.frombuffer(raw, dtype="<f4")
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite stereo payload: {path}")
            if channels == 1:
                positive += int(np.count_nonzero(values > 0))
            digest.update(raw)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Stereo result changed during validation: {path}")
    return {"size": after.st_size, "sha256": digest.hexdigest()}, positive, (width, height)


def validate_dense_selection(workspace, plan, references, registered_names, config):
    """Require a bounded, authenticated subset and a closed explicit source plan."""
    import math

    selection = plan.get("dense_selection")
    if not isinstance(selection, dict) or selection.get("config") != config:
        raise ValueError("Dense selection report does not match the enabled configuration")

    def names_in(field):
        values = selection.get(field)
        if (not isinstance(values, list) or any(not isinstance(name, str) for name in values)
                or len(values) != len(set(values))):
            raise ValueError(f"Invalid dense selection name list: {field}")
        return set(values)

    selected = names_in("selected_names")
    candidates = names_in("candidate_selected_names")
    removed = names_in("removed_unsupported_names")
    if selected != references or not selected <= candidates <= registered_names or removed != candidates - selected:
        raise ValueError("Dense selection views differ from the registered candidates or PatchMatch references")
    if config["max_images"] and len(candidates) > config["max_images"]:
        raise ValueError("Dense selection exceeds its configured image budget")
    if config["max_images"] == 0 and candidates != registered_names:
        raise ValueError("Unlimited dense selection must consider all registered views before geometric exclusions")
    if (selection.get("registered_view_count") != len(registered_names)
            or selection.get("selected_view_count") != len(selected)):
        raise ValueError("Dense selection view counts do not match its model and references")
    required = [workspace / "colmap_workspace_summary.json", workspace / "metadata/images_manifest.csv",
                Path(__file__).with_name("dense_keyframes.py"), Path(__file__).with_name("mvs_observations.py")]
    identities = plan.get("inputs", {})
    if not isinstance(identities, dict):
        raise ValueError("Dense selection has no authenticated input identities")
    for path in required:
        identity = file_identity(path)
        if identity is None or identities.get(str(path.resolve())) != identity:
            raise ValueError(f"Dense selection does not authenticate its input: {path.name}")
    sources = plan.get("sources")
    if not isinstance(sources, dict) or set(sources) != references or plan.get("status") != "ready":
        raise ValueError("Dense selection needs a ready source plan for every selected reference")
    for ref, rows in sources.items():
        if not isinstance(rows, list) or not 2 <= len(rows) <= config["max_source_images"]:
            raise ValueError("Dense selection source count exceeds its configured bounds")
        source_names = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Dense selection source evidence must be an object")
            angle, tracks = row.get("median_angle_degrees"), row.get("shared_tracks")
            if (row.get("usable") is not True or not isinstance(angle, (int, float))
                    or not math.isfinite(angle) or not 2 <= angle <= 60
                    or not isinstance(tracks, (int, float)) or not math.isfinite(tracks) or tracks < 30):
                raise ValueError("Dense selection source lacks usable shared-track and parallax evidence")
            source_names.append(row.get("source"))
        if (any(not isinstance(name, str) for name in source_names) or len(set(source_names)) != len(source_names)
                or ref in source_names or not set(source_names) <= references):
            raise ValueError("Dense selection sources are not closed over its selected references")
    if config.get("coverage_policy") is not None:
        validate_selection_coverage(workspace, selection, references, config["coverage_policy"])


def _coverage_equal(recorded, expected):
    """Compare complete reports while allowing only floating-point roundoff."""
    if isinstance(expected, dict):
        return (isinstance(recorded, dict) and set(recorded) == set(expected)
                and all(_coverage_equal(recorded[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(recorded, list) and len(recorded) == len(expected)
                and all(_coverage_equal(a, b) for a, b in zip(recorded, expected)))
    if type(expected) in (float, int):
        return (type(recorded) in (float, int) and math.isfinite(recorded) and math.isfinite(expected)
                and math.isclose(recorded, expected, rel_tol=1e-9, abs_tol=1e-9))
    return type(recorded) is type(expected) and recorded == expected


def validate_selection_coverage(workspace, selection, references, policy):
    """Recompute adaptive evidence; unmet targets do not invalidate its images."""
    import csv
    import mvs_observations
    from dense_keyframes import adaptive_coverage_report, assess_coverage_targets
    from reconstruction_metrics import Snapshot

    snapshot = Snapshot()
    views, _points = mvs_observations.read_model(workspace / "dense/sparse", snapshot)
    manifest = workspace / "metadata/images_manifest.csv"
    snapshot.identity(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        metadata = {row["image_file"]: row for row in csv.DictReader(handle)}
    selected = [view for view in views if view["name"] in references]
    report = adaptive_coverage_report(views, selected, metadata, policy)
    if not _coverage_equal(selection.get("coverage_report"), report):
        raise ValueError("Dense adaptive coverage report does not match its model and selected views")
    targets = assess_coverage_targets(report, policy)
    recorded = selection.get("coverage_targets")
    if (not isinstance(recorded, dict)
            or any(key not in recorded or not _coverage_equal(recorded[key], value) for key, value in targets.items())):
        raise ValueError("Dense adaptive coverage targets do not match the measured coverage")
    snapshot.verify_unchanged()


def validate_cache_inputs(workspace):
    """Snapshot the undistorted inputs before any PatchMatch output is accepted."""
    import cv2

    dense = checked_dense_path(workspace)
    for name in CACHE_FOLDERS:
        folder = dense / name
        if not folder.is_dir() or folder.resolve() != folder:
            raise ValueError(f"Missing or redirected MVS folder: {folder}")
    model = dense / "sparse"
    names = inspect_sparse_model(model)["names"]
    selected = dense_input_model(workspace)
    if selected is None:
        raise ValueError("No selected sparse model is available for MVS input validation")
    expected_names = inspect_sparse_model(selected)["names"]
    if names != expected_names:
        raise ValueError(
            "Undistorted views differ from the selected sparse model: "
            f"{len(expected_names - names)} missing, {len(names - expected_names)} unexpected"
        )
    config = dense / "stereo/patch-match.cfg"
    lines = [line.strip() for line in config.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    references = set(lines[::2])
    if len(lines) % 2 or len(lines[::2]) != len(references) or not references <= names or len(references) < 2:
        raise ValueError("Invalid or duplicated PatchMatch reference images")
    plan_path = dense / 'depth_source_summary.json'
    plan = read_record(plan_path)
    selection_config = dense_selection_config(workspace)
    if selection_config is not None:
        validate_dense_selection(workspace, plan, references, names, selection_config)
    if references != names or plan:
        if plan.get('status') != 'ready' or set(plan.get('sources', {})) != references:
            raise ValueError('Partial PatchMatch coverage requires its depth source plan')
        for path, identity in plan.get('inputs', {}).items():
            if file_identity(Path(path)) != identity:
                raise ValueError('Depth source plan inputs changed')
        for stem in ('cameras', 'images', 'points3D'):
            path = next((model / (stem+suffix) for suffix in ('.bin', '.txt')
                         if (model / (stem+suffix)).is_file()), None)
            if path is None or plan.get('inputs', {}).get(str(path.resolve())) != file_identity(path):
                raise ValueError('Depth source plan does not authenticate this model')
        for ref, line in zip(lines[::2], lines[1::2]):
            expected = [row['source'] for row in plan['sources'][ref]]
            if (len(expected) < 2 or len(set(expected)) != len(expected) or ref in expected
                    or line.split(', ') != expected or not set(expected) <= references):
                raise ValueError('PatchMatch sources differ from the checked depth plan')
        fusion_names = (dense / 'stereo/fusion.cfg').read_text(encoding='utf-8').splitlines()
        if set(fusion_names) != references or len(fusion_names) != len(references):
            raise ValueError('Fusion sources differ from the checked depth plan')
    for line in lines[1::2]:
        sources = [item.strip() for item in line.split(",")]
        if sources[0] == "__auto__":
            if len(sources) != 2 or not sources[1].isdigit() or int(sources[1]) < 1:
                raise ValueError("Invalid automatic PatchMatch source selection")
        elif sources != ["__all__"] and any(name not in names for name in sources):
            raise ValueError("PatchMatch sources reference an unknown image")
    files = {"stereo/patch-match.cfg": file_identity(config)}
    if plan:
        files['depth_source_summary.json'] = file_identity(plan_path)
        files['stereo/fusion.cfg'] = file_identity(dense / 'stereo/fusion.cfg')
    for path in sorted(model.iterdir()):
        if path.suffix in (".bin", ".txt"):
            files[path.relative_to(dense).as_posix()] = file_identity(path)
    image_sizes = {}
    for name in sorted(references):
        path = contained_path(dense / "images", name)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing/unreadable undistorted image: {name}")
        image_sizes[name] = [image.shape[1], image.shape[0]]
        files["images/" + name] = file_identity(path)
    return {"schema_version": 1, "view_count": len(references), "registered_view_count": len(names), "files": files,
            "image_sizes": image_sizes}


def patch_match_settings(workspace):
    settings = read_record(workspace / "colmap_workspace_summary.json")
    return settings.get("dense_profile_config", {}).get("patch_match", {})


def stereo_paths(dense, name, kind):
    paths = []
    for folder in ("depth_maps", "normal_maps", "consistency_graphs"):
        path = dense / "stereo" / folder / (name + "." + kind + ".bin")
        if path.is_symlink() or path.resolve() != path or not path.resolve().is_relative_to(dense):
            raise ValueError(f"Refusing redirected stereo output: {path}")
        paths.append(path)
    return paths


def check_consistency_graph(path, image_size, view_count):
    """Check COLMAP's mixed ASCII header and variable-length int32 graph rows."""
    import struct

    before = file_identity(path)
    with path.open("rb") as handle:
        header = bytearray()
        while header.count(b"&") < 3 and len(header) < 80:
            value = handle.read(1)
            if not value:
                break
            header.extend(value)
        try:
            width, height, channels = map(int, bytes(header).rstrip(b"&").split(b"&"))
        except ValueError as exc:
            raise ValueError(f"Invalid consistency graph header: {path}") from exc
        if header.count(b"&") != 3 or (width, height) != image_size or channels != 1:
            raise ValueError(f"Invalid consistency graph dimensions: {path}")
        while True:
            raw = handle.read(12)
            if not raw:
                break
            if len(raw) != 12:
                raise ValueError(f"Truncated consistency graph: {path}")
            row, column, count = struct.unpack("<iii", raw)
            if not (0 <= row < height and 0 <= column < width and 0 <= count <= view_count):
                raise ValueError(f"Invalid consistency graph entry: {path}")
            raw = handle.read(count * 4)
            if len(raw) != count * 4 or any(index < 0 or index >= view_count
                                         for (index,) in struct.iter_unpack("<i", raw)):
                raise ValueError(f"Invalid consistency graph sources: {path}")
    if file_identity(path) != before:
        raise ValueError(f"Consistency graph changed during validation: {path}")
    return before


def validate_view_outputs(dense, name, kind, image_size, settings, view_count):
    files, sizes = {}, []
    paths = stereo_paths(dense, name, kind)
    positive_depth = 0
    for path, channels in zip(paths[:2], (1, 3)):
        identity, positive, size = check_stereo_map(path, channels, image_size,
                                                   int(settings.get("max_image_size", -1)))
        files[path.relative_to(dense).as_posix()] = identity
        sizes.append(size)
        if channels == 1:
            positive_depth = positive
    if sizes[0] != sizes[1]:
        raise ValueError(f"Depth and normal dimensions differ: {name}")
    if str(settings.get("write_consistency_graph", False)).lower() in ("1", "true"):
        files[paths[2].relative_to(dense).as_posix()] = check_consistency_graph(paths[2], sizes[0], view_count)
    return files, positive_depth


def validate_cache_outputs(workspace):
    dense = checked_dense_path(workspace)
    inputs = validate_cache_inputs(workspace)
    files = dict(inputs["files"])
    positive_views = 0
    settings = patch_match_settings(workspace)
    for name, size in inputs["image_sizes"].items():
        view, positive = validate_view_outputs(dense, name, "geometric", size,
                                              settings, inputs.get("registered_view_count", inputs["view_count"]))
        files.update(view)
        positive_views += bool(positive)
    if positive_views < 2:
        raise ValueError("MVS output has fewer than two views with valid depth")
    return {"schema_version": 1, "view_count": inputs["view_count"], "files": files}


def sanitize_partial_outputs(workspace, inputs, recorded_files=None):
    """Leave only complete, structurally valid view pairs for COLMAP's skip check.

    COLMAP skips an existing depth/normal pair without validating its payload:
    https://github.com/colmap/colmap/blob/main/src/colmap/mvs/patch_match.cc
    Geometric maps consume photometric maps. If any photometric dependency must
    be recomputed, conservatively invalidate the geometric pass as a whole.
    """
    dense = checked_dense_path(workspace)
    recorded_files = recorded_files or {}
    settings = patch_match_settings(workspace)
    retained = {"photometric": {}, "geometric": {}}
    removed, invalid = [], set()
    paths_by_view = {}
    for kind in retained:
        for name, size in inputs["image_sizes"].items():
            # Check all deletion targets before making any changes.
            paths_by_view[kind, name] = stereo_paths(dense, name, kind)
            try:
                files, _ = validate_view_outputs(dense, name, kind, size, settings, inputs.get("registered_view_count", inputs["view_count"]))
                if any(key in recorded_files and recorded_files[key] != value for key, value in files.items()):
                    raise ValueError("Previously validated view changed")
            except (OSError, ValueError):
                invalid.add((kind, name))
            else:
                retained[kind][name] = files
    if len(retained["photometric"]) != inputs["view_count"]:
        invalid.update(("geometric", name) for name in inputs["image_sizes"])
        retained["geometric"].clear()
    for key in sorted(invalid):
        for path in paths_by_view[key]:
            if path.exists():
                path.unlink()
                removed.append(path.relative_to(dense).as_posix())
    files = {path: value for views in retained.values() for view in views.values() for path, value in view.items()}
    return {"files": files, "retained_views": {kind: len(views) for kind, views in retained.items()},
            "removed": removed}


@locked_workspace("workspace")
def prepare_cache(workspace):
    identity = cache_identity(workspace)
    dense = checked_dense_path(workspace)
    record_path = dense / CACHE_RECORD
    record = read_record(record_path)
    same_identity = record.get("identity") == identity
    status = record.get("status")
    cached_outputs = record.get("outputs")
    if status in ("complete", "prepared") and cached_outputs and same_identity:
        try:
            if validate_cache_outputs(workspace) != cached_outputs:
                raise ValueError("MVS outputs have no matching integrity record")
        except (OSError, ValueError) as exc:
            print(f"[dense cache] Incomplete/changed cache will be recomputed: {exc}", flush=True)
        else:
            write_provenance(record_path, {**record, "status": "prepared",
                                          "execution_runner": file_identity(workspace / "run_colmap_dense.bat"),
                                          "inputs": validate_cache_inputs(workspace)})
            return {"status": "reused", "removed": []}

    if status in ("prepared", "complete", "invalid") and same_identity:
        try:
            inputs = validate_cache_inputs(workspace)
            expected = record.get("inputs")
            # Upgrade completed receipts written before input checkpoints existed.
            if not expected and record.get("outputs"):
                old_files = record["outputs"].get("files", {})
                if all(old_files.get(name) == value for name, value in inputs["files"].items()):
                    expected = inputs
            if expected != inputs:
                raise ValueError("No matching undistorted input checkpoint")
            old_files = {**record.get("outputs", {}).get("files", {}),
                         **record.get("partial", {}).get("files", {})}
            partial = sanitize_partial_outputs(workspace, inputs, old_files)
        except (OSError, ValueError) as exc:
            print(f"[dense cache] Partial cache cannot be resumed: {exc}", flush=True)
        else:
            write_provenance(record_path, {"status": "prepared", "identity": identity,
                                          "execution_runner": file_identity(workspace / "run_colmap_dense.bat"),
                                          "inputs": inputs, "partial": partial})
            return {"status": "resumed", "removed": partial["removed"],
                    "retained_views": partial["retained_views"]}

    # Check every resolved target before removing any cache. PLY outputs stay intact.
    targets = [dense / name for name in CACHE_FOLDERS]
    for target in targets:
        if target.is_symlink() or target.resolve() != target or not target.resolve().is_relative_to(dense):
            raise ValueError(f"Refusing redirected dense cache path: {target}")
    removed = []
    for target in targets:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed.append(target.name)
    dense.mkdir(parents=True, exist_ok=True)
    write_provenance(record_path, {"status": "prepared", "identity": identity,
                                  "execution_runner": file_identity(workspace / "run_colmap_dense.bat")})
    return {"status": "invalidated", "removed": removed}


@locked_workspace("workspace")
def undistorted_cache(workspace):
    """Return whether the prepared cache can bypass image_undistorter entirely."""
    identity = cache_identity(workspace)
    record = read_record(checked_dense_path(workspace) / CACHE_RECORD)
    if record.get("status") != "prepared" or record.get("identity") != identity:
        raise ValueError("Dense cache must be prepared before checking undistortion")
    ready = record.get("inputs") is not None
    if ready and record["inputs"] != validate_cache_inputs(workspace):
        raise ValueError("Undistorted inputs changed after preparation")
    return {"status": "undistorted" if ready else "needs_undistortion", "ready": ready}


@locked_workspace("workspace")
def matched_cache(workspace):
    """Avoid running either PatchMatch pass when the full output receipt matches."""
    identity = cache_identity(workspace)
    record = read_record(checked_dense_path(workspace) / CACHE_RECORD)
    if record.get("status") != "prepared" or record.get("identity") != identity:
        raise ValueError("Dense cache must be prepared before checking PatchMatch")
    ready = bool(record.get("outputs"))
    if ready and record["outputs"] != validate_cache_outputs(workspace):
        raise ValueError("PatchMatch outputs changed after preparation")
    return {"status": "matched" if ready else "needs_patchmatch", "ready": ready}


@locked_workspace("workspace")
def checkpoint_cache(workspace):
    """Bind partial results to successful image undistortion before PatchMatch."""
    identity = cache_identity(workspace)
    dense = checked_dense_path(workspace)
    record_path = dense / CACHE_RECORD
    record = read_record(record_path)
    if record.get("status") != "prepared" or record.get("identity") != identity:
        raise ValueError("Dense inputs/settings changed during image undistortion")
    inputs = validate_cache_inputs(workspace)
    if record.get("inputs") is not None and record["inputs"] != inputs:
        raise ValueError("Undistortion changed the resumed inputs; run dense preparation again")
    write_provenance(record_path, {**record, "inputs": inputs})
    return {"status": "checkpointed", "view_count": inputs["view_count"]}


@locked_workspace("workspace")
def complete_cache(workspace):
    identity = cache_identity(workspace)
    dense = checked_dense_path(workspace)
    record_path = dense / CACHE_RECORD
    record = read_record(record_path)
    if record.get("status") not in ("prepared", "complete") or record.get("identity") != identity:
        raise ValueError("Dense inputs/settings changed during PatchMatch; refusing cache certification")
    try:
        inputs = validate_cache_inputs(workspace)
        if record.get("inputs") is not None and record["inputs"] != inputs:
            raise ValueError("Undistorted inputs changed during PatchMatch")
        outputs = validate_cache_outputs(workspace)
    except (OSError, ValueError) as exc:
        write_provenance(record_path, {**record, "status": "invalid", "error": str(exc)})
        raise
    write_provenance(record_path, {"status": "complete", "identity": identity,
                                  "execution_runner": file_identity(workspace / "run_colmap_dense.bat"),
                                  "inputs": inputs, "outputs": outputs})
    return {"status": "complete"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "undistorted", "matched", "checkpoint", "complete"))
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = {"prepare": prepare_cache, "undistorted": undistorted_cache,
                  "matched": matched_cache, "checkpoint": checkpoint_cache,
                  "complete": complete_cache}[args.action](args.workspace.resolve())
    except (OSError, ValueError, KeyError, WorkspaceBusyError) as exc:
        print(f"[error] Dense cache: {exc}", flush=True)
        return 2
    print("[dense cache] " + json.dumps(result), flush=True)
    if args.action in ("undistorted", "matched") and not result["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
