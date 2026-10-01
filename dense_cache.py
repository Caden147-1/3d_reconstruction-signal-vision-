"""Validate and resume completed PatchMatch views for identical reconstruction inputs."""

import argparse
import hashlib
import json
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


CACHE_RECORD = "mvs_cache_identity.json"
CACHE_IDENTITY_VERSION = 2
CACHE_FOLDERS = ("images", "sparse", "stereo")


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
    return {
        "schema_version": CACHE_IDENTITY_VERSION,
        "dense_input": model_identity,
        "sparse_provenance": identify(workspace / PROVENANCE_NAME),
        "dense_runner": identify(workspace / "run_colmap_dense.bat"),
        "colmap_executable": identify(executable) if executable is not None else None,
    }


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
            return {"cache_identity": record["identity"],
                    "dense_model": {name: file_identity(model / name) for name in names}}
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
    write_provenance(record_path, {"status": "prepared", "identity": identity})
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
