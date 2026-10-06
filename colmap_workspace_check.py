#!/usr/bin/env python3
"""Validate that a COLMAP workspace still matches its current input manifest.

The check is intentionally read-only. It prevents a dense run from silently
using a database or sparse model created before new video inputs were
prepared.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set


PROVENANCE_NAME = "sparse_run_provenance.json"
PENDING_NAME = "sparse_run_inputs.json"
ALIGNMENT_RECORD = "pdr_alignment_decision.json"
SPARSE_RUNNERS = {"run_colmap_sparse.bat", "run_colmap_automatic.bat"}
REQUIRED_MODEL_FILES = ("cameras", "images", "points3D")
OPTIONAL_MODEL_FILES = ("frames", "rigs")
VIDEO_PIPELINE_CODE = ("video_sparse_mapper.py", "video_intrinsics.py")
RESOURCE_PIPELINE_CODE = (*VIDEO_PIPELINE_CODE, "compute_resources.py", "portable_stage.py", "process_control.py")
OPTIMIZED_PIPELINE_CODE = (*RESOURCE_PIPELINE_CODE, "sparse_optimization.py")
RECOVERY_PIPELINE_CODE = (*OPTIMIZED_PIPELINE_CODE, "sparse_recovery.py", "sparse_quality.py")
FileIdentityReader = Callable[[Path], Dict[str, Any] | None]


def file_identity(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during provenance inspection: {path}")
    return {"size": after.st_size, "sha256": digest.hexdigest()}


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Input path escapes its root: {relative}")
    return path


def sparse_input_identity(workspace: Path, runner: str) -> Dict[str, Any]:
    if runner not in SPARSE_RUNNERS:
        raise ValueError("Unrecognized sparse runner")
    names = read_image_list(workspace / "image_list.txt")
    if not names or len(names) != len(set(names)) or not (workspace / runner).is_file():
        raise ValueError("Input list or sparse runner is missing or invalid")
    files: Dict[str, Any] = {}
    for name in sorted(names):
        identity = file_identity(contained_path(workspace / "images", name))
        if identity is None:
            raise ValueError(f"Missing input image: {name}")
        files[f"images/{name}"] = identity
    for name in ("image_list.txt", "match_pairs.txt", "pdr_alignment_refs.txt", "metadata/images_manifest.csv", runner):
        files[name] = file_identity(workspace / name)
    runner_text = (workspace / runner).read_text(encoding="utf-8")
    if any(name in runner_text for name in ("sparse_optimization.py", "compute_resources.py", "video_sparse_mapper.py")):
        dependencies = (RECOVERY_PIPELINE_CODE if "sparse_recovery.py" in runner_text else
                        OPTIMIZED_PIPELINE_CODE if "sparse_optimization.py" in runner_text else
                        RESOURCE_PIPELINE_CODE if "compute_resources.py" in runner_text else VIDEO_PIPELINE_CODE)
        for name in dependencies:
            files["pipeline_code/" + name] = file_identity(Path(__file__).resolve().parent / name)
    elif (workspace / "sparse_import/source_sparse_run_provenance.json").is_file():
        # Imported models archive their implementation bytes; their stub never
        # executes the current implementation or substitutes its newer hashes.
        for name in RECOVERY_PIPELINE_CODE:
            archived = workspace / "pipeline_code" / name
            if archived.exists():
                files["pipeline_code/" + name] = file_identity(archived)
    if 'rem portable-stage-v1' in runner_text.splitlines():
        for name in ('portable_stage.py', 'process_control.py'):
            files['pipeline_code/' + name] = file_identity(Path(__file__).resolve().parent / name)
    summary_path = workspace / "colmap_workspace_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    mask = summary.get("colmap_mask", {})
    mask_root = Path(mask["path"]) if mask.get("enabled") and mask.get("path") else None
    if mask_root is not None and not mask_root.is_dir():
        raise ValueError("Configured mask directory is missing")
    return {
        "runner": runner, "files": files,
        "mask_root": str(mask_root.resolve()) if mask_root is not None else None,
        "masks": {
            name: file_identity(contained_path(mask_root, name + ".png"))
            for name in sorted(names)
        } if mask_root is not None else {},
    }


def sparse_output_identity(workspace: Path) -> Dict[str, Any]:
    files = {name: file_identity(workspace / name) for name in ("database.db", "database.db-wal")}
    # SQLite readers may leave a zero-byte WAL without changing the database.
    # Only an empty WAL is equivalent to no WAL; any content remains hashed.
    if files["database.db-wal"] is not None and files["database.db-wal"]["size"] == 0:
        files["database.db-wal"] = None
    for name in (ALIGNMENT_RECORD, "pdr_alignment_reviewed_refs.txt", "video_intrinsics_summary.json", "sparse_optimization.json", "sparse_backbone_images.txt", "sparse_recovery_summary.json"):
        if (workspace / name).exists():
            files[name] = file_identity(workspace / name)
    for folder in ("sparse", "sparse_pdr_aligned", "sparse_pose_prior"):
        base = workspace / folder
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix in {".bin", ".txt"}:
                    files[path.relative_to(workspace).as_posix()] = file_identity(path)
    selected = dense_input_model(workspace)
    return {"files": files, "dense_input": selected.relative_to(workspace).as_posix() if selected else None}


def inspect_provenance(workspace: Path) -> Dict[str, Any]:
    receipt = workspace / PROVENANCE_NAME
    if (workspace / PENDING_NAME).exists():
        return {"status": "incomplete_run", "error": "Sparse input snapshot exists without successful completion."}
    if not receipt.is_file():
        return {"status": "unverified", "error": "No run-time input provenance was recorded; filenames alone cannot verify old models."}
    try:
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
        if recorded["schema_version"] != 1 or recorded["status"] != "complete":
            raise ValueError("Unsupported or incomplete provenance record")
        if recorded.get("provenance_kind"):
            from sparse_reuse import verify_import_record
            verify_import_record(workspace, recorded)
        if (workspace / "sparse_import_pending.json").exists():
            return {"status": "incomplete_import", "error": "Sparse artifact import has not completed verification."}
        current_inputs = sparse_input_identity(workspace, recorded["inputs"]["runner"])
        # A performance-only runner migration preserves historical model proof.
        # New runs still snapshot the real current runner in sparse_input_identity.
        from runner_migration import compatible_runner_identity
        runner = recorded["inputs"]["runner"]
        current_inputs["files"][runner] = compatible_runner_identity(
            workspace, runner, recorded["inputs"]["files"].get(runner))
        if current_inputs != recorded["inputs"]:
            return {"status": "input_mismatch", "error": "Image bytes, masks, matching pairs, pose references, manifest or sparse runner changed."}
        current_outputs = sparse_output_identity(workspace)
        # The receipt authenticates all model bytes. Its selected path records
        # the old policy; changing that policy does not invalidate unchanged SfM.
        recorded_files = dict(recorded["outputs"]["files"])
        wal = recorded_files.get("database.db-wal")
        if wal is not None and wal.get("size") == 0:
            recorded_files["database.db-wal"] = None
        if current_outputs["files"] != recorded_files:
            return {"status": "output_mismatch", "error": "The sparse model or database no longer matches the recorded run."}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {"status": "invalid_record", "error": str(exc)}
    return {"status": "verified", "error": "", "record": str(receipt),
            "dense_input": current_outputs["dense_input"],
            "selection_changed": current_outputs["dense_input"] != recorded["outputs"].get("dense_input")}


def write_provenance(path: Path, payload: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="provenance-", suffix=".tmp", delete=False) as handle:
        staged = Path(handle.name)
    try:
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def begin_sparse_run(workspace: Path, runner: str = "run_colmap_sparse.bat") -> None:
    report = inspect_workspace(workspace)
    failures = validation_failures(report, require_clean_database=True)
    if any((workspace / name).exists() for name in ("sparse", "sparse_pdr_aligned", "sparse_pose_prior")):
        failures.append("Existing sparse directories must be reset before recording a new run")
    if any((workspace / ("database.db" + suffix)).exists() for suffix in ("-wal", "-shm", "-journal")):
        failures.append("Old SQLite sidecar files must be reset before recording a new run")
    if failures:
        raise ValueError("; ".join(failures))
    inputs = sparse_input_identity(workspace, runner)
    from reconstruction_run_diagnostics import begin_run_log
    write_provenance(workspace / PENDING_NAME, {
        "schema_version": 1, "inputs": inputs, "run_log": begin_run_log(workspace),
    })


def complete_sparse_run(workspace: Path) -> None:
    pending = json.loads((workspace / PENDING_NAME).read_text(encoding="utf-8"))
    if pending["schema_version"] != 1 or sparse_input_identity(workspace, pending["inputs"]["runner"]) != pending["inputs"]:
        raise ValueError("Sparse inputs changed during reconstruction; refusing to certify the result")
    failures = validation_failures(inspect_workspace(workspace), require_database_coverage=True,
                                   require_sparse_model=True, require_provenance=False, require_dense_readiness=False)
    if failures:
        raise ValueError("; ".join(failures))
    from reconstruction_run_diagnostics import complete_run_log
    write_provenance(workspace / PROVENANCE_NAME, {
        "schema_version": 1, "status": "complete", "inputs": pending["inputs"],
        "outputs": sparse_output_identity(workspace),
        "run_log": complete_run_log(workspace, pending.get("run_log", {})),
    })
    (workspace / PENDING_NAME).unlink()


def normalize_name(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./")


def read_image_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [normalize_name(line) for line in handle if line.strip()]


def read_manifest_sources(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        return {
            normalize_name(row.get("image_file", "")): row.get("source_type", "unknown") or "unknown"
            for row in csv.DictReader(handle)
            if row.get("image_file")
        }


def read_database_names(path: Path) -> tuple[Set[str], str]:
    if not path.exists():
        return set(), ""
    try:
        connection = open_readonly_database(path)
        try:
            rows = connection.execute("SELECT name FROM images").fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return set(), str(exc)
    return {normalize_name(str(row[0])) for row in rows}, ""


def open_readonly_database(path: Path):
    """Avoid touching WAL sidecars when inspecting a checkpointed database."""
    path = Path(path).resolve()
    wal = Path(str(path) + "-wal")
    # Nonempty WAL data must be read through SQLite's WAL reader. With no
    # pending pages, immutable mode avoids creating/touching empty sidecars.
    options = "?mode=ro" if wal.exists() and wal.stat().st_size else "?mode=ro&immutable=1"
    return sqlite3.connect(path.as_uri() + options, uri=True)


def read_exact(handle: Any, size: int) -> bytes:
    if size < 0:
        raise ValueError("Invalid COLMAP record length")
    raw = handle.read(size)
    if len(raw) != size:
        raise ValueError("Truncated COLMAP model")
    return raw


def read_binary_image_names(path: Path, camera_ids: Set[int] | None = None) -> Set[str]:
    names: Set[str] = set()
    ids: Set[int] = set()
    with path.open("rb") as handle:
        file_size = path.stat().st_size
        count = struct.unpack("<Q", read_exact(handle, 8))[0]
        for _ in range(count):
            record = struct.unpack("<idddddddi", read_exact(handle, 64))
            image_id, camera_id = record[0], record[-1]
            if image_id in ids or image_id <= 0 or camera_id <= 0 or not all(math.isfinite(v) for v in record[1:8]) or sum(v*v for v in record[1:5]) < 1e-12:
                raise ValueError("Invalid COLMAP image pose or ID")
            ids.add(image_id)
            if camera_ids is not None:
                camera_ids.add(camera_id)
            name = bytearray()
            while True:
                byte = read_exact(handle, 1)
                if byte == b"\0":
                    break
                name.extend(byte)
            decoded = normalize_name(name.decode("utf-8"))
            if not decoded or decoded in names:
                raise ValueError("Empty or duplicate COLMAP image name")
            names.add(decoded)
            points = struct.unpack("<Q", read_exact(handle, 8))[0]
            if points * 24 > file_size - handle.tell():
                raise ValueError("Truncated COLMAP image observations")
            handle.seek(points * 24, 1)
        if handle.read(1):
            raise ValueError("Extra data in COLMAP images.bin")
    return names


def read_text_image_names(path: Path, camera_ids: Set[int] | None = None) -> Set[str]:
    names: Set[str] = set()
    ids: Set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split(maxsplit=9)
            if len(parts) != 10:
                raise ValueError("Invalid COLMAP image header")
            image_id, camera_id = int(parts[0]), int(parts[8])
            pose = [float(value) for value in parts[1:8]]
            name = normalize_name(parts[9])
            if image_id in ids or image_id <= 0 or camera_id <= 0 or not name or name in names or not all(math.isfinite(v) for v in pose) or sum(v*v for v in pose[:4]) < 1e-12:
                raise ValueError("Invalid COLMAP image pose or name")
            ids.add(image_id)
            names.add(name)
            if camera_ids is not None:
                camera_ids.add(camera_id)
            observations = handle.readline()
            if not observations:
                raise ValueError("Missing COLMAP image observation row")
            fields = observations.split()
            if len(fields) % 3:
                raise ValueError("Invalid COLMAP image observations")
            for index in range(0, len(fields), 3):
                if not math.isfinite(float(fields[index])) or not math.isfinite(float(fields[index + 1])):
                    raise ValueError("Non-finite COLMAP image observation")
                int(fields[index + 2])
    return names


def primary_sparse_model(workspace: Path) -> Path | None:
    """Choose a complete component by image count, point count, then folder name."""
    sparse = workspace / "sparse"
    if not sparse.is_dir():
        return None
    selected = None
    best_rank = (0, 0)
    for model in sorted(sparse.iterdir(), key=lambda path: path.name):
        if not model.is_dir():
            continue
        try:
            checked = inspect_sparse_model(model)
        except (OSError, ValueError, struct.error):
            continue
        rank = (len(checked["names"]), checked["point_count"])
        # Iteration order preserves the first folder when both counts tie.
        if rank > best_rank:
            selected, best_rank = model, rank
    return selected


def dense_input_model(workspace: Path) -> Path | None:
    aligned = workspace / "sparse_pdr_aligned" / "0"
    if verified_aligned_model(workspace):
        return aligned
    return primary_sparse_model(workspace)


def sparse_model_identity(
    model: Path, *, identify: FileIdentityReader = file_identity,
) -> Dict[str, Any]:
    """Hash required and optional model files, optionally through a read snapshot."""
    suffix = ".bin" if (model / "cameras.bin").is_file() else ".txt"
    identities = {}
    for stem in (*REQUIRED_MODEL_FILES, *OPTIONAL_MODEL_FILES):
        identity = identify(model / (stem + suffix))
        if identity is None and stem in REQUIRED_MODEL_FILES:
            raise ValueError(f"Incomplete sparse model: {model}")
        if identity is not None:
            identities[stem + suffix] = identity
    return identities


def verified_aligned_model(workspace: Path) -> bool:
    """Select only authenticated alignments of the current primary component."""
    aligned = workspace / "sparse_pdr_aligned" / "0"
    if not aligned.is_dir():
        return False
    try:
        raw = primary_sparse_model(workspace)
        if raw is None or inspect_sparse_model(aligned)["names"] != inspect_sparse_model(raw)["names"]:
            return False
        identity = sparse_model_identity(aligned)
        decision = workspace / ALIGNMENT_RECORD
        if decision.exists():
            receipt = json.loads(decision.read_text(encoding="utf-8"))
            return (receipt.get("schema_version") == 1 and receipt.get("status") == "aligned"
                    and receipt.get("aligned_model_identity") == identity
                    and receipt.get("source_model", raw.relative_to(workspace).as_posix()) == raw.relative_to(workspace).as_posix()
                    and receipt.get("source_model_identity") == sparse_model_identity(raw)
                    and receipt.get("manifest_identity") == file_identity(workspace / "metadata/images_manifest.csv")
                    and receipt.get("reviewed_refs_identity") == file_identity(workspace / "pdr_alignment_reviewed_refs.txt"))
        # Old completed runs predate alignment receipts. Their recorded model
        # bytes and selected path preserve the coordinate frame of existing data.
        # A fresh/unfinished run can never use this compatibility branch.
        if (workspace / PENDING_NAME).exists():
            return False
        legacy = json.loads((workspace / PROVENANCE_NAME).read_text(encoding="utf-8"))
        outputs = legacy.get("outputs", {})
        return (legacy.get("schema_version") == 1 and legacy.get("status") == "complete"
                and outputs.get("dense_input") == "sparse_pdr_aligned/0"
                and all(outputs.get("files", {}).get("sparse_pdr_aligned/0/" + name) == value
                        for name, value in identity.items()))
    except (OSError, ValueError, TypeError, AttributeError, struct.error):
        return False


# Camera layouts follow COLMAP's documented model IDs and binary format.
CAMERA_MODELS = [
    ("SIMPLE_PINHOLE", 3), ("PINHOLE", 4), ("SIMPLE_RADIAL", 4), ("RADIAL", 5),
    ("OPENCV", 8), ("OPENCV_FISHEYE", 8), ("FULL_OPENCV", 12), ("FOV", 5),
    ("SIMPLE_RADIAL_FISHEYE", 4), ("RADIAL_FISHEYE", 5), ("THIN_PRISM_FISHEYE", 12),
]


def inspect_sparse_model(model: Path) -> Dict[str, Any]:
    """Require complete image, camera and point records before dense execution."""
    binary = (model / "cameras.bin").exists()
    camera_ids: Set[int] = set()
    names = read_binary_image_names(model / "images.bin", camera_ids) if binary else read_text_image_names(model / "images.txt", camera_ids)
    cameras: Set[int] = set()

    def add_camera(camera_id: int, width: int, height: int, params: Sequence[float]) -> None:
        if camera_id <= 0 or camera_id in cameras or width <= 0 or height <= 0 or not params or params[0] <= 0 or not all(math.isfinite(v) for v in params):
            raise ValueError("Invalid COLMAP camera")
        cameras.add(camera_id)

    if binary:
        with (model / "cameras.bin").open("rb") as handle:
            count = struct.unpack("<Q", read_exact(handle, 8))[0]
            for _ in range(count):
                camera_id, model_id, width, height = struct.unpack("<iiQQ", read_exact(handle, 24))
                if not 0 <= model_id < len(CAMERA_MODELS):
                    raise ValueError(f"Unsupported COLMAP camera model ID: {model_id}")
                num_params = CAMERA_MODELS[model_id][1]
                add_camera(camera_id, width, height, struct.unpack("<" + "d" * num_params, read_exact(handle, 8 * num_params)))
            if handle.read(1):
                raise ValueError("Extra data in COLMAP cameras.bin")
        points_path = model / "points3D.bin"
        with points_path.open("rb") as handle:
            point_count = struct.unpack("<Q", read_exact(handle, 8))[0]
            file_size = points_path.stat().st_size
            for _ in range(point_count):
                record = struct.unpack("<QdddBBBd", read_exact(handle, 43))
                if not all(math.isfinite(v) for v in (*record[1:4], record[-1])):
                    raise ValueError("Non-finite COLMAP sparse point")
                tracks = struct.unpack("<Q", read_exact(handle, 8))[0]
                if tracks * 8 > file_size - handle.tell():
                    raise ValueError("Truncated COLMAP point tracks")
                handle.seek(tracks * 8, 1)
            if handle.read(1):
                raise ValueError("Extra data in COLMAP points3D.bin")
    else:
        camera_sizes = dict(CAMERA_MODELS)
        with (model / "cameras.txt").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4 or parts[1] not in camera_sizes or len(parts) != 4 + camera_sizes[parts[1]]:
                    raise ValueError("Invalid COLMAP camera record")
                add_camera(int(parts[0]), int(parts[2]), int(parts[3]), [float(v) for v in parts[4:]])
        point_count = 0
        with (model / "points3D.txt").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8 or (len(parts) - 8) % 2:
                    raise ValueError("Invalid COLMAP sparse point record")
                int(parts[0])
                if not all(math.isfinite(float(parts[i])) for i in (1, 2, 3, 7)):
                    raise ValueError("Non-finite COLMAP sparse point")
                for token in parts[4:7] + parts[8:]:
                    int(token)
                point_count += 1
    if not camera_ids <= cameras:
        raise ValueError("Sparse images reference missing cameras")
    if len(names) < 2 or not point_count:
        raise ValueError("Dense reconstruction requires registered views and sparse points")
    return {"names": names, "camera_count": len(cameras), "point_count": point_count}


def source_coverage(
    expected_names: Iterable[str],
    source_by_name: Dict[str, str],
    observed_names: Set[str],
) -> Dict[str, Dict[str, Any]]:
    expected_by_source: Dict[str, Set[str]] = {}
    for name in expected_names:
        expected_by_source.setdefault(source_by_name.get(name, "unknown"), set()).add(name)
    result: Dict[str, Dict[str, Any]] = {}
    for source, names in sorted(expected_by_source.items()):
        missing = sorted(names - observed_names)
        result[source] = {
            "expected_count": len(names),
            "present_count": len(names) - len(missing),
            "missing_count": len(missing),
            "missing_examples": missing[:10],
        }
    return result


def inspect_workspace(workspace: Path) -> Dict[str, Any]:
    from sparse_quality import assess_sparse_quality, summarize_coverage

    workspace = workspace.resolve()
    image_list = read_image_list(workspace / "image_list.txt")
    expected = set(image_list)
    images_dir = workspace / "images"
    actual = {
        normalize_name(str(path.relative_to(images_dir)))
        for path in images_dir.rglob("*")
        if path.is_file()
    } if images_dir.exists() else set()
    source_by_name = read_manifest_sources(workspace / "metadata" / "images_manifest.csv")
    database_names, database_error = read_database_names(workspace / "database.db")
    selected_model = dense_input_model(workspace)
    sparse_names: Set[str] = set()
    sparse_point_count = 0
    sparse_error = ""
    if selected_model is not None:
        try:
            sparse_info = inspect_sparse_model(selected_model)
            sparse_names = sparse_info["names"]
            sparse_point_count = sparse_info["point_count"]
        except (OSError, ValueError, struct.error) as exc:
            sparse_error = str(exc)
    manifest = workspace / "metadata" / "images_manifest.csv"
    metadata = {}
    if manifest.exists():
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            metadata = {row["image_file"]: row for row in csv.DictReader(stream)}
    coverage = summarize_coverage(expected, sparse_names, metadata)
    return {
        "workspace": str(workspace),
        "input": {
            "listed_count": len(image_list),
            "unique_listed_count": len(expected),
            "directory_file_count": len(actual),
            "missing_count": len(expected - actual),
            "extra_count": len(actual - expected),
            "missing_files": sorted(expected - actual)[:20],
            "extra_files": sorted(actual - expected)[:20],
            "duplicate_list_entries": len(image_list) - len(expected),
            "source_coverage": source_coverage(expected, source_by_name, actual),
        },
        "database": {
            "exists": (workspace / "database.db").exists(),
            "error": database_error,
            "image_count": len(database_names),
            "missing_count": len(expected - database_names),
            "extra_count": len(database_names - expected),
            "missing_names": sorted(expected - database_names)[:20],
            "extra_names": sorted(database_names - expected)[:20],
            "source_coverage": source_coverage(expected, source_by_name, database_names),
        },
        "dense_input_sparse": {
            "path": "" if selected_model is None else str(selected_model),
            "exists": selected_model is not None,
            "error": sparse_error,
            "registered_image_count": len(sparse_names),
            "point_count": sparse_point_count,
            "missing_count": len(expected - sparse_names),
            "extra_count": len(sparse_names - expected),
            "missing_names": sorted(expected - sparse_names)[:20],
            "extra_names": sorted(sparse_names - expected)[:20],
            "source_coverage": source_coverage(expected, source_by_name, sparse_names),
            "quality": assess_sparse_quality(coverage, metadata),
        },
        "provenance": inspect_provenance(workspace),
    }


def dense_readiness_failures(report: Dict[str, Any]) -> List[str]:
    """Reject poor camera coverage before expensive dense reconstruction."""
    from sparse_quality import assess_sparse_quality

    sparse = report.get("dense_input_sparse", {})
    if not sparse.get("exists") or sparse.get("error"):
        return []  # Missing/invalid models are reported by the structural checks.
    registered = sparse.get("registered_image_count", 0)
    expected = report.get("input", {}).get("unique_listed_count", 0)
    quality = sparse.get("quality")
    if quality is None:
        quality = assess_sparse_quality({"input_images": expected, "registered_images": registered}, {})
    return ["Sparse coverage is not ready for dense reconstruction: " + failure +
            ". Inspect sparse_recovery_summary.json and the mapper log; more depth maps cannot restore missing camera poses."
            for failure in quality["failures"]]


def validation_failures(
    report: Dict[str, Any],
    *,
    require_clean_database: bool = False,
    require_database_coverage: bool = False,
    require_sparse_model: bool = False,
    require_provenance: bool = True,
    require_dense_readiness: bool = True,
) -> List[str]:
    failures: List[str] = []
    input_info = report["input"]
    database = report["database"]
    sparse = report["dense_input_sparse"]
    if not input_info["listed_count"]:
        failures.append("image_list.txt is empty or missing")
    if input_info["missing_files"]:
        failures.append("some image_list entries are missing from workspace/images")
    if input_info["duplicate_list_entries"]:
        failures.append("image_list.txt contains duplicate names")
    if require_clean_database and database["exists"]:
        failures.append("database.db already exists; run the launcher reset before sparse reconstruction")
    if require_database_coverage:
        if not database["exists"]:
            failures.append("database.db is missing; run sparse reconstruction first")
        elif database["error"]:
            failures.append(f"database.db could not be read: {database['error']}")
        elif database["missing_names"] or database["extra_names"]:
            failures.append("database image names do not match the current image_list.txt")
    if require_sparse_model:
        if not sparse["exists"]:
            failures.append("the dense input sparse model is missing")
        elif sparse.get("error"):
            failures.append(f"the dense input sparse model is invalid: {sparse['error']}")
        elif sparse.get("registered_image_count", 0) < 2:
            failures.append("the dense input sparse model has fewer than two registered images")
        elif sparse.get("extra_names"):
            failures.append("the dense input sparse model contains images outside the current image list")
        if require_dense_readiness:
            failures.extend(dense_readiness_failures(report))
        if require_provenance and report.get("provenance", {}).get("status") != "verified":
            failures.append("sparse input provenance is not verified: " + report.get("provenance", {}).get("error", "missing run record"))
    return failures


def print_summary(report: Dict[str, Any]) -> None:
    input_info = report["input"]
    database = report["database"]
    sparse = report["dense_input_sparse"]
    print(f"[provenance] {report.get('provenance', {}).get('status', 'unverified')}", flush=True)
    print(
        "[coverage] input: "
        f"listed={input_info['listed_count']}, files={input_info['directory_file_count']}, "
        f"missing={input_info['missing_count']}, extra={input_info['extra_count']}",
        flush=True,
    )
    database_state = "missing" if not database["exists"] else str(database["image_count"])
    print(
        "[coverage] database: "
        f"images={database_state}, missing={database['missing_count']}, extra={database['extra_count']}",
        flush=True,
    )
    if sparse["exists"]:
        print(
            "[coverage] dense sparse input: "
            f"registered={sparse['registered_image_count']}, points={sparse.get('point_count', 'unknown')}, path={sparse['path']}",
            flush=True,
        )
    for source, values in database["source_coverage"].items():
        print(
            f"[coverage] database {source}: {values['present_count']}/{values['expected_count']}",
            flush=True,
        )
    if not sparse["exists"]:
        print("[coverage] Sparse registration has not been produced yet.", flush=True)
        return
    for source, values in sparse["source_coverage"].items():
        if values["expected_count"] and values["present_count"] < values["expected_count"]:
            print(
                f"[warn] sparse registered {source}: {values['present_count']}/{values['expected_count']}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--require-clean-database", action="store_true")
    parser.add_argument("--require-database-coverage", action="store_true")
    parser.add_argument("--require-sparse-model", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the complete inspection report as JSON.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--record-sparse-start", action="store_true", help="Snapshot inputs before feature extraction, requiring a clean workspace.")
    action.add_argument("--record-sparse-complete", action="store_true", help="Record completed output provenance only if the pre-run inputs are unchanged.")
    action.add_argument("--print-dense-input", action="store_true", help="Print the validated sparse model selected for dense reconstruction.")
    parser.add_argument("--runner", choices=sorted(SPARSE_RUNNERS), default="run_colmap_sparse.bat")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_dense_input:
        selected = dense_input_model(args.workspace)
        try:
            if selected is None:
                raise ValueError("No sparse model is available")
            inspect_sparse_model(selected)
        except (OSError, ValueError, struct.error) as exc:
            print(f"[error] Sparse selection: {exc}", file=sys.stderr)
            return 2
        print(selected.resolve())
        return 0
    if args.record_sparse_start or args.record_sparse_complete:
        try:
            if args.record_sparse_start:
                begin_sparse_run(args.workspace, args.runner)
            else:
                complete_sparse_run(args.workspace)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            print(f"[error] Sparse provenance: {exc}", flush=True)
            return 2
        print("[provenance] " + ("inputs recorded" if args.record_sparse_start else "completed run recorded"), flush=True)
        return 0
    report = inspect_workspace(args.workspace)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_summary(report)
    failures = validation_failures(
        report,
        require_clean_database=args.require_clean_database,
        require_database_coverage=args.require_database_coverage,
        require_sparse_model=args.require_sparse_model,
    )
    if failures:
        for failure in failures:
            print(f"[error] Workspace preflight: {failure}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
