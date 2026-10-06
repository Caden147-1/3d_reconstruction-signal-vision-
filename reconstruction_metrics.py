"""Shared reconstruction snapshots and sparse-model quality metrics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from colmap_workspace_check import (
    CAMERA_MODELS, file_identity, inspect_sparse_model, read_exact, sparse_model_identity,
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
