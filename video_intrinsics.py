"""Estimate a shared video focal initialization from verified two-view geometry.

For square pixels and a centered principal point, K.T @ F @ K must have two
equal nonzero singular values. Only informative, non-planar pair estimates with
broad temporal consensus can replace COLMAP's missing-EXIF default. This is an
initial estimate, not externally calibrated camera data. Confident shared video
intrinsics can be held during incremental mapping to avoid early degeneracy.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sqlite3

import numpy as np

from colmap_workspace_check import open_readonly_database

PAIR_BASE = 2147483647


def pair_focal(matrix, width, cx, cy):
    F = np.asarray(matrix, dtype=float)
    if F.shape != (3, 3) or not np.isfinite(F).all() or np.linalg.norm(F) < 1e-20:
        return None

    def errors(focals):
        K = np.repeat(np.eye(3)[None], len(focals), axis=0)
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, 0, 2], K[:, 1, 2] = cx, cy
        sv = np.linalg.svd(K.transpose(0, 2, 1) @ F @ K, compute_uv=False)
        return (sv[:, 0] - sv[:, 1] + sv[:, 2]) / np.maximum(sv[:, 1], 1e-20)

    grid = np.geomspace(.2, 2., 121) * width
    costs = errors(grid)
    index = int(np.argmin(costs))
    if index < 3 or index > 117 or costs[index] > .03:
        return None
    if min(costs[max(0, index-12)], costs[min(120, index+12)]) - costs[index] < .01:
        return None  # Pure translation/near-degenerate geometry cannot constrain f.
    fine = np.geomspace(grid[index-1], grid[index+1], 41)
    fine_costs = errors(fine)
    best = int(np.argmin(fine_costs))
    return float(fine[best]), float(fine_costs[best])


def focal_consensus(samples, frame_times):
    if len(samples) < 24:
        return {"status": "withheld", "reason": "too_few_informative_pairs"}
    values = np.array([s["focal"] for s in samples])
    median = float(np.median(values))
    keep = np.abs(np.log(values / median)) <= np.log(1.12)
    inliers = [s for s, accepted in zip(samples, keep) if accepted]
    values = values[keep]
    image_ids = {i for s in inliers for i in s["images"]}
    all_times = np.array(list(frame_times.values()), dtype=float)
    times = np.array([frame_times[i] for i in image_ids], dtype=float)
    span = float(np.ptp(all_times)) if len(all_times) else 0.
    coverage = float(np.ptp(times) / span) if len(times) and span > 0 else 0.
    quarters = set(np.minimum(3, ((times - all_times.min()) / span * 4).astype(int))) if span else set()
    iqr = float(np.ptp(np.percentile(values, [25, 75])) / np.median(values)) if len(values) else 1.
    if len(inliers) < 24 or len(inliers) / len(samples) < .6 or len(image_ids) < 24 or coverage < .5 or len(quarters) < 3 or iqr > .08:
        return {"status": "withheld", "reason": "insufficient_temporal_focal_consensus",
                "informative_pairs": len(samples), "consensus_pairs": len(inliers), "time_coverage": coverage}
    # A long fixed-focal part of a clip must not outvote a later zoom/lens change.
    # Check local evidence before holding one calibration for every video frame.
    # Sparse bins cannot establish a change; dispersed erroneous F estimates can
    # remain outliers as long as they do not dominate a supported time interval.
    bins = [[] for _ in range(8)]
    for sample, accepted in zip(samples, keep):
        midpoint = np.mean([frame_times[i] for i in sample["images"]])
        index = min(7, max(0, int((midpoint - all_times.min()) / span * 8)))
        bins[index].append((sample, bool(accepted)))
    temporal_evidence = []
    for index, local in enumerate(bins):
        local_ids = {i for sample, _ in local for i in sample["images"]}
        if len(local) < 6 or len(local_ids) < 8:
            continue
        support = sum(accepted for _, accepted in local) / len(local)
        temporal_evidence.append({"bin": index, "pair_count": len(local),
                                  "independent_image_count": len(local_ids),
                                  "consensus_fraction": support})
    conflicting = [row for row in temporal_evidence if row["consensus_fraction"] < .5]
    if conflicting:
        return {"status": "withheld", "reason": "temporally_inconsistent_focal_estimates",
                "informative_pairs": len(samples), "consensus_pairs": len(inliers),
                "time_coverage": coverage, "conflicting_time_bins": conflicting}
    return {"status": "estimated", "focal_px": float(np.median(values)),
            "informative_pairs": len(samples), "consensus_pairs": len(inliers),
            "independent_image_count": len(image_ids), "time_coverage": coverage, "relative_iqr": iqr,
            "temporal_consensus_bins": temporal_evidence}


def estimate_video_intrinsics(database: Path, manifest: Path, *, max_pairs=640):
    report = {"status": "withheld", "method": "shared_focal_two_view_consensus_v1", "cameras": [],
              "interpretation": "Visual focal initialization; no WiFi/IMU camera calibration is assumed."}
    if not manifest.is_file():
        report["reason"] = "missing_video_manifest"
        return report
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        metadata = {r["image_file"].replace("\\", "/"): r for r in csv.DictReader(handle)}
    connection = open_readonly_database(database)
    try:
        cameras = connection.execute("select camera_id, model, width, height, params, prior_focal_length from cameras").fetchall()
        images = connection.execute("select image_id,name,camera_id from images").fetchall()
        pairs = connection.execute("select pair_id,F from two_view_geometries where config in (2,3) and rows>=150 order by pair_id").fetchall()
    finally:
        connection.close()
    for cid, model, width, height, blob, prior in cameras:
        row = {"camera_id": cid, "status": "withheld"}
        report["cameras"].append(row)
        params = np.frombuffer(blob, dtype="<f8").copy()
        row["original_params"] = params.tolist()
        if prior or model not in (0, 2) or len(params) != (3 if model == 0 else 4):
            row["reason"] = "existing_focal_prior_or_unsupported_model"
            continue
        frames = {iid: metadata.get(name.replace("\\", "/"), {}) for iid, name, camera in images if camera == cid}
        clips = {r.get("video_name") for r in frames.values()}
        if len(frames) < 24 or len(clips) != 1 or not next(iter(clips), None) or any(r.get("source_type") != "video_base_frame" for r in frames.values()):
            row["reason"] = "requires_one_video_per_shared_camera"
            continue
        try:
            times = {iid: float(r["video_time_s"]) for iid, r in frames.items()}
            if not all(np.isfinite(t) and t >= 0 for t in times.values()):
                raise ValueError("invalid timestamp")
        except (ValueError, KeyError):
            row["reason"] = "invalid_video_times"
            continue
        eligible = [(pid, data) for pid, data in pairs if pid // PAIR_BASE in frames and pid % PAIR_BASE in frames and data and len(data) == 72]
        indices = np.linspace(0, len(eligible)-1, min(max_pairs, len(eligible)), dtype=int)
        samples = []
        for index in indices:
            pid, data = eligible[index]
            result = pair_focal(np.frombuffer(data, "<f8").reshape(3, 3), width, params[1], params[2])
            if result:
                samples.append({"focal": result[0], "error": result[1], "images": [pid // PAIR_BASE, pid % PAIR_BASE]})
        row.update(focal_consensus(samples, times))
        row["sampled_pair_count"] = len(indices)
        if row["status"] == "estimated":
            params[0] = row["focal_px"]
            row["initial_params"] = params.tolist()
    if any(r["status"] == "estimated" for r in report["cameras"]):
        report["status"] = "estimated"
    return report


def apply_video_intrinsics(database: Path, report: dict):
    """Apply only if the camera still has the exact uncalibrated values inspected."""
    with sqlite3.connect(database) as connection:
        for camera in report["cameras"]:
            if camera["status"] != "estimated":
                continue
            params = np.asarray(camera["initial_params"], dtype="<f8").tobytes()
            original = np.asarray(camera["original_params"], dtype="<f8").tobytes()
            cursor = connection.execute("update cameras set params=?, prior_focal_length=1 where camera_id=? and params=? and prior_focal_length=0",
                                        (params, camera["camera_id"], original))
            if cursor.rowcount != 1:
                raise ValueError("Camera changed while estimating video focal initialization")
