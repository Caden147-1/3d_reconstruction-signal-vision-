#!/usr/bin/env python3
"""Analyze visual similarity for generated COLMAP match-pair candidates.

The output is diagnostic: it explains whether constrained WiFi/PDR/video pairs
also have visual support. It does not remove pairs from COLMAP by itself.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
import math
import os
import statistics
import threading
from dataclasses import dataclass
from heapq import nsmallest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_SIMILARITY_FIELDS = [
    "image_a",
    "image_b",
    "reason",
    "source_type_a",
    "source_type_b",
    "edge_weight",
    "pair_status",
    "dhash_distance",
    "dhash_similarity",
    "color_hist_correlation",
    "orb_keypoints_a",
    "orb_keypoints_b",
    "orb_match_count",
    "orb_good_match_count",
    "fundamental_inlier_count",
    "fundamental_inlier_ratio",
    "visual_overlap_score",
    "visual_decision",
    "notes",
    "recovery_target",
    "wifi_retrieval_score",
]


@dataclass
class SimilarityConfig:
    max_pairs: int = 0
    max_image_size: int = 900
    orb_features: int = 1500
    min_visual_score: float = 0.22
    near_duplicate_hash_distance: int = 6
    near_duplicate_hist_correlation: float = 0.82
    geometry_min_inliers: int = 15
    geometry_min_inlier_ratio: float = 0.22
    feature_min_matches: int = 8
    recovery_neighbors: int = 12
    wifi_recovery_enabled: bool = True
    feature_workers: int = 0


@dataclass
class ImageFeatures:
    image_name: str
    status: str
    width: int = 0
    height: int = 0
    dhash: Optional[int] = None
    hist: Any = None
    keypoints: Any = None
    descriptors: Any = None
    keypoint_count: int = 0
    notes: str = ""


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_bit_count(value: int) -> int:
    try:
        return value.bit_count()
    except AttributeError:
        return bin(value).count("1")


def round_or_blank(value: Optional[float], digits: int = 6) -> Any:
    if value is None or math.isnan(value) or math.isinf(value):
        return ""
    return round(value, digits)


def summarize_numbers(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": round(min(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "median": round(statistics.median(values), 6),
        "max": round(max(values), 6),
    }


def select_rows(rows: Sequence[Dict[str, str]], max_pairs: int) -> List[Dict[str, str]]:
    if max_pairs <= 0 or len(rows) <= max_pairs:
        return list(rows)
    ranked = sorted(
        rows,
        key=lambda row: safe_float(row.get("edge_weight"), 0.0),
        reverse=True,
    )
    # Reserve a quarter of a bounded verification budget for RF proposals.
    # The remaining slots still compete on priority, and geometry is unchanged.
    wifi = [row for row in ranked if row.get("reason") == "wifi_fingerprint_recovery"]
    reserved = []
    checkpoint_uses, image_uses = {}, {}
    def checkpoint_keys(row):
        return [row.get("wifi_checkpoint_" + side) or "image:" + row.get("image_" + side, "") for side in ("a", "b")]
    # Share the reserved budget across stops before repeating already covered
    # stops. Within that coverage level, prefer underused photos and RF score.
    for _ in range(min(len(wifi), max(1, max_pairs // 4))):
        def priority(row):
            uses = [checkpoint_uses.get(key, 0) for key in checkpoint_keys(row)]
            images = [image_uses.get(row.get("image_" + side, ""), 0) for side in ("a", "b")]
            weight = safe_float(row.get("edge_weight"), 0.0)
            return (sum(use > 0 for use in uses), max(uses), sum(uses),
                    sum(use > 0 for use in images), sum(images),
                    -weight if math.isfinite(weight) else 0.0, row.get("image_a", ""), row.get("image_b", ""))
        chosen = min(wifi, key=priority)
        wifi.remove(chosen)
        reserved.append(chosen)
        for key in checkpoint_keys(chosen):
            checkpoint_uses[key] = checkpoint_uses.get(key, 0) + 1
        for side in ("a", "b"):
            name = chosen.get("image_" + side, "")
            image_uses[name] = image_uses.get(name, 0) + 1
    reserved_ids = {id(row) for row in reserved}
    return reserved + [row for row in ranked if id(row) not in reserved_ids][:max_pairs - len(reserved)]


def load_cv_modules() -> Tuple[Any, Any, Optional[str]]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        return None, None, str(exc)
    return cv2, np, None


def image_path_for(images_dir: Path, image_name: str) -> Path:
    return images_dir / Path(image_name)


def resized_image(cv2: Any, image: Any, max_image_size: int) -> Any:
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_image_size <= 0 or longest <= max_image_size:
        return image
    scale = max_image_size / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def difference_hash(cv2: Any, gray: Any) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def color_histogram(cv2: Any, image: Any) -> Any:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)
    return hist


def read_image_features(
    image_name: str,
    images_dir: Path,
    cv2: Any,
    orb: Any,
    config: SimilarityConfig,
) -> ImageFeatures:
    path = image_path_for(images_dir, image_name)
    if not path.exists():
        return ImageFeatures(image_name=image_name, status="missing", notes=str(path))

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return ImageFeatures(image_name=image_name, status="unreadable", notes=str(path))

    image = resized_image(cv2, image, config.max_image_size)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    keypoints = keypoints or []
    return ImageFeatures(
        image_name=image_name,
        status="ok",
        width=width,
        height=height,
        dhash=difference_hash(cv2, gray),
        hist=color_histogram(cv2, image),
        keypoints=keypoints,
        descriptors=descriptors,
        keypoint_count=len(keypoints),
    )


def prepare_image_features(
    image_names: Sequence[str],
    images_dir: Path,
    cv2: Any,
    config: SimilarityConfig,
) -> Dict[str, ImageFeatures]:
    """Read each image once with bounded concurrency and stable cache order.

    OpenCV releases the GIL while reading and extracting ORB features. Each
    worker owns its ORB object; no OpenCV thread or RNG settings are changed.
    Pair verification remains sequential so its RANSAC seed is unaffected.
    Set feature_workers=1 for the serial path, or 0 for automatic selection.
    Machines with at least 16 logical CPUs use up to eight workers; others four.
    Explicit worker counts are limited to eight concurrent image decodes.
    """
    if not isinstance(config.feature_workers, int) or not 0 <= config.feature_workers <= 8:
        raise ValueError("feature_workers must be an integer from 0 to 8 (0=auto, 1=serial)")
    names = list(dict.fromkeys(image_names))
    if not names:
        return {}
    cpus = os.cpu_count() or 1
    auto_workers = 8 if cpus >= 16 else min(4, cpus)
    workers = min(len(names), config.feature_workers or auto_workers)
    worker_state = threading.local()

    def read_one(name: str) -> ImageFeatures:
        if not hasattr(worker_state, "orb"):
            worker_state.orb = cv2.ORB_create(nfeatures=max(100, config.orb_features))
        return read_image_features(name, images_dir, cv2, worker_state.orb, config)

    if workers == 1:
        return {name: read_one(name) for name in names}

    # Keep at most one pending image per worker instead of queuing the entire
    # dataset. Refill as images finish so a slow decode cannot idle other workers.
    features: Dict[str, ImageFeatures] = {}
    remaining = iter(names)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="image-features") as pool:
        pending = {}
        for _ in range(workers):
            name = next(remaining)
            pending[pool.submit(read_one, name)] = name
        try:
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Observe every completed result before submitting more work;
                # a failure behind a slow image must stop replenishing the pool.
                for future in completed:
                    name = pending.pop(future)
                    features[name] = future.result()
                while len(pending) < workers:
                    name = next(remaining, None)
                    if name is None:
                        break
                    pending[pool.submit(read_one, name)] = name
        finally:
            for future in pending:
                future.cancel()
    return {name: features[name] for name in names}


def pair_hist_correlation(cv2: Any, features_a: ImageFeatures, features_b: ImageFeatures) -> Optional[float]:
    if features_a.hist is None or features_b.hist is None:
        return None
    return float(cv2.compareHist(features_a.hist, features_b.hist, cv2.HISTCMP_CORREL))


def pair_dhash(features_a: ImageFeatures, features_b: ImageFeatures) -> Tuple[Optional[int], Optional[float]]:
    if features_a.dhash is None or features_b.dhash is None:
        return None, None
    distance = int_bit_count(features_a.dhash ^ features_b.dhash)
    return distance, max(0.0, 1.0 - distance / 64.0)


def good_orb_matches(cv2: Any, descriptors_a: Any, descriptors_b: Any) -> Tuple[List[Any], List[Any], float]:
    if descriptors_a is None or descriptors_b is None:
        return [], [], 0.0
    if len(descriptors_a) == 0 or len(descriptors_b) == 0:
        return [], [], 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = list(matcher.match(descriptors_a, descriptors_b))
    matches.sort(key=lambda match: match.distance)
    if not matches:
        return [], [], 0.0
    distances = [float(match.distance) for match in matches]
    median_distance = statistics.median(distances)
    threshold = min(72.0, max(32.0, median_distance * 0.75))
    good = [match for match in matches if float(match.distance) <= threshold]
    return matches, good, threshold


def fundamental_inliers(
    cv2: Any,
    np: Any,
    features_a: ImageFeatures,
    features_b: ImageFeatures,
    matches: Sequence[Any],
) -> Tuple[int, Optional[float]]:
    if len(matches) < 8:
        return 0, None
    src = np.float32([features_a.keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst = np.float32([features_b.keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    try:
        # Pair verification must not depend on which sensor-derived candidates
        # happened to be evaluated before this pair in another experiment arm.
        cv2.setRNGSeed(0)
        matrix, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 3.0, 0.99)
    except cv2.error:
        return 0, None
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all() or mask is None:
        return 0, None
    if mask.size != len(matches):
        return 0, None
    inliers = int(mask.ravel().astype(bool).sum())
    return inliers, inliers / max(1, len(matches))


def bounded_ratio(value: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return max(0.0, min(1.0, value / target))


def visual_overlap_score(
    dhash_similarity: Optional[float],
    hist_correlation: Optional[float],
    good_match_count: int,
    inlier_count: int,
    inlier_ratio: Optional[float],
) -> float:
    hash_score = dhash_similarity if dhash_similarity is not None else 0.0
    hist_score = 0.0
    if hist_correlation is not None:
        hist_score = max(0.0, min(1.0, (hist_correlation + 1.0) * 0.5))
    feature_score = bounded_ratio(float(good_match_count), 60.0)
    geometry_ratio = inlier_ratio if inlier_ratio is not None else 0.0
    geometry_score = bounded_ratio(float(inlier_count), 30.0) * bounded_ratio(geometry_ratio, 0.35)
    return max(0.0, min(1.0, 0.20 * hash_score + 0.20 * hist_score + 0.30 * feature_score + 0.30 * geometry_score))


def visual_decision(
    config: SimilarityConfig,
    dhash_distance: Optional[int],
    hist_correlation: Optional[float],
    good_match_count: int,
    inlier_count: int,
    inlier_ratio: Optional[float],
    score: float,
) -> str:
    if (
        inlier_ratio is not None
        and good_match_count >= config.geometry_min_inliers
        and inlier_count >= config.geometry_min_inliers
        and inlier_ratio >= config.geometry_min_inlier_ratio
    ):
        return "geometry_supported"
    if (
        good_match_count >= config.feature_min_matches
        and dhash_distance is not None
        and hist_correlation is not None
        and dhash_distance <= config.near_duplicate_hash_distance
        and hist_correlation >= config.near_duplicate_hist_correlation
    ):
        return "near_duplicate"
    if good_match_count >= config.feature_min_matches and (score >= config.min_visual_score or good_match_count >= 30):
        return "feature_overlap"
    if good_match_count == 0:
        return "no_feature_overlap"
    return "weak_visual_support"


def analyze_pair(
    row: Dict[str, str],
    features_a: ImageFeatures,
    features_b: ImageFeatures,
    cv2: Any,
    np: Any,
    config: SimilarityConfig,
) -> Dict[str, Any]:
    base = {
        "image_a": row.get("image_a", ""),
        "image_b": row.get("image_b", ""),
        "reason": row.get("reason", ""),
        "source_type_a": row.get("source_type_a", ""),
        "source_type_b": row.get("source_type_b", ""),
        "edge_weight": row.get("edge_weight", ""),
        "recovery_target": row.get("recovery_target", ""),
        "wifi_retrieval_score": row.get("wifi_retrieval_score", ""),
    }
    if features_a.status != "ok" or features_b.status != "ok":
        notes = "; ".join(
            item
            for item in (
                f"{features_a.image_name}:{features_a.status}" if features_a.status != "ok" else "",
                f"{features_b.image_name}:{features_b.status}" if features_b.status != "ok" else "",
            )
            if item
        )
        return {
            **base,
            "pair_status": "missing_or_unreadable_image",
            "orb_keypoints_a": features_a.keypoint_count,
            "orb_keypoints_b": features_b.keypoint_count,
            "visual_decision": "not_evaluated",
            "notes": notes,
        }

    dhash_distance, dhash_similarity = pair_dhash(features_a, features_b)
    hist_correlation = pair_hist_correlation(cv2, features_a, features_b)
    all_matches, good_matches, _threshold = good_orb_matches(cv2, features_a.descriptors, features_b.descriptors)
    inlier_count, inlier_ratio = fundamental_inliers(cv2, np, features_a, features_b, good_matches)
    score = visual_overlap_score(
        dhash_similarity=dhash_similarity,
        hist_correlation=hist_correlation,
        good_match_count=len(good_matches),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
    )
    decision = visual_decision(
        config=config,
        dhash_distance=dhash_distance,
        hist_correlation=hist_correlation,
        good_match_count=len(good_matches),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        score=score,
    )
    return {
        **base,
        "pair_status": "ok",
        "dhash_distance": "" if dhash_distance is None else dhash_distance,
        "dhash_similarity": round_or_blank(dhash_similarity),
        "color_hist_correlation": round_or_blank(hist_correlation),
        "orb_keypoints_a": features_a.keypoint_count,
        "orb_keypoints_b": features_b.keypoint_count,
        "orb_match_count": len(all_matches),
        "orb_good_match_count": len(good_matches),
        "fundamental_inlier_count": inlier_count,
        "fundamental_inlier_ratio": round_or_blank(inlier_ratio),
        "visual_overlap_score": round(score, 6),
        "visual_decision": decision,
        "notes": "",
    }


def count_by(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def summarize_by_reason(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        score = safe_float(row.get("visual_overlap_score"), math.nan)
        if math.isnan(score):
            continue
        buckets.setdefault(str(row.get("reason") or "unknown"), []).append(score)
    return {reason: summarize_numbers(values) for reason, values in sorted(buckets.items())}


def build_summary(
    rows: Sequence[Dict[str, Any]],
    edge_rows: Sequence[Dict[str, str]],
    output_csv: Path,
    config: SimilarityConfig,
    status: str = "ok",
    reason: str = "",
) -> Dict[str, Any]:
    scores = [
        safe_float(row.get("visual_overlap_score"), math.nan)
        for row in rows
        if not math.isnan(safe_float(row.get("visual_overlap_score"), math.nan))
    ]
    decision_counts = count_by(rows, "visual_decision")
    visual_supported = sum(
        decision_counts.get(name, 0)
        for name in ("geometry_supported", "feature_overlap", "near_duplicate")
    )
    summary = {
        "status": status,
        "reason": reason,
        "output_csv": str(output_csv),
        "input_pair_count": len(edge_rows),
        "evaluated_pair_count": len(rows),
        "diagnostic_only": True,
        "visual_supported_pair_count": visual_supported,
        "weak_visual_support_pair_count": decision_counts.get("weak_visual_support", 0),
        "no_feature_overlap_pair_count": decision_counts.get("no_feature_overlap", 0),
        "near_duplicate_pair_count": decision_counts.get("near_duplicate", 0),
        "geometry_supported_pair_count": decision_counts.get("geometry_supported", 0),
        "pair_status_counts": count_by(rows, "pair_status"),
        "visual_decision_counts": decision_counts,
        "visual_overlap_score": summarize_numbers(scores),
        "visual_overlap_score_by_reason": summarize_by_reason(rows),
        "config": {
            "max_pairs": config.max_pairs,
            "max_image_size": config.max_image_size,
            "orb_features": config.orb_features,
            "min_visual_score": config.min_visual_score,
            "near_duplicate_hash_distance": config.near_duplicate_hash_distance,
            "near_duplicate_hist_correlation": config.near_duplicate_hist_correlation,
            "geometry_min_inliers": config.geometry_min_inliers,
            "geometry_min_inlier_ratio": config.geometry_min_inlier_ratio,
            "feature_min_matches": config.feature_min_matches,
            "recovery_neighbors": config.recovery_neighbors,
            "feature_workers": config.feature_workers,
        },
        "explanation": (
            "Each constrained match-pair candidate is checked with perceptual hash, "
            "HSV color histogram correlation, ORB feature matches, and optional "
            "fundamental-matrix RANSAC inliers. Workspace preparation consumes this "
            "table to select evaluated P1-supported pairs; standalone analysis only "
            "writes evidence and does not rewrite match_pairs.txt."
        ),
    }
    return summary


def recovery_candidates(
    features: Sequence[ImageFeatures],
    existing_pairs: set[Tuple[str, str]],
    cv2: Any,
    neighbors: int,
) -> List[Dict[str, Any]]:
    """Use a fixed visual top-k pool, then omit pairs already in the route graph."""
    if neighbors <= 0:
        return []
    valid = [f for f in features if f.status == "ok" and f.keypoint_count >= 15]
    candidates: Dict[Tuple[str, str], float] = {}
    for left in valid:
        ranked: List[Tuple[float, str]] = []
        for right in valid:
            pair = tuple(sorted((left.image_name, right.image_name)))
            if left is right:
                continue
            _distance, hash_similarity = pair_dhash(left, right)
            correlation = pair_hist_correlation(cv2, left, right)
            score = 0.5 * (hash_similarity or 0.0) + 0.5 * max(0.0, correlation or 0.0)
            ranked.append((score, right.image_name))
        for score, name in sorted(ranked, key=lambda item: (-item[0], item[1]))[:neighbors]:
            pair = tuple(sorted((left.image_name, name)))
            if pair not in existing_pairs:
                candidates[pair] = score
    return [
        {"image_a": a, "image_b": b, "reason": "visual_recovery", "edge_weight": score}
        for (a, b), score in sorted(candidates.items())
    ]


def targeted_recovery_candidates(
    features: Sequence[ImageFeatures], existing_pairs: set[Tuple[str, str]],
    cv2: Any, neighbors: int, targets: set[str], references: set[str],
    metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    wifi_recovery_enabled: bool = True,
) -> List[Dict[str, Any]]:
    """Spend new comparisons on weak/unregistered images against supported ones.

    Exclude already evaluated pairs before top-k so an exhausted old top-k does
    not starve recovery. RF similarity is a ranking cue, never a pose constraint.
    """
    from wifi_observations import fingerprint_similarity
    if neighbors <= 0:
        return []
    metadata = metadata or {}
    valid = {f.image_name: f for f in features if f.status == "ok" and f.keypoint_count >= 15}
    existing = {tuple(sorted(pair)) for pair in existing_pairs}
    reference_names = sorted((references & valid.keys()) - targets)
    target_names = sorted(targets & valid.keys())
    if not target_names or not reference_names:
        return []

    # A checkpoint can contain different RF observations. Cache by the evidence
    # itself, including validity and measurement IDs, rather than its location.
    rf_keys = {
        name: json.dumps(
            {field: value for field, value in metadata.get(name, {}).items()
             if field == "session" or field.startswith(("ap_", "wifi_"))},
            sort_keys=True,
        )
        for name in (*target_names, *reference_names)
    } if wifi_recovery_enabled else {}
    candidates = {}
    rf_cache = {}
    for name in target_names:
        left = valid[name]
        ranked = []
        for other in reference_names:
            pair = tuple(sorted((name, other)))
            if pair in existing:
                continue
            right = valid[other]
            _, hash_similarity = pair_dhash(left, right)
            correlation = pair_hist_correlation(cv2, left, right)
            visual = .5 * (hash_similarity or 0.0) + .5 * max(0.0, correlation or 0.0)
            a, b = metadata.get(name, {}), metadata.get(other, {})
            rf = 0.0
            if wifi_recovery_enabled:
                key = (rf_keys[name], rf_keys[other])
                if key not in rf_cache:
                    rf_cache[key] = fingerprint_similarity(a, b)["score"]
                rf = rf_cache[key]
            # Missing RF evidence neither excludes an image nor gives it a fake fingerprint.
            score = visual + .25 * rf
            same_stop = bool(a.get("session") and a.get("global_point_index") not in (None, "")
                             and a.get("session") == b.get("session")
                             and a.get("global_point_index") == b.get("global_point_index"))
            ranked.append((score, other, rf, same_stop))
        # Spend recovery on different viewpoints before more rotations at one stop.
        # This is a retrieval priority, never a substitute for geometric checks.
        selected = nsmallest(neighbors, ranked, key=lambda item: (item[3], -item[0], item[1]))
        for score, other, rf, _ in selected:
            pair = tuple(sorted((name, other)))
            candidates[pair] = {"image_a": pair[0], "image_b": pair[1], "reason": "visual_recovery",
                                "edge_weight": score, "recovery_target": name,
                                "wifi_retrieval_score": rf}
    return [candidates[key] for key in sorted(candidates)]


def weak_geometry_targets(names: Sequence[str], rows: Sequence[Dict[str, Any]]) -> Tuple[set[str], set[str]]:
    """Find weak vertices and small components using measured geometry only."""
    adjacency = {name: set() for name in names}
    for row in rows:
        a, b = row.get("image_a"), row.get("image_b")
        if row.get("pair_status") == "ok" and row.get("visual_decision") == "geometry_supported" and a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    remaining = set(adjacency)
    components = []
    while remaining:
        stack = [min(remaining)]; group = set()
        while stack:
            name = stack.pop()
            if name in group:
                continue
            group.add(name); remaining.discard(name)
            stack.extend(adjacency[name] - group)
        components.append(group)
    largest = min(components, key=lambda group: (-len(group), sorted(group))) if components else set()
    references = {name for name in largest if len(adjacency[name]) >= 2}
    return set(names) - references, references


def analyze_image_similarity_edges(
    images_dir: Path,
    edge_rows: Sequence[Dict[str, str]],
    output_csv: Path,
    summary_path: Optional[Path] = None,
    config: Optional[SimilarityConfig] = None,
    recovery_image_names: Optional[Sequence[str]] = None,
    recovery_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    config = config or SimilarityConfig()
    cv2, np, import_error = load_cv_modules()
    if cv2 is None or np is None:
        write_csv_rows(output_csv, IMAGE_SIMILARITY_FIELDS, [])
        summary = build_summary(
            rows=[],
            edge_rows=edge_rows,
            output_csv=output_csv,
            config=config,
            status="skipped",
            reason=f"missing_opencv_or_numpy: {import_error}",
        )
        if summary_path:
            write_json(summary_path, summary)
        return summary

    selected_rows = select_rows(edge_rows, config.max_pairs)
    feature_cache = prepare_image_features(
        [row[key] for row in selected_rows for key in ("image_a", "image_b")],
        images_dir, cv2, config,
    )
    rows = [analyze_pair(row, feature_cache[row["image_a"]], feature_cache[row["image_b"]], cv2, np, config)
            for row in selected_rows]

    recovery_rows: List[Dict[str, Any]] = []
    if recovery_image_names is not None and config.recovery_neighbors > 0:
        recovery_names = sorted(set(recovery_image_names))
        feature_cache.update(prepare_image_features(
            [name for name in recovery_names if name not in feature_cache],
            images_dir, cv2, config,
        ))
        existing_pairs = {tuple(sorted((row["image_a"], row["image_b"]))) for row in edge_rows}
        recovery_edges = recovery_candidates(
            [feature_cache[name] for name in recovery_names],
            existing_pairs,
            cv2,
            config.recovery_neighbors,
        )
        for row in recovery_edges:
            recovery_rows.append(analyze_pair(
                row, feature_cache[row["image_a"]], feature_cache[row["image_b"]], cv2, np, config,
            ))
    rows.extend(recovery_rows)
    targeted_edges = []
    targeted_rows = []
    weak_names = set()
    if recovery_image_names is not None and config.recovery_neighbors > 0:
        weak_names, references = weak_geometry_targets(recovery_image_names, rows)
        # Include even budget-skipped inputs in the exclusion set; do not bypass max_pairs.
        evaluated = {tuple(sorted((row["image_a"], row["image_b"]))) for row in [*edge_rows, *rows]}
        targeted_edges = targeted_recovery_candidates(
            list(feature_cache.values()), evaluated, cv2, config.recovery_neighbors,
            weak_names, references, recovery_metadata, config.wifi_recovery_enabled,
        )
        for row in targeted_edges:
            result = analyze_pair(row, feature_cache[row["image_a"]], feature_cache[row["image_b"]], cv2, np, config)
            result.update(recovery_target=row["recovery_target"], wifi_retrieval_score=row["wifi_retrieval_score"])
            targeted_rows.append(result)
        rows.extend(targeted_rows)
    write_csv_rows(output_csv, IMAGE_SIMILARITY_FIELDS, rows)
    summary = build_summary(
        rows=rows,
        edge_rows=edge_rows,
        output_csv=output_csv,
        config=config,
    )
    summary.update(
        {
            "images_dir": str(images_dir),
            "unique_image_count": len(feature_cache),
            "input_pair_count": len(edge_rows) + len(recovery_rows) + len(targeted_rows),
            "route_input_pair_count": sum(row.get("reason") != "wifi_fingerprint_recovery" for row in edge_rows),
            "route_evaluated_pair_count": sum(row.get("reason") not in {"wifi_fingerprint_recovery", "visual_recovery"} for row in rows),
            "wifi_input_pair_count": sum(row.get("reason") == "wifi_fingerprint_recovery" for row in edge_rows),
            "wifi_evaluated_pair_count": sum(row.get("reason") == "wifi_fingerprint_recovery" for row in rows),
            "max_pairs_scope": "route_and_wifi_candidates; visual recovery uses a separate per-image neighbor budget",
            "wifi_budget_reservation_fraction": 0.25,
            "wifi_budget_allocation": ("all_candidates" if config.max_pairs <= 0 or len(edge_rows) <= config.max_pairs
                                       else "checkpoint_and_photo_coverage_then_RF_priority"),
            "targeted_recovery": {
                "target_image_count": len(weak_names),
                "candidate_pair_count": len(targeted_rows),
                "geometry_supported_pair_count": sum(r["visual_decision"] == "geometry_supported" for r in targeted_rows),
                "geometry_supported_target_count": len({r["recovery_target"] for r in targeted_rows if r["visual_decision"] == "geometry_supported"}),
                "policy": "weak_or_disconnected_images_to_supported_component; exclude_evaluated_before_top_k",
                "budget_scope": "at_most_recovery_neighbors_per_target_additional_pairs",
            },
            "visual_recovery": {
                "status": "evaluated" if recovery_image_names is not None and config.recovery_neighbors > 0 else "disabled",
                "candidate_pair_count": len(recovery_rows),
                "geometry_supported_pair_count": sum(row["visual_decision"] == "geometry_supported" for row in recovery_rows),
                "neighbors_per_image": config.recovery_neighbors,
                "explanation": "Appearance retrieves candidates outside route/slot windows; only geometry-supported candidates can be added to the COLMAP list.",
            },
            "image_status_counts": count_by(
                [
                    {"status": features.status}
                    for features in feature_cache.values()
                ],
                "status",
            ),
        }
    )
    if summary_path:
        write_json(summary_path, summary)
    return summary


def analyze_workspace_image_similarity(
    workspace_dir: Path,
    edge_rows: Optional[Sequence[Dict[str, str]]] = None,
    max_pairs: int = 0,
    max_image_size: int = 900,
    orb_features: int = 1500,
    min_visual_score: float = 0.22,
    recovery_image_names: Optional[Sequence[str]] = None,
    recovery_neighbors: int = 12,
    recovery_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    wifi_recovery_enabled: bool = True,
    feature_workers: int = 0,
) -> Dict[str, Any]:
    if edge_rows is None:
        _fields, edge_rows = read_csv(workspace_dir / "match_graph_edges.csv")
    config = SimilarityConfig(
        max_pairs=max_pairs,
        max_image_size=max_image_size,
        orb_features=orb_features,
        min_visual_score=min_visual_score,
        recovery_neighbors=recovery_neighbors,
        wifi_recovery_enabled=wifi_recovery_enabled,
        feature_workers=feature_workers,
    )
    return analyze_image_similarity_edges(
        images_dir=workspace_dir / "images",
        edge_rows=edge_rows,
        output_csv=workspace_dir / "image_similarity_edges.csv",
        summary_path=workspace_dir / "image_similarity_summary.json",
        config=config,
        recovery_image_names=recovery_image_names,
        recovery_metadata=recovery_metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("colmap_workspace"))
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--match-graph", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--max-pairs", type=int, default=0, help="Analyze all pairs by default; set a positive sample cap.")
    parser.add_argument("--max-image-size", type=int, default=900)
    parser.add_argument("--orb-features", type=int, default=1500)
    parser.add_argument("--feature-workers", type=int, choices=range(9), default=0,
                        help="Concurrent image feature readers (0=auto, up to 8 on 16+ logical CPUs; 1=serial; maximum 8).")
    parser.add_argument("--min-visual-score", type=float, default=0.22)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_dir = args.workspace.resolve()
    images_dir = args.images_dir.resolve() if args.images_dir else workspace_dir / "images"
    match_graph = args.match_graph.resolve() if args.match_graph else workspace_dir / "match_graph_edges.csv"
    output = args.output.resolve() if args.output else workspace_dir / "image_similarity_edges.csv"
    summary_path = args.summary.resolve() if args.summary else workspace_dir / "image_similarity_summary.json"
    _fields, edge_rows = read_csv(match_graph)
    summary = analyze_image_similarity_edges(
        images_dir=images_dir,
        edge_rows=edge_rows,
        output_csv=output,
        summary_path=summary_path,
        config=SimilarityConfig(
            max_pairs=args.max_pairs,
            max_image_size=args.max_image_size,
            orb_features=args.orb_features,
            feature_workers=args.feature_workers,
            min_visual_score=args.min_visual_score,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
