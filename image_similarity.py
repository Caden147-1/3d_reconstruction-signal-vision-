#!/usr/bin/env python3
"""Analyze visual similarity for generated COLMAP match-pair candidates.

The output is diagnostic: it explains whether constrained WiFi/PDR/video pairs
also have visual support. It does not remove pairs from COLMAP by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
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
    return sorted(
        rows,
        key=lambda row: safe_float(row.get("edge_weight"), 0.0),
        reverse=True,
    )[:max_pairs]


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
        _matrix, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 3.0, 0.99)
    except cv2.error:
        return 0, None
    if mask is None:
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
        dhash_distance is not None
        and hist_correlation is not None
        and dhash_distance <= config.near_duplicate_hash_distance
        and hist_correlation >= config.near_duplicate_hist_correlation
    ):
        return "near_duplicate"
    if (
        inlier_ratio is not None
        and inlier_count >= config.geometry_min_inliers
        and inlier_ratio >= config.geometry_min_inlier_ratio
    ):
        return "geometry_supported"
    if score >= config.min_visual_score or good_match_count >= 30:
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
    geometry_matches = good_matches if len(good_matches) >= 8 else all_matches[: min(100, len(all_matches))]
    inlier_count, inlier_ratio = fundamental_inliers(cv2, np, features_a, features_b, geometry_matches)
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
        },
        "explanation": (
            "Each constrained match-pair candidate is checked with perceptual hash, "
            "HSV color histogram correlation, ORB feature matches, and optional "
            "fundamental-matrix RANSAC inliers. The table is used for diagnostics "
            "and ablation planning; it does not filter COLMAP pairs automatically."
        ),
    }
    return summary


def analyze_image_similarity_edges(
    images_dir: Path,
    edge_rows: Sequence[Dict[str, str]],
    output_csv: Path,
    summary_path: Optional[Path] = None,
    config: Optional[SimilarityConfig] = None,
) -> Dict[str, Any]:
    config = config or SimilarityConfig()
    selected_rows = select_rows(edge_rows, config.max_pairs)
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

    orb = cv2.ORB_create(nfeatures=max(100, config.orb_features))
    feature_cache: Dict[str, ImageFeatures] = {}

    def features_for(image_name: str) -> ImageFeatures:
        cached = feature_cache.get(image_name)
        if cached is not None:
            return cached
        features = read_image_features(
            image_name=image_name,
            images_dir=images_dir,
            cv2=cv2,
            orb=orb,
            config=config,
        )
        feature_cache[image_name] = features
        return features

    rows: List[Dict[str, Any]] = []
    for row in selected_rows:
        image_a = row.get("image_a", "")
        image_b = row.get("image_b", "")
        rows.append(
            analyze_pair(
                row=row,
                features_a=features_for(image_a),
                features_b=features_for(image_b),
                cv2=cv2,
                np=np,
                config=config,
            )
        )

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
) -> Dict[str, Any]:
    if edge_rows is None:
        _fields, edge_rows = read_csv(workspace_dir / "match_graph_edges.csv")
    config = SimilarityConfig(
        max_pairs=max_pairs,
        max_image_size=max_image_size,
        orb_features=orb_features,
        min_visual_score=min_visual_score,
    )
    return analyze_image_similarity_edges(
        images_dir=workspace_dir / "images",
        edge_rows=edge_rows,
        output_csv=workspace_dir / "image_similarity_edges.csv",
        summary_path=workspace_dir / "image_similarity_summary.json",
        config=config,
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
            min_visual_score=args.min_visual_score,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
