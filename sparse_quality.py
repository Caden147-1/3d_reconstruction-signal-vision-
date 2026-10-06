"""Read-only camera coverage checks before expensive dense reconstruction.

A passing result describes retention of the supplied images and video timeline.
It is not a claim about pose accuracy, scene coverage or surface completeness.
No point-cloud read or reconstruction package is needed by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from statistics import median


DEFAULT_POLICY = {
    "min_registered_images": 3,
    "min_images_for_ratio": 20,
    "min_registered_ratio": 0.75,
    "min_clip_frames": 20,
    "min_clip_registered_ratio": 0.75,
    "min_gap_tolerance_s": 10.0,
    "max_gap_fraction": 0.02,
    "input_gap_multiplier": 3.0,
}


def _number(value, label, *, integer=False, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite {'integer' if integer else 'number'}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < minimum or (integer and int(value) != value):
        raise ValueError(f"{label} must be finite and >= {minimum:g}" + (" (integer)" if integer else ""))
    return int(value) if integer else float(value)


def validate_policy(policy=None):
    """Return a complete validated policy; reject misspellings and NaN values."""
    if policy is not None and not isinstance(policy, Mapping):
        raise ValueError("Sparse quality policy must be a mapping")
    unknown = set(policy or {}) - set(DEFAULT_POLICY)
    if unknown:
        raise ValueError("Unknown sparse quality policy fields: " + ", ".join(sorted(map(str, unknown))))
    result = dict(DEFAULT_POLICY)
    result.update(policy or {})
    for key in ("min_registered_images", "min_images_for_ratio", "min_clip_frames"):
        result[key] = _number(result[key], key, integer=True, minimum=1)
    for key in ("min_registered_ratio", "min_clip_registered_ratio", "max_gap_fraction"):
        result[key] = _number(result[key], key)
        if not 0 < result[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    result["min_gap_tolerance_s"] = _number(result["min_gap_tolerance_s"], "min_gap_tolerance_s")
    if result["min_gap_tolerance_s"] <= 0:
        raise ValueError("min_gap_tolerance_s must be positive")
    result["input_gap_multiplier"] = _number(result["input_gap_multiplier"], "input_gap_multiplier", minimum=1)
    return result


def _name(value):
    return str(value).replace("\\", "/") if value is not None else ""


def _metadata_index(metadata):
    if metadata is None:
        return {}, []
    errors, result = [], {}
    if isinstance(metadata, Mapping):
        records = metadata.items()
    elif isinstance(metadata, (list, tuple)):
        records = ((row.get("image_file") if isinstance(row, Mapping) else None, row) for row in metadata)
    else:
        return {}, ["Image metadata must be a mapping or a list of rows"]
    for key, row in records:
        name = _name(key)
        if not name or not isinstance(row, Mapping):
            errors.append("Image metadata has an unnamed image or a non-object row")
        elif name in result:
            errors.append(f"Duplicate image metadata: {name}")
        else:
            result[name] = row
    return result, errors


def _video_groups(metadata):
    groups, errors = {}, []
    for name, row in metadata.items():
        clip = row.get("video_name")
        if not clip:
            if str(row.get("source_type", "")).startswith("video"):
                errors.append(f"Video image has no video_name: {name}")
            continue
        if not isinstance(clip, str):
            errors.append(f"Invalid video_name for image: {name}")
            continue
        entry = groups.setdefault(clip, {"names": [], "times": []})
        entry["names"].append(name)
        try:
            raw = row.get("video_time_s")
            stamp = float(raw) if not isinstance(raw, bool) else float("nan")
            if not math.isfinite(stamp) or stamp < 0:
                raise ValueError("invalid timestamp")
        except (TypeError, ValueError, OverflowError):
            errors.append(f"Missing or invalid video_time_s: {name}")
            continue
        entry["times"].append((stamp, name))
    return groups, errors


def _time_baseline(times):
    stamps = sorted(stamp for stamp, _ in times)
    gaps = [right - left for left, right in zip(stamps, stamps[1:])]
    positive = [gap for gap in gaps if gap > 0]
    return {
        "timed_input_frames": len(stamps),
        "input_time_span_s": stamps[-1] - stamps[0] if stamps else None,
        "input_max_gap_s": max(gaps, default=0.0) if stamps else None,
        "input_median_gap_s": median(positive) if positive else (0.0 if stamps else None),
    }


def summarize_coverage(input_names, registered_names, metadata):
    """Build a model_summary-compatible report using image names only.

    The denominator is exactly the supplied input list, not every metadata row.
    Unrecognized registered names are reported separately and cannot increase
    registration coverage. Lost first/last video intervals count as gaps.
    """
    inputs = {_name(name) for name in input_names}
    registered = {_name(name) for name in registered_names}
    index, errors = _metadata_index(metadata)
    if "" in inputs or "" in registered:
        errors.append("Input or registered image list contains an empty name")
    scoped = {name: row for name, row in index.items() if name in inputs}
    groups, video_errors = _video_groups(scoped)
    errors.extend(video_errors)
    coverage = {}
    for clip, group in groups.items():
        baseline = _time_baseline(group["times"])
        all_times = sorted(stamp for stamp, _ in group["times"])
        used_times = sorted(stamp for stamp, name in group["times"] if name in registered)
        anchors = [all_times[0], *used_times, all_times[-1]] if all_times else []
        coverage[clip] = {
            "input_frames": len(group["names"]),
            "registered_frames": sum(name in registered for name in group["names"]),
            "max_uncovered_gap_s": max((b - a for a, b in zip(anchors, anchors[1:])), default=0.0) if all_times else None,
            **baseline,
        }
    return {
        "input_images": len(inputs),
        "registered_images": len(inputs & registered),
        "unexpected_registered_images": len(registered - inputs),
        "unexpected_registered_names": sorted(registered - inputs)[:20],
        "metadata_input_images": len(scoped),
        "coverage_errors": errors,
        "video_coverage": coverage,
    }


def assess_sparse_quality(summary, metadata=None, *, policy=None):
    """Assess supplied camera retention, returning failures suitable for a gate.

    Accepts either summarize_coverage output or sparse_optimization.model_summary
    with its original metadata. Bad summary evidence fails closed. A malformed
    policy raises ValueError rather than silently weakening the gate.
    """
    thresholds = validate_policy(policy)
    failures, warnings, codes = [], [], []
    report = {
        "schema_version": 1,
        "scope": "supplied_image_and_video_timeline_coverage",
        "policy": thresholds,
        "ready_for_dense": False,
        "status": "failed",
        "failures": failures,
        "failure_codes": codes,
        "warnings": warnings,
        "global_coverage": {},
        "video_coverage": {},
        "interpretation": "Coverage passing does not establish accurate camera poses or complete scene geometry.",
    }

    def fail(code, message):
        codes.append(code)
        failures.append(message)

    if not isinstance(summary, Mapping):
        fail("invalid_summary", "Sparse quality summary must be an object")
        return report
    index, metadata_errors = _metadata_index(metadata)
    # Adapted summaries have already scoped metadata to the actual input list.
    adapted = "metadata_input_images" in summary
    groups, video_errors = _video_groups(index) if not adapted else ({}, [])
    for error in metadata_errors + video_errors:
        fail("invalid_metadata", error)
    supplied_errors = summary.get("coverage_errors", [])
    if not isinstance(supplied_errors, list) or any(not isinstance(error, str) for error in supplied_errors):
        fail("invalid_summary", "coverage_errors must be a list of strings")
    else:
        for error in supplied_errors:
            fail("invalid_metadata", error)
    try:
        expected = _number(summary.get("input_images", len(index)), "input_images", integer=True)
        registered = _number(summary.get("registered_images"), "registered_images", integer=True)
        unexpected = _number(summary.get("unexpected_registered_images", 0), "unexpected_registered_images", integer=True)
    except ValueError as exc:
        fail("invalid_counts", str(exc))
        return report
    ratio = registered / expected if expected else None
    ratio_applies = expected >= thresholds["min_images_for_ratio"]
    report["global_coverage"] = {
        "input_images": expected, "registered_images": registered,
        "registered_ratio": ratio, "ratio_gate_applies": ratio_applies,
        "min_registered_ratio": thresholds["min_registered_ratio"],
        "unexpected_registered_images": unexpected,
    }
    if not expected:
        fail("no_input_images", "No input images are available to assess sparse coverage")
    if registered > expected or unexpected:
        fail("unexpected_registered_images", "Registered camera names/counts exceed the supplied input image set")
    if registered < thresholds["min_registered_images"]:
        fail("too_few_registered_images", f"Only {registered} images registered; at least {thresholds['min_registered_images']} are required")
    if ratio_applies and ratio < thresholds["min_registered_ratio"]:
        fail("low_global_registration", f"Registered {registered}/{expected} images ({ratio:.1%}); required {thresholds['min_registered_ratio']:.0%}")
    elif expected and not ratio_applies and registered < expected:
        warnings.append(f"Small input set: {registered}/{expected} registered; the percentage gate is not applied below {thresholds['min_images_for_ratio']} images")
    if adapted:
        try:
            metadata_count = _number(summary["metadata_input_images"], "metadata_input_images", integer=True)
            if metadata_count > expected:
                fail("invalid_metadata", "Metadata image count exceeds the supplied input set")
            elif metadata_count < expected:
                warnings.append(f"Metadata is unavailable for {expected - metadata_count} input images; their video timeline is not evaluated")
        except ValueError as exc:
            fail("invalid_metadata", str(exc))
    videos = summary.get("video_coverage", {})
    if not isinstance(videos, Mapping):
        fail("invalid_video_coverage", "video_coverage must be an object")
        return report
    for clip in sorted(set(videos) | set(groups), key=str):
        value = videos.get(clip)
        if not isinstance(clip, str) or not isinstance(value, Mapping):
            fail("invalid_video_coverage", f"Missing or invalid coverage summary for video {clip!r}")
            continue
        try:
            count = _number(value.get("input_frames"), f"{clip}: input_frames", integer=True, minimum=1)
            used = _number(value.get("registered_frames"), f"{clip}: registered_frames", integer=True)
            gap = _number(value.get("max_uncovered_gap_s"), f"{clip}: max_uncovered_gap_s")
            if used > count or count > expected or used > registered:
                raise ValueError(f"{clip}: video frame counts are inconsistent with input/registered image counts")
            baseline_keys = ("timed_input_frames", "input_time_span_s", "input_max_gap_s", "input_median_gap_s")
            if any(key in value for key in baseline_keys):
                baseline = {key: value.get(key) for key in baseline_keys}
            elif clip in groups:
                if count != len(groups[clip]["names"]):
                    raise ValueError(f"{clip}: summary input frame count disagrees with metadata")
                baseline = _time_baseline(groups[clip]["times"])
            else:
                baseline = None
            limit = None
            if baseline is not None:
                timed = _number(baseline["timed_input_frames"], f"{clip}: timed_input_frames", integer=True)
                span = _number(baseline["input_time_span_s"], f"{clip}: input_time_span_s")
                original_gap = _number(baseline["input_max_gap_s"], f"{clip}: input_max_gap_s")
                median_gap = _number(baseline["input_median_gap_s"], f"{clip}: input_median_gap_s")
                if (timed != count or original_gap > span + 1e-6 or median_gap > original_gap + 1e-6
                        or gap > span + 1e-6 or gap + 1e-6 < original_gap):
                    raise ValueError(f"{clip}: invalid/incomplete video timeline evidence")
                if count > 1 and (span <= 0 or original_gap <= 0):
                    raise ValueError(f"{clip}: multiple video frames have no usable time span")
                if ((used == count and abs(gap - original_gap) > 1e-6)
                        or (used == 0 and abs(gap - span) > 1e-6)):
                    raise ValueError(f"{clip}: uncovered interval contradicts registered frame count")
                limit = max(thresholds["min_gap_tolerance_s"], span * thresholds["max_gap_fraction"],
                            median_gap * thresholds["input_gap_multiplier"], original_gap)
        except ValueError as exc:
            fail("invalid_video_coverage", str(exc))
            continue
        clip_ratio = used / count
        gate_applies = count >= thresholds["min_clip_frames"]
        entry = {
            "input_frames": count, "registered_frames": used,
            "registered_ratio": clip_ratio, "max_uncovered_gap_s": gap,
            "allowed_max_gap_s": limit, "gate_applies": gate_applies,
            "min_registered_ratio": thresholds["min_clip_registered_ratio"],
            "timeline_status": "evaluated" if baseline is not None else "not_evaluable",
            "failures": [],
            **(baseline or {}),
        }
        report["video_coverage"][clip] = entry
        if gate_applies and clip_ratio < thresholds["min_clip_registered_ratio"]:
            message = f"{clip}: registered {used}/{count} frames ({clip_ratio:.1%}); required {thresholds['min_clip_registered_ratio']:.0%}"
            entry["failures"].append(message)
            fail("low_clip_registration", message)
        if gate_applies and limit is not None and gap > limit + 1e-6:
            message = f"{clip}: uncovered video interval {gap:.3f}s exceeds {limit:.3f}s allowed by input sampling and clip duration"
            entry["failures"].append(message)
            fail("long_video_gap", message)
        if not gate_applies and used < count:
            warnings.append(f"{clip}: short input sequence ({count} frames), {used} registered; percentage/gap gates are not applied below {thresholds['min_clip_frames']} frames")
        if limit is None:
            warnings.append(f"{clip}: input timing baseline is unavailable; temporal gap cannot be assessed")
        entry["status"] = "failed" if entry["failures"] else "passed" if gate_applies and limit is not None else "limited_evidence"
    # Per-clip totals must not count the same images more than once.
    assessed = list(report["video_coverage"].values())
    if sum(item["input_frames"] for item in assessed) > expected or sum(item["registered_frames"] for item in assessed) > registered:
        fail("invalid_video_coverage", "Per-video frame totals exceed the supplied image counts")
    report["ready_for_dense"] = not failures
    report["status"] = "failed" if failures else "passed_with_warnings" if warnings else "passed"
    return report
