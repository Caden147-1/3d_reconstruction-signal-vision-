"""Infer ordered capture batches from camera-clock pauses and a soft count prior.

Camera times are never synchronized to WiFi/PDR. Weak or ambiguous evidence
keeps the original count-based grouping; inferred boundaries are not ground truth.
"""

from datetime import datetime
import math
import statistics
import struct


def capture_timing_diagnostics(groups):
    """Audit ordered (image name, camera time) groups without changing labels.

    Long pauses are review cues, not evidence of a different physical location.
    Camera-clock times must never be compared directly with sensor-clock times.
    """
    gaps = [b[1] - a[1] for members in groups.values()
            for a, b in zip(members, members[1:])
            if all(t is not None and math.isfinite(t) for t in (a[1], b[1]))
            and b[1] > a[1]]
    cadence = statistics.median(gaps) if gaps else None
    threshold = None
    if cadence is not None:
        mad = statistics.median(abs(gap - cadence) for gap in gaps)
        threshold = max(10 * cadence, cadence + 6 * 1.4826 * mad)
    reports = []
    for key, members in groups.items():
        missing = sum(t is None or not math.isfinite(t) for _, t in members)
        pauses, reversed_pairs = [], []
        for left, right in zip(members, members[1:]):
            if any(t is None or not math.isfinite(t) for t in (left[1], right[1])):
                continue
            gap = right[1] - left[1]
            if gap < 0:
                reversed_pairs.append([left[0], right[0]])
            elif threshold is not None and gap > threshold:
                pauses.append(dict(before=left[0], after=right[0], gap_seconds=gap))
        reports.append(dict(session=key[0], checkpoint=key[1], image_count=len(members),
                            missing_timestamp_count=missing, long_internal_pauses=pauses,
                            reversed_pairs=reversed_pairs))
    flagged = sum(bool(r['long_internal_pauses'] or r['reversed_pairs']) for r in reports)
    return dict(status='needs_review' if flagged else 'incomplete_timestamps' if any(
                    r['missing_timestamp_count'] for r in reports) else 'no_timing_flags' if gaps else 'unavailable',
                cadence_seconds=cadence, long_pause_threshold_seconds=threshold,
                flagged_checkpoint_count=flagged, checkpoints=reports,
                interpretation='Camera-clock consistency only; no flags does not verify photo-to-checkpoint assignments.')


def workspace_capture_diagnostics(workspace):
    """Read the workspace copies whose bytes are covered by quality provenance."""
    import csv
    from collections import defaultdict
    from colmap_workspace_check import contained_path

    manifest = workspace / 'metadata/images_manifest.csv'
    if not manifest.is_file():
        return dict(status='unavailable', reason='missing_manifest')
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        if row.get('source_type') == 'wifi_anchor_photo':
            grouped[(row.get('session', ''), row.get('global_point_index', ''))].append(row)
    groups, ambiguous = {}, []
    for key, members in sorted(grouped.items()):
        try:
            slots = [int(row['image_slot']) for row in members]
            if len(set(slots)) != len(slots) or min(slots) < 1:
                raise ValueError('Invalid slots')
        except (KeyError, TypeError, ValueError):
            ambiguous.append(dict(session=key[0], checkpoint=key[1]))
            continue
        ordered = sorted(zip(slots, members), key=lambda item: item[0])
        groups[key] = [(row['image_file'], jpeg_capture_time(
            contained_path(workspace / 'images', row['image_file']))) for _, row in ordered]
    result = capture_timing_diagnostics(groups)
    result.update(ambiguous_order_checkpoints=ambiguous,
                  visual_only_photo_count=sum(row.get('source_type') == 'visual_only_photo' for row in rows))
    if ambiguous:
        result['status'] = 'needs_review'
    return result


def jpeg_capture_time(path):
    """Read EXIF DateTimeOriginal without decoding pixels or adding dependencies."""
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return None
            for _ in range(256):
                if handle.read(1) != b"\xff":
                    return None
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in (b"", b"\xda", b"\xd9"):
                    return None
                if marker == b"\x01" or b"\xd0" <= marker <= b"\xd7":
                    continue
                size = struct.unpack(">H", handle.read(2))[0] - 2
                if size < 0:
                    return None
                data = handle.read(size)
                if len(data) != size:
                    return None
                if marker != b"\xe1" or not data.startswith(b"Exif\x00\x00"):
                    continue
                tiff = data[6:]
                order = {b"II": "<", b"MM": ">"}.get(tiff[:2])
                if order is None or struct.unpack_from(order + "H", tiff, 2)[0] != 42:
                    return None

                def directory(offset):
                    count = struct.unpack_from(order + "H", tiff, offset)[0]
                    if offset + 2 + 12 * count > len(tiff):
                        raise ValueError("Truncated EXIF directory")
                    return {struct.unpack_from(order + "H", tiff, offset + 2 + i * 12)[0]: offset + 2 + i * 12
                            for i in range(count)}

                def text(entries, tag):
                    pos = entries.get(tag)
                    if pos is None:
                        return ""
                    kind, count = struct.unpack_from(order + "HI", tiff, pos + 2)
                    if kind != 2 or not 0 < count <= 64:
                        return ""
                    start = pos + 8 if count <= 4 else struct.unpack_from(order + "I", tiff, pos + 8)[0]
                    if start + count > len(tiff):
                        return ""
                    return tiff[start:start + count].rstrip(b"\x00").decode("ascii")

                entries = directory(struct.unpack_from(order + "I", tiff, 4)[0])
                pointer = entries.get(0x8769)
                if pointer is None or struct.unpack_from(order + "HI", tiff, pointer + 2) != (4, 1):
                    return None
                entries = directory(struct.unpack_from(order + "I", tiff, pointer + 8)[0])
                stamp = datetime.strptime(text(entries, 0x9003), "%Y:%m:%d %H:%M:%S")
                subsecond = text(entries, 0x9291)
                fraction = float("0." + subsecond) if subsecond.isascii() and subsecond.isdigit() else 0.0
                return (stamp - datetime(1970, 1, 1)).total_seconds() + fraction
    except (OSError, ValueError, struct.error):
        return None
    return None


def infer_capture_boundaries(times, point_count, expected=20, tolerance=None):
    """Dynamic programming over ordered groups, keeping two paths per endpoint.

    Pause significance is relative to the camera's normal shooting cadence.
    Ambiguity is a score margin, not a calibrated probability. A tied solution
    cannot silently rewrite the fixed-count baseline.
    """
    tolerance = max(0, expected // 2) if tolerance is None else tolerance
    if expected <= 0 or point_count < 0 or not 0 <= tolerance < expected:
        raise ValueError("Grouping requires a positive expected count and 0 <= tolerance < expected")
    count = len(times)
    baseline = [min(count, (i + 1) * expected) for i in range(point_count)]
    result = {"status": "fallback", "reason": "insufficient_capture_times", "boundaries": baseline,
              "expected_count": expected, "count_tolerance": tolerance,
              "timestamp_count": sum(t is not None and math.isfinite(t) for t in times),
              "confidence_basis": "relative_pause_evidence_and_best_alternative_score_margin"}
    if not point_count or not tolerance or count < point_count * (expected - tolerance):
        return {**result, "reason": "fixed_mode_or_insufficient_photos"}
    valid = [(i, t) for i, t in enumerate(times) if t is not None and math.isfinite(t)]
    if any(b[1] < a[1] for a, b in zip(valid, valid[1:])):
        return {**result, "reason": "camera_clock_or_filename_order_reversed"}
    gaps = {i: times[i] - times[i - 1] for i in range(1, count)
            if times[i] is not None and times[i - 1] is not None
            and math.isfinite(times[i]) and math.isfinite(times[i - 1]) and times[i] > times[i - 1]}
    if len(gaps) < max(4, point_count):
        return result
    cadence = statistics.median(gaps.values())
    mad = statistics.median(abs(gap - cadence) for gap in gaps.values())
    threshold = max(3 * cadence, cadence + 3 * 1.4826 * mad)
    evidence = {i: min(6.0, math.log1p(gap / cadence)) for i, gap in gaps.items() if gap > threshold}
    result.update(cadence_seconds=cadence, pause_threshold_seconds=threshold,
                  pause_indices=sorted(evidence))
    if not evidence:
        return {**result, "reason": "no_distinct_capture_pauses"}
    states = {0: [(0.0, ())]}
    for group in range(point_count):
        updated = {}
        for start, paths in states.items():
            for length in range(expected - tolerance, expected + tolerance + 1):
                end = start + length
                remaining = point_count - group - 1
                if end > count or end + remaining * (expected - tolerance) > count:
                    continue
                # End of input is a possible final endpoint, not a measured pause.
                reward = 2.0 if end == count and not remaining else evidence.get(end, 0.0)
                penalty = .75 * ((length - expected) / tolerance) ** 2
                proposals = updated.setdefault(end, [])
                proposals.extend((score + reward - penalty, path + (end,)) for score, path in paths)
                proposals.sort(key=lambda item: (-item[0], item[1]))
                del proposals[2:]
        states = updated
    ranked = sorted((entry for paths in states.values() for entry in paths), key=lambda item: (-item[0], item[1]))
    if not ranked:
        return {**result, "reason": "no_feasible_partition"}
    score, boundaries = ranked[0]
    margin = score - ranked[1][0] if len(ranked) > 1 else None
    baseline_score = sum((2.0 if end == count and i == point_count - 1 else evidence.get(end, 0.0))
                         for i, end in enumerate(baseline))
    changed = [i for i, end in enumerate(boundaries) if end != baseline[i]]
    result.update(proposed_boundaries=list(boundaries), score=score, baseline_score=baseline_score,
                  alternative_score_margin=margin,
                  unexplained_pause_indices=sorted(set(evidence) - set(boundaries)),
                  prior_limit_reached=any(end-start in (expected-tolerance, expected+tolerance)
                                          for start, end in zip((0,) + boundaries[:-1], boundaries)),
                  boundary_evidence=[{"after_image_index": end, "gap_seconds": gaps.get(end),
                                      "distinct_pause": end in evidence, "end_of_input": end == count}
                                     for end in boundaries])
    if not changed:
        return {**result, "reason": "default_boundaries_retained"}
    if any(boundaries[i] not in evidence and boundaries[i] != count for i in changed):
        return {**result, "reason": "changed_boundary_without_pause_evidence"}
    if score - baseline_score < .5 or (margin is not None and margin < .35):
        return {**result, "reason": "ambiguous_boundary_evidence"}
    return {**result, "status": "adjusted", "reason": "distinct_capture_pauses", "boundaries": list(boundaries)}


def fuzzy_checkpoint_groups(images, point_count, expected, tolerance=None):
    deviation = expected // 2 if tolerance is None else tolerance
    # One extra image supplies the gap after the latest feasible endpoint.
    # The artificial read limit cannot be mistaken for the end of a group.
    limit = point_count * (expected + deviation) + 1 if point_count else 0
    times = [jpeg_capture_time(path) for path in images[:limit]]
    evidence = infer_capture_boundaries(times, point_count, expected, tolerance)
    groups, start = [], 0
    for end in evidence["boundaries"]:
        groups.append(list(images[start:end]))
        start = end
    return groups, list(images[start:]), {
        "method": "fuzzy_capture_gaps" if evidence["status"] == "adjusted" else "fixed_count",
        "source": "camera_EXIF_DateTimeOriginal_relative_gaps", "inference": evidence,
        "interpretation": "Count prior plus camera-clock pauses; no WiFi clock synchronization. Inferred grouping requires review and is not a measured position.",
    }
