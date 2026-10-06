"""Associate video and sensor clocks with explicitly uncertain matching windows.

Embedded recording timestamps are device-clock evidence, not synchronization.
No frame receives a sensor sample, camera pose, checkpoint, or WiFi fingerprint.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess


FUZZY_FIELDS = ["video_sensor_fuzzy_source_sha256", "video_sensor_window_start_ms",
                "video_sensor_window_end_ms", "video_sensor_time_uncertainty_s"]
DECODED_TIME_SOURCES = {"opencv_decoded_timestamp", "ffprobe_frame_pts", "ffmpeg_pts", "decoded_pts"}
MAX_CLOCK_RESIDUAL_MS = 100.
MAX_MOV_METADATA_BYTES = 64 * 1024 * 1024


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _read_csv(path):
    if not path.is_file():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _identity(path):
    if not path.is_file():
        return None
    raw = path.read_bytes()
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _epoch_ms(text):
    try:
        # Apple records offsets as +0800; Python 3.10 requires +08:00.
        normalized = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", str(text).replace("Z", "+00:00"))
        value = datetime.fromisoformat(normalized)
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        result = value.timestamp() * 1000
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _boxes(data, start=0):
    """Yield bounded ISO BMFF atoms; malformed lengths never escape metadata."""
    while start + 8 <= len(data):
        size, name = struct.unpack_from(">I4s", data, start)
        header = 8
        if size == 1:
            if start + 16 > len(data):
                return
            size = struct.unpack_from(">Q", data, start + 8)[0]
            header = 16
        elif size == 0:
            size = len(data) - start
        if size < header or start + size > len(data):
            return
        yield name, data[start + header:start + size]
        start += size


def _quicktime_tags(meta):
    tags = {}
    # ISO meta has a FullBox header; older QuickTime meta may omit it.
    for offset in (4, 0):
        boxes = dict(_boxes(meta, offset))
        keys, values = boxes.get(b"keys", b""), boxes.get(b"ilst", b"")
        if len(keys) < 8 or not values:
            continue
        names = {}
        for index, (_namespace, key) in enumerate(_boxes(keys, 8), 1):
            names[index] = key.decode("utf-8", "replace")
        for index, item in _boxes(values):
            name = names.get(int.from_bytes(index, "big"))
            if name:
                for kind, content in _boxes(item):
                    if kind == b"data" and len(content) >= 8:
                        tags[name] = content[8:].decode("utf-8", "replace").rstrip("\x00")
    return tags


def _mov_metadata(path):
    """Read only MOV/MP4 container metadata, seeking over compressed media."""
    size = path.stat().st_size
    moov = None
    with path.open("rb") as handle:
        while handle.tell() + 8 <= size:
            position = handle.tell()
            length, kind = struct.unpack(">I4s", handle.read(8))
            header = 8
            if length == 1:
                raw = handle.read(8)
                if len(raw) != 8:
                    break
                length, header = struct.unpack(">Q", raw)[0], 16
            elif length == 0:
                length = size - position
            if length < header or position + length > size:
                break
            if kind == b"moov":
                if length - header > MAX_MOV_METADATA_BYTES:
                    raise ValueError("MOV metadata exceeds the bounded read size")
                moov = handle.read(length - header)
                break
            handle.seek(position + length)
    if moov is None:
        return None
    tags, movie_epoch, duration = {}, None, None
    pending = [moov]
    while pending:
        for kind, payload in _boxes(pending.pop()):
            if kind in (b"moov", b"udta"):
                pending.append(payload)
            elif kind == b"meta":
                tags.update(_quicktime_tags(payload))
            elif kind == b"mvhd":
                try:
                    if payload[0] == 0:
                        created, _modified, timescale, ticks = struct.unpack_from(">IIII", payload, 4)
                    elif payload[0] == 1:
                        created, _modified, timescale, ticks = struct.unpack_from(">QQIQ", payload, 4)
                    else:
                        continue
                    stamp = datetime(1904, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=created)
                    if created and stamp.year >= 1970:
                        movie_epoch = stamp.timestamp() * 1000
                    if timescale and ticks:
                        duration = ticks / timescale
                except (IndexError, struct.error, OverflowError):
                    continue
    explicit = tags.get("com.apple.quicktime.creationdate")
    explicit_epoch = _epoch_ms(explicit) if explicit else None
    epoch = explicit_epoch if explicit_epoch is not None else movie_epoch
    if epoch is None or duration is None or not math.isfinite(duration) or duration <= 0:
        return None
    return dict(video_start_epoch_ms=epoch, duration_s=duration,
                evidence=dict(source="embedded_mov_atoms", timestamp_basis=("com.apple.quicktime.creationdate"
                              if explicit_epoch is not None else "mvhd_creation_time_utc"),
                              embedded_creationdate=explicit, mvhd_creation_epoch_ms=movie_epoch,
                              creationdate_status=("valid" if explicit_epoch is not None else
                                                   "invalid_used_mvhd" if explicit else "absent_used_mvhd"),
                              device_make=tags.get("com.apple.quicktime.make", ""),
                              device_model=tags.get("com.apple.quicktime.model", ""),
                              metadata_sha256=hashlib.sha256(moov).hexdigest(), video_size_bytes=size,
                              cross_device_clock_offset_verified=False))


def read_video_recording_metadata(path, ffmpeg_exe=None):
    """Return embedded recording time/duration or None; never use filesystem dates."""
    path = Path(path)
    candidates = []
    if ffmpeg_exe:
        candidates.append(Path(ffmpeg_exe).with_name("ffprobe" + Path(ffmpeg_exe).suffix))
    found = shutil.which("ffprobe")
    if found:
        candidates.append(Path(found))
    for executable in dict.fromkeys(candidates):
        if not executable.is_file():
            continue
        try:
            result = subprocess.run([str(executable), "-v", "error", "-show_entries",
                                     "format=duration:format_tags:stream=duration:stream_tags", "-of", "json", str(path)],
                                    capture_output=True, text=True, check=True, timeout=30)
            payload = json.loads(result.stdout)
            items = [payload.get("format", {}), *payload.get("streams", [])]
            duration = next((n for item in items if math.isfinite(n := _number(item.get("duration"))) and n > 0), None)
            tags = {key: value for item in reversed(items) for key, value in item.get("tags", {}).items()}
            preferred = tags.get("com.apple.quicktime.creationdate")
            preferred_epoch = _epoch_ms(preferred) if preferred else None
            tag = "com.apple.quicktime.creationdate" if preferred_epoch is not None else "creation_time"
            epoch = preferred_epoch if preferred_epoch is not None else _epoch_ms(tags.get(tag))
            if epoch is not None and duration is not None:
                return dict(video_start_epoch_ms=epoch, duration_s=duration,
                            evidence=dict(source="embedded_ffprobe_tags", timestamp_basis=tag,
                                          embedded_creationdate=tags.get(tag),
                                          original_quicktime_creationdate=preferred,
                                          creationdate_status=("valid" if preferred_epoch is not None else
                                                               "invalid_used_creation_time" if preferred else "absent_used_creation_time"),
                                          device_make=tags.get("com.apple.quicktime.make", ""),
                                          device_model=tags.get("com.apple.quicktime.model", ""),
                                          metadata_sha256=hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
                                          video_size_bytes=path.stat().st_size, cross_device_clock_offset_verified=False))
        except (OSError, ValueError, subprocess.SubprocessError, TypeError):
            pass
    try:
        return _mov_metadata(path)
    except (OSError, ValueError, struct.error):
        return None


def has_fuzzy_video_sensor_window(row):
    """Validate window syntax only; consumers also authenticate the context hash/model."""
    start, end = _number(row.get("video_sensor_window_start_ms")), _number(row.get("video_sensor_window_end_ms"))
    uncertainty, video = _number(row.get("video_sensor_time_uncertainty_s")), _number(row.get("video_time_s"))
    digest = str(row.get("video_sensor_fuzzy_source_sha256", ""))
    return (row.get("video_sensor_sync_status") == "fuzzy_time_window" and bool(str(row.get("session", "")).strip())
            and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
            and all(math.isfinite(value) for value in (start, end, uncertainty, video))
            and 0 <= start < end and 0 < uncertainty <= 300 and end - start <= uncertainty * 2000 + 1e-3
            and video >= 0 and row.get("video_time_source") in DECODED_TIME_SOURCES)


def _session_clocks(rows):
    grouped = defaultdict(list)
    for row in rows:
        session = str(row.get("session", "")).strip()
        elapsed, epoch = _number(row.get("elapsed_ts")), _number(row.get("epoch_ts"))
        if session and all(math.isfinite(n) and n >= 0 for n in (elapsed, epoch)):
            grouped[session].append((elapsed, epoch))
    clocks, rejected = {}, {}
    for session, samples in grouped.items():
        if len({item[0] for item in samples}) < 2:
            rejected[session] = "insufficient_clock_samples"
            continue
        offset = statistics.median(epoch - elapsed for elapsed, epoch in samples)
        residual = max(abs(epoch - elapsed - offset) for elapsed, epoch in samples)
        if residual > MAX_CLOCK_RESIDUAL_MS:
            rejected[session] = "unstable_epoch_elapsed_clock"
            continue
        clocks[session] = dict(epoch_elapsed_offset_ms=offset, epoch_elapsed_max_residual_ms=residual,
                               sensor_start_ms=min(s[0] for s in samples), sensor_end_ms=max(s[0] for s in samples),
                               clock_sample_count=len(samples))
    return clocks, rejected


def annotate_fuzzy_video_sensor_priors(processed_dir, root, video_dir, *, uncertainty_s=30., offset_s=0., ffmpeg_exe=None):
    """Write only uncertain windows for clips with one overlapping sensor session."""
    from video_sensor_fusion import VIDEO_SENSOR_PRIOR_FIELDS, clear_video_sensor_priors

    uncertainty_s, offset_s = _number(uncertainty_s), _number(offset_s)
    if not math.isfinite(uncertainty_s) or not 0 < uncertainty_s <= 300:
        raise ValueError("Fuzzy time uncertainty must be finite and in (0,300] seconds")
    if not math.isfinite(offset_s) or abs(offset_s) > 3600:
        raise ValueError("Fuzzy time offset must be finite and within +/-3600 seconds")
    processed_dir, root = Path(processed_dir).resolve(), Path(root).resolve()
    video_dir = Path(video_dir)
    video_dir = video_dir.resolve() if video_dir.is_absolute() else (root / video_dir).resolve()
    fields, frames = _read_csv(processed_dir / "video_frames_manifest.csv")
    route_path, wifi_path = processed_dir / "pdr_trajectory_contour.csv", processed_dir / "wifi_pdr_features.csv"
    identities = {p.name: _identity(p) for p in (route_path, wifi_path)}
    _, route = _read_csv(route_path)
    _, wifi = _read_csv(wifi_path)
    clocks, rejected = _session_clocks([*route, *wifi])
    models, video_status = {}, {}
    for name in sorted({str(frame.get("video_name", "")) for frame in frames}):
        path = (video_dir / name).resolve()
        if not name or not path.is_relative_to(video_dir) or not path.is_file():
            video_status[name] = "video_file_unavailable"
            continue
        metadata = read_video_recording_metadata(path, ffmpeg_exe)
        if not metadata:
            video_status[name] = "embedded_recording_time_unavailable"
            continue
        candidates = []
        for session, clock in clocks.items():
            offset = metadata["video_start_epoch_ms"] - clock["epoch_elapsed_offset_ms"] + offset_s * 1000
            first, last = offset - uncertainty_s * 1000, offset + (metadata["duration_s"] + uncertainty_s) * 1000
            if max(first, clock["sensor_start_ms"]) < min(last, clock["sensor_end_ms"]):
                candidates.append(dict(session=session, offset_ms=offset, scale=1., **clock,
                                       video_duration_s=metadata["duration_s"],
                                       metadata_evidence={**metadata["evidence"],
                                                          "video_start_epoch_ms": metadata["video_start_epoch_ms"]}))
        if len(candidates) != 1:
            video_status[name] = "ambiguous_sensor_sessions" if candidates else "no_overlapping_sensor_session"
            continue
        models[name], video_status[name] = candidates[0], "fuzzy_time_window"
    if any(_identity(processed_dir / name) != value for name, value in identities.items()):
        raise ValueError("Sensor tables changed while preparing fuzzy matching windows")
    context = dict(schema_version=1, method="fuzzy_time_window", models=models, route_samples=route, wifi_scans=wifi,
                   uncertainty_s=uncertainty_s, offset_adjustment_s=offset_s, temporal_uncertainty_is_assumed=True,
                   input_identities=identities, video_status=video_status, rejected_sensor_clocks=rejected,
                   interpretation="Assumed cross-device clock uncertainty supplies soft visual matching candidates only; not synchronization or camera poses.")
    context_bytes = (json.dumps(context, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(context_bytes).hexdigest()
    output = []
    for frame in frames:
        row = clear_video_sensor_priors(frame, "fuzzy_time_window_unavailable")
        row.update({field: "" for field in FUZZY_FIELDS})
        model = models.get(str(frame.get("video_name", "")))
        timestamp = _number(frame.get("video_time_s"))
        if model is None:
            row["video_sensor_sync_status"] = video_status.get(str(frame.get("video_name", "")), "fuzzy_time_window_unavailable")
        elif not math.isfinite(timestamp) or timestamp < 0 or frame.get("video_time_source") not in DECODED_TIME_SOURCES:
            row["video_sensor_sync_status"] = "decoded_video_timestamp_unavailable"
        elif timestamp > model["video_duration_s"] + 1e-6:
            row["video_sensor_sync_status"] = "outside_video_interval"
        else:
            center = model["offset_ms"] + timestamp * 1000
            start = max(model["sensor_start_ms"], center - uncertainty_s * 1000)
            end = min(model["sensor_end_ms"], center + uncertainty_s * 1000)
            if start >= end:
                row["video_sensor_sync_status"] = "outside_observed_sensor_interval"
            else:
                row.update(video_sensor_sync_status="fuzzy_time_window", session=model["session"],
                           video_sensor_fuzzy_source_sha256=digest, video_sensor_window_start_ms=round(start, 6),
                           video_sensor_window_end_ms=round(end, 6), video_sensor_time_uncertainty_s=uncertainty_s,
                           bind_type="fuzzy_video_sensor_candidates")
        output.append(row)
    summary = dict(status="annotated" if any(has_fuzzy_video_sensor_window(r) for r in output) else "no_fuzzy_bindings",
                   schema_version=1, method="fuzzy_time_window", frame_count=len(output),
                   fuzzy_frame_count=sum(has_fuzzy_video_sensor_window(r) for r in output), synchronized_frame_count=0,
                   models=models, status_counts=dict(Counter(r["video_sensor_sync_status"] for r in output)),
                   video_status=video_status, rejected_sensor_clocks=rejected, temporal_uncertainty_is_assumed=True,
                   uncertainty_s=uncertainty_s, offset_adjustment_s=offset_s, context_sha256=digest,
                   requires_visual_geometric_verification=True, camera_orientation_from_phone=False)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "video_sensor_fuzzy_context.json").write_bytes(context_bytes)
    for name in ("video_sensor_sync.csv", "video_sensor_sync_summary.json"):
        (processed_dir / name).unlink(missing_ok=True)
    if fields:
        output_fields = list(dict.fromkeys([*fields, *VIDEO_SENSOR_PRIOR_FIELDS, *FUZZY_FIELDS]))
        with (processed_dir / "video_frames_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(output)
    (processed_dir / "video_sensor_fuzzy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
