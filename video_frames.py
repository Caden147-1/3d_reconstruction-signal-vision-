"""Extract route video frames and bind them as weak visual base images."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from process_control import managed_process
from frame_io import read_frame, write_frame
from video_decode import open_video
from progress_output import WorkProgress
from gpu_image_ops import laplacian_variance


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
VIDEO_FRAME_PROVENANCE_FIELDS = ["video_raw_frame_index"]
VIDEO_QUALITY_FIELDS = ["quality_sharpness_640", "quality_contrast_640"]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
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


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def natural_sort_key(path: Path) -> List[Tuple[int, Any]]:
    parts = re.split(r"(\d+)", path.name.lower())
    key: List[Tuple[int, Any]] = []
    for part in parts:
        key.append((0, int(part)) if part.isdigit() else (1, part))
    return key


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "video"


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def find_video_files(video_dir: Path) -> List[Path]:
    if not video_dir.exists() or not video_dir.is_dir():
        return []
    return sorted(
        (path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=natural_sort_key,
    )


def resolve_ffmpeg(root: Path, ffmpeg_exe: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if ffmpeg_exe is not None:
        candidates.append(ffmpeg_exe if ffmpeg_exe.is_absolute() else root / ffmpeg_exe)

    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))

    if os.name == 'nt':
        candidates.extend([
            root / "ffmpeg.exe",
            root / "ffmpeg" / "bin" / "ffmpeg.exe",
            root / "colmap-x64-windows-cuda" / "bin" / "ffmpeg.exe",
        ])
    else:
        candidates.extend([root / 'ffmpeg', root / 'ffmpeg' / 'bin' / 'ffmpeg'])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def extract_with_ffmpeg(
    ffmpeg_exe: Path,
    video_path: Path,
    output_dir: Path,
    frame_interval_s: float,
    max_frames: int,
    decode_threads: int = 0,
) -> Tuple[str, str]:
    from compute_resources import worker_count
    threads = str(worker_count(decode_threads))
    fps = 1.0 / frame_interval_s
    command = [
        str(ffmpeg_exe),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats", "-progress", "pipe:1",
        "-y",
        "-threads", threads,
        "-i",
        str(video_path),
        "-threads", threads,
        "-filter_threads", threads,
        "-vf",
        f"fps={fps:.12g}:start_time=0",
        "-q:v",
        "2",
    ]
    if max_frames > 0:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(output_dir / "frame_%06d.jpg"))

    # Extraction can run for minutes. Keep its process tree owned until exit,
    # including when the IDE forcibly stops main and releases its output locks.
    total = estimated_sample_count(probe_video_duration(video_path, ffmpeg_exe),
                                   frame_interval_s, max_frames)
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as errors, WorkProgress(
            f"Video: extract {video_path.name}", total, "frames") as progress:
        with managed_process(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=errors, text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                key, _, value = line.strip().partition('=')
                if key == 'frame' and value.strip().isdigit():
                    progress.update(max(progress.completed, int(value)))
            returncode = process.wait()
        if returncode != 0:
            progress.close('failed')
        else:
            progress.close('complete')
        errors.seek(0)
        stderr = errors.read(1000)
    if returncode != 0:
        message = stderr.strip()
        return "failed", message[:1000]
    return "ok", ""


def estimated_sample_count(duration_s, interval_s, max_frames):
    """Estimate only the known video span; an unknown span has no fixed total."""
    if duration_s is None or not math.isfinite(duration_s) or duration_s <= 0:
        return None
    count = max(1, math.ceil(duration_s / interval_s - 1e-9))
    return min(count, max_frames) if max_frames > 0 else count


def extract_with_opencv(
    video_path: Path,
    output_dir: Path,
    frame_interval_s: float,
    max_frames: int,
    frame_timestamps: Optional[Dict[str, Dict[str, Any]]] = None,
    decode_threads: int = 0,
    decode_hardware: str = 'auto',
    decode_device: int = -1,
    decode_diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return "missing_opencv", "OpenCV is not installed."

    capture = open_video(video_path, decode_threads, hardware=decode_hardware,
                         device=decode_device, diagnostics=decode_diagnostics)
    if not capture.isOpened():
        capture.release()
        return "failed", "OpenCV could not open the video."

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        return "failed", "No valid video frame rate; cannot assign sample times."
    written = 0
    frame_index = 0
    previous_time = -1.0
    total = estimated_sample_count(capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps,
                                   frame_interval_s, max_frames)
    try:
        with WorkProgress(f"Video: extract {video_path.name}", total, "frames") as progress:
            while True:
                if not capture.grab():
                    break
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                time_source = "opencv_decoded_timestamp"
                if not math.isfinite(timestamp) or timestamp <= previous_time:
                    timestamp = frame_index / fps
                    time_source = "opencv_frame_index_fps_estimate"
                previous_time = timestamp
                if timestamp + 1e-6 >= written * frame_interval_s:
                    ok, frame = capture.retrieve()
                    if not ok:
                        progress.close('failed')
                        return "failed", "Could not decode a selected video frame."
                    written += 1
                    output_path = output_dir / f"frame_{written:06d}.jpg"
                    if not write_frame(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                        progress.close('failed')
                        return "failed", "Could not write a selected video frame."
                    if frame_timestamps is not None:
                        frame_timestamps[output_path.name] = {"video_time_s": timestamp, "video_time_source": time_source,
                                                               "video_raw_frame_index": frame_index}
                    progress.update(written)
                    if max_frames > 0 and written >= max_frames:
                        break
                frame_index += 1
            if not written:
                progress.close('failed')
            else:
                progress.close('complete')
    finally:
        capture.release()
    print(f"[video] {video_path.name}: extracted {written} frames.", flush=True)
    return ("ok", "") if written else ("failed", "No frames were written.")


def probe_video_duration(video_path: Path, ffmpeg_exe: Optional[Path] = None) -> Optional[float]:
    ffprobe = ffmpeg_exe.with_name("ffprobe" + ffmpeg_exe.suffix) if ffmpeg_exe else None
    if ffprobe is None or not ffprobe.is_file():
        found = shutil.which("ffprobe")
        ffprobe = Path(found) if found else None
    if ffprobe is not None:
        try:
            result = subprocess.run([
                str(ffprobe), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration:format=duration", "-of", "json", str(video_path),
            ], capture_output=True, text=True, check=True)
            payload = json.loads(result.stdout)
            for item in [*payload.get("streams", []), payload.get("format", {})]:
                duration = safe_float(item.get("duration"))
                if math.isfinite(duration) and duration > 0:
                    return duration
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    capture = open_video(video_path, 1, report=False, hardware='off')
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if capture.isOpened() and math.isfinite(fps) and math.isfinite(frames) and fps > 0 and frames > 0:
            return frames / fps
    finally:
        capture.release()
    return None


def full_clip_interval(duration_s: Optional[float], requested_interval_s: float, max_frames: int) -> float:
    if not math.isfinite(requested_interval_s) or requested_interval_s <= 0:
        raise ValueError("Video frame interval must be finite and greater than zero.")
    if max_frames <= 0:
        return requested_interval_s
    if duration_s is None or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Cannot apply a whole-clip frame budget without a valid video duration.")
    return max(requested_interval_s, duration_s / max_frames)


def frame_original_index(frame_path: Path, fallback: int) -> int:
    matches = re.findall(r"\d+", frame_path.stem)
    if not matches:
        return fallback
    return safe_int(matches[-1], fallback)


def temporal_coverage(frames, frame_timestamps, interval_s, duration_s, sampled_count):
    """Report retained sampling bins and temporal gaps, not 3D scene coverage."""
    times, sources, bins = [], set(), set()
    for position, frame in enumerate(frames):
        index = frame_original_index(frame, position + 1)
        stamp = frame_timestamps.get(frame.name, {})
        value = safe_float(stamp.get("video_time_s", (index - 1) * interval_s))
        if not math.isfinite(value) or value < 0:
            continue
        times.append(value)
        sources.add(stamp.get("video_time_source", "resampled_time_grid_not_sensor_sync"))
        bins.add(index)
    times.sort()
    known_duration = duration_s is not None and math.isfinite(duration_s) and duration_s > 0
    within_duration = not known_duration or not times or times[-1] <= duration_s + 1e-6
    gaps = [b-a for a, b in zip(times, times[1:])]
    if known_duration and within_duration:
        boundaries = [0., *times, float(duration_s)]
        gaps = [b-a for a, b in zip(boundaries, boundaries[1:])]
    return {
        "sampled_bins": sampled_count,
        "retained_bins": len(bins),
        "retained_bin_fraction": len(bins) / max(sampled_count, 1),
        "valid_timestamp_count": len(times),
        "first_retained_time_s": times[0] if times else None,
        "last_retained_time_s": times[-1] if times else None,
        "max_gap_s": max(gaps) if gaps else None,
        "includes_clip_edges": bool(known_duration and within_duration),
        "start_gap_s": times[0] if times else (float(duration_s) if known_duration else None),
        "end_gap_s": float(duration_s)-times[-1] if known_duration and within_duration and times else None,
        "timestamp_sources": sorted(sources),
        "interpretation": "Temporal sampling coverage only; neither 3D surface coverage nor sensor synchronization",
    }


def filter_keyframes(
    frames: Sequence[Path],
    blur_threshold: float,
    min_visual_change: float,
    min_gap: int,
    *,
    blur_policy: str = "fixed",
    keyframe_max_gap_s: float = 2.0,
    frame_interval_s: float = 1.0,
    frame_timestamps: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Path], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if blur_policy not in {"fixed", "adaptive"}:
        raise ValueError("Video blur policy must be 'fixed' or 'adaptive'.")
    if blur_policy == "adaptive":
        return filter_adaptive_keyframes(frames, blur_threshold, min_visual_change, min_gap,
                                         keyframe_max_gap_s, frame_interval_s, frame_timestamps or {})
    if not frames:
        return [], {}, {
            "status": "no_frames",
            "input_frame_count": 0,
            "kept_frame_count": 0,
            "rejected_blurry_count": 0,
            "rejected_low_change_count": 0,
            "rejected_too_close_count": 0,
            "rejected_unreadable_count": 0,
        }

    needs_cv = blur_threshold > 0 or min_visual_change > 0
    if not needs_cv and min_gap <= 1:
        return list(frames), {}, {
            "status": "disabled",
            "input_frame_count": len(frames),
            "kept_frame_count": len(frames),
            "rejected_blurry_count": 0,
            "rejected_low_change_count": 0,
            "rejected_too_close_count": 0,
            "rejected_unreadable_count": 0,
        }

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return list(frames), {}, {
            "status": "missing_opencv",
            "input_frame_count": len(frames),
            "kept_frame_count": len(frames),
            "rejected_blurry_count": 0,
            "rejected_low_change_count": 0,
            "rejected_too_close_count": 0,
            "rejected_unreadable_count": 0,
        }

    kept: List[Path] = []
    metrics: Dict[str, Dict[str, Any]] = {}
    last_kept_gray = None
    last_kept_frame_pos = -10**9
    rejected_blurry = 0
    rejected_low_change = 0
    rejected_too_close = 0
    rejected_unreadable = 0
    min_gap = max(1, min_gap)

    with WorkProgress("Video: keyframe selection", len(frames), "frames") as progress:
        for frame_pos, frame_path in enumerate(progress.iterate(frames)):
            gray = read_frame(frame_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                rejected_unreadable += 1
                continue

            blur_score = laplacian_variance(gray) if blur_threshold > 0 else 0.0
            if blur_threshold > 0 and blur_score < blur_threshold:
                rejected_blurry += 1
                continue

            if kept and frame_pos - last_kept_frame_pos < min_gap:
                rejected_too_close += 1
                continue

            visual_change = ""
            if last_kept_gray is not None and min_visual_change > 0:
                compare_gray = gray
                if gray.shape != last_kept_gray.shape:
                    compare_gray = cv2.resize(gray, (last_kept_gray.shape[1], last_kept_gray.shape[0]))
                visual_change_value = float(cv2.mean(cv2.absdiff(compare_gray, last_kept_gray))[0])
                visual_change = round(visual_change_value, 6)
                if visual_change_value < min_visual_change:
                    rejected_low_change += 1
                    continue

            kept.append(frame_path)
            metrics[frame_path.name] = {
                "blur_score": round(blur_score, 6) if blur_threshold > 0 else "",
                "visual_change": visual_change,
                "keyframe_filter_status": "kept",
            }
            last_kept_gray = gray
            last_kept_frame_pos = frame_pos

    return kept, metrics, {
        "status": "ok",
        "input_frame_count": len(frames),
        "kept_frame_count": len(kept),
        "rejected_blurry_count": rejected_blurry,
        "rejected_low_change_count": rejected_low_change,
        "rejected_too_close_count": rejected_too_close,
        "rejected_unreadable_count": rejected_unreadable,
        "fallback_used": False,
        "blur_threshold": blur_threshold,
        "min_visual_change": min_visual_change,
        "min_gap": min_gap,
    }


def filter_adaptive_keyframes(frames, blur_threshold, min_visual_change, min_gap,
                             max_gap_s, interval_s, timestamps):
    """Calibrate clarity per clip and protect temporal overlap with observed frames.

    Native Laplacian scores vary strongly with resolution and camera processing.
    The supplied blur threshold is a ceiling for the clip's 20th-percentile
    threshold. A separate 640-pixel measurement rejects blank/unusable frames.
    Coverage anchors can bypass relative gates, but never that absolute floor.
    """
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("Adaptive keyframe maximum gap must be finite and positive.")
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("Adaptive keyframe interval must be finite and positive.")
    records = []
    unreadable = 0
    previous_timestamp = None

    def thumbnail(gray):
        scale = min(1., 640. / max(gray.shape))
        return cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                                 max(1, round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)

    with WorkProgress("Video: adaptive quality measurement", len(frames), "frames") as progress:
        for position, path in enumerate(progress.iterate(frames)):
            gray = read_frame(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                unreadable += 1
                continue
            small = thumbnail(gray)
            sharpness = laplacian_variance(small)
            contrast = float(small.std())
            raw_score = laplacian_variance(gray) if blur_threshold > 0 else 0.
            # Exclude almost blank, clipped, or severely detail-free images even
            # when the entire clip has poor quality. These floors are deliberately
            # conservative; a texture-poor wall must not calibrate itself as sharp.
            usable = (contrast >= 6. and sharpness >= 2. and
                      float(np.mean(small <= 8)) < .98 and float(np.mean(small >= 247)) < .98)
            fallback_time = (frame_original_index(path, position + 1) - 1) * interval_s
            stamp = safe_float(timestamps.get(path.name, {}).get("video_time_s", fallback_time))
            if not math.isfinite(stamp) or stamp < 0:
                raise ValueError(f"Invalid source video timestamp for {path.name}: {stamp!r}; re-extract frames.")
            if previous_timestamp is not None and stamp <= previous_timestamp:
                raise ValueError(f"Source video timestamps must increase in frame order ({path.name}); re-extract frames.")
            previous_timestamp = stamp
            records.append({"path": path, "position": position, "time": stamp, "usable": usable,
                            "blur_score": raw_score, "sharpness": sharpness, "contrast": contrast})

    usable_records = [record for record in records if record["usable"]]
    effective_threshold = (min(float(blur_threshold), float(np.percentile(
        [record["blur_score"] for record in usable_records], 20)))
        if blur_threshold > 0 and usable_records else 0.)
    detail_threshold = (max(2., float(np.percentile(
        [record["sharpness"] for record in usable_records], 20))) if usable_records else 2.)

    # One best observation in each half-gap time bin bounds adjacent anchor
    # separation by max_gap_s whenever those bins contain a usable image.
    # This also protects the beginning/end instead of silently losing minutes.
    anchors = {}
    for record in usable_records:
        bucket = int(record["time"] / (max_gap_s * .5))
        previous = anchors.get(bucket)
        if previous is None or record["sharpness"] > previous["sharpness"]:
            anchors[bucket] = record
    anchor_positions = {record["position"] for record in anchors.values()}
    kept, metrics = [], {}
    last_gray, last_position = None, -10**9
    rejected_blurry = rejected_low_change = rejected_too_close = coverage_rescued = 0
    min_gap = max(1, min_gap)
    with WorkProgress("Video: adaptive keyframe selection", len(records), "frames") as progress:
        for record in progress.iterate(records):
            if not record["usable"]:
                continue
            path, position = record["path"], record["position"]
            anchor = position in anchor_positions
            relative_blur = bool(blur_threshold > 0 and
                                 (record["blur_score"] < effective_threshold or record["sharpness"] < detail_threshold))
            too_close = bool(kept and position - last_position < min_gap)
            if not anchor and (relative_blur or too_close):
                rejected_blurry += int(relative_blur)
                rejected_too_close += int(not relative_blur and too_close)
                continue
            gray = read_frame(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                unreadable += 1
                continue
            small = thumbnail(gray)
            visual_change = ""
            low_change = False
            if last_gray is not None and min_visual_change > 0:
                compare = small if small.shape == last_gray.shape else cv2.resize(small, (last_gray.shape[1], last_gray.shape[0]))
                visual_change = float(cv2.mean(cv2.absdiff(compare, last_gray))[0])
                low_change = visual_change < min_visual_change
            if low_change and not anchor:
                rejected_low_change += 1
                continue
            rescued = anchor and (relative_blur or too_close or low_change)
            coverage_rescued += int(rescued)
            kept.append(path)
            metrics[path.name] = {"blur_score": round(record["blur_score"], 6) if blur_threshold > 0 else "",
                                  "visual_change": round(visual_change, 6) if visual_change != "" else "",
                                  "quality_sharpness_640": round(record["sharpness"], 6),
                                  "quality_contrast_640": round(record["contrast"], 6),
                                  "keyframe_filter_status": "coverage_anchor" if rescued else "kept"}
            last_gray, last_position = small, position

    times = [record["time"] for record in records if record["path"].name in metrics]
    actual_gap = max((b-a for a, b in zip(times, times[1:])), default=None)
    return kept, metrics, {
        "status": "ok" if frames else "no_frames", "blur_policy": "adaptive",
        "input_frame_count": len(frames), "kept_frame_count": len(kept),
        "rejected_blurry_count": rejected_blurry, "rejected_low_change_count": rejected_low_change,
        "rejected_too_close_count": rejected_too_close, "rejected_unreadable_count": unreadable,
        "rejected_unusable_count": sum(not record["usable"] for record in records),
        "coverage_anchor_count": len(anchor_positions), "coverage_rescued_count": coverage_rescued,
        "fallback_used": bool(coverage_rescued), "blur_threshold": blur_threshold,
        "effective_blur_threshold": effective_threshold, "quality_sharpness_threshold_640": detail_threshold,
        "calibration_percentile": 20, "quality_max_dimension": 640,
        "min_visual_change": min_visual_change, "min_gap": min_gap,
        "keyframe_max_gap_s": max_gap_s, "actual_internal_max_gap_s": actual_gap,
        "coverage_policy": "best usable frame per half-gap bin; blank/detail-free frames are never rescued",
        "internal_coverage_target_met": bool(times) and (actual_gap is None or actual_gap <= max_gap_s + 1e-6),
        "absolute_quality_floor": {"contrast_640": 6., "sharpness_640": 2., "max_clipped_fraction": .98},
        "coverage_caveat": "Temporal overlap only; missing/unusable bins can exceed the requested gap and no geometric baseline is guaranteed",
    }


def clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def unassociated_frame_fields() -> Dict[str, Any]:
    """Sensor association is supplied later by explicit sync or fuzzy windows."""
    return {
        "bind_type": "visual_only",
        "wifi_label_weight": 0.0,
        "nearest_global_point_index": "",
        "left_global_point_index": "",
        "right_global_point_index": "",
        "route_ratio": "",
        "pdr_x": "",
        "pdr_y": "",
        "pdr_bind_source": "video_sensor_sync_unavailable",
        "trajectory_contour_id": "",
        "trajectory_contour_distance_m": "",
        "trajectory_distance_m": "",
        "trajectory_position_source": "unavailable",
    }


def build_frame_rows(
    root: Path,
    video_path: Path,
    frames: Sequence[Path],
    frame_interval_s: float,
    start_sequence_index: int,
    source_frame_count: Optional[int] = None,
    frame_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    frame_timestamps: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    frame_count = source_frame_count or len(frames)
    frame_metrics = frame_metrics or {}
    for idx, frame_path in enumerate(frames):
        original_index = frame_original_index(frame_path, idx + 1)
        binding = unassociated_frame_fields()
        metrics = frame_metrics.get(frame_path.name, {})
        timestamp = (frame_timestamps or {}).get(frame_path.name, {})
        rows.append(
            {
                "source_type": "video_base_frame",
                "video_name": video_path.name,
                "video_path": relative_path(video_path, root),
                "frame_name": frame_path.name,
                "frame_path": relative_path(frame_path, root),
                "video_frame_index": original_index,
                "video_sequence_index": start_sequence_index + idx,
                "video_time_s": f"{timestamp.get('video_time_s', (original_index - 1) * frame_interval_s):.6f}",
                "video_time_source": timestamp.get("video_time_source", "resampled_time_grid_not_sensor_sync"),
                "video_raw_frame_index": timestamp.get("video_raw_frame_index", ""),
                "blur_score": metrics.get("blur_score", ""),
                "visual_change": metrics.get("visual_change", ""),
                "keyframe_filter_status": metrics.get("keyframe_filter_status", "kept"),
                **{field: metrics.get(field, "") for field in VIDEO_QUALITY_FIELDS},
                **binding,
            }
        )
    return rows


def run_video_frame_pipeline(
    root: Path,
    processed_dir: Path,
    video_dir: Path,
    frame_interval_s: float = 1.5,
    max_frames: int = 120,
    ffmpeg_exe: Optional[Path] = None,
    blur_threshold: float = 80.0,
    keyframe_min_change: float = 6.0,
    keyframe_min_gap: int = 2,
    refine_window_s: float = 0.,
    refine_budget_hours: float = 8.,
    refine_coverage_search: bool = True,
    blur_policy: str = "fixed",
    keyframe_max_gap_s: float = 2.0,
    decode_threads: int = 0,
    decode_hardware: str = 'auto',
    decode_device: int = -1,
) -> Dict[str, Any]:
    if frame_interval_s <= 0:
        raise ValueError("--video-frame-interval-s must be greater than zero")

    resolved_video_dir = video_dir if video_dir.is_absolute() else root / video_dir
    frames_root = processed_dir / "video_frames"
    manifest_path = processed_dir / "video_frames_manifest.csv"
    summary_path = processed_dir / "video_frames_summary.json"

    if not resolved_video_dir.exists():
        summary = {
            "status": "skipped",
            "reason": "missing_video_dir",
            "video_dir": str(resolved_video_dir),
            "frame_manifest": str(manifest_path),
        }
        write_csv(manifest_path, VIDEO_FRAME_FIELDS, [])
        write_json(summary_path, summary)
        return summary

    videos = find_video_files(resolved_video_dir)
    if not videos:
        summary = {
            "status": "skipped",
            "reason": "no_video_files",
            "video_dir": str(resolved_video_dir),
            "frame_manifest": str(manifest_path),
        }
        write_csv(manifest_path, VIDEO_FRAME_FIELDS, [])
        write_json(summary_path, summary)
        return summary

    refinement = None
    if refine_window_s > 0:
        from video_refinement import NativeFrameRefiner

        refinement = NativeFrameRefiner(processed_dir, refine_window_s, refine_budget_hours,
                                        coverage_search=refine_coverage_search, blur_threshold=blur_threshold,
                                        decode_threads=decode_threads, decode_hardware=decode_hardware,
                                        decode_device=decode_device)
    ffmpeg = resolve_ffmpeg(root, ffmpeg_exe)
    rows: List[Dict[str, Any]] = []
    video_summaries: List[Dict[str, Any]] = []

    for clip_index, video_path in enumerate(videos, 1):
        # Stems alone collide for clip.MOV/clip.mp4, Unicode names and, on
        # Windows, case variants. Each source needs its own extraction tree.
        output_dir = frames_root / f"{clip_index:04d}_{safe_stem(video_path.name)[:80]}"
        duration_s = probe_video_duration(video_path, ffmpeg)
        try:
            effective_interval_s = full_clip_interval(duration_s, frame_interval_s, max_frames)
        except ValueError as exc:
            video_summaries.append({"video": relative_path(video_path, root), "status": "failed", "message": str(exc)})
            continue
        clean_output_dir(output_dir)
        frame_timestamps: Dict[str, Dict[str, Any]] = {}
        decode_diagnostics: Dict[str, Any] = {}
        # Nearby native selection needs decoded timestamps, not an ffmpeg fps-grid time.
        if ffmpeg is not None and refinement is None and blur_policy == "fixed" and decode_hardware == 'off':
            extract_status, message = extract_with_ffmpeg(
                ffmpeg_exe=ffmpeg,
                video_path=video_path,
                output_dir=output_dir,
                frame_interval_s=effective_interval_s,
                max_frames=max_frames,
                decode_threads=decode_threads,
            )
            extractor = "ffmpeg"
            decode_diagnostics = {'requested_hardware': decode_hardware, 'actual_backend': 'ffmpeg_software', 'hardware_active': False}
        else:
            extract_status, message = extract_with_opencv(
                video_path=video_path,
                output_dir=output_dir,
                frame_interval_s=effective_interval_s,
                max_frames=max_frames,
                frame_timestamps=frame_timestamps,
                decode_threads=decode_threads,
                decode_hardware=decode_hardware,
                decode_device=decode_device,
                decode_diagnostics=decode_diagnostics,
            )
            extractor = "opencv"

        frames = sorted(output_dir.glob("*.jpg"), key=natural_sort_key) if extract_status == "ok" else []
        if refinement is not None:
            frames = refinement.refine(video_path, frames, frame_timestamps, effective_interval_s)
        selected_frames, frame_metrics, keyframe_summary = filter_keyframes(
            frames=frames,
            blur_threshold=blur_threshold,
            min_visual_change=keyframe_min_change,
            min_gap=keyframe_min_gap,
            blur_policy=blur_policy,
            keyframe_max_gap_s=keyframe_max_gap_s,
            frame_interval_s=effective_interval_s,
            frame_timestamps=frame_timestamps,
        )
        video_rows = build_frame_rows(
            root=root,
            video_path=video_path,
            frames=selected_frames,
            frame_interval_s=effective_interval_s,
            start_sequence_index=len(rows) + 1,
            source_frame_count=len(frames),
            frame_metrics=frame_metrics,
            frame_timestamps=frame_timestamps,
        )
        rows.extend(video_rows)
        coverage = temporal_coverage(selected_frames, frame_timestamps, effective_interval_s,
                                     duration_s, len(frames))
        largest_gap = coverage.get("max_gap_s")
        gap_label = f"{largest_gap:.2f}s" if largest_gap is not None else "unavailable"
        print(f"[video] {video_path.name}: retained {len(selected_frames)}/{len(frames)} frames "
              f"({blur_policy}); largest temporal gap={gap_label}.", flush=True)
        if frames and (len(selected_frames) < len(frames) * .5 or
                       (largest_gap is not None and largest_gap > max(5., 3. * effective_interval_s))):
            print("[warn] Video sampling has low retention or long gaps. Review video_frames_summary.json "
                  "and sparse registration before relying on dense reconstruction.", flush=True)
        video_summaries.append(
            {
                "video": relative_path(video_path, root),
                "extractor": extractor,
                "decoder": decode_diagnostics,
                "status": extract_status if extract_status != "ok" or selected_frames else "no_frames",
                "message": message,
                "video_duration_s": duration_s,
                "effective_frame_interval_s": effective_interval_s,
                "extracted_frame_count": len(frames),
                "frame_count": len(selected_frames),
                "keyframe_filter": keyframe_summary,
                "temporal_coverage": coverage,
                "output_dir": str(output_dir),
            }
        )

    write_csv(manifest_path, VIDEO_FRAME_FIELDS, rows)
    failed_videos = [item["video"] for item in video_summaries if item["status"] != "ok"]
    status = "partial_failure" if failed_videos and rows else "extract_failed" if failed_videos else "ok"
    if failed_videos:
        print(f"[error] Video extraction/selection failed for {len(failed_videos)} clip(s): "
              + ", ".join(failed_videos) + ". Review video_frames_summary.json before reconstruction.", flush=True)
    summary = {
        "status": status,
        "video_dir": str(resolved_video_dir),
        "video_count": len(videos),
        "failed_videos": failed_videos,
        "frame_count": len(rows),
        "frame_interval_s": frame_interval_s,
        "max_frames_per_video": max_frames,
        "sampling_policy": "whole_clip_uniform_budget",
        "route_binding": "disabled_without_video_sensor_synchronization",
        "blur_threshold": blur_threshold,
        "keyframe_min_change": keyframe_min_change,
        "keyframe_min_gap": keyframe_min_gap,
        "blur_policy": blur_policy,
        "keyframe_max_gap_s": keyframe_max_gap_s,
        "ffmpeg_exe": "" if ffmpeg is None else str(ffmpeg),
        "decode_threads": decode_threads,
        "decode_hardware": decode_hardware,
        "decode_device": decode_device,
        "frame_manifest": str(manifest_path),
        "videos": video_summaries,
        "refinement": refinement.summary() if refinement is not None else {"method": "off"},
    }
    write_json(summary_path, summary)
    return summary


VIDEO_FRAME_FIELDS = [
    "source_type",
    "video_name",
    "video_path",
    "frame_name",
    "frame_path",
    "video_frame_index",
    "video_sequence_index",
    "video_time_s",
    "video_time_source",
    "blur_score",
    "visual_change",
    "keyframe_filter_status",
    *VIDEO_QUALITY_FIELDS,
    *VIDEO_FRAME_PROVENANCE_FIELDS,
    "bind_type",
    "wifi_label_weight",
    "nearest_global_point_index",
    "left_global_point_index",
    "right_global_point_index",
    "route_ratio",
    "pdr_x",
    "pdr_y",
    "pdr_bind_source",
    "trajectory_contour_id",
    "trajectory_contour_distance_m",
    "trajectory_distance_m",
    "trajectory_position_source",
]
