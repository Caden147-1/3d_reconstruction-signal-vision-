"""Select sharper nearby native frames without generating image content."""

import json
import math
from pathlib import Path
import time
import uuid

import cv2


class NativeFrameRefiner:
    def __init__(self, processed: Path, window_s=0., budget_hours=8., *,
                 coverage_search=False, blur_threshold=0.):
        if not math.isfinite(window_s) or not 0 <= window_s <= 2:
            raise ValueError("Refinement window must be finite and in [0,2]")
        if not math.isfinite(budget_hours) or not 0 < budget_hours <= 20:
            raise ValueError("Refinement budget must be finite and in (0,20] hours")
        if not math.isfinite(blur_threshold):
            raise ValueError("Blur threshold must be finite")
        self.window_s = window_s
        self.coverage_search = bool(coverage_search)
        self.blur_threshold = max(0., float(blur_threshold))
        self.started = time.monotonic()
        self.deadline = self.started + budget_hours * 3600
        self.output = processed / "video_refinement" / uuid.uuid4().hex
        self.output.mkdir(parents=True, exist_ok=False)
        self.refinements = []

    def refine(self, video, frames, timestamps, interval):
        if self.window_s <= 0 or not frames:
            return frames
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Sampling interval must be finite and positive")
        folder = self.output / f"refined_{len(self.refinements):04d}"
        folder.mkdir()
        result, records = [], []
        selected_indices = set()
        radius = min(self.window_s, interval * .45)
        fallback_radius = min(2., interval * .45) if self.coverage_search else radius
        budget_exhausted = False
        capture = cv2.VideoCapture(str(video))
        try:
            for position, path in enumerate(frames):
                if time.monotonic() >= self.deadline:
                    result.extend(frames[position:])
                    budget_exhausted = True
                    break
                original = cv2.imread(str(path))
                original_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if original is None or original_gray is None:
                    result.append(path)
                    continue
                stamp = timestamps.get(path.name, {})
                center = float(stamp.get("video_time_s", position * interval))
                if not math.isfinite(center) or center < 0:
                    result.append(path)
                    continue
                original_score = float(cv2.Laplacian(original_gray, cv2.CV_64F).var())
                # Retain the sampled image unless a saved native candidate is
                # actually sharper. Comparing after JPEG encoding also matches
                # the downstream blur gate, which reads the saved grayscale.
                best = (original_score, None, center, stamp.get("video_raw_frame_index"), original_score)
                seen = set()
                attempted = set()

                def search(deltas, limit):
                    nonlocal best, budget_exhausted
                    for delta in deltas:
                        if time.monotonic() >= self.deadline:
                            budget_exhausted = True
                            return
                        target = max(0., center + delta)
                        target_key = round(target, 9)
                        if target_key in attempted or not capture.isOpened():
                            continue
                        attempted.add(target_key)
                        capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000)
                        ok, image = capture.read()
                        actual = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        decoded_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                        if not math.isfinite(decoded_position):
                            continue
                        raw_index = int(round(decoded_position)) - 1
                        if (not ok or image is None or image.shape != original.shape or
                            not math.isfinite(actual) or actual < 0 or abs(actual - center) > limit + 1e-6 or
                            raw_index in seen or raw_index in selected_indices or raw_index < 0):
                            continue
                        seen.add(raw_index)
                        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if not ok:
                            continue
                        gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                        if gray is None:
                            continue
                        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        if score > best[0]:
                            native_score = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                            best = score, encoded.tobytes(), actual, raw_index, native_score

                search((0., -radius / 2, radius / 2, -radius, radius), radius)
                initial_score = best[0]
                expand = (self.coverage_search and initial_score < self.blur_threshold and
                          fallback_radius > radius + 1e-6 and not budget_exhausted)
                if expand:
                    # Eight additional native seeks at most; adjacent sampling
                    # bins never overlap. This recovers missed sharp moments,
                    # without accepting a frame below the existing blur floor.
                    search((sign * fallback_radius * fraction
                            for fraction in (.25, .5, .75, 1.) for sign in (-1, 1)), fallback_radius)
                if best[1] is None:
                    result.append(path)
                else:
                    destination = folder / path.name
                    destination.write_bytes(best[1])
                    timestamps[path.name] = {"video_time_s": best[2], "video_time_source": "opencv_decoded_timestamp",
                                             "video_raw_frame_index": best[3]}
                    result.append(destination)
                if best[3] is not None and best[3] != "":
                    selected_indices.add(int(best[3]))
                records.append({"frame": path.name, "sample_time_s": center, "selected_time_s": best[2],
                                "raw_frame_index": best[3], "candidate_count": len(seen),
                                "attempted_candidate_count": len(attempted), "native_blur_score": best[4],
                                "original_blur_score": original_score, "selected_blur_score": best[0],
                                "selected_original": best[1] is None,
                                "coverage_search_triggered": expand,
                                "coverage_search_recovered": bool(expand and best[0] >= self.blur_threshold),
                                "passes_blur_threshold": best[0] >= self.blur_threshold})
        finally:
            capture.release()
        self.refinements.append({"video": str(video), "input_frame_count": len(frames),
                                 "initial_search_radius_s": radius, "coverage_search_radius_s": fallback_radius,
                                 "candidate_limit_per_frame": 13 if self.coverage_search else 5,
                                 "budget_exhausted": budget_exhausted, "frames": records})
        (self.output / "refinement.json").write_text(json.dumps(self.refinements, indent=2) + "\n", encoding="utf-8")
        replaced = sum(not record["selected_original"] for record in records)
        recovered = sum(record["coverage_search_recovered"] for record in records)
        print(f"[video] Compared {len(records)}/{len(frames)} samples; replaced {replaced} with sharper native frames; recovered {recovered} blur-failed bins.", flush=True)
        return result

    def summary(self):
        value = {"method": "native_frame_selection", "refine_window_s": self.window_s,
                 "coverage_search": self.coverage_search, "blur_threshold": self.blur_threshold,
                 "coverage_search_max_radius_s": 2. if self.coverage_search else None,
                 "coverage_search_radius_policy": "min(2 seconds, 0.45 * sampling interval), only below blur threshold" if self.coverage_search else "disabled",
                 "selected_frames": sum(len(r["frames"]) for r in self.refinements),
                 "replaced_frames": sum(not f["selected_original"] for r in self.refinements for f in r["frames"]),
                 "recovered_blur_failed_bins": sum(f["coverage_search_recovered"] for r in self.refinements for f in r["frames"]),
                 "budget_exhausted": any(r["budget_exhausted"] for r in self.refinements),
                 "elapsed_s": time.monotonic() - self.started, "output_dir": str(self.output),
                 "caveat": "Observed frames only; decoded timestamps do not establish video/sensor synchronization"}
        (self.output / "summary.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return value
