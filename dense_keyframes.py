"""Budget dense views using sparse tracks and capture coverage, before stereo.

The sparse reconstruction stays intact. Only PatchMatch references and their
closed set of source views are reduced. Legacy workspaces use mvs_observations
unchanged, including its authenticated source file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np

import mvs_observations
from colmap_workspace_check import write_provenance
from reconstruction_metrics import Snapshot
from workspace_lock import workspace_lock


TRACK_OBSERVATION_REWARDS = (1., .7, .4)


def selection_config(settings):
    config = settings.get("dense_selection_config")
    if config is not None and not isinstance(config, dict):
        raise ValueError("Dense selection configuration must be an object")
    if not config or not config.get("enabled", False):
        return None
    if config.get("enabled") is not True:
        raise ValueError("Dense selection enabled must be a boolean")
    base_keys = {"enabled", "max_images", "max_source_images"}
    if set(config) not in (base_keys, base_keys | {"coverage_policy"}):
        raise ValueError("Dense selection requires enabled, max_images, max_source_images and optional coverage_policy")
    result = dict(config)
    limit, sources = result["max_images"], result["max_source_images"]
    if type(limit) is not int or limit < 0 or 0 < limit < 3:
        raise ValueError("Dense max_images must be 0 (all views) or at least 3")
    if type(sources) is not int or not 2 <= sources <= 64:
        raise ValueError("Dense max_source_images must be between 2 and 64")
    if "coverage_policy" in config:
        policy = config["coverage_policy"]
        fields = {"mode", "initial_images", "min_single_view_coverage", "min_three_view_coverage", "max_video_gap_s"}
        if not isinstance(policy, dict) or set(policy) != fields or policy["mode"] != "adaptive":
            raise ValueError("Invalid adaptive dense coverage policy")
        initial = policy["initial_images"]
        if type(initial) is not int or initial < 3 or (limit and initial > limit):
            raise ValueError("Adaptive initial_images must be at least 3 and not exceed the hard image limit")
        for field in ("min_single_view_coverage", "min_three_view_coverage", "max_video_gap_s"):
            value = policy[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid positive adaptive target: {field}")
            if field != "max_video_gap_s" and value > 1:
                raise ValueError(f"Adaptive coverage fraction exceeds 1: {field}")
        result["coverage_policy"] = dict(policy)
    return result


def _finite_time(row):
    try:
        value = float(row.get("video_time_s", ""))
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def _timelines(views, metadata):
    groups = {}
    for index, view in enumerate(views):
        row = metadata.get(view["name"], {})
        group = row.get("video_name") or "capture:" + str(row.get("session") or "unknown")
        groups.setdefault(group, []).append(index)
    result = []
    for name, indices in sorted(groups.items()):
        times = {index: _finite_time(metadata.get(views[index]["name"], {})) for index in indices}
        timed = all(value is not None for value in times.values())
        ordered = sorted(indices, key=lambda index: (times[index], views[index]["name"])) if timed else indices
        values = np.array([times[index] for index in ordered], dtype=float) if timed else np.arange(len(ordered), dtype=float)
        span = float(np.ptp(values)) if len(values) else 0.
        coordinates = (values - values[0]) / span if span > 0 else np.zeros(len(values))
        result.append(dict(name=name, indices=np.array(ordered, dtype=int), values=values,
                           coordinates=coordinates, basis="video_time_s" if timed else "image_name_rank"))
    return result


def _point_incidence(views):
    arrays = [np.unique(np.asarray(view["point_ids"], dtype=np.int64)) for view in views]
    lengths = np.array([len(ids) for ids in arrays], dtype=np.int64)
    flat = np.concatenate(arrays) if arrays else np.array([], dtype=np.int64)
    point_ids, indexed = np.unique(flat, return_inverse=True)
    starts = np.concatenate(([0], np.cumsum(lengths)))
    order = np.argsort(indexed, kind="stable")
    observing_views = np.repeat(np.arange(len(views)), lengths)[order]
    point_starts = np.concatenate(([0], np.cumsum(np.bincount(indexed, minlength=len(point_ids)))))
    return lengths, indexed, starts, point_ids, observing_views, point_starts


def select_dense_keyframes(views, points, metadata, max_images=480):
    """Choose representative views; geometric source verification follows later.

    A third of the budget seeds the full capture order. Remaining selections
    reward sparse-point coverage up to three observations, camera separation,
    and capture-time separation. Track gains are updated through an inverted
    index, so selecting hundreds of views does not rescan every observation on
    every iteration. Scene-unit distances are normalized within this model.
    """
    if type(max_images) is not int or max_images < 0 or 0 < max_images < 3:
        raise ValueError("Dense max_images must be 0 or at least 3")
    ordered = sorted(views, key=lambda view: view["name"])
    if not ordered or len({view["name"] for view in ordered}) != len(ordered):
        raise ValueError("Dense selection requires distinct registered views")
    limit = len(ordered) if max_images == 0 else min(max_images, len(ordered))
    if limit == len(ordered):
        return ordered, {"algorithm": "all_registered_views", "candidate_selected_names": [v["name"] for v in ordered]}
    lengths, indexed, starts, point_ids, observers, point_starts = _point_incidence(ordered)
    counts = np.zeros(len(point_ids), dtype=np.uint8)
    marginal = np.array([*TRACK_OBSERVATION_REWARDS, 0.])
    gains = lengths.astype(float) * marginal[0]
    gain_scale = max(1., float(lengths.max(initial=0)) * float(marginal.max()))
    centers = np.array([view["C"] for view in ordered], dtype=float)
    if centers.shape != (len(ordered), 3) or not np.isfinite(centers).all():
        raise ValueError("Invalid camera centers for dense selection")
    spatial_scale = float(np.linalg.norm(np.quantile(centers, .95, axis=0) - np.quantile(centers, .05, axis=0)))
    spatial_scale = max(spatial_scale, 1e-12)
    spatial_distance = np.full(len(ordered), np.inf)
    temporal_distance = np.ones(len(ordered))
    timelines = _timelines(ordered, metadata)
    timeline_by_index = {}
    for timeline in timelines:
        for offset, index in enumerate(timeline["indices"]):
            timeline_by_index[int(index)] = (timeline, offset)
    selected = np.zeros(len(ordered), dtype=bool)

    def add(index):
        if selected[index]:
            return
        selected[index] = True
        affected = indexed[starts[index]:starts[index + 1]]
        affected = affected[counts[affected] < 3]
        before = counts[affected]
        delta = marginal[before] - marginal[before + 1]
        counts[affected] += 1
        widths = point_starts[affected + 1] - point_starts[affected]
        if len(widths) and widths.sum():
            offsets = np.cumsum(widths) - widths
            positions = (np.repeat(point_starts[affected] - offsets, widths)
                         + np.arange(int(widths.sum())))
            gains[:] -= np.bincount(observers[positions], weights=np.repeat(delta, widths), minlength=len(ordered))
        spatial_distance[:] = np.minimum(spatial_distance, np.linalg.norm(centers - centers[index], axis=1))
        timeline, offset = timeline_by_index[index]
        indices = timeline["indices"]
        temporal_distance[indices] = np.minimum(
            temporal_distance[indices], abs(timeline["coordinates"] - timeline["coordinates"][offset]))

    seed_count = max(1, limit // 3)
    quotas = np.array([seed_count * len(t["indices"]) / len(ordered) for t in timelines])
    whole = np.floor(quotas).astype(int)
    for index in np.argsort(-(quotas - whole), kind="stable")[:seed_count - int(whole.sum())]:
        whole[index] += 1
    for timeline, quota in zip(timelines, whole):
        for bucket in np.array_split(timeline["indices"], int(quota)) if quota else []:
            # Use track-rich representatives throughout each clip, with a stable
            # preference for the middle of a bucket when track counts are tied.
            middle = (len(bucket) - 1) / 2
            best = max(range(len(bucket)), key=lambda offset: (lengths[bucket[offset]], -abs(offset - middle), -offset))
            add(int(bucket[best]))
    while int(selected.sum()) < limit:
        spatial = np.minimum(1., spatial_distance / spatial_scale * math.sqrt(limit))
        temporal = np.minimum(1., temporal_distance * limit)
        scores = .65 * np.maximum(0., gains) / gain_scale + .20 * spatial + .15 * temporal
        scores[selected] = -np.inf
        add(int(np.argmax(scores)))
    chosen = [view for index, view in enumerate(ordered) if selected[index]]
    return chosen, {
        "algorithm": "stratified_capture_and_greedy_three_view_track_coverage_v1",
        "capture_seed_count": seed_count,
        "track_observation_rewards": list(TRACK_OBSERVATION_REWARDS),
        "score_weights": {"sparse_tracks": .65, "camera_position": .20, "capture_time": .15},
        "candidate_selected_names": [view["name"] for view in chosen],
        "candidate_selection_requires_source_geometry_verification": True,
    }


def coverage_report(views, selected_views, metadata):
    """Describe retained evidence, not surface completeness or pose accuracy."""
    ordered = sorted(views, key=lambda view: view["name"])
    selected_names = {view["name"] for view in selected_views}
    lengths, indexed, _, point_ids, _, _ = _point_incidence(ordered)
    all_counts = np.bincount(indexed, minlength=len(point_ids))
    selected_flat = np.concatenate([view["point_ids"] for view in selected_views]) if selected_views else np.array([], dtype=np.int64)
    selected_counts = np.bincount(np.searchsorted(point_ids, selected_flat), minlength=len(point_ids))
    support = {}
    for count in (1, 2, 3):
        available = int(np.count_nonzero(all_counts >= count))
        retained = int(np.count_nonzero(selected_counts >= count))
        support[str(count)] = {"registered_points": available, "selected_points": retained,
                               "retained_fraction": retained / available if available else None}
    temporal = []
    for timeline in _timelines(ordered, metadata):
        mask = np.array([ordered[index]["name"] in selected_names for index in timeline["indices"]])
        full, kept = timeline["values"], timeline["values"][mask]
        temporal.append({
            "capture": timeline["name"], "basis": timeline["basis"],
            "registered_view_count": len(full), "selected_view_count": len(kept),
            "registered_range": [float(full[0]), float(full[-1])],
            "selected_range": [float(kept[0]), float(kept[-1])] if len(kept) else None,
            "max_unrepresented_interval": float(np.diff(np.concatenate(([full[0]], kept, [full[-1]]))).max(initial=0)),
        })
    centers = np.array([view["C"] for view in ordered])
    distances = np.full(len(ordered), np.inf)
    for view in selected_views:
        distances = np.minimum(distances, np.linalg.norm(centers - view["C"], axis=1))
    spatial = {"units": "sparse_model_scene_units", "nearest_selected_camera_distance": {
        "median": float(np.median(distances)) if len(selected_views) else None,
        "p95": float(np.quantile(distances, .95)) if len(selected_views) else None,
        "max": float(distances.max()) if len(selected_views) else None,
    }}
    return {"sparse_points_at_least_n_views": support, "capture_coverage": temporal,
            "camera_position_coverage": spatial,
            "registered_observation_count": int(lengths.sum()),
            "selected_observation_count": int(len(selected_flat)),
            "interpretation": "Sparse track and capture coverage only; these fractions do not measure reconstructed surface completeness or camera accuracy."}


def _adaptive_regions(views, metadata, policy):
    """Local point sets; supporting observations may come from any segment."""
    timelines = _timelines(views, metadata)
    regions = []
    width = 8. * policy["max_video_gap_s"]
    for timeline in timelines:
        indices, values = timeline["indices"], timeline["values"]
        timed = timeline["basis"] == "video_time_s"
        buckets = np.floor((values - values[0]) / width).astype(int) if timed else np.arange(len(indices)) // 64
        timeline["video_capture"] = any(
            metadata.get(views[int(index)]["name"], {}).get("video_name")
            or metadata.get(views[int(index)]["name"], {}).get("source_type") == "video_base_frame"
            or views[int(index)]["name"].startswith("video/") for index in indices)
        for bucket in np.unique(buckets):
            members = indices[buckets == bucket]
            ids = np.unique(np.concatenate([views[int(index)]["point_ids"] for index in members]))
            regions.append({"region_id": f'{timeline["name"]}:{int(bucket)}',
                            "capture": timeline["name"], "basis": timeline["basis"],
                            "range": [float(values[buckets == bucket][0]), float(values[buckets == bucket][-1])],
                            "indices": members, "point_ids": ids})
    return timelines, regions


def _adaptive_counts_report(views, point_ids, all_counts, counts, selected, timelines, regions):
    def support(indices):
        result = {}
        for level in (1, 2, 3):
            eligible = all_counts[indices] >= level
            available = int(np.count_nonzero(eligible))
            retained = int(np.count_nonzero(eligible & (counts[indices] >= level)))
            result[str(level)] = {"registered_points": available, "selected_points": retained,
                                  "retained_fraction": retained / available if available else None}
        return result

    captures = []
    for timeline in timelines:
        values, indices = timeline["values"], timeline["indices"]
        kept = values[selected[indices]]
        captures.append({"capture": timeline["name"], "basis": timeline["basis"],
                         "video_capture": bool(timeline["video_capture"]),
                         "registered_view_count": len(values), "selected_view_count": len(kept),
                         "registered_range": [float(values[0]), float(values[-1])],
                         "selected_range": [float(kept[0]), float(kept[-1])] if len(kept) else None,
                         "registered_max_gap_s": float(np.diff(values).max(initial=0)) if timeline["basis"] == "video_time_s" else None,
                         "max_unrepresented_interval": float(np.diff(np.concatenate(([values[0]], kept, [values[-1]]))).max(initial=0))})
    local = []
    for region in regions:
        indices = region.get("point_indices")
        if indices is None:
            indices = np.searchsorted(point_ids, region["point_ids"])
        local.append({key: region[key] for key in ("region_id", "capture", "basis", "range")} | {
            "registered_view_count": len(region["indices"]),
            "selected_view_count": int(np.count_nonzero(selected[region["indices"]])),
            "sparse_points_at_least_n_views": support(indices),
            "support_scope": "all_selected_registered_views_including_other_segments",
        })
    return {"sparse_points_at_least_n_views": support(np.arange(len(point_ids))),
            "capture_coverage": captures, "local_regions": local}


def adaptive_coverage_report(views, selected_views, metadata, policy):
    """Pure, reproducible final report used by planning and cache validation."""
    ordered = sorted(views, key=lambda view: view["name"])
    point_ids, all_counts = np.unique(np.concatenate([view["point_ids"] for view in ordered]), return_counts=True)
    flat = np.concatenate([view["point_ids"] for view in selected_views]) if selected_views else np.array([], dtype=np.int64)
    counts = np.bincount(np.searchsorted(point_ids, flat), minlength=len(point_ids))
    selected_names = {view["name"] for view in selected_views}
    mask = np.array([view["name"] in selected_names for view in ordered])
    timelines, regions = _adaptive_regions(ordered, metadata, policy)
    report = coverage_report(ordered, selected_views, metadata)
    report.update(_adaptive_counts_report(ordered, point_ids, all_counts, counts, mask, timelines, regions))
    report["local_region_policy"] = {"video_window_s": 8. * policy["max_video_gap_s"],
                                     "fallback_frame_window": 64,
                                     "three_view_denominator": "points_observed_in_at_least_three_original_registered_views"}
    return report


def assess_coverage_targets(report, policy):
    """Coverage readiness is independent of a usable geometric source graph."""
    support = report["sparse_points_at_least_n_views"]
    single, triple = support["1"]["retained_fraction"], support["3"]["retained_fraction"]
    reasons = []
    if single is None:
        reasons.append("single_view_coverage_unavailable")
    elif single + 1e-12 < policy["min_single_view_coverage"]:
        reasons.append("global_single_view_coverage_below_target")
    if triple is None:
        reasons.append("three_view_coverage_unavailable_no_original_three_view_tracks")
    elif triple + 1e-12 < policy["min_three_view_coverage"]:
        reasons.append("global_three_view_coverage_below_target")
    unmet = []
    unavailable_regions = []
    for region in report["local_regions"]:
        local = region["sparse_points_at_least_n_views"]
        if local["1"]["retained_fraction"] is None:
            unavailable_regions.append(region["region_id"])
            continue
        if (local["1"]["retained_fraction"] + 1e-12 < policy["min_single_view_coverage"]
                or (local["3"]["retained_fraction"] is not None
                    and local["3"]["retained_fraction"] + 1e-12 < policy["min_three_view_coverage"])):
            unmet.append(region["region_id"])
    if unmet:
        reasons.append("local_sparse_coverage_below_target")
    if unavailable_regions:
        reasons.append("local_sparse_coverage_unavailable")
    gaps, missing_time, irreducible = [], [], []
    for capture in report["capture_coverage"]:
        if not capture.get("video_capture"):
            continue
        if capture["basis"] != "video_time_s":
            missing_time.append(capture["capture"])
            continue
        gaps.append(capture["max_unrepresented_interval"])
        if capture["max_unrepresented_interval"] > policy["max_video_gap_s"] + 1e-9:
            if capture["registered_max_gap_s"] > policy["max_video_gap_s"] + 1e-9:
                irreducible.append(capture["capture"])
    if gaps and max(gaps) > policy["max_video_gap_s"] + 1e-9:
        reasons.append("video_gap_above_target")
    if irreducible:
        reasons.append("original_registered_video_gap_above_target")
    if missing_time:
        reasons.append("video_time_coverage_unavailable")
    return {"target_met": not reasons, "reasons": reasons,
            "global_single_view_coverage": single, "global_three_view_coverage": triple,
            "three_view_status": "available" if triple is not None else "not_evaluable",
            "unmet_local_region_count": len(unmet), "unmet_local_regions": unmet,
            "unavailable_local_regions": unavailable_regions,
            "unevaluable_time_capture_count": len(missing_time), "unevaluable_time_captures": missing_time,
            "irreducible_video_gap_captures": irreducible,
            "max_video_gap_s": max(gaps) if gaps else None}


class _AdaptiveSelection:
    """Incremental sparse-track deficits, including poorly covered segments."""

    def __init__(self, views, metadata, policy, initial):
        self.views, self.policy = views, policy
        self.lengths, self.indexed, self.starts, self.point_ids, self.observers, self.point_starts = _point_incidence(views)
        self.all_counts = np.diff(self.point_starts)
        self.view_observers = np.repeat(np.arange(len(views)), self.lengths)
        self.timelines, self.regions = _adaptive_regions(views, metadata, policy)
        self.weights = np.ones(len(self.point_ids))
        typical = float(np.median([len(region["point_ids"]) for region in self.regions])) if self.regions else 1.
        for region in self.regions:
            region["point_indices"] = np.searchsorted(self.point_ids, region["point_ids"])
            weight = min(6., max(1., math.sqrt(typical / max(1, len(region["point_ids"])))))
            self.weights[region["point_indices"]] = np.maximum(self.weights[region["point_indices"]], weight)
        self.centers = np.array([view["C"] for view in views], dtype=float)
        if self.centers.shape != (len(views), 3) or not np.isfinite(self.centers).all():
            raise ValueError("Invalid camera centers for adaptive dense selection")
        self.chosen = np.zeros(len(views), dtype=bool)
        self.chosen[initial] = True
        self.focus_id = None
        self.focus = np.zeros(len(self.point_ids), dtype=bool)
        self.rebase(initial)

    def marginal(self, indices):
        counts = np.minimum(self.counts[indices], 3)
        # Completion of the third observation earns the largest reward. Points
        # without three original observations contribute only to single coverage.
        triple = np.array([.6, .85, 1.2, 0.])[counts]
        single = (counts == 0).astype(float)
        return self.weights[indices] * np.where(self.all_counts[indices] >= 3, triple, single)

    def rebase(self, supporting):
        self.supporting = np.zeros(len(self.views), dtype=bool)
        self.supporting[supporting] = True
        mask = np.repeat(self.supporting, self.lengths)
        self.counts = np.bincount(self.indexed[mask], minlength=len(self.point_ids)).astype(np.int32)
        rewards = self.marginal(np.arange(len(self.point_ids)))
        self.gains = np.bincount(self.view_observers, weights=rewards[self.indexed], minlength=len(self.views))
        self.local_gains = np.bincount(self.view_observers, weights=(rewards * self.focus)[self.indexed], minlength=len(self.views))
        self.spatial_distance = np.full(len(self.views), np.inf)
        for index in supporting:
            self.spatial_distance = np.minimum(self.spatial_distance, np.linalg.norm(self.centers - self.centers[index], axis=1))

    def add(self, index):
        self.chosen[index] = self.supporting[index] = True
        affected = self.indexed[self.starts[index]:self.starts[index + 1]]
        before = self.marginal(affected)
        self.counts[affected] += 1
        delta = before - self.marginal(affected)
        useful = delta != 0
        affected, delta = affected[useful], delta[useful]
        widths = self.point_starts[affected + 1] - self.point_starts[affected]
        if len(widths) and widths.sum():
            offsets = np.cumsum(widths) - widths
            positions = np.repeat(self.point_starts[affected] - offsets, widths) + np.arange(int(widths.sum()))
            observers = self.observers[positions]
            self.gains -= np.bincount(observers, weights=np.repeat(delta, widths), minlength=len(self.views))
            self.local_gains -= np.bincount(observers, weights=np.repeat(delta * self.focus[affected], widths), minlength=len(self.views))
        self.spatial_distance = np.minimum(self.spatial_distance, np.linalg.norm(self.centers - self.centers[index], axis=1))

    def report(self):
        return _adaptive_counts_report(self.views, self.point_ids, self.all_counts, self.counts,
                                       self.supporting, self.timelines, self.regions)

    def set_focus(self, report):
        deficient = []
        for region in report["local_regions"]:
            support = region["sparse_points_at_least_n_views"]
            ratios = [support[level]["retained_fraction"] / self.policy[target]
                      for level, target in (("1", "min_single_view_coverage"), ("3", "min_three_view_coverage"))
                      if support[level]["retained_fraction"] is not None]
            if ratios and min(ratios) < 1 - 1e-12:
                deficient.append((min(ratios), region["region_id"]))
        focus_id = min(deficient)[1] if deficient else None
        if focus_id == self.focus_id:
            return
        self.focus_id = focus_id
        self.focus[:] = False
        for region in self.regions:
            if region["region_id"] == focus_id:
                self.focus[region["point_indices"]] = True
                break
        rewards = self.marginal(np.arange(len(self.point_ids))) * self.focus
        self.local_gains = np.bincount(self.view_observers, weights=rewards[self.indexed], minlength=len(self.views))

    def gap_candidate(self):
        best = None
        for timeline in self.timelines:
            if not timeline["video_capture"] or timeline["basis"] != "video_time_s":
                continue
            indices, values = timeline["indices"], timeline["values"]
            kept = values[self.supporting[indices]]
            bounds = np.concatenate(([values[0]], kept, [values[-1]]))
            for left, right in zip(bounds, bounds[1:]):
                if right - left <= self.policy["max_video_gap_s"] + 1e-9:
                    continue
                candidates = np.flatnonzero((values >= left) & (values <= right) & ~self.chosen[indices])
                if not len(candidates):
                    continue
                offset = min(candidates, key=lambda i: (abs(values[i] - (left + right) / 2), -self.lengths[indices[i]], int(indices[i])))
                proposal = (-(right - left), abs(values[offset] - (left + right) / 2), int(indices[offset]))
                if best is None or proposal < best:
                    best = proposal
        return best[-1] if best is not None else None

    def best_track_candidate(self):
        global_gain = np.maximum(0., self.gains)
        local_gain = np.maximum(0., self.local_gains)
        global_gain[self.chosen] = local_gain[self.chosen] = 0
        if global_gain.max(initial=0) < 1e-9:
            return None
        scores = .35 * global_gain / max(1., global_gain.max()) + .6 * local_gain / max(1., local_gain.max())
        distance = np.nan_to_num(self.spatial_distance, posinf=0.)
        scores += .05 * distance / max(1e-12, distance.max(initial=0))
        scores[self.chosen] = -np.inf
        return int(np.argmax(scores))


class _IncrementalSourceGraph:
    """Validate each camera pair once even when coverage requires extra views."""

    def __init__(self, points, metadata, max_sources):
        self.points, self.metadata, self.max_sources = points, metadata, max_sources
        self.views, self.candidates = {}, {}
        self.pair_evaluation_count = 0

    def extend(self, selected):
        for view in sorted(selected, key=lambda row: row["name"]):
            name = view["name"]
            if name in self.views:
                continue
            self.candidates[name] = []
            for other_name, other in self.views.items():
                self.pair_evaluation_count += 1
                evidence = mvs_observations.pair_depth_support(view, other, self.points)
                if not evidence["usable"]:
                    continue
                a, b = self.metadata.get(name, {}), self.metadata.get(other_name, {})
                same = bool(a.get("session") and a.get("global_point_index") not in (None, "")
                            and a.get("session") == b.get("session")
                            and a.get("global_point_index") == b.get("global_point_index"))
                self.candidates[name].append(dict(source=other_name, same_checkpoint=same, **evidence))
                self.candidates[other_name].append(dict(source=name, same_checkpoint=same, **evidence))
            self.views[name] = view

    def plan(self):
        stationary = {name for name in self.views
                      if self.metadata.get(name, {}).get("image_association_basis") == "ordered_checkpoint_same_location"
                      and self.metadata.get(name, {}).get("session")
                      and self.metadata.get(name, {}).get("global_point_index") not in (None, "")}
        active, no_translation = set(self.views), set()
        while active:
            weak = set()
            for name in active:
                rows = [row for row in self.candidates[name] if row["source"] in active]
                if len(rows) < 2:
                    weak.add(name)
                if name in stationary and not any(not row["same_checkpoint"] for row in rows):
                    weak.add(name)
                    no_translation.add(name)
            if not weak:
                break
            active -= weak
        sources = {name: sorted((row for row in self.candidates[name] if row["source"] in active),
                                key=lambda row: (row["same_checkpoint"], -row["shared_tracks"], row["source"]))[:self.max_sources]
                   for name in sorted(active)}
        return dict(status="ready" if len(active) >= 3 else "insufficient_parallax",
                    minimum_shared_tracks=30, minimum_median_angle_degrees=2.,
                    max_sources=self.max_sources, view_count=len(self.views), depth_reference_count=len(active),
                    weak_views=sorted(set(self.views) - active), sources=sources,
                    stationary_views_without_external_source=sorted(no_translation),
                    stationary_source_policy="Known same-location capture requires a geometrically usable source outside its checkpoint; unknown source locations are not certified translations.",
                    interpretation="Internal geometric support only; coverage target results are reported separately and do not establish true scene shape.")


def adaptive_dense_selection_plan(views, points, metadata, config):
    config = selection_config({"dense_selection_config": config})
    policy = config["coverage_policy"]
    ordered = sorted(views, key=lambda view: view["name"])
    hard_limit = min(config["max_images"] or len(ordered), len(ordered))
    initial_budget = hard_limit if config["max_images"] == 0 else min(policy["initial_images"], hard_limit)
    if initial_budget == len(ordered):
        # A valid sparse component can contain fewer than three views. Report
        # unavailable three-view support rather than treating its size as an
        # invalid user-supplied image budget.
        initial = ordered
    else:
        initial, _ = select_dense_keyframes(ordered, points, metadata, initial_budget)
    index_by_name = {view["name"]: index for index, view in enumerate(ordered)}
    state = _AdaptiveSelection(ordered, metadata, policy, [index_by_name[view["name"]] for view in initial])
    initial_assessment = assess_coverage_targets(state.report(), policy)
    graph = _IncrementalSourceGraph(points, metadata, config["max_source_images"])
    passes, growth_history = 0, []
    while True:
        before_count = int(state.chosen.sum())
        while int(state.chosen.sum()) < hard_limit:
            report = state.report()
            assessment = assess_coverage_targets(report, policy)
            if assessment["target_met"]:
                break
            gap = state.gap_candidate()
            if gap is not None:
                state.add(gap)
                continue
            sparse_deficit = any(reason in assessment["reasons"] for reason in (
                "global_single_view_coverage_below_target", "global_three_view_coverage_below_target", "local_sparse_coverage_below_target"))
            if not sparse_deficit:
                break
            state.set_focus(report)
            added = 0
            for _ in range(min(8, hard_limit - int(state.chosen.sum()))):
                candidate = state.best_track_candidate()
                if candidate is None:
                    break
                state.add(candidate)
                added += 1
            if not added:
                break
        before_geometry = assess_coverage_targets(state.report(), policy)
        candidates = [view for index, view in enumerate(ordered) if state.chosen[index]]
        graph.extend(candidates)
        plan = graph.plan()
        passes += 1
        state.rebase([index_by_name[name] for name in plan["sources"]])
        after_geometry = assess_coverage_targets(state.report(), policy)
        growth_history.append({"candidate_count": len(candidates), "reference_count": len(plan["sources"]),
                               "before_geometry": before_geometry, "after_geometry": after_geometry})
        if (after_geometry["target_met"] or len(candidates) >= hard_limit
                or (passes > 1 and len(candidates) == before_count)):
            break
        # A geometry pass may remove supporting views; their coverage is no
        # longer counted while adding new candidates. Cached pairs remain valid.
        if state.gap_candidate() is None and not any(reason in after_geometry["reasons"] for reason in (
                "global_single_view_coverage_below_target", "global_three_view_coverage_below_target", "local_sparse_coverage_below_target")):
            break
    retained = [ordered[index_by_name[name]] for name in plan["sources"]]
    final_report = adaptive_coverage_report(ordered, retained, metadata, policy)
    targets = assess_coverage_targets(final_report, policy)
    targets.update(initial_assessment=initial_assessment, before_geometry=before_geometry,
                   hard_limit_reached=bool(config["max_images"] and len(candidates) >= config["max_images"]),
                   initial_view_count=len(initial), added_view_count=len(candidates) - len(initial),
                   geometry_pair_evaluation_count=graph.pair_evaluation_count, geometry_passes=passes,
                   all_images_requested=config["max_images"] == 0,
                   registered_views_exhausted=len(candidates) == len(ordered))
    counts = [len(rows) for rows in plan["sources"].values()]
    warnings = [] if targets["target_met"] else ["Adaptive dense coverage targets were not met: " + ", ".join(targets["reasons"])]
    removed = sorted(set(view["name"] for view in candidates) - set(plan["sources"]))
    if removed:
        warnings.append("Views without a closed set of two geometric sources were excluded; coverage was rechecked after exclusion.")
    plan["view_count"] = len(ordered)
    plan["dense_selection"] = {
        "algorithm": "incremental_local_and_global_track_coverage_v2",
        "config": config, "registered_view_count": len(ordered), "selected_view_count": len(retained),
        "candidate_selected_names": [view["name"] for view in candidates], "selected_names": sorted(plan["sources"]),
        "removed_unsupported_names": removed, "coverage_report": final_report, "coverage_targets": targets,
        "growth_history": growth_history, "warnings": warnings,
        "sources_per_reference": {"min": min(counts, default=0), "median": float(np.median(counts)) if counts else None,
                                  "max": max(counts, default=0)},
        "source_graph_passed_internal_geometry_checks": plan["status"] == "ready",
    }
    return plan


def dense_selection_plan(views, points, metadata, config):
    if "coverage_policy" in config:
        return adaptive_dense_selection_plan(views, points, metadata, config)
    selected, report = select_dense_keyframes(views, points, metadata, config["max_images"])
    plan = mvs_observations.depth_source_plan(selected, points, metadata, max_sources=config["max_source_images"])
    active = set(plan["sources"])
    retained = [view for view in selected if view["name"] in active]
    report.update(config=dict(config), registered_view_count=len(views), selected_view_count=len(active),
                  selected_names=sorted(active), removed_unsupported_names=sorted(set(report["candidate_selected_names"]) - active),
                  coverage_report=coverage_report(views, retained, metadata))
    source_counts = [len(rows) for rows in plan["sources"].values()]
    report["sources_per_reference"] = {
        "min": min(source_counts, default=0),
        "median": float(np.median(source_counts)) if source_counts else None,
        "max": max(source_counts, default=0),
    }
    report["source_graph_passed_internal_geometry_checks"] = plan["status"] == "ready"
    report["warnings"] = []
    for level, coverage in report["coverage_report"]["sparse_points_at_least_n_views"].items():
        if coverage["retained_fraction"] is not None and coverage["retained_fraction"] < .9:
            report["warnings"].append(f"Retained fewer than 90% of sparse points with at least {level} selected views; consider increasing max_images.")
    if report["removed_unsupported_names"]:
        report["warnings"].append("Candidate views without a closed set of at least two geometrically usable sources were excluded.")
    plan["view_count"] = len(views)
    plan["dense_selection"] = report
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select representative dense keyframes with closed geometric sources.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Write source configs only before any stereo maps exist.")
    parser.add_argument("--summary", type=Path, help="Separate JSON diagnostics; inspection otherwise prints only.")
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    summary_path = workspace / "colmap_workspace_summary.json"
    settings = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    config = selection_config(settings)
    if config is None:
        return mvs_observations.main(argv)
    with workspace_lock(workspace) if args.apply else nullcontext():
        dense = workspace / "dense"
        if args.apply and (any((dense / "stereo/depth_maps").rglob("*.bin"))
                           or any((dense / "stereo/normal_maps").rglob("*.bin"))):
            raise ValueError("Existing depth maps must not be reinterpreted with different source views")
        snapshot = Snapshot()
        snapshot.identity(Path(__file__))
        snapshot.identity(Path(mvs_observations.__file__))
        if selection_config(snapshot.read_json(summary_path, required=True)) != config:
            raise ValueError("Dense selection configuration changed before planning")
        views, points = mvs_observations.read_model(dense / "sparse", snapshot)
        manifest = workspace / "metadata/images_manifest.csv"
        snapshot.identity(manifest)
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            metadata = {row["image_file"]: row for row in csv.DictReader(handle)}
        plan = dense_selection_plan(views, points, metadata, config)
        plan["inputs"] = {str(path): identity for path, (identity, _) in snapshot.files.items()}
        snapshot.verify_unchanged()
        output = args.summary or (dense / "depth_source_summary.json" if args.apply else None)
        if output is not None:
            output = output.resolve()
            if output.is_relative_to(dense / "sparse") or output in snapshot.files or output.suffix != ".json":
                raise ValueError("Dense selection report must be separate from model inputs")
            if not args.apply and output == (dense / "depth_source_summary.json").resolve():
                raise ValueError("Inspection must not overwrite the applied depth-source receipt")
        if args.apply and plan["status"] != "ready":
            raise ValueError("Insufficient closed geometric sources after dense selection; inspect without --apply or increase max_images")
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            write_provenance(output, plan)
        overview = {key: value for key, value in plan["dense_selection"].items()
                    if key not in {"selected_names", "candidate_selected_names", "removed_unsupported_names"}}
        print(json.dumps({key: plan[key] for key in ("status", "view_count", "depth_reference_count")} | {
            "dense_selection": overview}), flush=True)
        if args.apply:
            lines = []
            for name, rows in plan["sources"].items():
                lines.extend((name, ", ".join(row["source"] for row in rows)))
            for filename, content in (("patch-match.cfg", "\n".join(lines) + "\n"),
                                      ("fusion.cfg", "\n".join(plan["sources"]) + "\n")):
                target = dense / "stereo" / filename
                staged = target.with_suffix(".pending")
                staged.write_text(content, encoding="utf-8")
                staged.replace(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
