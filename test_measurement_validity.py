"""Regression coverage for measurement provenance and signal-dependent matching."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colmap_bridge import (
    build_constrained_match_pairs, build_wifi_recovery_edges,
    copy_metadata, prepare_colmap_workspace, select_visually_supported_match_pairs,
)
from image_mapping import build_image_wifi_tables, nearest_fresh_fingerprint
from preprocess_pipeline import build_wifi_tables, parse_args
from reconstruction_export import export_reconstruction_dataset
from video_frames import (
    build_frame_rows, extract_with_ffmpeg, full_clip_interval,
    run_video_frame_pipeline, write_csv,
)
from wifi_observations import valid_fingerprint
from wifi_positioning import estimate_position, load_fingerprints, run_wifi_positioning_pipeline


def fingerprint(point=1, timestamp=1000, rssi=-60):
    return {
        "session": "session.sess", "scan_index": str(point), "elapsed_ts": str(timestamp + 100),
        "epoch_ts": str(timestamp + 1000000), "_elapsed_ts_int": timestamp + 100,
        "global_point_index": str(point), "point_index": str(point),
        "image_file": f"a{point}.jpg", "image_slot": "1", "source_type": "wifi_anchor_photo",
        "pdr_x": str(point * 20), "pdr_y": "0", "trajectory_contour_id": "route",
        "trajectory_contour_distance_m": str(point * 20),
        "wifi_observation_status": "fresh", "wifi_fresh_ap_count": "3", "wifi_max_age_ms": "100",
        "wifi_measurement_min_elapsed_ts": str(timestamp),
        "wifi_measurement_max_elapsed_ts": str(timestamp),
        "ap_A": str(rssi), "ap_B": str(rssi - 1), "ap_C": str(rssi - 2),
    }


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class WifiMeasurementTests(unittest.TestCase):
    def test_cached_readings_do_not_inflate_statistics_or_move_to_new_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "session.sess"
            session.mkdir()
            def aps(timestamp, rssi=-60):
                return [{"mac": mac, "dbm": rssi, "elap_ts": timestamp} for mac in "ABC"]
            records = [(1100, aps(1000)), (1200, aps(1000)), (30000, aps(1000)), (30100, aps(30100)),
                       (30200, aps(40000)), (30300, [{"mac": "A", "dbm": -60}]),
                       (31000, [{"mac": "A", "dbm": "inf", "elap_ts": 31000}])]
            (session / "wifiScanner.dat").write_text("".join(
                f"{elapsed}\t{elapsed + 1000000}\t{json.dumps(payload)}\n" for elapsed, payload in records
            ))
            summary = build_wifi_tables([session], root / "out", 0, -100, 50, 1)
            self.assertEqual(summary["session_scan_counts"][session.name], 7)
            self.assertEqual(summary["session_independent_measurement_record_counts"][session.name], 2)
            self.assertEqual(summary["observation_status_counts"]["cached_duplicate"], 3)
            self.assertEqual(summary["observation_status_counts"]["stale_or_future"], 6)
            stats = csv_rows(root / "out" / "ap_stats.csv")
            self.assertTrue(all(row["entry_count"] == "2" and row["total_scans"] == "2" for row in stats))
            rows = csv_rows(root / "out" / "wifi_wide_selected.csv")
            self.assertEqual(len(rows), 7)  # Keep logger-time rows for independent PDR association.
            self.assertTrue(valid_fingerprint(rows[0]))
            self.assertFalse(valid_fingerprint(rows[1]))
            self.assertFalse(valid_fingerprint(rows[2]))
            chosen, gap = nearest_fresh_fingerprint(rows, 29900)
            self.assertEqual(chosen["scan_index"], "4")
            self.assertEqual(gap, "200")
            self.assertIsNone(nearest_fresh_fingerprint(rows, 20000)[0])

    def test_freshness_is_based_on_every_ap_and_requires_provenance(self):
        row = fingerprint()
        self.assertTrue(valid_fingerprint(row))
        self.assertIsNone(nearest_fresh_fingerprint([row], 20000)[0])
        self.assertIsNotNone(nearest_fresh_fingerprint([row], 2000)[0])
        for updates in ({"wifi_max_age_ms": "nan"}, {"wifi_max_age_ms": "6000"},
                        {"wifi_measurement_min_elapsed_ts": "-10000"},
                        {"wifi_measurement_min_elapsed_ts": "-100", "wifi_measurement_max_elapsed_ts": "-100"},
                        {"wifi_gap_ms": "6000"},
                        {"wifi_observation_status": ""}, {"ap_C": "inf"}):
            with self.subTest(updates=updates):
                self.assertFalse(valid_fingerprint({**row, **updates}))

    def test_missing_wifi_preserves_photo_and_pdr_but_export_does_not_claim_exact_wifi(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "session.sess"
            session.mkdir()
            (session / "movements.dat").write_text('30000\t1000000\t"CHECKPOINT_PAUSE"\n')
            (root / "pic").mkdir()
            (root / "pic" / "1.jpg").write_bytes(b"photo")
            processed = root / "processed"
            old = fingerprint(timestamp=1000)
            current = {**fingerprint(timestamp=30000), "wifi_observation_status": "insufficient_fresh_aps",
                       "ap_A": "-110", "ap_B": "-110", "ap_C": "-110", "pdr_x": "42"}
            write_csv(processed / "wifi_pdr_features.csv", list(old), [old, current])
            summary = build_image_wifi_tables(root, [session], processed, Path("pic"), 1, "CHECKPOINT_PAUSE")
            point = csv_rows(processed / "image_wifi_points.csv")[0]
            self.assertEqual(summary["mapped_image_count"], 1)
            self.assertEqual(summary["fresh_wifi_point_count"], 0)
            self.assertEqual(point["pdr_x"], "42")
            self.assertEqual(point["trajectory_contour_id"], "route")
            self.assertIn("missing_wifi", point["mapping_status"])
            self.assertEqual(point["wifi_scan_index"], "")
            export_reconstruction_dataset(root, processed, root / "export", include_video_frames=False)
            image = csv_rows(root / "export" / "images_manifest.csv")[0]
            self.assertEqual(image["wifi_label_weight"], "0.0")
            self.assertEqual(image["bind_type"], "checkpoint_photo_without_valid_wifi")
            self.assertEqual(float(image["pdr_x"]), 42)
            self.assertIn("ap_A", image)
            self.assertIn("wifi_measurement_min_elapsed_ts", image)
            self.assertEqual(load_fingerprints(root / "export" / "wifi_fingerprint_library.csv")[1], [])

            fresh = {**fingerprint(timestamp=30000), "pdr_x": "42"}
            write_csv(processed / "wifi_pdr_features.csv", list(fresh), [fresh])
            build_image_wifi_tables(root, [session], processed, Path("pic"), 1, "CHECKPOINT_PAUSE")
            export_reconstruction_dataset(root, processed, root / "export", include_video_frames=False)
            image = csv_rows(root / "export" / "images_manifest.csv")[0]
            self.assertEqual(image["wifi_label_weight"], "1.0")
            self.assertEqual(image["bind_type"], "checkpoint_photo_with_fresh_wifi")
            self.assertTrue(valid_fingerprint(image))
            self.assertEqual(len(load_fingerprints(root / "export" / "wifi_fingerprint_library.csv")[1]), 1)

    def test_localization_has_no_success_without_observations_or_valid_library(self):
        fp = {"session": "s", "point_index": "1", "global_point_index": "1", "pdr_x": 0,
              "pdr_y": 0, "vector": {"ap_A": -60}}
        for scan, active in [({"ap_A": -110}, None), ({"ap_A": math.inf}, None), ({"ap_A": -60}, [])]:
            result = estimate_position(scan, ["ap_A"], [fp], active_fields=active)
            self.assertEqual(result["status"], "no_known_aps")
            self.assertIsNone(result["estimated_x"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "wifi_localization_model.json").write_text('{"old":true}')
            row = {**fingerprint(), "wifi_observation_status": "stale"}
            write_csv(root / "library.csv", list(row), [row])
            summary = run_wifi_positioning_pipeline(root / "library.csv", root)
            self.assertEqual(summary["status"], "no_valid_fingerprints")
            self.assertFalse((root / "wifi_localization_model.json").exists())
            metadata = root / "workspace" / "metadata"
            metadata.mkdir(parents=True)
            (metadata / "wifi_localization_model.json").write_text('{"old":true}')
            (metadata / "user_notes.txt").write_text("keep")
            copy_metadata(root, root / "workspace")
            self.assertFalse((metadata / "wifi_localization_model.json").exists())
            self.assertTrue((metadata / "user_notes.txt").exists())


class WifiRetrievalTests(unittest.TestCase):
    def test_rssi_changes_candidates_without_replacing_route_or_setting_geometry(self):
        left, right = fingerprint(), fingerprint(9, 30000, -62)
        right["trajectory_contour_id"] = "return_after_contour_break"
        pairs, route_edges, _ = build_constrained_match_pairs([left, right], 0, 0, 1, 1, 0, 1)
        self.assertEqual(pairs, [])
        edges, summary = build_wifi_recovery_edges([left, right], pairs)
        self.assertEqual(len(edges), 1)
        self.assertEqual(summary["valid_checkpoint_count"], 2)
        self.assertEqual(edges[0]["reason"], "wifi_fingerprint_recovery")
        self.assertEqual(build_wifi_recovery_edges([left, fingerprint(9, 30000, -90)], pairs)[0], [])
        self.assertEqual(build_wifi_recovery_edges([left, right], [("a1.jpg", "a9.jpg")])[0], [])
        self.assertEqual(build_wifi_recovery_edges([left, right], pairs, enabled=False)[0], [])
        self.assertEqual(build_wifi_recovery_edges([left, fingerprint(9, 1000)], pairs)[0], [])
        self.assertEqual(build_wifi_recovery_edges([left, {**right, "wifi_observation_status": ""}], pairs)[0], [])

    def test_wifi_candidates_require_geometry_even_if_p1_is_missing_or_capped(self):
        edges, _ = build_wifi_recovery_edges([fingerprint(), fingerprint(9, 30000)], [])
        pair = [(edges[0]["image_a"], edges[0]["image_b"])]
        for status, decision in [("disabled", None), ("ok", None), ("ok", "feature_supported"),
                                 ("ok", "geometry_supported")]:
            similarity = [] if decision is None else [{**edges[0], "pair_status": "ok", "visual_decision": decision}]
            selected, _, summary = select_visually_supported_match_pairs(pair, edges, similarity, {"status": status})
            self.assertEqual(bool(selected), decision == "geometry_supported")
            self.assertEqual(summary["selected_visual_recovery_pair_count"], 0)
            self.assertEqual(summary["selected_wifi_recovery_pair_count"], len(selected))

    def test_workspace_wires_wifi_retrieval_into_geometry_verified_pair_list(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reconstruction = root / "reconstruction"
            (reconstruction / "images").mkdir(parents=True)
            rows = [fingerprint(), fingerprint(9, 30000)]
            rows[1]["trajectory_contour_id"] = "return_after_contour_break"
            for row in rows:
                (reconstruction / "images" / row["image_file"]).write_bytes(b"image")
                row["image_rel_path"] = "images/" + row["image_file"]
            write_csv(reconstruction / "images_manifest.csv", list(rows[0]), rows)
            def analyze(workspace_dir, edge_rows, **kwargs):
                wifi = [row for row in edge_rows if row["reason"] == "wifi_fingerprint_recovery"]
                self.assertEqual(len(wifi), 1)
                evaluated = [{**row, "pair_status": "ok", "visual_decision": "geometry_supported"} for row in wifi]
                write_csv(workspace_dir / "image_similarity_edges.csv", list(evaluated[0]), evaluated)
                return {"status": "ok"}
            with patch("colmap_bridge.analyze_workspace_image_similarity", side_effect=analyze):
                summary = prepare_colmap_workspace(
                    root, reconstruction, root / "workspace", constrained_point_window=0,
                    constrained_slot_window=0, constrained_max_pdr_distance_m=1,
                )
            report = summary["constrained_match_pairs"]["wifi_recovery"]
            self.assertEqual(report["candidate_image_pair_count"], 1)
            self.assertEqual(report["selected_image_pair_count"], 1)
            self.assertTrue((root / "workspace" / "match_pairs.txt").read_text().strip())
            self.assertTrue(parse_args(["--skip-wifi-recovery"]).skip_wifi_recovery)

    def test_real_visual_evaluation_reports_wifi_separately_from_route(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/numpy unavailable")
        from image_similarity import analyze_workspace_image_similarity
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            (workspace / "images").mkdir()
            pixels = np.random.default_rng(13).integers(0, 256, (256, 256, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(workspace / "images" / "a1.jpg"), pixels))
            self.assertTrue(cv2.imwrite(str(workspace / "images" / "a9.jpg"), pixels))
            edges, _ = build_wifi_recovery_edges([fingerprint(), fingerprint(9, 30000)], [])
            summary = analyze_workspace_image_similarity(workspace, edges, recovery_neighbors=0)
            evidence = csv_rows(workspace / "image_similarity_edges.csv")
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["route_input_pair_count"], 0)
            self.assertEqual(summary["wifi_input_pair_count"], 1)
            self.assertEqual(summary["wifi_evaluated_pair_count"], 1)
            self.assertEqual(evidence[0]["reason"], "wifi_fingerprint_recovery")
            pair = [(edges[0]["image_a"], edges[0]["image_b"])]
            selected, _, _ = select_visually_supported_match_pairs(pair, edges, evidence, summary)
            self.assertEqual(bool(selected), evidence[0]["visual_decision"] == "geometry_supported")


class VideoTimingTests(unittest.TestCase):
    def test_budget_spans_clip_and_filtering_preserves_original_sample_times(self):
        interval = full_clip_interval(1005, 1, 240)
        self.assertGreater(interval * 239, 1000)
        self.assertEqual(full_clip_interval(10, 1, 240), 1)
        self.assertEqual(full_clip_interval(None, 1, 0), 1)
        for duration in (None, 0, math.nan):
            with self.assertRaises(ValueError):
                full_clip_interval(duration, 1, 240)
        rows = build_frame_rows(Path("."), Path("clip.mov"), [Path("frame_000003.jpg"), Path("frame_000240.jpg")],
                                [fingerprint()], interval, 1, source_frame_count=240,
                                frame_timestamps={"frame_000003.jpg": {"video_time_s": 8.4, "video_time_source": "decoded"}})
        self.assertEqual(float(rows[0]["video_time_s"]), 8.4)
        self.assertGreater(float(rows[1]["video_time_s"]), 1000)
        for row in rows:
            self.assertEqual(row["bind_type"], "visual_only")
            self.assertEqual(row["wifi_label_weight"], 0)
            self.assertEqual(row["pdr_x"], "")
            self.assertEqual(row["nearest_global_point_index"], "")

    def test_legacy_order_bound_video_does_not_reenter_route_matching(self):
        video = {**fingerprint(2), "source_type": "video_base_frame", "video_sequence_index": "1",
                 "nearest_global_point_index": "1", "trajectory_position_source": "checkpoint_order_interpolation"}
        pairs, _, _ = build_constrained_match_pairs([fingerprint(), video], 1, 1, 100, 1, 1, 10)
        self.assertEqual(pairs, [])

    def test_ffmpeg_uses_whole_clip_interval(self):
        with patch("video_frames.subprocess.run") as run:
            run.return_value.returncode = 0
            status, _ = extract_with_ffmpeg(Path("ffmpeg.exe"), Path("a.mov"), Path("frames"),
                                            full_clip_interval(1000, 1, 10), 10)
            self.assertEqual(status, "ok")
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-vf") + 1], "fps=0.01:start_time=0")
            self.assertEqual(command[command.index("-frames:v") + 1], "10")

    def test_unknown_duration_does_not_reuse_stale_frames_or_truncate_a_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "videos").mkdir()
            (root / "videos" / "clip.mov").write_bytes(b"unreadable")
            old = root / "processed" / "video_frames" / "clip" / "frame_000001.jpg"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"old")
            with patch("video_frames.probe_video_duration", return_value=None), patch("video_frames.extract_with_opencv") as extract:
                result = run_video_frame_pipeline(root, root / "processed", root / "videos")
            self.assertEqual(result["status"], "extract_failed")
            extract.assert_not_called()
            self.assertEqual(csv_rows(root / "processed" / "video_frames_manifest.csv"), [])

    def test_real_opencv_pipeline_samples_whole_video_without_route_binding(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/numpy unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            videos = root / "videos"
            videos.mkdir()
            writer = cv2.VideoWriter(str(videos / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(100):
                    writer.write(np.full((32, 32, 3), index * 2, dtype=np.uint8))
            finally:
                writer.release()
            with patch("video_frames.resolve_ffmpeg", return_value=None):
                summary = run_video_frame_pipeline(root, root / "processed", videos, .1, 4,
                                                   blur_threshold=0, keyframe_min_change=0, keyframe_min_gap=1)
            rows = csv_rows(root / "processed" / "video_frames_manifest.csv")
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(len(rows), 4)
            self.assertAlmostEqual(float(rows[-1]["video_time_s"]), 7.5, places=1)
            self.assertTrue(all(row["pdr_x"] == "" for row in rows))
            self.assertTrue(all(row["video_time_source"] == "opencv_decoded_timestamp" for row in rows))


if __name__ == "__main__":
    unittest.main()
