"""Coverage gates must detect lost video segments without assuming dense input."""
import math
import unittest

from sparse_quality import assess_sparse_quality, summarize_coverage, validate_policy


def video_fixture(times, clip="walk.mov", prefix="frame"):
    return {f"video/{prefix}_{index}.jpg": {"image_file": f"video/{prefix}_{index}.jpg",
            "video_name": clip, "video_time_s": str(stamp), "source_type": "video_base_frame"}
            for index, stamp in enumerate(times)}


def assess(metadata, registered=None, *, policy=None):
    names = list(metadata)
    summary = summarize_coverage(names, names if registered is None else registered, metadata)
    return assess_sparse_quality(summary, metadata, policy=policy)


class SparseQualityTests(unittest.TestCase):
    def test_healthy_dense_video_passes(self):
        metadata = video_fixture([index * 0.5 for index in range(200)])
        report = assess(metadata)
        self.assertTrue(report["ready_for_dense"])
        self.assertEqual(report["video_coverage"]["walk.mov"]["allowed_max_gap_s"], 10.0)
        self.assertIn("does not establish", report["interpretation"])

    def test_latest_like_low_registration_and_long_hole_fail(self):
        metadata = video_fixture([index * 1004.508333 / 1702 for index in range(1703)])
        names = list(metadata)
        # Retain endpoints but leave 73 seconds in the middle unregistered.
        keep = [name for name in names if not 531 < float(metadata[name]["video_time_s"]) < 604]
        registered = [keep[round(index * (len(keep) - 1) / 982)] for index in range(983)]
        report = assess(metadata, registered)
        self.assertFalse(report["ready_for_dense"])
        self.assertIn("low_global_registration", report["failure_codes"])
        self.assertIn("long_video_gap", report["failure_codes"])
        self.assertAlmostEqual(report["global_coverage"]["registered_ratio"], 983 / 1703)
        self.assertAlmostEqual(report["video_coverage"]["walk.mov"]["allowed_max_gap_s"], 20.09016666)
        self.assertGreater(report["video_coverage"]["walk.mov"]["max_uncovered_gap_s"], 73)
        self.assertLess(report["video_coverage"]["walk.mov"]["max_uncovered_gap_s"], 76)

    def test_high_global_registration_does_not_hide_missing_clip(self):
        metadata = video_fixture(range(200), "first.mov", "a")
        metadata.update(video_fixture(range(25), "second.mov", "b"))
        registered = [name for name, row in metadata.items() if row["video_name"] == "first.mov"]
        report = assess(metadata, registered)
        self.assertNotIn("low_global_registration", report["failure_codes"])
        self.assertIn("low_clip_registration", report["failure_codes"])
        self.assertFalse(report["ready_for_dense"])

    def test_lost_start_and_end_are_measured(self):
        metadata = video_fixture(range(200))
        names = list(metadata)
        for registered in (names[20:], names[:-20]):
            with self.subTest(first=registered[0]):
                report = assess(metadata, registered)
                self.assertNotIn("low_global_registration", report["failure_codes"])
                self.assertIn("long_video_gap", report["failure_codes"])
                self.assertEqual(report["video_coverage"]["walk.mov"]["max_uncovered_gap_s"], 20)

    def test_sparse_sampling_does_not_trigger_dense_video_gap_limit(self):
        metadata = video_fixture([index * 30 for index in range(30)])
        registered = [name for index, name in enumerate(metadata) if index % 5 != 2]
        report = assess(metadata, registered)
        self.assertTrue(report["ready_for_dense"], report["failures"])
        self.assertEqual(report["video_coverage"]["walk.mov"]["allowed_max_gap_s"], 90)
        self.assertEqual(report["video_coverage"]["walk.mov"]["max_uncovered_gap_s"], 60)

    def test_original_input_gap_is_not_an_sfm_loss(self):
        metadata = video_fixture(list(range(50)) + list(range(150, 200)))
        report = assess(metadata)
        self.assertTrue(report["ready_for_dense"])
        self.assertEqual(report["video_coverage"]["walk.mov"]["input_max_gap_s"], 101)
        self.assertEqual(report["video_coverage"]["walk.mov"]["allowed_max_gap_s"], 101)

    def test_short_sequence_has_warning_not_percentage_failure(self):
        metadata = video_fixture([index * 10 for index in range(10)])
        report = assess(metadata, list(metadata)[3:8])
        self.assertTrue(report["ready_for_dense"], report["failures"])
        self.assertTrue(report["warnings"])
        self.assertFalse(report["video_coverage"]["walk.mov"]["gate_applies"])
        self.assertFalse(assess(metadata, list(metadata)[:2])["ready_for_dense"])

    def test_old_model_summary_uses_metadata_baseline(self):
        metadata = video_fixture(range(100))
        summary = {"registered_images": 85, "points3D": 1000,
                   "video_coverage": {"walk.mov": {"input_frames": 100,
                   "registered_frames": 85, "max_uncovered_gap_s": 16}}}
        report = assess_sparse_quality(summary, metadata)
        self.assertEqual(report["global_coverage"]["input_images"], 100)
        self.assertEqual(report["video_coverage"]["walk.mov"]["input_median_gap_s"], 1)
        self.assertIn("long_video_gap", report["failure_codes"])

    def test_counts_only_inputs_are_supported(self):
        report = assess_sparse_quality({"input_images": 100, "registered_images": 85}, {})
        self.assertTrue(report["ready_for_dense"])
        self.assertEqual(report["video_coverage"], {})

    def test_unknown_timing_is_explicit_not_invented(self):
        summary = {"input_images": 100, "registered_images": 100,
                   "video_coverage": {"walk.mov": {"input_frames": 100,
                   "registered_frames": 100, "max_uncovered_gap_s": 60}}}
        report = assess_sparse_quality(summary)
        self.assertTrue(report["ready_for_dense"])
        self.assertEqual(report["video_coverage"]["walk.mov"]["timeline_status"], "not_evaluable")
        self.assertTrue(any("cannot be assessed" in message for message in report["warnings"]))

    def test_adapter_scopes_metadata_and_normalizes_paths(self):
        metadata = video_fixture(range(25))
        metadata.update(video_fixture(range(500), "unused.mov", "unused"))
        names = list(metadata)[:25]
        registered = [name.replace("/", "\\") for name in names]
        summary = summarize_coverage(names, registered, metadata)
        self.assertEqual(summary["registered_images"], 25)
        self.assertEqual(list(summary["video_coverage"]), ["walk.mov"])
        self.assertTrue(assess_sparse_quality(summary, metadata)["ready_for_dense"])

    def test_unexpected_registered_names_do_not_inflate_ratio(self):
        metadata = video_fixture(range(100))
        summary = summarize_coverage(metadata, list(metadata)[:60] + ["unknown.jpg"], metadata)
        report = assess_sparse_quality(summary, metadata)
        self.assertEqual(report["global_coverage"]["registered_images"], 60)
        self.assertIn("unexpected_registered_images", report["failure_codes"])

    def test_metadata_rows_and_duplicate_names(self):
        metadata = video_fixture(range(25))
        rows = list(metadata.values())
        summary = summarize_coverage(metadata, metadata, rows)
        self.assertTrue(assess_sparse_quality(summary, rows)["ready_for_dense"])
        summary = summarize_coverage(metadata, metadata, rows + rows[:1])
        self.assertFalse(assess_sparse_quality(summary, rows + rows[:1])["ready_for_dense"])

    def test_invalid_timestamp_cannot_disappear_from_denominator(self):
        for stamp in ("nan", "inf", "-1", "", None, True):
            with self.subTest(stamp=stamp):
                metadata = video_fixture(range(25))
                metadata[next(iter(metadata))]["video_time_s"] = stamp
                summary = summarize_coverage(metadata, metadata, metadata)
                self.assertEqual(summary["video_coverage"]["walk.mov"]["input_frames"], 25)
                self.assertFalse(assess_sparse_quality(summary, metadata)["ready_for_dense"])

    def test_invalid_summary_values_fail_closed(self):
        for value in (None, "100", True, -1, 20.5, math.nan, math.inf, 10 ** 400):
            with self.subTest(value=value):
                report = assess_sparse_quality({"input_images": 100, "registered_images": value})
                self.assertFalse(report["ready_for_dense"])
                self.assertTrue(report["failures"])
        for value in (None, "bad", [], {"registered_images": 100, "input_images": 10}):
            with self.subTest(summary=value):
                self.assertFalse(assess_sparse_quality(value)["ready_for_dense"])

    def test_invalid_gap_or_baseline_fail_closed(self):
        metadata = video_fixture(range(25))
        for key in ("max_uncovered_gap_s", "input_time_span_s", "input_max_gap_s", "input_median_gap_s", "timed_input_frames"):
            for value in (math.nan, math.inf, -1, None):
                with self.subTest(key=key, value=value):
                    summary = summarize_coverage(metadata, metadata, metadata)
                    summary["video_coverage"]["walk.mov"][key] = value
                    self.assertFalse(assess_sparse_quality(summary, metadata)["ready_for_dense"])

    def test_invalid_policy_never_silently_relaxes_gate(self):
        policies = [{"unknown": 1}, {"min_registered_ratio": math.nan}, {"max_gap_fraction": math.inf},
                    {"min_registered_ratio": 1.01}, {"min_clip_frames": 0}, {"input_gap_multiplier": 0.9},
                    {"min_gap_tolerance_s": 0}, {"min_images_for_ratio": True}, {"min_clip_frames": 2.5}]
        for policy in policies:
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    assess_sparse_quality({"input_images": 100, "registered_images": 100}, policy=policy)

    def test_contradictory_video_coverage_fails_closed(self):
        metadata = video_fixture(range(25))
        for key, value in (("max_uncovered_gap_s", 0), ("max_uncovered_gap_s", 2),
                           ("max_uncovered_gap_s", 26), ("timed_input_frames", 24),
                           ("input_median_gap_s", 2), ("registered_frames", 26)):
            with self.subTest(key=key, value=value):
                summary = summarize_coverage(metadata, metadata, metadata)
                summary["video_coverage"]["walk.mov"][key] = value
                self.assertFalse(assess_sparse_quality(summary, metadata)["ready_for_dense"])

    def test_custom_policy_is_recorded_and_applied(self):
        metadata = video_fixture(range(100))
        registered = list(metadata)[::2]
        report = assess(metadata, registered, policy={"min_registered_ratio": 0.5, "min_clip_registered_ratio": 0.5})
        self.assertTrue(report["ready_for_dense"])
        self.assertEqual(report["policy"]["min_registered_ratio"], 0.5)
        self.assertEqual(validate_policy()["min_registered_ratio"], 0.75)


if __name__ == "__main__":
    unittest.main()
