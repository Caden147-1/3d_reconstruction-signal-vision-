"""Code-freeze regressions for input identity, data validity and bounded sampling."""

import contextlib
import csv
import io
import json
import math
import os
import sqlite3
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np

import dense_surface_filter as surface
import run_colmap
from colmap_bridge import mask_candidates, mask_dir_summary, prepare_colmap_workspace
from colmap_workspace_check import (
    PENDING_NAME, PROVENANCE_NAME, begin_sparse_run, complete_sparse_run,
    inspect_workspace, validation_failures,
)
from ply_io import write_ply_vertices
from preprocess_pipeline import (
    Event, HeadingUnavailableError, build_pdr_tables, interpolate_angle,
    interpolate_pdr_state, quaternion_to_yaw, wrap_angle,
)
from reconstruction_quality import build_report, count_ply_vertices, summarize_dense
from test_review_fixes import PROPERTIES, make_text_model
from trajectory_contour import build_trajectory_contour, link_status, write_csv


def prepare_inputs(workspace):
    (workspace / "images").mkdir()
    for name in ("a.jpg", "b.jpg"):
        (workspace / "images" / name).write_bytes(b"old image")
    (workspace / "image_list.txt").write_text("a.jpg\nb.jpg\n")
    (workspace / "match_pairs.txt").write_text("a.jpg b.jpg\n")
    (workspace / "run_colmap_sparse.bat").write_text("@echo off\n")


def create_sparse_outputs(workspace):
    with contextlib.closing(sqlite3.connect(workspace / "database.db")) as db:
        db.execute("CREATE TABLE images (name TEXT)")
        db.executemany("INSERT INTO images VALUES (?)", [("a.jpg",), ("b.jpg",)])
        db.commit()
    make_text_model(workspace / "sparse" / "0")


def verified_workspace(workspace):
    prepare_inputs(workspace)
    begin_sparse_run(workspace)
    create_sparse_outputs(workspace)
    complete_sparse_run(workspace)


class ProvenanceTests(unittest.TestCase):
    def test_old_runner_is_rejected_before_rebuild_cleanup(self):
        for mode in ("full", "sparse", "automatic"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                workspace = Path(temp)
                for name in run_colmap.SCRIPT_BY_MODE.values():
                    (workspace / name).write_text("@echo off\n")
                final = workspace / "final.ply"
                final.write_bytes(b"existing reconstruction")
                with patch.object(run_colmap, "reset_existing_colmap_outputs") as reset, patch.object(run_colmap, "call_batch") as call, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run_colmap.run_colmap_script(workspace, mode), 2)
                reset.assert_not_called()
                call.assert_not_called()
                self.assertEqual(final.read_bytes(), b"existing reconstruction")

    def test_successful_run_and_repeat_read_only_inspection(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            verified_workspace(workspace)
            receipt = (workspace / PROVENANCE_NAME).read_bytes()
            for _ in range(2):
                report = inspect_workspace(workspace)
                self.assertEqual(report["provenance"]["status"], "verified")
                self.assertEqual(validation_failures(report, require_database_coverage=True, require_sparse_model=True), [])
            self.assertEqual((workspace / PROVENANCE_NAME).read_bytes(), receipt)
            self.assertFalse((workspace / PENDING_NAME).exists())

    def test_same_name_size_and_mtime_do_not_hide_changed_image_content(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            verified_workspace(workspace)
            image = workspace / "images" / "a.jpg"
            before = image.stat()
            image.write_bytes(b"new image")
            os.utime(image, ns=(before.st_atime_ns, before.st_mtime_ns))
            report = inspect_workspace(workspace)
            self.assertEqual(report["database"]["missing_names"], [])
            self.assertEqual(report["dense_input_sparse"]["extra_names"], [])
            self.assertEqual(report["provenance"]["status"], "input_mismatch")
            self.assertTrue(validation_failures(report, require_sparse_model=True))

    def test_matching_and_runner_changes_invalidate_the_record(self):
        for name in ("match_pairs.txt", "run_colmap_sparse.bat", "pdr_alignment_refs.txt", "metadata/images_manifest.csv"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                workspace = Path(temp)
                verified_workspace(workspace)
                path = workspace / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("changed")
                self.assertEqual(inspect_workspace(workspace)["provenance"]["status"], "input_mismatch")

    def test_changed_model_or_database_cannot_reuse_input_proof(self):
        for kind in ("camera", "database", "aligned"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                workspace = Path(temp)
                verified_workspace(workspace)
                if kind == "camera":
                    path = workspace / "sparse" / "0" / "cameras.txt"
                    path.write_text(path.read_text().replace("500", "501"))
                elif kind == "database":
                    with contextlib.closing(sqlite3.connect(workspace / "database.db")) as db:
                        db.execute("CREATE TABLE changed (value TEXT)")
                        db.commit()
                else:
                    make_text_model(workspace / "sparse_pdr_aligned" / "0")
                self.assertEqual(inspect_workspace(workspace)["provenance"]["status"], "output_mismatch")

    def test_old_outputs_cannot_be_certified_after_the_fact(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            prepare_inputs(workspace)
            create_sparse_outputs(workspace)
            self.assertEqual(inspect_workspace(workspace)["provenance"]["status"], "unverified")
            with self.assertRaises(ValueError):
                begin_sparse_run(workspace)
            with self.assertRaises(OSError):
                complete_sparse_run(workspace)
            self.assertFalse((workspace / PROVENANCE_NAME).exists())

    def test_sqlite_sidecars_prevent_recording_a_clean_start(self):
        for suffix in ("-wal", "-shm", "-journal"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temp:
                workspace = Path(temp)
                prepare_inputs(workspace)
                sidecar = workspace / ("database.db" + suffix)
                sidecar.write_bytes(b"old database state")
                with self.assertRaisesRegex(ValueError, "SQLite sidecar"):
                    begin_sparse_run(workspace)
                self.assertFalse((workspace / PENDING_NAME).exists())
                self.assertEqual(sidecar.read_bytes(), b"old database state")

    def test_changed_inputs_during_run_leave_no_completion_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            prepare_inputs(workspace)
            begin_sparse_run(workspace)
            create_sparse_outputs(workspace)
            (workspace / "images" / "a.jpg").write_bytes(b"changed during run")
            with self.assertRaises(ValueError):
                complete_sparse_run(workspace)
            self.assertFalse((workspace / PROVENANCE_NAME).exists())
            self.assertTrue((workspace / "sparse" / "0" / "images.txt").exists())
            self.assertEqual(inspect_workspace(workspace)["provenance"]["status"], "incomplete_run")

    def test_invalid_receipt_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            verified_workspace(workspace)
            for content in ("null", "{}", '{"schema_version":2,"status":"complete"}'):
                (workspace / PROVENANCE_NAME).write_text(content)
                self.assertEqual(inspect_workspace(workspace)["provenance"]["status"], "invalid_record")

    def test_mask_changes_are_part_of_input_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            prepare_inputs(workspace)
            masks = workspace / "masks"
            masks.mkdir()
            (masks / "a.jpg.png").write_bytes(b"mask")
            (workspace / "colmap_workspace_summary.json").write_text(json.dumps({"colmap_mask": {"enabled": True, "path": str(masks)}}))
            begin_sparse_run(workspace)
            create_sparse_outputs(workspace)
            complete_sparse_run(workspace)
            (masks / "b.jpg.png").write_bytes(b"new mask")
            self.assertEqual(inspect_workspace(workspace)["provenance"]["status"], "input_mismatch")


class DenseValidationTests(unittest.TestCase):
    def test_header_only_or_nonfinite_ply_cannot_make_quality_report_ok(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            verified_workspace(workspace)
            dense = workspace / "dense"
            dense.mkdir()
            header = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
            final = dense / "final.ply"
            for payload in (header, header + "nan 0 0\n"):
                final.write_text(payload)
                report = build_report(workspace, workspace / "processed", workspace / "reconstruction", workspace)
                self.assertEqual(report["status"], "dense_needs_review")
                self.assertIsNone(report["dense"]["final_vertex_count"])
                self.assertEqual(report["dense"]["ply_validation"]["final"]["status"], "invalid")
                self.assertIsNone(count_ply_vertices(final))
            write_ply_vertices(final, PROPERTIES, [(0, 0, 0)])
            self.assertEqual(build_report(workspace, workspace / "processed", workspace / "reconstruction", workspace)["status"], "ok")

    def test_dense_validation_is_read_only_and_checks_each_existing_cloud(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            dense = workspace / "dense"
            dense.mkdir()
            write_ply_vertices(dense / "final.ply", PROPERTIES, [(1, 2, 3)])
            (dense / "fused.ply").write_bytes(b"truncated")
            before = {path.name: path.read_bytes() for path in dense.iterdir()}
            summary = summarize_dense(workspace)
            self.assertEqual(summary["ply_validation"]["final"]["status"], "validated")
            self.assertEqual(summary["ply_validation"]["fused"]["status"], "invalid")
            self.assertEqual(summary["ply_validation"]["filtered"]["status"], "missing")
            self.assertEqual(before, {path.name: path.read_bytes() for path in dense.iterdir()})

    def test_statistics_buffer_is_bounded_and_preserves_the_old_global_stride(self):
        references, alive_counts = [], []
        original_positions = surface.positions
        def chunks(_path):
            for start in range(0, 100000, 1000):
                chunk = np.zeros(1000, dtype=[(axis, "f8") for axis in "xyz"])
                chunk["x"] = np.arange(start, start + 1000)
                yield chunk
        def tracked_positions(chunk):
            xyz = original_positions(chunk)
            references.append(weakref.ref(xyz))
            alive_counts.append(sum(ref() is not None for ref in references))
            return xyz
        with patch.object(surface, "vertex_chunks", side_effect=chunks), patch.object(surface, "positions", side_effect=tracked_positions):
            sample, stride, finite = surface.statistics_sample(Path("unused"), 100000, max_samples=17)
        np.testing.assert_array_equal(sample[:, 0], np.arange(100000)[::stride])
        self.assertEqual(finite, 100000)
        self.assertLessEqual(len(sample), 17)
        self.assertLessEqual(max(alive_counts), 2)
        self.assertFalse(any(ref() is not None for ref in references))


class TrajectoryValidityTests(unittest.TestCase):
    def test_missing_nonfinite_or_stale_heading_is_not_zero_heading(self):
        with self.assertRaises(HeadingUnavailableError):
            interpolate_angle([], 1000)
        self.assertEqual(interpolate_angle([(1000, 0, 0.0)], 1000), 0.0)
        for samples, timestamp in [([(1000, 0, 0)], 3000), ([(3000, 0, 0)], 1000),
                                   ([(0, 0, 0), (3000, 0, 1)], 1500), ([(0, 0, math.nan)], 0)]:
            with self.assertRaises(HeadingUnavailableError):
                interpolate_angle(samples, timestamp)
        self.assertIsNone(quaternion_to_yaw([0, 0, math.nan, 1]))
        with self.assertRaises(ValueError):
            wrap_angle(math.inf)

    def test_heading_failure_does_not_export_a_synthetic_straight_route(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "s.sess"
            events = [Event(session.name, 0, 1000, "START"), Event(session.name, 2000, 3000, "CHECKPOINT_PAUSE")]
            with patch("preprocess_pipeline.read_events", return_value=events), patch("preprocess_pipeline.read_sensor", return_value=[]), patch("preprocess_pipeline.read_heading_samples", return_value=[]), patch("preprocess_pipeline.detect_steps_in_segment", return_value=[(1000, 2000, 1.0, 1.0)]):
                summary = build_pdr_tables([session], root, .7, 0)
            self.assertEqual(summary[session.name]["status"], "unavailable_heading")
            with (root / "pdr_path.csv").open(newline="") as handle:
                self.assertTrue(all(row["pdr_x"] == "" and row["pdr_y"] == "" for row in csv.DictReader(handle)))

    def test_long_gap_breaks_unless_documented_pause_and_no_displacement(self):
        left = {"elapsed_ts": 0, "pdr_x": 0, "pdr_y": 0, "step_index": "1", "point_type": "STEP"}
        right = {**left, "elapsed_ts": 20000, "step_index": "2", "pdr_x": .7}
        self.assertEqual(link_status(left, right, 10000, 1.5, .05)[:2], (False, "time_gap_break"))
        self.assertFalse(link_status(left, {**right, "pdr_x": 0}, 10000, 1.5, .05)[0])
        paused = {**left, "point_type": "CHECKPOINT_PAUSE"}
        restarted = {**left, "point_type": "RESTART", "elapsed_ts": 60000}
        self.assertEqual(link_status(paused, restarted, 10000, 1.5, .05)[:2], (True, "documented_checkpoint_pause"))
        self.assertFalse(link_status(paused, {**restarted, "step_index": "2"}, 10000, 1.5, .05)[0])
        self.assertTrue(link_status(left, {**right, "elapsed_ts": 500}, 10000, 1.5, .05)[0])

    def test_contour_gap_is_not_interpolated_into_an_xy_pose(self):
        points = [{"elapsed_ts": 0, "pdr_x": 0, "pdr_y": 0, "point_type": "STEP", "trajectory_contour_id": "first",
                   "trajectory_contour_distance_m": 0, "trajectory_distance_m": 0},
                  {"elapsed_ts": 20000, "pdr_x": .7, "pdr_y": 0, "point_type": "STEP", "trajectory_contour_id": "second",
                   "trajectory_contour_distance_m": 0, "trajectory_distance_m": 0},
                  {"elapsed_ts": 20500, "pdr_x": 1.4, "pdr_y": 0, "point_type": "STEP", "trajectory_contour_id": "second",
                   "trajectory_contour_distance_m": .7, "trajectory_distance_m": .7}]
        self.assertIsNone(interpolate_pdr_state(points, 10000)["pdr_x"])
        self.assertEqual(interpolate_pdr_state(points, 10000)["status"], "contour_break")
        self.assertAlmostEqual(interpolate_pdr_state(points, 20000)["pdr_x"], .7)
        self.assertAlmostEqual(interpolate_pdr_state(points, 20250)["pdr_x"], 1.05)

    def test_contour_output_has_no_edge_across_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"session": "s", "elapsed_ts": 0, "point_type": "STEP", "step_index": 1, "pdr_x": 0, "pdr_y": 0},
                    {"session": "s", "elapsed_ts": 20000, "point_type": "STEP", "step_index": 2, "pdr_x": .7, "pdr_y": 0},
                    {"session": "s", "elapsed_ts": 20500, "point_type": "STEP", "step_index": 3, "pdr_x": 1.4, "pdr_y": 0}]
            write_csv(root / "path.csv", list(rows[0]), rows)
            report = build_trajectory_contour(root / "path.csv", root / "contour.csv", root / "edges.csv", root / "summary.json")
            self.assertEqual(report["contour_count"], 2)
            self.assertEqual(report["break_counts"]["time_gap_break"], 1)
            self.assertEqual(report["observed_edge_count"], 1)


class MaskConventionTests(unittest.TestCase):
    def test_only_colmap_relative_name_plus_png_counts_as_a_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "anchors").mkdir()
            (root / "anchors" / "a.png").write_bytes(b"wrong name")
            (root / "anchors" / "a.jpg").write_bytes(b"wrong name")
            self.assertEqual(mask_dir_summary(root, ["anchors/a.jpg"])["matched_image_count"], 0)
            (root / "anchors" / "a.jpg.png").write_bytes(b"correct path")
            self.assertEqual(mask_dir_summary(root, ["anchors/a.jpg", "anchors/b.jpg"])["matched_image_count"], 1)
            self.assertEqual(mask_candidates(root, "anchors/a.jpg"), [root / "anchors" / "a.jpg.png"])

    def test_generated_batches_record_inputs_before_features_and_outputs_after_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reconstruction = root / "reconstruction"
            (reconstruction / "images").mkdir(parents=True)
            (reconstruction / "images" / "a.jpg").write_bytes(b"image")
            (reconstruction / "images_manifest.csv").write_text("image_file,source_type\na.jpg,wifi_anchor_photo\n")
            workspace = root / "workspace"
            prepare_colmap_workspace(root, reconstruction, workspace, matcher="exhaustive", run_image_similarity=False)
            sparse = (workspace / "run_colmap_sparse.bat").read_text()
            self.assertLess(sparse.index("--record-sparse-start"), sparse.index('"%COLMAP_EXE%" feature_extractor'))
            self.assertLess(sparse.index('"%COLMAP_EXE%" mapper'), sparse.index("--record-sparse-complete"))
            self.assertLess(sparse.index("--record-sparse-complete"), sparse.index("reconstruction_quality.py"))
            automatic = (workspace / "run_colmap_automatic.bat").read_text()
            self.assertIn("--runner run_colmap_automatic.bat", automatic)
            self.assertIn("--record-sparse-complete", automatic)
            for name in ("run_colmap_sparse.bat", "run_colmap_automatic.bat"):
                self.assertEqual(run_colmap.validate_sparse_runner_provenance(workspace / name), 0)


if __name__ == "__main__":
    unittest.main()
