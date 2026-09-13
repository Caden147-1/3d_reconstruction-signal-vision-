"""Behavioral regressions for candidate recall, model integrity and launch wiring."""

import argparse
import contextlib
import csv
import io
import json
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as launcher
import finalize_dense_outputs as finalizer
from colmap_bridge import build_constrained_match_pairs, prepare_colmap_workspace, select_visually_supported_match_pairs
from colmap_workspace_check import (
    dense_input_model, inspect_sparse_model, inspect_workspace,
    read_binary_image_names, read_text_image_names, validation_failures,
)
from image_similarity import (
    IMAGE_SIMILARITY_FIELDS, ImageFeatures, SimilarityConfig,
    analyze_image_similarity_edges, load_cv_modules, recovery_candidates,
    visual_decision, visual_overlap_score, write_csv_rows,
)
from ply_io import validate_ply, write_ply_vertices
from point_cloud_filter import reservoir_sample_rows, sampling_quality_warning
from preprocess_pipeline import parse_args
from reconstruction_quality import build_report, render_markdown


PROPERTIES = [("x", "float"), ("y", "float"), ("z", "float")]


def anchor(index, distance, contour="route", slot=1):
    return {
        "image_file": f"a{index}.jpg", "source_type": "wifi_anchor_photo", "session": "s",
        "global_point_index": str(index), "image_slot": str(slot), "anchor_slot_count": "20",
        "pdr_x": str(distance), "pdr_y": "0", "trajectory_contour_id": contour,
        "trajectory_contour_distance_m": str(distance),
    }


def make_text_model(model, names=("a.jpg", "b.jpg")):
    model.mkdir(parents=True)
    (model / "cameras.txt").write_text("1 SIMPLE_RADIAL 640 480 500 320 240 0\n", encoding="ascii")
    (model / "images.txt").write_text(
        "".join(f"{i} 1 0 0 0 0 0 0 1 {name}\n1 2 -1 3 4 -1 5 6 -1 7 8 -1\n" for i, name in enumerate(names, 1)),
        encoding="ascii",
    )
    (model / "points3D.txt").write_text("1 0 0 1 100 100 100 0.1 1 0 2 0\n", encoding="ascii")


def make_binary_model(model):
    model.mkdir(parents=True)
    (model / "cameras.bin").write_bytes(struct.pack("<QiiQQdddd", 1, 1, 2, 640, 480, 500, 320, 240, 0))
    images = struct.pack("<Q", 2)
    for i, name in enumerate(("a.jpg", "b.jpg"), 1):
        images += struct.pack("<idddddddi", i, 1, 0, 0, 0, 0, 0, 0, 1) + name.encode() + b"\0" + struct.pack("<Q", 1) + struct.pack("<ddq", 1, 2, 1)
    (model / "images.bin").write_bytes(images)
    (model / "points3D.bin").write_bytes(struct.pack("<QQdddBBBdQiiii", 1, 1, 0, 0, 1, 100, 100, 100, .1, 2, 1, 0, 2, 0))


class CandidateTests(unittest.TestCase):
    def test_appearance_cannot_claim_features_or_duplicates_without_matches(self):
        for hash_distance, hist in ((32, .5), (0, 1.0)):
            score = visual_overlap_score(1 - hash_distance / 64, hist, 0, 0, 0.0)
            self.assertGreater(score, .22)
            self.assertEqual(visual_decision(SimilarityConfig(), hash_distance, hist, 0, 0, 0.0, score), "no_feature_overlap")
        self.assertEqual(visual_decision(SimilarityConfig(), 0, 1, 0, 50, 1, 1), "no_feature_overlap")

    def test_all_rejected_pairs_stay_rejected(self):
        pairs = [("a", "b"), ("b", "c")]
        rows = [{"image_a": a, "image_b": b, "pair_status": "ok", "visual_decision": "no_feature_overlap"} for a, b in pairs]
        selected, audit, summary = select_visually_supported_match_pairs(pairs, [], rows, {"status": "ok"})
        self.assertEqual(selected, [])
        self.assertEqual(summary["status"], "empty_selection")
        self.assertEqual(summary["rejected_pair_count"], 2)
        self.assertTrue(all(row["selected"] == 0 for row in audit))

    def test_unknown_route_pairs_can_remain_but_unknown_recovery_cannot(self):
        pairs = [("a", "b"), ("a", "c")]
        edges = [{"image_a": "a", "image_b": "c", "reason": "visual_recovery"}]
        selected, _, summary = select_visually_supported_match_pairs(pairs, edges, [], {"status": "skipped"})
        self.assertEqual(selected, [("a", "b")])
        self.assertEqual(summary["selected_visual_recovery_pair_count"], 0)

    def test_one_missing_anchor_does_not_disable_other_contours(self):
        rows = [anchor(1, 0), anchor(2, 1), anchor(3, 1, "other"), anchor(4, 1, ""), anchor(5, 2, "")]
        pairs, _, summary = build_constrained_match_pairs(rows, 1, 1, 12, 1, 1, 8)
        self.assertIn(("a1.jpg", "a2.jpg"), pairs)
        self.assertNotIn(("a2.jpg", "a3.jpg"), pairs)
        self.assertNotIn(("a3.jpg", "a4.jpg"), pairs)
        self.assertIn(("a4.jpg", "a5.jpg"), pairs)
        self.assertEqual(summary["trajectory_constraint_mode"], "partial_ordered_contour")

    def test_recovery_retrieves_pairs_outside_route_or_slot_gate(self):
        class Histogram:
            HISTCMP_CORREL = 0
            @staticmethod
            def compareHist(a, b, mode):
                return 1.0 if a == b else 0.0
        features = [ImageFeatures(name, "ok", dhash=value, hist=value, keypoint_count=30) for name, value in (("a", 0), ("b", 0), ("c", 255))]
        rows = recovery_candidates(features, {("a", "c")}, Histogram(), 1)
        pairs = {(r["image_a"], r["image_b"]) for r in rows}
        self.assertIn(("a", "b"), pairs)
        self.assertNotIn(("a", "c"), pairs)

    def test_workspace_imports_verified_recovery_and_reports_matcher_scope(self):
        for matcher in ("constrained", "exhaustive", "hybrid", "sequential"):
            with self.subTest(matcher=matcher), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                reconstruction = root / "reconstruction"
                (reconstruction / "images").mkdir(parents=True)
                rows = [anchor(1, 0), anchor(2, 1), anchor(3, 100, slot=10)]
                write_csv_rows(reconstruction / "images_manifest.csv", list(rows[0]), rows)
                for row in rows:
                    (reconstruction / "images" / row["image_file"]).write_bytes(b"test")

                def analyze(**kwargs):
                    evidence = [dict(e, pair_status="ok", visual_decision="geometry_supported") for e in kwargs["edge_rows"]]
                    evidence.extend([
                        {"image_a": "anchors/a1.jpg", "image_b": "anchors/a3.jpg", "reason": "visual_recovery", "pair_status": "ok", "visual_decision": "geometry_supported"},
                        {"image_a": "anchors/a2.jpg", "image_b": "anchors/a3.jpg", "reason": "visual_recovery", "pair_status": "ok", "visual_decision": "feature_overlap"},
                    ])
                    write_csv_rows(kwargs["workspace_dir"] / "image_similarity_edges.csv", IMAGE_SIMILARITY_FIELDS, evidence)
                    return {"status": "ok"}

                workspace = root / "workspace"
                with patch("colmap_bridge.analyze_workspace_image_similarity", side_effect=analyze):
                    summary = prepare_colmap_workspace(root, reconstruction, workspace, colmap_exe=root / "none.exe", matcher=matcher, run_meshing=matcher == "hybrid")
                pair_list = (workspace / "match_pairs.txt").read_text()
                self.assertIn("anchors/a1.jpg anchors/a3.jpg", pair_list)
                self.assertNotIn("anchors/a2.jpg anchors/a3.jpg", pair_list)
                self.assertNotIn("anchors/a1.jpg anchors/a3.jpg", (workspace / "match_pairs_route.txt").read_text())
                selected = summary["constrained_match_pairs"]["p1_pair_filter"]
                self.assertEqual(selected["selected_visual_recovery_pair_count"], 1)
                self.assertEqual(selected["pair_gate_enforced"], matcher == "constrained")
                self.assertEqual(selected["pair_list_used_by_sparse_matcher"], matcher in {"constrained", "hybrid"})
                self.assertEqual(summary["image_similarity"]["used_for_colmap_pair_selection"], matcher in {"constrained", "hybrid"})
                self.assertIn("Pair gate enforced", render_markdown(build_report(root, root / "processed", reconstruction, workspace)))
                dense = (workspace / "run_colmap_dense.bat").read_text()
                self.assertEqual('"%COLMAP_EXE%" poisson_mesher' in dense, matcher == "hybrid")
                self.assertEqual('set "FINALIZE_KEEP=--keep-intermediate"' in dense.split('if "%KEEP_INTERMEDIATE_PLY%"')[0], matcher == "hybrid")
                self.assertEqual(summary["cleanup_intermediate_ply"], matcher != "hybrid")
                self.assertIn('if "%KEEP_INTERMEDIATE_PLY%"=="1"', dense)

    def test_real_opencv_recovery_and_counts(self):
        cv2, np, error = load_cv_modules()
        if cv2 is None or np is None:
            self.skipTest(error)
        cv2.setRNGSeed(42)
        rng = np.random.default_rng(42)
        textured = rng.integers(0, 256, (320, 480, 3), dtype=np.uint8)
        transform = np.float32([[1, .015, 12], [.01, 1, 5], [.00002, .00001, 1]])
        shifted = cv2.warpPerspective(textured, transform, (480, 320))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, pixels in (("a.png", textured), ("b.png", shifted), ("c.png", np.zeros_like(textured))):
                self.assertTrue(cv2.imwrite(str(root / name), pixels))
            route = [{"image_a": "a.png", "image_b": "c.png", "reason": "route"}]
            output = root / "evidence.csv"
            summary = analyze_image_similarity_edges(root, route, output, config=SimilarityConfig(recovery_neighbors=1), recovery_image_names=["a.png", "b.png", "c.png"])
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            recovery = [row for row in rows if row["reason"] == "visual_recovery"]
            self.assertEqual(len(recovery), 1)
            self.assertEqual(recovery[0]["visual_decision"], "geometry_supported")
            self.assertGreaterEqual(int(recovery[0]["fundamental_inlier_count"]), SimilarityConfig().geometry_min_inliers)
            self.assertEqual((summary["input_pair_count"], summary["route_input_pair_count"], summary["evaluated_pair_count"]), (2, 1, 2))
            self.assertEqual(summary["visual_recovery"]["geometry_supported_pair_count"], 1)
            disabled = analyze_image_similarity_edges(root, route, output, config=SimilarityConfig(recovery_neighbors=0), recovery_image_names=["a.png", "b.png", "c.png"])
            self.assertEqual(disabled["visual_recovery"]["status"], "disabled")
            self.assertEqual(disabled["unique_image_count"], 2)


class OutputTests(unittest.TestCase):
    def test_sampling_fills_budget_and_does_not_amplify_sparse_outliers(self):
        # Dense room surfaces occupy little of a large 3D bounding box.
        wall = [(x / 50, y / 50, 0.0) for x in range(100) for y in range(100)]
        outliers = [(float(i), float(i % 17), 10.0) for i in range(100)]
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "surfaces.ply"
            write_ply_vertices(source, PROPERTIES, wall + outliers)
            args = (source, PROPERTIES, [(-1, 101), (-1, 101), (-1, 11)], (0, 0, 0), 1000, 2000)
            rows, summary = reservoir_sample_rows(*args)
            repeated, _ = reservoir_sample_rows(*args)
            self.assertEqual(rows, repeated)
            self.assertEqual(len(rows), 2000)
            self.assertEqual(summary["budget_utilization"], 1.0)
            self.assertLess(sum(row[2] == 10 for row in rows), 50)
            self.assertGreater(len({row[0] for row in rows if row[2] == 0}), 90)
            self.assertGreater(len({row[1] for row in rows if row[2] == 0}), 90)

    def test_recorded_underfilled_filter_is_not_promoted(self):
        audit = {"max_points": 1200000, "source_vertex_count": 18059032, "output_vertex_count": 133560}
        self.assertTrue(sampling_quality_warning(audit))
        with tempfile.TemporaryDirectory() as temp:
            dense = Path(temp)
            filtered = dense / "fused_filtered.ply"
            write_ply_vertices(filtered, PROPERTIES, [(1, 2, 3)])
            write_ply_vertices(dense / "fused.ply", PROPERTIES, [(1, 2, 3), (4, 5, 6)])
            audit.update(output=str(filtered), output_vertex_count=1)
            (dense / "point_filter_summary.json").write_text(json.dumps(audit))
            rejected = []
            self.assertEqual(finalizer.choose_source(dense, rejected=rejected)[0], "raw_point_cloud")
            self.assertEqual(len(rejected), 1)
    def test_validator_rejects_truncated_nonfinite_and_out_of_range_meshes(self):
        header = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        mesh = header + "element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n"
        invalid = [b"", (header + "end_header\n0 0 0\n").encode(), (header + "end_header\nnan 0 0\n1 0 0\n0 1 0\n").encode(), mesh.encode(), (mesh + "3 0 1 3\n").encode()]
        invalid.append((header + "element face 1\nproperty int bogus\nend_header\n0 0 0\n1 0 0\n0 1 0\n1\n").encode())
        for endian, name in (("<", "little"), (">", "big")):
            invalid.append(f"ply\nformat binary_{name}_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n".encode() + struct.pack(endian + "fff", 0, 0, 0))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mesh.ply"
            for payload in invalid:
                with self.subTest(payload=payload[:50]):
                    path.write_bytes(payload)
                    with self.assertRaises(ValueError):
                        validate_ply(path)
            path.write_text(mesh + "3 0 1 2\n")
            self.assertEqual(validate_ply(path), {"vertex": 3, "face": 1})

    def test_binary_vertex_validation_in_both_byte_orders(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cloud.ply"
            for endian, name in (("<", "little"), (">", "big")):
                header = f"ply\nformat binary_{name}_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n".encode()
                path.write_bytes(header + struct.pack(endian + "fff", 1, 2, 3))
                self.assertEqual(validate_ply(path), {"vertex": 1})
                path.write_bytes(header + struct.pack(endian + "fff", 1, float("nan"), 3))
                with self.assertRaises(ValueError):
                    validate_ply(path)

    def test_invalid_filtered_cloud_falls_back_to_raw_and_keeps_unrelated_ply(self):
        with tempfile.TemporaryDirectory() as temp:
            dense = Path(temp)
            raw = dense / "fused.ply"
            write_ply_vertices(raw, PROPERTIES, [(1, 2, 3)])
            expected = raw.read_bytes()
            (dense / "fused_filtered.ply").write_bytes(b"")
            (dense / "user_capture.ply").write_bytes(b"user")
            args = argparse.Namespace(dense_dir=dense, summary=dense / "summary.json", keep_intermediate=False, prefer_mesh=False)
            with patch.object(finalizer, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(finalizer.main(), 0)
            self.assertEqual((dense / "final.ply").read_bytes(), expected)
            self.assertEqual(raw.read_bytes(), expected)
            self.assertEqual((dense / "user_capture.ply").read_bytes(), b"user")
            summary = json.loads(args.summary.read_text())
            self.assertEqual(summary["source_kind"], "raw_point_cloud")
            self.assertEqual(len(summary["rejected_sources"]), 1)

    def test_failed_promotion_preserves_previous_final_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            dense = Path(temp)
            write_ply_vertices(dense / "fused.ply", PROPERTIES, [(1, 2, 3)])
            write_ply_vertices(dense / "fused_filtered.ply", PROPERTIES, [(1, 2, 3)])
            write_ply_vertices(dense / "final.ply", PROPERTIES, [(4, 5, 6)])
            previous = (dense / "final.ply").read_bytes()
            args = argparse.Namespace(dense_dir=dense, summary=None, keep_intermediate=False, prefer_mesh=False)
            with patch.object(finalizer, "parse_args", return_value=args), patch.object(finalizer.shutil, "copy2", side_effect=OSError("test copy failure")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(finalizer.main(), 1)
            self.assertEqual((dense / "final.ply").read_bytes(), previous)
            self.assertTrue((dense / "fused.ply").exists())
            self.assertEqual(list(dense.glob("*.tmp")), [])

    def test_repeated_finalization_preserves_filter_and_accepts_new_filtered_result(self):
        with tempfile.TemporaryDirectory() as temp:
            dense = Path(temp)
            raw = dense / "fused.ply"
            filtered = dense / "fused_filtered.ply"
            final = dense / "final.ply"
            write_ply_vertices(raw, PROPERTIES, [(1, 2, 3), (4, 5, 6)])
            write_ply_vertices(filtered, PROPERTIES, [(1, 2, 3)])
            expected = filtered.read_bytes()
            args = argparse.Namespace(dense_dir=dense, summary=None, keep_intermediate=False, prefer_mesh=False)
            with patch.object(finalizer, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(finalizer.main(), 0)
                self.assertFalse(filtered.exists())
                self.assertEqual(finalizer.main(), 0)
                self.assertEqual(final.read_bytes(), expected)
                write_ply_vertices(filtered, PROPERTIES, [(4, 5, 6)])
                updated = filtered.read_bytes()
                self.assertEqual(finalizer.main(), 0)
                self.assertEqual(final.read_bytes(), updated)
                args.allow_raw_replacement = True
                self.assertEqual(finalizer.main(), 0)
                self.assertEqual(final.read_bytes(), raw.read_bytes())

    def test_invalid_filtered_cannot_downgrade_existing_final_to_valid_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            dense = Path(temp)
            write_ply_vertices(dense / "fused.ply", PROPERTIES, [(1, 2, 3), (4, 5, 6)])
            write_ply_vertices(dense / "final.ply", PROPERTIES, [(1, 2, 3)])
            (dense / "fused_filtered.ply").write_bytes(b"partial")
            expected = (dense / "final.ply").read_bytes()
            args = argparse.Namespace(dense_dir=dense, summary=None, keep_intermediate=False, prefer_mesh=False)
            with patch.object(finalizer, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(finalizer.main(), 1)
            self.assertEqual((dense / "final.ply").read_bytes(), expected)
            self.assertTrue((dense / "fused_filtered.ply").exists())

    def test_no_valid_sources_causes_no_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            dense = Path(temp)
            (dense / "fused.ply").write_bytes(b"partial")
            (dense / "final.ply").write_bytes(b"old")
            args = argparse.Namespace(dense_dir=dense, summary=None, keep_intermediate=False, prefer_mesh=False)
            with patch.object(finalizer, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(finalizer.main(), 1)
            self.assertEqual((dense / "fused.ply").read_bytes(), b"partial")
            self.assertEqual((dense / "final.ply").read_bytes(), b"old")

    def test_old_final_cannot_disguise_failed_new_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            dense = Path(temp)
            (dense / "fused.ply").write_bytes(b"partial")
            write_ply_vertices(dense / "final.ply", PROPERTIES, [(1, 2, 3)])
            previous = (dense / "final.ply").read_bytes()
            args = argparse.Namespace(dense_dir=dense, summary=dense / "summary.json", keep_intermediate=False, prefer_mesh=False)
            with patch.object(finalizer, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(finalizer.main(), 1)
            self.assertEqual((dense / "fused.ply").read_bytes(), b"partial")
            self.assertEqual((dense / "final.ply").read_bytes(), previous)
            self.assertEqual(json.loads(args.summary.read_text())["status"], "no_valid_new_source")


class SparseTests(unittest.TestCase):
    def test_text_model_coordinates_are_not_image_names(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            make_text_model(model)
            self.assertEqual(read_text_image_names(model / "images.txt"), {"a.jpg", "b.jpg"})
            self.assertEqual(inspect_sparse_model(model)["point_count"], 1)
            (model / "images.txt").write_text("1 1 0 0 0 0 0 0 1 a.jpg\n\n2 1 0 0 0 0 0 0 1 b.jpg\n\n")
            self.assertEqual(read_text_image_names(model / "images.txt"), {"a.jpg", "b.jpg"})

    def test_binary_model_truncation_is_not_partial_success(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            make_binary_model(model)
            self.assertEqual(inspect_sparse_model(model)["names"], {"a.jpg", "b.jpg"})
            original = (model / "images.bin").read_bytes()
            for payload in (b"", original[:-1], original[:80]):
                (model / "images.bin").write_bytes(payload)
                with self.assertRaises(ValueError):
                    read_binary_image_names(model / "images.bin")
            (model / "images.bin").write_bytes(original)
            for name in ("cameras.bin", "points3D.bin"):
                path = model / name
                original = path.read_bytes()
                path.write_bytes(original[:-1])
                with self.assertRaises(ValueError):
                    inspect_sparse_model(model)
                path.write_bytes(original)

    def test_dense_model_selection_matches_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            raw = workspace / "sparse" / "0"
            make_text_model(raw)
            make_text_model(workspace / "sparse_pose_prior" / "0")
            self.assertEqual(dense_input_model(workspace), raw)
            aligned = workspace / "sparse_pdr_aligned" / "0"
            make_binary_model(aligned)
            self.assertEqual(dense_input_model(workspace), aligned)

    def test_preflight_rejects_stale_sparse_even_when_database_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            (workspace / "images").mkdir()
            for name in ("a.jpg", "b.jpg"):
                (workspace / "images" / name).write_bytes(b"image")
            (workspace / "image_list.txt").write_text("a.jpg\nb.jpg\n")
            with contextlib.closing(sqlite3.connect(workspace / "database.db")) as db:
                db.execute("CREATE TABLE images (name TEXT)")
                db.executemany("INSERT INTO images VALUES (?)", [("a.jpg",), ("b.jpg",)])
                db.commit()
            model = workspace / "sparse" / "0"
            make_text_model(model, ("old_a.jpg", "old_b.jpg"))
            failures = validation_failures(inspect_workspace(workspace), require_database_coverage=True, require_sparse_model=True)
            self.assertTrue(any("outside" in message for message in failures))
            (model / "images.txt").write_bytes(b"")
            failures = validation_failures(inspect_workspace(workspace), require_database_coverage=True, require_sparse_model=True)
            self.assertTrue(any("invalid" in message for message in failures))

    def test_missing_or_bad_camera_and_points_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            make_text_model(model)
            camera = model / "cameras.txt"
            original = camera.read_text()
            for content in ("", "2 SIMPLE_RADIAL 640 480 500 320 240 0\n", "1 SIMPLE_RADIAL 640 480 nan 320 240 0\n"):
                camera.write_text(content)
                with self.assertRaises(ValueError):
                    inspect_sparse_model(model)
            camera.write_text(original)
            (model / "points3D.txt").write_bytes(b"")
            with self.assertRaises(ValueError):
                inspect_sparse_model(model)


class LaunchTests(unittest.TestCase):
    def test_load_choice_is_applied_before_one_preprocessing_call(self):
        with patch.object(launcher.sys, "argv", ["main.py"]), patch.object(launcher, "ask_colmap_run_plan", return_value=(True, False)), patch.object(launcher, "run_pipeline", return_value=0) as pipeline, patch.object(launcher, "run_colmap", return_value=0):
            self.assertEqual(launcher.main(), 0)
            pipeline.assert_called_once()
            args = parse_args(pipeline.call_args.args[0])
            self.assertEqual((args.colmap_load_profile, args.colmap_num_threads, args.colmap_stage_cooldown_s), ("max", 0, 0))

    def test_mesh_generation_is_explicit(self):
        self.assertTrue(parse_args([]).skip_colmap_meshing)
        self.assertFalse(parse_args(["--colmap-meshing"]).skip_colmap_meshing)
        self.assertTrue(parse_args(["--skip-colmap-meshing"]).skip_colmap_meshing)


if __name__ == "__main__":
    unittest.main()
