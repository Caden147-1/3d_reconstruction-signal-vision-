#!/usr/bin/env python3
"""Small standard-library regression checks for the reconstruction pipeline."""

from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from colmap_bridge import (
    build_constrained_match_pairs,
    build_workspace_manifest,
    camera_grouping_summary,
    prepare_colmap_workspace,
    select_visually_supported_match_pairs,
)
from colmap_workspace_check import inspect_workspace, validation_failures
from finalize_dense_outputs import choose_source
from ply_io import read_ply_vertices, write_ply_vertices
from point_cloud_filter import filter_point_cloud
from run_colmap import build_upfront_runtime_estimate
from video_frames import VIDEO_FRAME_FIELDS, bind_frame_to_route, write_csv


class PipelineRegressionTests(unittest.TestCase):
    def test_video_manifest_keeps_contour_coordinates(self) -> None:
        points = [
            {
                "global_point_index": "1",
                "pdr_x": "0",
                "pdr_y": "0",
                "trajectory_contour_id": "route_001",
                "trajectory_contour_distance_m": "0",
                "trajectory_distance_m": "0",
            },
            {
                "global_point_index": "2",
                "pdr_x": "10",
                "pdr_y": "0",
                "trajectory_contour_id": "route_001",
                "trajectory_contour_distance_m": "10",
                "trajectory_distance_m": "10",
            },
        ]
        binding = bind_frame_to_route(points, frame_index_zero_based=1, frame_count=3)
        self.assertEqual(binding["trajectory_contour_id"], "route_001")
        self.assertEqual(binding["trajectory_position_source"], "checkpoint_order_interpolation")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "video.csv"
            write_csv(path, VIDEO_FRAME_FIELDS, [binding])
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["trajectory_contour_id"], "route_001")
        self.assertEqual(row["trajectory_contour_distance_m"], "5.000000")

    def test_mixed_sources_use_two_camera_groups(self) -> None:
        rows = build_workspace_manifest(
            [
                {"image_file": "anchor.jpg", "source_type": "wifi_anchor_photo"},
                {"image_file": "frame.jpg", "source_type": "video_base_frame"},
            ]
        )
        summary = camera_grouping_summary(rows)
        self.assertEqual([row["image_file"] for row in rows], ["anchors/anchor.jpg", "video/frame.jpg"])
        self.assertFalse(summary["single_camera"])
        self.assertTrue(summary["single_camera_per_folder"])

    def test_workspace_generation_uses_source_folder_camera_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reconstruction = root / "reconstruction"
            images = reconstruction / "images"
            images.mkdir(parents=True)
            (images / "anchor.jpg").write_bytes(b"anchor")
            (images / "frame.jpg").write_bytes(b"frame")
            (reconstruction / "images_manifest.csv").write_text(
                "image_file,source_type,global_point_index,image_slot,nearest_global_point_index\n"
                "anchor.jpg,wifi_anchor_photo,1,1,\nframe.jpg,video_base_frame,,,1\n",
                encoding="utf-8",
            )
            workspace = root / "workspace"
            summary = prepare_colmap_workspace(
                root=root,
                reconstruction_dir=reconstruction,
                workspace_dir=workspace,
                colmap_exe=root / "missing-colmap.exe",
                visual_only=True,
                run_image_similarity=False,
                run_meshing=False,
                use_gpu=False,
                dense_profile="complete",
            )
            sparse_batch = (workspace / "run_colmap_sparse.bat").read_text(encoding="utf-8")
            automatic_batch = (workspace / "run_colmap_automatic.bat").read_text(encoding="utf-8")
            dense_batch = (workspace / "run_colmap_dense.bat").read_text(encoding="utf-8")
            unfiltered_pairs_exists = (workspace / "match_pairs_unfiltered.txt").exists()
            filter_summary_exists = (workspace / "match_pair_filter_summary.json").exists()
        self.assertEqual(summary["camera_grouping"]["groups"], {"anchors": 1, "video": 1})
        self.assertIn("--ImageReader.single_camera 0", sparse_batch)
        self.assertIn("--ImageReader.single_camera_per_folder 1", sparse_batch)
        self.assertIn("--require-clean-database", automatic_batch)
        self.assertTrue(unfiltered_pairs_exists)
        self.assertTrue(filter_summary_exists)
        self.assertIn("--method connected-density", dense_batch)
        self.assertIn("--max-points 0", dense_batch)
        self.assertIn("stopping before final promotion", dense_batch)

    def test_p2_contour_topology_creates_route_first_edges(self) -> None:
        def anchor(point: int, route_distance_m: float) -> dict[str, str]:
            return {
                "image_file": f"anchors/point_{point:03d}.jpg",
                "source_type": "wifi_anchor_photo",
                "session": "route_a",
                "global_point_index": str(point),
                "image_slot": "1",
                "anchor_slot_count": "1",
                "pdr_x": str(route_distance_m),
                "pdr_y": "0",
                "trajectory_contour_id": "route_a_001",
                "trajectory_contour_distance_m": str(route_distance_m),
            }

        pairs, edges, summary = build_constrained_match_pairs(
            manifest_rows=[anchor(10, 0.0), anchor(30, 50.0), anchor(90, 100.0)],
            point_window=1,
            slot_window=1,
            max_pdr_distance_m=12.0,
            video_frame_window=1,
            video_anchor_point_window=1,
            video_anchor_limit=8,
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual({row["reason"] for row in edges}, {"trajectory_contour_neighbors"})
        self.assertEqual(summary["p2_contour_edge_count"], 2)
        self.assertEqual(summary["reason_counts"].get("nearby_wifi_points", 0), 0)

    def test_p1_selection_filters_evaluated_weak_pairs_but_fails_open(self) -> None:
        pairs = [("anchors/a.jpg", "anchors/b.jpg"), ("anchors/a.jpg", "anchors/c.jpg")]
        edges = [
            {"image_a": "anchors/a.jpg", "image_b": "anchors/b.jpg", "reason": "trajectory_contour_neighbors", "edge_weight": 0.9},
            {"image_a": "anchors/a.jpg", "image_b": "anchors/c.jpg", "reason": "trajectory_contour_neighbors", "edge_weight": 0.9},
        ]
        similarity = [
            {"image_a": "anchors/a.jpg", "image_b": "anchors/b.jpg", "pair_status": "ok", "visual_decision": "geometry_supported", "visual_overlap_score": "0.8"},
            {"image_a": "anchors/a.jpg", "image_b": "anchors/c.jpg", "pair_status": "ok", "visual_decision": "no_feature_overlap", "visual_overlap_score": "0.0"},
        ]
        selected, rows, summary = select_visually_supported_match_pairs(pairs, edges, similarity, {"status": "ok"})
        self.assertEqual(selected, [("anchors/a.jpg", "anchors/b.jpg")])
        self.assertEqual(summary["status"], "applied")
        self.assertEqual(summary["rejected_pair_count"], 1)
        self.assertEqual(rows[1]["selected"], 0)

        fallback_selected, _fallback_rows, fallback_summary = select_visually_supported_match_pairs(
            pairs,
            edges,
            [],
            {"status": "skipped"},
        )
        self.assertEqual(fallback_selected, pairs)
        self.assertEqual(fallback_summary["status"], "not_applied")

    def test_final_ply_prefers_filtered_observed_cloud_over_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dense = Path(temp_dir)
            write_ply_vertices(dense / "fused_filtered.ply", [("x", "float"), ("y", "float"), ("z", "float")], [(0, 0, 0)])
            (dense / "meshed-poisson.ply").write_text(
                "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
                "element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
                encoding="ascii",
            )
            observed = choose_source(dense)
            mesh = choose_source(dense, prefer_mesh=True)
        self.assertIsNotNone(observed)
        self.assertIsNotNone(mesh)
        self.assertEqual(observed[0], "filtered_point_cloud")
        self.assertEqual(mesh[0], "poisson_mesh")

    def test_runtime_estimate_uses_p1_selected_pair_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "image_list.txt").write_text("a.jpg\nb.jpg\nc.jpg\n", encoding="utf-8")
            (workspace / "match_pairs.txt").write_text("a.jpg b.jpg\na.jpg c.jpg\n", encoding="utf-8")
            (workspace / "match_pairs_unfiltered.txt").write_text(
                "a.jpg b.jpg\na.jpg c.jpg\nb.jpg c.jpg\n",
                encoding="utf-8",
            )
            (workspace / "colmap_workspace_summary.json").write_text(
                json.dumps(
                    {
                        "image_count": 3,
                        "matcher": "constrained",
                        "constrained_match_pairs": {
                            "pair_count": 3,
                            "unfiltered_pair_count": 3,
                            "colmap_pair_count": 2,
                            "p1_pair_filter": {"selected_pair_count": 2},
                        },
                    }
                ),
                encoding="utf-8",
            )
            estimate = build_upfront_runtime_estimate(workspace)
        self.assertEqual(estimate["unfiltered_constrained_pairs"], 3)
        self.assertEqual(estimate["constrained_pairs"], 2)
        self.assertEqual(estimate["matching_pairs"], 2)

    def test_dense_preflight_rejects_stale_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "images" / "anchors").mkdir(parents=True)
            (workspace / "images" / "video").mkdir(parents=True)
            (workspace / "images" / "anchors" / "anchor.jpg").write_bytes(b"x")
            (workspace / "images" / "video" / "frame.jpg").write_bytes(b"x")
            (workspace / "metadata").mkdir()
            (workspace / "image_list.txt").write_text("anchors/anchor.jpg\nvideo/frame.jpg\n", encoding="utf-8")
            (workspace / "metadata" / "images_manifest.csv").write_text(
                "image_file,source_type\nanchors/anchor.jpg,wifi_anchor_photo\nvideo/frame.jpg,video_base_frame\n",
                encoding="utf-8",
            )
            connection = sqlite3.connect(workspace / "database.db")
            connection.execute("CREATE TABLE images (name TEXT)")
            connection.execute("INSERT INTO images VALUES ('anchors/anchor.jpg')")
            connection.commit()
            connection.close()
            report = inspect_workspace(workspace)
            failures = validation_failures(report, require_database_coverage=True)
        self.assertTrue(any("database image names" in failure for failure in failures))

    def test_reservoir_filter_uses_budget_without_stride_sampling(self) -> None:
        rows = [
            (float(x), float(y), 0.0, 100, 120, 140)
            for x in range(20)
            for y in range(20)
        ]
        properties = [
            ("x", "float"),
            ("y", "float"),
            ("z", "float"),
            ("red", "uchar"),
            ("green", "uchar"),
            ("blue", "uchar"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.ply"
            output = Path(temp_dir) / "filtered.ply"
            write_ply_vertices(source, properties, rows)
            summary = filter_point_cloud(
                input_path=source,
                output_path=output,
                summary_path=None,
                max_points=16,
                bbox_quantile=0.0,
                bbox_padding_ratio=0.10,
                radial_quantile=1.0,
                radius_ratio=0.0,
                min_neighbors=0,
                skip_density=True,
            )
            _properties, filtered, _io = read_ply_vertices(output)
        self.assertEqual(summary["sampling"]["method"], "uniform_reservoir")
        self.assertEqual(len(filtered), 16)
        self.assertEqual(summary["sample_stride"], 1)
        self.assertIn("removed_vertex_count", summary)
        self.assertIn("removed_ratio", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
