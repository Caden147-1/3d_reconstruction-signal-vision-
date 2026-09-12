"""Observed-surface selection must remove noise without inventing geometry."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from ply_io import read_ply_vertices, validate_ply, write_ply_vertices
from point_cloud_filter import sampling_quality_warning


@unittest.skipIf(np is None, "Connected-density filtering requires NumPy")
class SurfaceFilterTests(unittest.TestCase):
    def test_full_resolution_preserves_surface_rows_and_rejects_rays(self):
        from dense_surface_filter import filter_connected_cloud
        properties = [(axis, "float") for axis in ("x", "y", "z", "nx", "ny", "nz")]
        properties += [(name, "uchar") for name in ("red", "green", "blue")]
        plane = [(x*.01, y*.01, 0., 0., 0., 1., 20, 90, 150) for x in range(101) for y in range(101)]
        rays = [(1+i*.04, y, 0., 1., 0., 0., 255, 0, 0) for i in range(200) for y in (0., .5, 1.)]
        remote = [(4+x*.005, 4+y*.005, 1., 0., 0., 1., 255, 255, 0) for x in range(20) for y in range(20)]
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp)/"source.ply", Path(temp)/"filtered.ply"
            write_ply_vertices(source, properties, plane+rays+remote)
            original = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                report = filter_connected_cloud(source, output, None)
            _, selected, _ = read_ply_vertices(output)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(validate_ply(output), {"vertex": len(selected)})
            self.assertEqual(report["sampling"]["method"], "full_resolution")
            self.assertEqual(report["budget_omitted_vertex_count"], 0)
            self.assertEqual(report["geometry_removed_vertex_count"]+len(selected), len(plane+rays+remote))
            self.assertEqual(sampling_quality_warning(report), "")
            self.assertGreater(sum(row[-3:] == (20, 90, 150) for row in selected), len(plane)*.98)
            self.assertLess(sum(row[-3:] == (255, 0, 0) for row in selected), len(rays)*.1)
            self.assertFalse(any(row[-3:] == (255, 255, 0) for row in selected))
            self.assertTrue(all(row[3:6] == (0., 0., 1.) for row in selected if row[-3:] == (20, 90, 150)))
            # Binary float rounding is allowed; source observations are not moved.
            original_xyz = np.asarray([row[:3] for row in plane+rays+remote], dtype=np.float32)
            original_set = {tuple(row) for row in original_xyz}
            self.assertTrue(all(tuple(np.float32(row[:3])) in original_set for row in selected))

    def test_sampling_is_after_geometry_and_reproducible(self):
        from dense_surface_filter import filter_connected_cloud
        props = [(a, "float") for a in "xyz"]
        rows = [(x*.01, y*.01, 0) for x in range(101) for y in range(101)]
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/"source.ply"
            write_ply_vertices(source, props, rows)
            outputs = [Path(temp)/f"sample_{i}.ply" for i in range(2)]
            with contextlib.redirect_stdout(io.StringIO()):
                reports = [filter_connected_cloud(source, p, None, max_points=100, cell_ratio=.02) for p in outputs]
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            for report in reports:
                self.assertEqual(report["output_vertex_count"], 100)
                self.assertGreater(report["geometry_candidate_count"], 9000)
                self.assertEqual(report["sampling"]["budget_utilization"], 1)
                self.assertEqual(sampling_quality_warning(report), "")

    def test_binary_reader_preserves_property_types_in_both_orders(self):
        from dense_surface_filter import vertex_chunks
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"source.ply"
            for endian, order in (("<", "little"), (">", "big")):
                header = f"ply\nformat binary_{order}_endian 1.0\nelement vertex 1\nproperty double x\nproperty float y\nproperty float z\nproperty uchar red\nend_header\n"
                path.write_bytes(header.encode()+struct.pack(endian+"dffB", 1.123456789, 2, 3, 241))
                chunk, = list(vertex_chunks(path))
                self.assertEqual(float(chunk["x"][0]), 1.123456789)
                self.assertEqual(int(chunk["red"][0]), 241)
                path.write_bytes(header.encode()+b"short")
                with self.assertRaisesRegex(ValueError, "Truncated"):
                    list(vertex_chunks(path))

    def test_ambiguous_components_fail_and_halo_does_not_fill_empty_voxels(self):
        from dense_surface_filter import connected_core, expand_mask
        grid = np.zeros((12,12,12), dtype=int)
        grid[1:4,1:4,1] = 50
        grid[7:10,7:10,7] = 50
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            connected_core(grid, 20)
        grid[7:10,7:10,7] = 1
        core, report = connected_core(grid, 20)
        expanded = expand_mask(core, 1)
        self.assertEqual(report["selected_core_vertex_count"], 450)
        self.assertEqual(int(grid[expanded].sum()), 450)
        self.assertEqual(expanded.sum(), 75)

    def test_invalid_input_keeps_existing_output_and_disallows_in_place(self):
        from dense_surface_filter import filter_connected_cloud
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/"bad.ply", Path(temp)/"old.ply"
            source.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1000\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
            target.write_bytes(b"previous result")
            with self.assertRaises(ValueError):
                filter_connected_cloud(source, target, None)
            self.assertEqual(target.read_bytes(), b"previous result")
            with self.assertRaisesRegex(ValueError, "preserved"):
                filter_connected_cloud(source, source, None)

    def test_audit_distinguishes_filtering_from_sampling_failure(self):
        audit = {"source_vertex_count": 10000, "max_points": 10000, "output_vertex_count": 3000,
                 "sampling": {"post_filter_candidate_count": 3000}}
        self.assertFalse(sampling_quality_warning(audit))
        audit["output_vertex_count"] = 1000
        self.assertTrue(sampling_quality_warning(audit))
        audit["max_points"] = 0
        self.assertTrue(sampling_quality_warning(audit))
        audit["sampling"]["post_filter_candidate_count"] = 0
        self.assertTrue(sampling_quality_warning(audit))


if __name__ == "__main__":
    unittest.main()
