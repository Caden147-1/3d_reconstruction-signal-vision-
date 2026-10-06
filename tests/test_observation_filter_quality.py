"""Observation coverage must not corrupt source indices or reject unknowns."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import observation_filter as observed
from dense_surface_filter import vertex_chunks


DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                  ('red', 'u1'), ('source_id', '<i4')])


def write_cloud(path, vertices):
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(vertices)}\nproperty float x\n'
              'property float y\nproperty float z\nproperty uchar red\n'
              'property int source_id\nend_header\n')
    with path.open('wb') as handle:
        handle.write(header.encode('ascii'))
        vertices.tofile(handle)


class ObservationQualityTests(unittest.TestCase):
    def test_vectorized_conditioning_matches_scalar_track_geometry(self):
        rng = np.random.default_rng(91)
        views = [dict(id=i, C=rng.normal(size=3), point_ids=np.arange(20)) for i in range(6)]
        points = {i: (rng.normal(size=3), 1., np.array([*range(6), 999])) for i in range(20)}
        actual = observed.conditioning(views, points)
        for view in views:
            values = []
            for xyz, _, track in points.values():
                a = xyz-view['C']
                values.append(max(np.sqrt(max(0., 1-np.clip(a @ (xyz-other['C']) /
                    (np.linalg.norm(a)*np.linalg.norm(xyz-other['C'])), -1, 1)**2))
                    for other in views if other['id'] != view['id']))
            self.assertAlmostEqual(actual[view['id']], float(np.median(values)), places=14)

    def camera_fixture(self, supporting_last=False):
        views = [dict(id=i+1, name=chr(97+i), C=np.array([x, 0., 0.]),
                      R=np.eye(3), K=np.array([[200., 0., 8.], [0., 200., 8.], [0., 0., 1.]]),
                      size=(16, 16), noise_px=1.) for i, x in enumerate((-.1, .1, 0.))]
        normals = np.zeros((16, 16, 3), np.float32)
        normals[:, :, 2] = 1.
        pairs = {view['id']: (np.full((16, 16), 5. if supporting_last and view['id'] == 3 else 6.,
                                      np.float32), normals) for view in views}

        class Store:
            def get(self, view):
                return pairs.get(view['id'])

        return views, Store(), {view['id']: 1. for view in views}

    def test_one_support_outside_local_view_cap_vetoes_deletion(self):
        views, store, quality = self.camera_fixture(supporting_last=True)
        xyz = np.array([[0., 0., 5.]])
        evidence, _ = observed.assess_points(xyz, views, store, quality, {'max_views': 2})
        self.assertEqual(evidence['status'][0], observed.CONFLICT)
        evidence, _, report = observed.confirm_free_space_conflicts(xyz, evidence, views, store, quality)
        self.assertEqual(evidence['status'][0], observed.UNKNOWN)
        self.assertEqual(evidence['support_a'][0], 3)
        self.assertEqual(report['rescued_by_support_count'], 1)
        self.assertEqual(report['confirmed_conflict_count'], 0)

    def test_conflict_requires_original_diverse_votes_and_complete_support_search(self):
        views, store, quality = self.camera_fixture()
        xyz = np.array([[0., 0., 5.], [.01, 0., 5.], [.02, 0., 5.]])
        evidence, _ = observed.assess_points(xyz, views, store, quality, {'max_views': 2})
        evidence['status'][1:] = [observed.UNKNOWN, observed.SUPPORTED]
        evidence, _, report = observed.confirm_free_space_conflicts(xyz, evidence, views, store, quality)
        np.testing.assert_array_equal(evidence['status'], [observed.CONFLICT, observed.UNKNOWN, observed.SUPPORTED])
        self.assertEqual(report['confirmed_conflict_count'], 1)
        self.assertEqual(report['checked_view_count'], 3)

    def test_incomplete_confirmation_retains_conflict_as_unknown(self):
        views, store, quality = self.camera_fixture()
        xyz = np.array([[0., 0., 5.]])
        evidence, _ = observed.assess_points(xyz, views, store, quality, {'max_views': 2})
        with patch.object(observed.time, 'monotonic', side_effect=lambda: 2.):
            evidence, _, report = observed.confirm_free_space_conflicts(
                xyz, evidence, views, store, quality, deadline=1.)
        self.assertEqual(evidence['status'][0], observed.UNKNOWN)
        self.assertTrue(report['budget_exhausted'])
        self.assertEqual(report['unresolved_budget_count'], 1)

    def test_missing_expected_confirmation_map_does_not_certify_conflict(self):
        views, store, quality = self.camera_fixture()
        xyz = np.array([[0., 0., 5.]])
        evidence, _ = observed.assess_points(xyz, views, store, quality, {'max_views': 2})
        original = store.get
        with patch.object(store, 'get', side_effect=lambda view: None if view['id'] == 3 else original(view)):
            evidence, _, report = observed.confirm_free_space_conflicts(xyz, evidence, views, store, quality)
        self.assertEqual(evidence['status'][0], observed.UNKNOWN)
        self.assertEqual(report['unavailable_observation_count'], 1)

    def test_batches_are_local_complete_and_visit_distant_regions_early(self):
        rng = np.random.default_rng(16)
        xyz = rng.normal(0, .01, (256, 3))
        xyz[:, 0] += np.tile(np.arange(8), 32) * 10
        batches = observed.spatial_assessment_batches(xyz, 32)
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(len(xyz)))
        self.assertEqual(len(batches), 8)
        self.assertTrue(all(len(batch) <= 32 for batch in batches))
        self.assertTrue(all(np.ptp(xyz[batch, 0]) < .1 for batch in batches))
        self.assertGreater(abs(xyz[batches[0], 0].mean()-xyz[batches[1], 0].mean()), 30)
        for first, second in zip(batches, observed.spatial_assessment_batches(xyz.copy(), 32)):
            np.testing.assert_array_equal(first, second)

    def test_duplicate_positions_and_empty_samples_do_not_lose_indices(self):
        batches = observed.spatial_assessment_batches(np.zeros((107, 3)), 16)
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(107))
        self.assertEqual(observed.spatial_assessment_batches(np.empty((0, 3))), [])

    def _filter(self, root, *, count=257, max_candidates=80, exhaust=False):
        vertices = np.zeros(count, dtype=DTYPE)
        vertices['source_id'] = np.arange(count)
        vertices['red'] = np.arange(count) % 255
        vertices['x'] = np.random.default_rng(47).permutation(count)
        vertices['y'], vertices['z'] = 1., 5.
        source = root/'source.ply'
        write_cloud(source, vertices)
        before = source.read_bytes()
        clock = [0.]
        calls = []

        class Assessor:
            available, reason = True, ''

            def __init__(self, *args, **kwargs):
                self.count = 0

            def assess(self, chunk):
                calls.append(chunk['source_id'].copy())
                self.count += len(chunk)
                evidence = np.zeros(len(chunk), dtype=observed.EVIDENCE_DTYPE)
                evidence['tested'] = 2
                evidence['status'][chunk['source_id'] % 5 == 0] = observed.SUPPORTED
                evidence['status'][chunk['source_id'] % 7 == 0] = observed.CONFLICT
                if exhaust:
                    clock[0] += 1.
                return evidence

            def summary(self):
                return dict(observations_checked=bool(self.count), tested_view_ids=[1, 2],
                            available_view_count=8, conditioned_view_count=8,
                            tested_view_fraction=.25, tested_candidate_count=self.count)

            def confirm_conflicts(self, chunk, evidence, deadline=None):
                return evidence

            def close(self):
                pass

        with patch.object(observed, 'DensityObservationProtector', Assessor), \
                patch.object(observed.time, 'monotonic', side_effect=lambda: clock[0]), \
                redirect_stdout(io.StringIO()):
            report = observed.filter_bounded_observed_cloud(
                source, root/'candidate.ply', root/'summary.json', root/'dense',
                binding={}, expected={'fixture': {}}, max_candidates=max_candidates,
                budget_seconds=2.5, assessment_batch_size=8)
        self.assertEqual(source.read_bytes(), before)
        return vertices, report, calls

    def test_spatial_reordering_preserves_exact_original_index_evidence_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, report, calls = self._filter(root)
            tested = np.sort(np.concatenate(calls))
            with np.load(report['evidence']['path']) as evidence:
                np.testing.assert_array_equal(evidence['source_indices'], tested)
                expected = np.zeros(len(tested), dtype=np.uint8)
                expected[tested % 5 == 0] = observed.SUPPORTED
                expected[tested % 7 == 0] = observed.CONFLICT
                np.testing.assert_array_equal(evidence['evidence']['status'], expected)
            keep = np.ones(len(original), dtype=bool)
            keep[tested[tested % 7 == 0]] = False
            np.testing.assert_array_equal(np.concatenate(list(vertex_chunks(root/'candidate.ply'))), original[keep])
            self.assertEqual(report['assessment']['assessed_candidate_count'], len(tested))
            self.assertEqual(report['assessment']['tested_source_fraction'], len(tested)/len(original))
            self.assertEqual(report['available_view_count'], 8)
            self.assertEqual(report['tested_view_fraction'], .25)

    def test_timeout_retains_all_unassessed_original_points_across_regions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, report, calls = self._filter(root, max_candidates=257, exhaust=True)
            self.assertEqual(len(calls), 2)
            self.assertTrue(report['assessment']['budget_exhausted'])
            tested = np.sort(np.concatenate(calls))
            keep = np.ones(len(original), dtype=bool)
            keep[tested[tested % 7 == 0]] = False
            np.testing.assert_array_equal(np.concatenate(list(vertex_chunks(root/'candidate.ply'))), original[keep])
            self.assertEqual(report['assessment']['unevaluated_candidate_count'], len(original)-len(tested))
            self.assertGreater(abs(original['x'][calls[0]].mean()-original['x'][calls[1]].mean()), 80)

    def test_disabled_assessment_retains_every_original_point(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, report, calls = self._filter(root, max_candidates=0)
            self.assertFalse(calls)
            self.assertEqual(report['assessment']['status'], 'disabled')
            np.testing.assert_array_equal(np.concatenate(list(vertex_chunks(root/'candidate.ply'))), original)


if __name__ == '__main__':
    unittest.main()
