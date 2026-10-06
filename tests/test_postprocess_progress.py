"""Progress must count completed work without changing retained geometry."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import dense_surface_filter as density
import observation_filter as observed


class RecordingProgress:
    records = []

    def __init__(self, label, total=None, unit='items', **kwargs):
        self.label, self.total, self.unit = label, total, unit
        self.updates, self.current, self.outcome = [], 0, None
        self.records.append(self)

    def __enter__(self):
        return self

    def update(self, completed):
        if completed < self.current:
            raise AssertionError('Progress moved backwards')
        self.current = completed
        self.updates.append(completed)

    def __exit__(self, kind, error, traceback):
        self.outcome = 'error' if kind else 'complete' if self.current >= self.total else 'incomplete'


def write_cloud(path, vertices):
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(vertices)}\nproperty float x\nproperty float y\n'
              'property float z\nproperty uchar red\nend_header\n')
    with path.open('wb') as handle:
        handle.write(header.encode('ascii'))
        vertices.tofile(handle)


def vertices(count):
    data = np.zeros(count, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1')])
    data['x'] = np.arange(count)
    data['y'], data['z'], data['red'] = np.arange(count) % 17, 1, np.arange(count) % 256
    return data


class PostprocessProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source.ply'
        RecordingProgress.records = []
        for context in (patch.object(density, 'WorkProgress', RecordingProgress),
                        patch.object(observed, 'WorkProgress', RecordingProgress)):
            context.start()
            self.addCleanup(context.stop)

    def progress(self, label):
        return next(record for record in RecordingProgress.records if record.label == label)

    def test_sampling_counts_original_points_and_preserves_source_index_selection(self):
        original = vertices(501)
        write_cloud(self.source, original)
        with patch.object(density, 'BLOCK_SIZE', 73):
            sample, indices, stride = density.sample_original_vertices(self.source, len(original), 80)
        expected = np.arange(0, len(original), 7)
        self.assertEqual(stride, 7)
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_array_equal(sample, original[expected])
        progress = self.progress('Sampling original observations')
        self.assertEqual(progress.total, len(original))
        self.assertEqual(progress.current, len(original))
        self.assertEqual(progress.outcome, 'complete')
        self.assertEqual(len(progress.updates), 7)

    def test_failed_sampling_closes_progress_without_completing_the_pass(self):
        original = vertices(501)
        original['x'][-1] = np.nan
        write_cloud(self.source, original)
        with patch.object(density, 'BLOCK_SIZE', 128), self.assertRaisesRegex(ValueError, 'Non-finite'):
            density.sample_original_vertices(self.source, len(original), 80)
        progress = self.progress('Sampling original observations')
        self.assertEqual(progress.current, 384)
        self.assertEqual(progress.outcome, 'error')

    def run_observed(self, *, exhaust=False):
        original = vertices(32770)
        write_cloud(self.source, original)
        clock = [0.]
        assessors = []

        class Assessor:
            available, reason = True, ''

            def __init__(self, *args, **kwargs):
                self.closed = False
                assessors.append(self)

            def assess(self, chunk):
                evidence = np.zeros(len(chunk), dtype=observed.EVIDENCE_DTYPE)
                evidence['status'][chunk['x'] == 0] = observed.CONFLICT
                evidence['status'][chunk['x'] == 5] = observed.SUPPORTED
                if exhaust:
                    clock[0] = 2.
                return evidence

            def summary(self):
                return dict(observations_checked=True, tested_view_ids=[])

            def confirm_conflicts(self, chunk, evidence, deadline=None):
                return evidence

            def close(self):
                self.closed = True

        output = self.root / 'candidate.ply'
        with patch.object(observed, 'DensityObservationProtector', Assessor), \
                patch.object(observed.time, 'monotonic', side_effect=lambda: clock[0]):
            report = observed.filter_bounded_observed_cloud(
                self.source, output, self.root / 'summary.json', self.root / 'dense',
                expected={'fixture': {}}, binding={}, max_candidates=len(original), budget_seconds=1.)
        np.testing.assert_array_equal(np.concatenate(list(density.vertex_chunks(output))), original[1:])
        self.assertTrue(assessors[0].closed)
        return report

    def test_observation_batches_and_streaming_finish_with_original_geometry(self):
        report = self.run_observed()
        assessment = self.progress('Assessing sampled observations')
        self.assertEqual(assessment.current, 32770)
        self.assertEqual(assessment.updates, sorted(set(assessment.updates)))
        self.assertTrue(all(0 < b-a <= 4096 for a, b in
                            zip([0, *assessment.updates], assessment.updates)))
        self.assertEqual(assessment.outcome, 'complete')
        publication = self.progress('Writing observation classification layers')
        self.assertEqual(publication.current, 32770)
        self.assertEqual(publication.outcome, 'complete')
        self.assertEqual(report['geometry_removed_vertex_count'], 1)
        self.assertEqual(report['classification'], dict(unknown=32768, supported=1, conflict=1))

    def test_budget_limited_assessment_reports_incomplete_and_retains_unknown(self):
        report = self.run_observed(exhaust=True)
        assessment = self.progress('Assessing sampled observations')
        self.assertGreater(assessment.current, 0)
        self.assertLessEqual(assessment.current, 4096)
        self.assertEqual(assessment.total, 32770)
        self.assertEqual(assessment.outcome, 'incomplete')
        self.assertTrue(report['assessment']['budget_exhausted'])
        self.assertEqual(report['assessment']['unevaluated_candidate_count'],
                         assessment.total - assessment.current)
        self.assertEqual(report['geometry_removed_vertex_count'], 1)

    def test_density_passes_count_all_input_and_keep_original_properties(self):
        xyz = np.repeat(np.array(list(np.ndindex(5, 5, 5)), dtype=np.float32), 10, axis=0)
        original = vertices(len(xyz))
        for index, axis in enumerate('xyz'):
            original[axis] = xyz[:, index]
        write_cloud(self.source, original)
        output = self.root / 'density.ply'
        with patch.object(density, 'BLOCK_SIZE', 77):
            report = density.filter_connected_cloud(
                self.source, output, None, bbox_quantile=0, bbox_padding_ratio=0,
                cell_ratio=.15, core_quantile=0)
        np.testing.assert_array_equal(np.concatenate(list(density.vertex_chunks(output))), original)
        self.assertEqual(report['geometry_removed_vertex_count'], 0)
        for label in ('Measuring cloud statistics', 'Counting density-grid observations', 'Writing filtered observations'):
            progress = self.progress(label)
            self.assertEqual(progress.total, len(original))
            self.assertEqual(progress.current, len(original))
            self.assertEqual(progress.outcome, 'complete')
        core = self.progress('Finding connected density surfaces')
        self.assertEqual(core.current, core.total)
        self.assertEqual(core.outcome, 'complete')


if __name__ == '__main__':
    unittest.main()
