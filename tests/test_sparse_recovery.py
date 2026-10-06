"""Recovery must retain valid geometry and leave failed trials unselected."""
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from colmap_workspace_check import sparse_model_identity
from sparse_recovery import (REPORT_NAME, candidate_gate, evaluate_model, intrinsic_checks,
                             native_exit_status, project_camera, promote_candidate, run_recovery)


def write_model(path, *, names=('a.jpg', 'b.jpg'), shift=0., kept=20, focal=200., k=0.):
    path.mkdir(parents=True, exist_ok=True)
    camera = {'model': 'SIMPLE_RADIAL', 'width': 640, 'height': 480,
              'params': [focal, 320., 240., k]}
    true_camera = dict(camera, params=[200., 320., 240., 0.])
    xyz = np.array([[-1. + index*.1, (index % 3)*.1, 5.] for index in range(20)])
    with (path / 'cameras.bin').open('wb') as stream:
        stream.write(struct.pack('<QiiQQ4d', 1, 1, 2, 640, 480, *camera['params']))
    with (path / 'images.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(names)))
        for iid, name in enumerate(names, 1):
            translation = np.array([-.2 * (iid-1), 0., 0.])
            stream.write(struct.pack('<i7di', iid, 1., 0., 0., 0., *translation, 1))
            stream.write(name.encode() + b'\0' + struct.pack('<Q', 20))
            observed = project_camera(xyz + translation, true_camera) + [1., 0.]
            for index, xy in enumerate(observed):
                stream.write(struct.pack('<2dq', *xy, index+1 if index < kept else -1))
    with (path / 'points3D.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', kept))
        for index, point in enumerate(xyz[:kept], 1):
            # Deliberately stale errors must never determine acceptance.
            stream.write(struct.pack('<Q3d3BdQ', index, *(point + [shift, 0., 0.]),
                                     127, 127, 127, 999., len(names)))
            for iid in range(1, len(names)+1):
                stream.write(struct.pack('<ii', iid, index-1))


class SparseRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve()
        self.baseline = self.workspace / 'sparse' / '0'
        write_model(self.baseline)
        self.metadata = {name: {'image_file': name, 'video_name': 'clip.MOV', 'video_time_s': str(index)}
                         for index, name in enumerate(('a.jpg', 'b.jpg'))}
        folder = self.workspace / 'metadata'
        folder.mkdir()
        (folder / 'images_manifest.csv').write_text('image_file,video_name,video_time_s\na.jpg,clip.MOV,0\nb.jpg,clip.MOV,1\n')
        (self.workspace / 'sparse_run_inputs.json').write_text(json.dumps({'schema_version': 1, 'inputs': {}}))
        (self.workspace / 'video_intrinsics_summary.json').write_text(json.dumps({'intrinsics_held_during_mapping': True}))
        (self.workspace / 'compute_resources.json').write_text(json.dumps({'ba_gpu_probe': {'supported': False}}))
        self.before = sparse_model_identity(self.baseline)

    def run_stage(self, launch, **kwargs):
        with patch('sparse_recovery.inherited_lock_held', return_value=True), redirect_stdout(io.StringIO()):
            return run_recovery(self.workspace, self.workspace / 'colmap.exe', launch=launch, **kwargs)

    def test_reprojection_recomputed_despite_stale_native_errors(self):
        baseline = evaluate_model(self.baseline, self.metadata)
        self.assertAlmostEqual(baseline['summary']['mean_reprojection_error_px'], 1.)
        candidate = self.workspace / 'candidate'
        write_model(candidate, shift=.025)
        after = evaluate_model(candidate, self.metadata)
        self.assertLess(after['summary']['mean_reprojection_error_px'], 1e-10)
        self.assertTrue(candidate_gate(baseline, after)['accepted'])

    def test_lost_image_points_or_wrong_intrinsics_are_rejected(self):
        before = evaluate_model(self.baseline, self.metadata)
        for key, options in (('lost', {'names': ('a.jpg',)}), ('points', {'kept': 18}),
                             ('focal', {'focal': 280.}), ('distortion', {'k': -.25})):
            path = self.workspace / key
            write_model(path, **options)
            if key == 'lost':
                candidate = dict(before, observations={'a.jpg': before['observations']['a.jpg']})
            else:
                candidate = evaluate_model(path, self.metadata)
            self.assertFalse(candidate_gate(before, candidate)['accepted'], key)

    def test_nonzero_radial_projection_matches_opencv(self):
        import cv2
        points = np.array([[.2, -.1, 2.], [1., .3, 3.], [-.5, 1., 4.]])
        camera = {'model': 'SIMPLE_RADIAL', 'params': [500., 320., 240., -.02]}
        matrix = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
        expected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix,
                                        np.array([-.02, 0., 0., 0., 0.]))
        np.testing.assert_allclose(project_camera(points, camera), expected.reshape(-1, 2), atol=1e-10)

    def test_completed_ba_promotes_only_improved_candidate_and_backs_up(self):
        commands = []
        def launch(command, label, timeout):
            commands.append(command)
            write_model(Path(command[command.index('--output_path')+1]), shift=.025)
            return 0
        report = self.run_stage(launch)
        self.assertTrue(report['accepted'])
        self.assertEqual(len(commands), 2)  # Two views fail the separate three-view quality floor.
        self.assertNotIn('--BundleAdjustment.num_threads', commands[0])
        self.assertEqual(sparse_model_identity(self.workspace / report['baseline_backup']), self.before)
        self.assertNotEqual(sparse_model_identity(self.baseline), self.before)
        self.assertTrue((self.workspace / REPORT_NAME).is_file())

    def test_no_improvement_retains_baseline(self):
        def launch(command, label, timeout):
            write_model(Path(command[command.index('--output_path')+1]))
            return 0
        report = self.run_stage(launch)
        self.assertFalse(report['accepted'])
        self.assertEqual(report['status'], 'baseline_retained')
        self.assertEqual(sparse_model_identity(self.baseline), self.before)

    def test_timeout_never_promotes_even_if_partial_output_exists(self):
        def launch(command, label, timeout):
            write_model(Path(command[command.index('--output_path')+1]), shift=.025)
            raise subprocess.TimeoutExpired(command, timeout)
        report = self.run_stage(launch)
        self.assertFalse(report['accepted'])
        self.assertEqual(report['stages'][0]['status'], 'budget_exhausted')
        self.assertEqual(sparse_model_identity(self.baseline), self.before)

    def test_native_soft_budget_is_export_eligible_but_unbounded_interrupt_is_not(self):
        result = native_exit_status(0, 'Reached maximum runtime of 675 seconds.\nKeeping reconstruction due to interrupt')
        self.assertTrue(result['export_eligible'])
        self.assertTrue(result['budget_limited'])
        self.assertTrue(result['final_global_refinement_may_be_skipped'])
        self.assertFalse(native_exit_status(0, 'Keeping reconstruction due to interrupt')['export_eligible'])
        self.assertFalse(native_exit_status(1, 'Reached maximum runtime')['export_eligible'])

    def test_soft_budget_export_runs_geometry_gate_and_retains_native_output(self):
        (self.workspace / 'metadata/images_manifest.csv').write_text(
            'image_file,video_name,video_time_s\na.jpg,clip.MOV,0\nb.jpg,clip.MOV,1\nc.jpg,clip.MOV,2\n')
        @contextmanager
        def managed(command, **kwargs):
            retry = command[1] == 'mapper'
            names = ('a.jpg', 'b.jpg', 'c.jpg') if retry else ('a.jpg', 'b.jpg')
            write_model(Path(command[command.index('--output_path')+1]), names=names, shift=.025)
            log = 'Reached maximum runtime of 675 seconds.\nKeeping reconstruction due to interrupt\n' if retry else 'Global bundle adjustment\n'
            yield SimpleNamespace(stdout=io.StringIO(log), wait=lambda timeout: 0)
        with patch('sparse_recovery.managed_process', managed):
            report = self.run_stage(None)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['result']['registered_images'], 3)
        self.assertEqual(report['stages'][-1]['status'], 'budget_limited')
        self.assertIn('Keeping reconstruction', (self.workspace / report['stages'][-1]['log']).read_text())

    def test_poor_coverage_retries_all_frames_with_refined_intrinsics_held(self):
        (self.workspace / 'metadata/images_manifest.csv').write_text(
            'image_file,video_name,video_time_s\na.jpg,clip.MOV,0\nb.jpg,clip.MOV,1\nc.jpg,clip.MOV,2\n')
        commands = []
        def launch(command, label, timeout):
            commands.append(command)
            names = ('a.jpg', 'b.jpg', 'c.jpg') if label == 'register_remaining' else ('a.jpg', 'b.jpg')
            output = Path(command[command.index('--output_path')+1])
            if label == 'register_remaining':
                output = output / '0'  # Accept native output variants without assuming a suffix.
            write_model(output, names=names, shift=.025)
            return 0
        report = self.run_stage(launch)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['result']['registered_images'], 3)
        self.assertEqual(len(commands), 2)
        retry = commands[-1]
        self.assertNotIn('--Mapper.image_list_path', retry)
        self.assertEqual(retry[retry.index('--Mapper.ba_refine_focal_length')+1], '0')
        self.assertEqual(retry[retry.index('--Mapper.ba_refine_extra_params')+1], '0')
        self.assertIn('--Mapper.max_runtime_seconds', retry)

    def test_second_stage_timeout_never_promotes_aborted_model_but_can_keep_completed_ba(self):
        (self.workspace / 'metadata/images_manifest.csv').write_text(
            'image_file,video_name,video_time_s\na.jpg,clip.MOV,0\nb.jpg,clip.MOV,1\nc.jpg,clip.MOV,2\n')
        def launch(command, label, timeout):
            if label == 'register_remaining':
                write_model(Path(command[command.index('--output_path')+1]),
                            names=('a.jpg', 'b.jpg', 'c.jpg'), shift=.025)
                raise subprocess.TimeoutExpired(command, timeout)
            write_model(Path(command[command.index('--output_path')+1]), shift=.025)
            return 0
        report = self.run_stage(launch)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['result']['registered_images'], 2)
        self.assertIn('did_not_complete', report['registration_retry_warning'])
        self.assertEqual(sparse_model_identity(self.workspace / report['baseline_backup']), self.before)

    def test_completed_or_unlocked_workspace_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'lock'):
            run_recovery(self.workspace, self.workspace / 'colmap.exe')
        (self.workspace / 'sparse_run_provenance.json').write_text('{}')
        with patch('sparse_recovery.inherited_lock_held', return_value=True):
            with self.assertRaisesRegex(ValueError, 'certified'):
                run_recovery(self.workspace, self.workspace / 'colmap.exe')
        self.assertEqual(sparse_model_identity(self.baseline), self.before)

    def test_largest_component_is_selected_instead_of_assuming_zero(self):
        larger = self.baseline.parent / '2'
        self.baseline.rename(larger)
        write_model(self.baseline, kept=2)
        self.baseline = larger
        def launch(command, label, timeout):
            if label == 'late_intrinsics':
                self.assertEqual(Path(command[command.index('--input_path')+1]), larger)
            write_model(Path(command[command.index('--output_path')+1]), shift=.025)
            return 0
        report = self.run_stage(launch)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['baseline_model'], 'sparse\\2' if '\\' in str(larger) else 'sparse/2')
        self.assertEqual(report['baseline']['points3D'], 20)
        self.assertEqual(evaluate_model(larger.parent / '0', self.metadata)['summary']['points3D'], 2)

    def test_promotion_rename_failure_rolls_back_baseline(self):
        attempt = self.workspace / 'sparse_recovery' / 'test'
        candidate, backup = attempt / 'candidate', attempt / 'backup'
        write_model(candidate, shift=.025)
        original_rename = Path.rename
        def rename(path, destination):
            if path == candidate:
                raise OSError('simulated locked candidate')
            return original_rename(path, destination)
        with patch('sparse_recovery.inherited_lock_held', return_value=True), patch.object(Path, 'rename', rename):
            with self.assertRaisesRegex(OSError, 'locked candidate'):
                promote_candidate(self.workspace, self.baseline, candidate, backup)
        self.assertEqual(sparse_model_identity(self.baseline), self.before)
        self.assertTrue(candidate.exists())
        self.assertFalse(backup.exists())


if __name__ == '__main__':
    unittest.main()
