"""Sharpness acceleration must preserve decisions and survive missing GPUs."""

import os
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from gpu_image_ops import SharpnessScorer, configure_gpu_image_ops


class SharpnessTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.random.default_rng(417).integers(0, 256, (96, 127), dtype=np.uint8)

    def reference(self, image):
        return float(cv2.Laplacian(image, cv2.CV_64F).var())

    def test_cpu_scores_match_reference_for_shapes_and_supported_input_types(self):
        scorer = SharpnessScorer(allow_gpu=False, report=False)
        examples = [self.gray, self.gray[::2, ::2], self.gray[:1], self.gray[:, :1],
                    np.zeros((16, 16), np.uint8), np.full((17, 19), 255, np.uint8),
                    ((np.indices((17, 19)).sum(axis=0) % 2) * 255).astype(np.uint8),
                    self.gray.astype(np.float64) * .23,
                    np.stack([self.gray, self.gray // 2, self.gray // 3], axis=-1)]
        for image in examples:
            with self.subTest(shape=image.shape, dtype=image.dtype):
                self.assertAlmostEqual(scorer(image), self.reference(image), delta=1e-8)

    def test_no_opencl_preserves_full_cpu_budget(self):
        threads = cv2.getNumThreads()
        scorer = SharpnessScorer(report=False, minimum_pixels=0)
        with patch.object(cv2.ocl, 'haveOpenCL', return_value=False), \
                patch.object(cv2, 'setNumThreads', side_effect=AssertionError('Thread budget changed')):
            self.assertAlmostEqual(scorer(self.gray), self.reference(self.gray), delta=1e-8)
        self.assertEqual(cv2.getNumThreads(), threads)
        self.assertEqual(scorer.diagnostics()['calibrations'][0]['reason'], 'opencl_unavailable')

    def _gpu_context(self):
        device = Mock()
        device.name.return_value = 'test GPU'
        device.available.return_value = True
        device.type.return_value = cv2.ocl.Device_TYPE_GPU
        device.doubleFPConfig.return_value = 1
        return device

    def _clock_values(self, gpu_duration):
        values = []
        now = 0.
        for _ in range(5):
            values.extend((now, now + .01, now + .01, now + .01 + gpu_duration))
            now += .01 + gpu_duration
        return values

    def _probe(self, scorer, gpu_duration, score=None):
        with patch.object(cv2.ocl, 'haveOpenCL', return_value=True), \
                patch.object(cv2.ocl, 'useOpenCL', return_value=True), \
                patch.object(cv2.ocl, 'Device_getDefault', return_value=self._gpu_context()), \
                patch.object(scorer, '_gpu_variance', return_value=self.reference(self.gray) if score is None else score), \
                patch('gpu_image_ops.time.perf_counter', side_effect=self._clock_values(gpu_duration)):
            return scorer(self.gray)

    def test_slower_gpu_falls_back_to_cpu(self):
        scorer = SharpnessScorer(report=False, minimum_pixels=0)
        self.assertAlmostEqual(self._probe(scorer, .02), self.reference(self.gray), delta=1e-8)
        self.assertEqual(scorer.last_backend, 'cpu')
        self.assertEqual(scorer.diagnostics()['calibrations'][0]['reason'], 'gpu_transfer_cost_not_faster')

    def test_gpu_without_fp64_keeps_cpu_without_running_kernel(self):
        scorer = SharpnessScorer(report=False, minimum_pixels=0)
        device = self._gpu_context()
        device.doubleFPConfig.return_value = 0
        with patch.object(cv2.ocl, 'haveOpenCL', return_value=True), \
                patch.object(cv2.ocl, 'useOpenCL', return_value=True), \
                patch.object(cv2.ocl, 'Device_getDefault', return_value=device), \
                patch.object(scorer, '_gpu_variance', side_effect=AssertionError('Unsupported kernel')):
            self.assertAlmostEqual(scorer(self.gray), self.reference(self.gray), delta=1e-8)
        self.assertEqual(scorer.diagnostics()['calibrations'][0]['reason'], 'gpu_fp64_unavailable')

    def test_faster_verified_gpu_is_selected(self):
        scorer = SharpnessScorer(report=False, minimum_pixels=0)
        self.assertAlmostEqual(self._probe(scorer, .001), self.reference(self.gray), delta=1e-8)
        self.assertEqual(scorer.last_backend, 'opencl')

    def test_gpu_precision_failure_keeps_cpu_scores(self):
        scorer = SharpnessScorer(report=False, minimum_pixels=0)
        self.assertAlmostEqual(self._probe(scorer, .001, self.reference(self.gray) + .01),
                               self.reference(self.gray), delta=1e-8)
        self.assertEqual(scorer.last_backend, 'cpu')
        self.assertEqual(scorer.diagnostics()['calibrations'][0]['reason'], 'gpu_score_validation_failed')

    def test_runtime_gpu_error_falls_back_without_repeated_probes(self):
        scorer = SharpnessScorer(report=False, minimum_pixels=0)
        self._probe(scorer, .001)
        with patch.object(cv2.ocl, 'useOpenCL', return_value=True), \
                patch.object(scorer, '_gpu_variance', side_effect=cv2.error('GPU lost')) as gpu:
            for _ in range(2):
                self.assertAlmostEqual(scorer(self.gray), self.reference(self.gray), delta=1e-8)
            self.assertEqual(gpu.call_count, 1)
        self.assertEqual(scorer.last_backend, 'cpu')
        self.assertEqual(scorer.diagnostics()['gpu_failure'], 'error')

    def test_opencl_disabled_after_calibration_falls_back(self):
        scorer = SharpnessScorer(report=False, minimum_pixels=0)
        self._probe(scorer, .001)
        with patch.object(cv2.ocl, 'useOpenCL', return_value=False), \
                patch.object(scorer, '_gpu_variance', side_effect=AssertionError('OpenCL is disabled')):
            self.assertAlmostEqual(scorer(self.gray), self.reference(self.gray), delta=1e-8)
        self.assertEqual(scorer.last_backend, 'cpu')

    def test_explicit_opencl_device_is_respected(self):
        with patch.dict(os.environ, {'OPENCV_OPENCL_DEVICE': 'AMD:GPU:'}):
            configure_gpu_image_ops(prefer_nvidia=True)
            self.assertEqual(os.environ['OPENCV_OPENCL_DEVICE'], 'AMD:GPU:')


if __name__ == '__main__':
    unittest.main()
