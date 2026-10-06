"""Measured GPU acceleration for native-frame sharpness, with CPU fallback.

The score remains the population variance of the default OpenCV Laplacian.
Large grayscale images use OpenCL only when its FP64 result matches the CPU
reference and a short local measurement shows an advantage. CPU thread limits
are never changed here.
"""

from collections import OrderedDict
import math
import os
from statistics import median
import threading
import time


def configure_gpu_image_ops(*, prefer_nvidia=False):
    """Select the discrete OpenCL device before importing cv2, respecting users.

    Some builds read this preference at import time. An existing context cannot
    be switched using this environment variable; the scorer always reports the
    actual device instead of assuming this preference won.
    """
    if prefer_nvidia:
        os.environ.setdefault('OPENCV_OPENCL_DEVICE', 'NVIDIA:GPU:')


def _cpu_variance(gray):
    import cv2
    import numpy as np

    if gray.ndim == 2 and gray.dtype == np.uint8:
        # The ksize=1 Laplacian of uint8 is an exact integer in [-1020, 1020].
        # Keep that intermediate compact; meanStdDev accumulates in double.
        # This also avoids the old full-image NumPy squared-deviation array.
        laplacian = cv2.Laplacian(gray, cv2.CV_16S)
        _, deviation = cv2.meanStdDev(laplacian)
        return float(deviation[0, 0] ** 2)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class SharpnessScorer:
    """A bounded, thread-safe backend cache for full-resolution blur scores."""

    def __init__(self, *, allow_gpu=True, report=True, minimum_pixels=1_000_000):
        self.allow_gpu = allow_gpu
        self.report = report
        self.minimum_pixels = minimum_pixels
        self._lock = threading.RLock()
        self._decisions = OrderedDict()
        self._mask = None
        self._mask_shape = None
        self._failure = None
        self.last_backend = 'cpu'

    def _gpu_variance(self, gray):
        import cv2
        import numpy as np

        mask_key = (threading.get_ident(), gray.shape)
        if self._mask_shape != mask_key:
            self._mask = cv2.UMat(np.ones(gray.shape, dtype=np.uint8))
            self._mask_shape = mask_key
        laplacian = cv2.Laplacian(cv2.UMat(gray), cv2.CV_64F)
        # An explicit all-valid mask avoids OpenCV 4.13's empty-mask OpenCL
        # kernel failure, which would silently execute the reduction on CPU.
        # FP64 is required: the CV_16S/CV_32F GPU reductions lose precision.
        _, deviation = cv2.meanStdDev(laplacian, mask=self._mask)
        score = float(deviation.get()[0, 0] ** 2)
        if not math.isfinite(score) or score < 0:
            raise ValueError('OpenCL returned an invalid sharpness score')
        return score

    def _choose(self, gray):
        import cv2

        decision = {'backend': 'cpu', 'reason': 'opencl_unavailable'}
        if not cv2.ocl.haveOpenCL() or not cv2.ocl.useOpenCL():
            return decision
        device = cv2.ocl.Device_getDefault()
        decision['device'] = device.name()
        if not device.available() or not (device.type() & cv2.ocl.Device_TYPE_GPU):
            return dict(decision, reason='opencl_device_is_not_a_gpu')
        if not device.doubleFPConfig():
            return dict(decision, reason='gpu_fp64_unavailable')

        expected = _cpu_variance(gray)
        candidate = self._gpu_variance(gray)  # Includes one-time kernel setup.
        if not math.isclose(candidate, expected, rel_tol=1e-12, abs_tol=1e-9):
            return dict(decision, reason='gpu_score_validation_failed')
        cpu_times, gpu_times = [], []
        # Consecutive frames arrive in different host buffers. Repeatedly
        # timing the same image, interleaved with CPU work, overstates the GPU
        # advantage on some drivers. Include buffer rotation and steady work.
        samples = (gray, gray.copy(), gray.copy())
        for sample in samples:
            self._gpu_variance(sample)
        # Synchronization is included: getting the tiny result waits for GPU
        # work, and every call includes the host-to-device image upload.
        for _ in range(5):
            started = time.perf_counter()
            for sample in samples:
                _cpu_variance(sample)
            cpu_times.append((time.perf_counter() - started) / len(samples))
            started = time.perf_counter()
            for sample in samples:
                self._gpu_variance(sample)
            gpu_times.append((time.perf_counter() - started) / len(samples))
        decision.update(cpu_seconds=median(cpu_times), gpu_seconds=median(gpu_times))
        if decision['gpu_seconds'] < decision['cpu_seconds'] * .85:
            decision.update(backend='opencl', reason='gpu_score_and_speed_verified')
        else:
            decision['reason'] = 'gpu_transfer_cost_not_faster'
        return decision

    def _announce(self, decision):
        if not self.report:
            return
        if decision['backend'] == 'opencl':
            ratio = decision['cpu_seconds'] / max(decision['gpu_seconds'], 1e-12)
            print(f"[resources] Native-frame sharpness: OpenCL GPU {decision['device']} "
                  f"(FP64, measured {ratio:.1f}x versus CPU scoring); CPU thread budget unchanged.", flush=True)
        else:
            print(f"[resources] Native-frame sharpness: CPU ({decision['reason']}); "
                  'CPU thread budget unchanged.', flush=True)

    def __call__(self, gray):
        import cv2
        import numpy as np

        if (not self.allow_gpu or gray.ndim != 2 or gray.dtype != np.uint8 or
                gray.size < self.minimum_pixels):
            self.last_backend = 'cpu'
            return _cpu_variance(gray)
        with self._lock:
            if self._failure:
                self.last_backend = 'cpu'
                return _cpu_variance(gray)
            try:
                # OpenCL execution contexts belong to threads. Never reuse a
                # context's mask or decision from another calling thread.
                key = (threading.get_ident(), gray.shape)
                if key not in self._decisions:
                    self._mask = self._mask_shape = None
                    self._decisions[key] = self._choose(gray)
                    if len(self._decisions) > 4:
                        self._decisions.popitem(last=False)
                    self._announce(self._decisions[key])
                decision = self._decisions[key]
                self._decisions.move_to_end(key)
                self.last_backend = decision['backend']
                if decision['backend'] == 'opencl':
                    if not cv2.ocl.useOpenCL():
                        raise RuntimeError('OpenCL was disabled after calibration')
                    return self._gpu_variance(gray)
            except (cv2.error, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                self._failure = type(exc).__name__
                self._mask = self._mask_shape = None
                self.last_backend = 'cpu'
                self._announce({'backend': 'cpu', 'reason': f'gpu_unavailable_{self._failure}'})
            return _cpu_variance(gray)

    def diagnostics(self):
        with self._lock:
            return {'last_backend': self.last_backend, 'gpu_failure': self._failure,
                    'calibrations': [dict(value, shape=list(key[1]))
                                     for key, value in self._decisions.items()]}


_scorer = SharpnessScorer()


def laplacian_variance(gray):
    return _scorer(gray)


def sharpness_diagnostics():
    return _scorer.diagnostics()
