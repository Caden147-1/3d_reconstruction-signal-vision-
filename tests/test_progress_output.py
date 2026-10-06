"""Progress is truthful on interruption and does not flood redirected logs."""

from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from progress_output import WorkProgress
from pipeline_output import Tee
from image_similarity import bounded_parallel_map
from video_frames import estimated_sample_count, extract_with_opencv


class Clock:
    now = 0.

    def __call__(self):
        return self.now


class ProgressTests(unittest.TestCase):
    def progress(self, total=10, **kwargs):
        self.clock = Clock()
        self.output = io.StringIO()
        return WorkProgress('Example', total, 'frames', clock=self.clock,
                            stream=self.output, **kwargs)

    def test_eta_samples_then_accounts_for_stall(self):
        progress = self.progress()
        self.clock.now = 2.
        progress.update(1)
        self.assertIsNone(progress.eta)
        self.clock.now = 4.
        progress.update(2)
        self.assertEqual(progress.eta, 16.)
        self.clock.now = 20.
        progress.update(2)
        self.assertEqual(progress.eta, 80.)
        progress.close()

    def test_logs_are_throttled_plain_and_final_count_is_not_forced(self):
        progress = self.progress()
        for count in range(1, 7):
            self.clock.now = float(count)
            progress.update(count)
        self.assertEqual(len(self.output.getvalue().splitlines()), 1)
        progress.close()
        text = self.output.getvalue()
        self.assertIn('6/10 frames', text)
        self.assertIn('incomplete', text)
        self.assertIn('ETA unavailable', text)
        self.assertNotIn('10/10', text)
        self.assertNotIn('\r', text)
        self.assertNotIn('###', text)

    def test_unknown_total_and_disabled_estimate(self):
        progress = self.progress(total=None)
        progress.update(5)
        self.assertIsNone(progress.eta)
        progress.close()
        self.assertIn('5/? frames', self.output.getvalue())
        self.assertIn('ETA unavailable', self.output.getvalue())
        progress = self.progress(estimate=False)
        self.clock.now = 4
        progress.update(2)
        self.clock.now = 8
        progress.update(4)
        self.assertIsNone(progress.eta)
        progress.close()
        self.assertIn('ETA disabled', self.output.getvalue())

    def test_estimate_environment_default_and_explicit_override(self):
        with patch.dict('os.environ', {'FYP_PROGRESS_ESTIMATE': '0'}):
            progress = self.progress()
            self.assertFalse(progress.estimate)
            progress.close()
            progress = self.progress(estimate=True)
            self.assertTrue(progress.estimate)
            progress.close()
        with patch.dict('os.environ', {'FYP_PROGRESS_ESTIMATE': '1'}):
            progress = self.progress()
            self.assertTrue(progress.estimate)
            progress.close()
            progress = self.progress(estimate=False)
            self.assertFalse(progress.estimate)
            progress.close()

    def test_iterator_counts_continue_but_not_unfinished_body(self):
        progress = self.progress(total=3)
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            with progress:
                for value in progress.iterate(range(3)):
                    if value == 0:
                        continue
                    raise RuntimeError('failed')
        self.assertEqual(progress.completed, 1)
        self.assertEqual(progress.status, 'failed')
        self.assertIn('1/3 frames', self.output.getvalue())

    def test_complete_and_early_budget_status(self):
        progress = self.progress(total=2)
        with progress:
            for _ in progress.iterate([1, 2]):
                pass
        self.assertEqual(progress.status, 'complete')
        self.assertEqual(progress.eta, 0.)
        progress = self.progress(total=3)
        with progress:
            progress.update(1)
            progress.close('budget exhausted')
        self.assertEqual(progress.completed, 1)
        self.assertEqual(progress.status, 'budget exhausted')

    def test_parallel_results_keep_input_order_and_count_completion(self):
        started = threading.Event()
        second_done = threading.Event()
        def task(value):
            if value == 0:
                started.set()
                self.assertTrue(second_done.wait(timeout=5))
            else:
                self.assertTrue(started.wait(timeout=5))
                second_done.set()
            return value * 10
        progress = self.progress(total=2)
        with progress:
            result = bounded_parallel_map(task, [0, 1], 2, 'test', progress)
        self.assertEqual(result, [0, 10])
        self.assertEqual(progress.completed, 2)
        self.assertEqual(progress.status, 'complete')

    def test_interactive_uses_optional_log_safe_writer(self):
        class Terminal(io.StringIO):
            progress_writes = 0
            def isatty(self):
                return True
            def write_progress(self, value):
                self.progress_writes += 1
                return super().write(value)
        terminal = Terminal()
        with WorkProgress('Interactive', 1, stream=terminal) as progress:
            progress.label = 'Renamed'
            progress.update(1)
        self.assertGreater(terminal.progress_writes, 0)
        self.assertIn('1/1', terminal.getvalue())
        self.assertIn('Renamed', terminal.getvalue())

    def test_narrow_terminal_retains_eta_and_log_keeps_full_snapshot(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        terminal, log, clock = Terminal(), io.StringIO(), Clock()
        tee = Tee(terminal, log, threading.RLock(), clock=clock)
        label = 'Video: sharper native frames (IMG_1735.MOV)'
        with patch('progress_output.shutil.get_terminal_size', return_value=os.terminal_size((66, 24))):
            progress = WorkProgress(label, 10, 'frames', stream=tee, clock=clock)
            clock.now = 2.
            progress.update(1)
            clock.now = 4.
            progress.update(2)
            clock.now = 20.
            progress.refresh()
            rendered = terminal.getvalue().split('\r')[-1].strip()
            self.assertIn('2/10 frames', rendered)
            self.assertIn('ETA ~00:01:20', rendered)
            self.assertLessEqual(len(rendered), 65)
            self.assertNotIn(label, rendered)
            record = log.getvalue().splitlines()[-1]
            self.assertIn(label, record)
            self.assertIn('2/10 frames | elapsed 00:00:20 | ETA ~00:01:20', record)
            progress.close('budget exhausted')
        final = log.getvalue().splitlines()[-1]
        self.assertIn(label, final)
        self.assertIn('2/10 frames', final)
        self.assertIn('budget exhausted', final)
        self.assertNotIn('\r', log.getvalue())
        self.assertNotIn('\x1b', log.getvalue())

    def test_narrow_unknown_total_keeps_honest_eta_and_next_task_start_logged(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        terminal, log, clock = Terminal(), io.StringIO(), Clock()
        tee = Tee(terminal, log, threading.RLock(), clock=clock)
        with patch('progress_output.shutil.get_terminal_size', return_value=os.terminal_size((66, 24))):
            with WorkProgress('Sparse: global bundle adjustment with no known duration',
                              stream=tee, clock=clock) as progress:
                progress.update(1)
                clock.now = 2.
                progress.refresh()
                self.assertIn('ETA unavailable', terminal.getvalue().split('\r')[-1])
                self.assertNotIn('ETA ~', terminal.getvalue())
            with WorkProgress('Next task', 10, stream=tee, clock=clock):
                pass
        self.assertIn('[progress] Next task | 0/10 items', log.getvalue())

    def test_invalid_progress_rejected(self):
        progress = self.progress()
        progress.update(2)
        with self.assertRaises(ValueError):
            progress.update(1)
        with self.assertRaises(ValueError):
            progress.update(float('nan'))
        progress.close()

    def test_bursts_keep_measured_window_and_bad_metadata_loses_total(self):
        progress = self.progress(total=100)
        self.clock.now = 10.
        for count in range(1, 33):
            progress.update(count)
        self.assertAlmostEqual(progress.eta, 68 * 10 / 32)
        progress.update(101)
        self.assertIsNone(progress.total)
        self.assertIsNone(progress.eta)
        progress.close()


class VideoProgressTests(unittest.TestCase):
    def test_known_span_and_budget(self):
        self.assertEqual(estimated_sample_count(2., .5, 0), 4)
        self.assertEqual(estimated_sample_count(2., .5, 3), 3)
        self.assertIsNone(estimated_sample_count(None, .5, 3))

    def test_extraction_keeps_frames_timestamps_and_releases_on_error(self):
        import cv2
        class Capture:
            index = -1
            released = False
            def isOpened(self):
                return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 4., cv2.CAP_PROP_FRAME_COUNT: 8.,
                        cv2.CAP_PROP_POS_MSEC: self.index * 250.}.get(prop, 0.)
            def grab(self):
                self.index += 1
                return self.index < 8
            def retrieve(self):
                return True, self.index
            def release(self):
                self.released = True
        capture = Capture()
        written = []
        timestamps = {}
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()), \
                patch('video_frames.open_video', return_value=capture), \
                patch('video_frames.write_frame', side_effect=lambda path, frame, args: written.append(frame) or True):
            status, _ = extract_with_opencv(Path('clip.MOV'), Path(directory), .5, 0,
                                            frame_timestamps=timestamps)
        self.assertEqual(status, 'ok')
        self.assertEqual(written, [0, 2, 4, 6])
        self.assertEqual([item['video_time_s'] for item in timestamps.values()], [0., .5, 1., 1.5])
        self.assertTrue(capture.released)
        capture = Capture()
        output = io.StringIO()
        with redirect_stderr(output), patch('video_frames.open_video', return_value=capture), \
                patch('video_frames.write_frame', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                extract_with_opencv(Path('clip.MOV'), Path('.'), .5, 0)
        self.assertTrue(capture.released)
        self.assertIn('failed', output.getvalue())


if __name__ == '__main__':
    unittest.main()
