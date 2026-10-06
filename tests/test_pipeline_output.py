import io
import threading
import unittest

from pipeline_output import CompactProgress, StageTimings, Tee


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value

    def step(self, seconds=20.):
        self.value += seconds


class Terminal(io.StringIO):
    def isatty(self):
        return True


class PipelineProgressTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.stream = io.StringIO()
        self.display = CompactProgress(stream=self.stream, clock=self.clock, scope='sparse')

    def tearDown(self):
        self.display.close()

    def test_matching_counts_completed_blocks_and_last_block(self):
        self.display.observe('[2/4] Matching selected pairs...')
        self.display.observe('Processing block [1/3]')
        self.assertEqual(self.display.work.completed, 0)
        self.clock.step(6.)
        self.display.observe('in 6.037s')
        self.assertEqual(self.display.work.completed, 1)
        self.display.observe('Processing block [2/3]')
        self.clock.step(8.)
        self.display.observe('in 8.000s')
        self.assertAlmostEqual(self.display.work.eta, 7.)
        self.display.observe('Processing block [3/3]')
        self.assertEqual(self.display.work.completed, 2)
        self.clock.step(10.)
        self.display.observe('in 10.000s')
        self.assertEqual(self.display.work.completed, 3)
        self.display.observe('Elapsed time: 0.4 [minutes]')
        self.assertIsNone(self.display.work)
        self.assertIn('[stage] sparse [2/4]', self.stream.getvalue())

    def test_undistortion_last_start_is_not_completion(self):
        self.display.observe('Undistorting image [1/2]')
        self.clock.step()
        self.display.observe('Undistorting image [2/2]')
        self.assertEqual(self.display.work.completed, 1)
        self.display.observe('Writing reconstruction...')
        self.assertIsNone(self.display.work)
        self.assertIn('2/2 images', self.stream.getvalue())

    def test_patchmatch_out_of_order_duplicate_and_new_phase(self):
        self.display.observe('Processing view 2 / 3 for b.jpg')
        self.display.observe('geom_consistency: 0')
        self.display.observe('Processing view 1 / 3 for a.jpg')
        self.assertEqual(self.display.work.completed, 0)
        self.clock.step()
        self.display.observe('Writing photometric output for b.jpg')
        self.display.observe('Writing photometric output for b.jpg')
        self.assertEqual(self.display.work.completed, 1)
        self.clock.step()
        self.display.observe('Writing photometric output for a.jpg')
        self.assertEqual(self.display.work.completed, 2)
        self.assertGreater(self.display.work.eta, 0)
        self.display.observe('Processing view 3 / 3 for c.jpg')
        self.display.observe('Writing photometric output for c.jpg')
        self.display.observe('Processing view 1 / 3 for a.jpg')
        self.display.observe('geom_consistency: 1')
        self.assertEqual(self.display.work.completed, 0)
        self.display.observe('Writing geometric output for a.jpg')
        self.assertEqual(self.display.work.completed, 1)
        self.assertIn('geometric', self.display.work.label)

    def test_sparse_count_is_not_total_progress_or_predicted_eta(self):
        self.display.observe('Registering image #752 (num_reg_frames=2)')
        self.clock.step()
        self.display.observe('Registering image #758 (num_reg_frames=3)')
        self.display.observe('Retriangulation and Global bundle adjustment')
        self.assertIsNone(self.display.work.total)
        self.assertIsNone(self.display.work.eta)
        self.assertIn('global BA #1', self.display.work.label)
        self.clock.step(3600)
        self.display.progress()
        self.assertIn('ETA unavailable', self.stream.getvalue())
        self.display.observe('Registering image #1 (num_reg_frames=2)')
        self.assertEqual(self.display.work.completed, 2)

    def test_failure_does_not_force_completion_and_logs_no_cursor_codes(self):
        self.display.observe('Processed file [1/100]')
        self.clock.step()
        self.display.observe('Processed file [2/100]')
        self.display.close('failed')
        output = self.stream.getvalue()
        self.assertIn('2/100 images', output)
        self.assertIn('failed', output)
        self.assertNotIn('100/100', output)
        self.assertNotIn('\r', output)

    def test_child_progress_and_warnings_preserved_without_second_bar(self):
        self.display.observe('[progress] Observation evidence | 2/10 points | failed')
        self.display.observe('W20261005 17:54:53.454759 3724 solver.cc:123] Test warning')
        self.display.observe('W20261005 17:54:53.454859 3724 solver.cc:123] Test warning')
        self.display.close()
        self.assertIn('2/10 points | failed', self.stream.getvalue())
        self.assertIn('repeated 1 more times', self.stream.getvalue())
        self.assertIsNone(self.display.work)

    def test_verbose_does_not_add_progress(self):
        display = CompactProgress(stream=self.stream, verbose=True)
        display.observe('Processed file [1/10]')
        display.observe('Processing view 1 / 10 for a.jpg')
        display.close()
        self.assertEqual(self.stream.getvalue(), '')

    def test_estimate_flag_applies_to_sparse_feature_work(self):
        display = CompactProgress(stream=self.stream, estimate=False, clock=self.clock)
        display.observe('Processed file [1/10]')
        self.clock.step()
        display.observe('Processed file [2/10]')
        self.assertIsNone(display.work.eta)
        display.close()

    def test_tee_retains_tty_and_throttles_control_free_log(self):
        terminal, log = Terminal(), io.StringIO()
        tee = Tee(terminal, log, threading.RLock(), clock=self.clock)
        self.assertTrue(tee.isatty())
        tee.write_progress('\rfirst 1/10')
        self.clock.step(1.)
        tee.write_progress('\rsecond 2/10')
        self.assertNotIn('second', log.getvalue())
        self.clock.step(15.)
        tee.write_progress('\rthird 3/10\x1b[K')
        tee.write_progress('\rfinal 4/10')
        tee.write_progress('\n')
        self.assertIn('third', log.getvalue())
        self.assertIn('final', log.getvalue())
        self.assertNotIn('\r', log.getvalue())
        self.assertNotIn('\x1b', log.getvalue())
        self.assertIn('\r', terminal.getvalue())

    def test_stage_times_keep_finalization_and_unknown_tasks(self):
        timing = StageTimings(clock=self.clock)
        self.clock.step(10)
        timing.observe('[1/4] Extracting features...')
        self.clock.step(20)
        timing.observe('[report] Generating reconstruction quality report...')
        self.clock.step(30)
        report = timing.finish('completed')
        self.assertEqual([r['elapsed_seconds'] for r in report['stages']], [10, 20, 30])
        self.assertEqual(report['elapsed_seconds'], 60)


if __name__ == '__main__':
    unittest.main()
