"""Destructive cleanup must preserve captures, outputs and active workspaces."""
from contextlib import redirect_stdout
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from startup_cleanup import cleanup_project, archive_results
from workspace_lock import workspace_lock


class StartupCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def write(self, name, value=b'fixture'):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def workspace(self, name='experiments/old/workspace'):
        self.write(name+'/colmap_workspace_summary.json', b'{}')
        self.write(name+'/images/original.jpg')
        self.write(name+'/dense/sparse/images.bin')
        self.write(name+'/dense/images/copy.jpg')
        self.write(name+'/dense/stereo/depth_maps/view.bin')
        self.write(name+'/dense/stereo/normal_maps/view.bin')
        self.write(name+'/dense/stereo/patch-match.cfg')
        self.write(name+'/dense/final.ply', b'important model')
        return self.root/name

    def clean(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return cleanup_project(self.root, **kwargs)

    def test_history_caches_removed_but_images_results_configs_and_code_retained(self):
        workspace=self.workspace()
        for name in ('video/original.MOV','capture.sess/data.dat','VideoScene/checkpoints/weights.bin','.git/objects/data','main.py'):
            self.write(name)
        result=self.clean()
        self.assertEqual(result['status'],'completed')
        self.assertFalse((workspace/'dense/stereo/depth_maps').exists())
        self.assertFalse((workspace/'dense/images').exists())
        for name in ('images/original.jpg','dense/final.ply','dense/sparse/images.bin','dense/stereo/patch-match.cfg'):
            self.assertTrue((workspace/name).exists())
        self.assertTrue((self.root/'video/original.MOV').exists())
        self.assertTrue((self.root/'VideoScene/checkpoints/weights.bin').exists())

    def test_default_latest_and_explicit_inputs_are_protected(self):
        latest=self.workspace('generated/colmap_workspace')
        custom=self.workspace('experiments/custom/workspace')
        result=self.clean(protected=[custom/'dense/stereo'])
        self.assertEqual(result['removed_files'],0)
        self.assertTrue((latest/'dense/stereo/depth_maps/view.bin').exists())
        self.assertTrue((custom/'dense/stereo/depth_maps/view.bin').exists())

    def test_dry_run_and_idempotence(self):
        workspace=self.workspace()
        result=self.clean(apply=False)
        self.assertGreater(result['candidate_logical_bytes'],0)
        self.assertTrue((workspace/'dense/images/copy.jpg').exists())
        self.clean()
        self.assertEqual(self.clean()['removed_files'],0)

    def test_unknown_files_stop_directory_deletion(self):
        workspace=self.workspace()
        (workspace/'dense/stereo/depth_maps/notes.txt').write_text('keep')
        self.clean()
        self.assertTrue((workspace/'dense/stereo/depth_maps/view.bin').exists())

    def test_keep_two_most_recent_snapshots(self):
        workspace=self.workspace()
        for index in range(5):
            for name in ('images.bin','cameras.bin','points3D.bin'):
                path=self.write(str(workspace.relative_to(self.root))+f'/sparse_snapshots/run/{index}/{name}')
                os.utime(path,(100+index,100+index))
        self.clean()
        self.assertEqual({p.name for p in (workspace/'sparse_snapshots/run').iterdir()},{'3','4'})

    def test_only_reextractable_video_images_deleted(self):
        self.write('video/input.MOV')
        self.write('processed/video_frames_summary.json',json.dumps({'videos':[{'video':'video/input.MOV'}]}).encode())
        frame=self.write('processed/video_frames/clip/001.jpg')
        report=self.write('processed/video_refinement/summary.json',b'{}')
        self.clean()
        self.assertFalse(frame.exists());self.assertTrue(report.exists())
        self.write('processed/video_frames/clip/001.jpg')
        (self.root/'video/input.MOV').unlink()
        self.clean()
        self.assertTrue(frame.exists())

    def test_locked_workspace_defers_every_deletion(self):
        workspace=self.workspace()
        ready,release=threading.Event(),threading.Event()
        def hold():
            with workspace_lock(workspace):
                ready.set();release.wait(20)
        thread=threading.Thread(target=hold)
        thread.start();self.assertTrue(ready.wait(5))
        try:
            result=self.clean()
            self.assertEqual(result['status'],'deferred')
            self.assertEqual(result['removed_files'],0)
            self.assertTrue((workspace/'dense/images/copy.jpg').exists())
        finally:
            release.set();thread.join(5)

    def test_archive_survives_reset_and_log_rewrite(self):
        workspace=self.workspace()
        log=workspace/'main_console.log';log.write_text('old log')
        with redirect_stdout(io.StringIO()):
            archive=archive_results(self.root,workspace)
        log.write_text('new log')
        (workspace/'dense/final.ply').unlink()
        self.assertEqual((archive/'dense/final.ply').read_bytes(),b'important model')
        self.assertEqual((archive/'main_console.log').read_text(),'old log')
        self.assertFalse((archive/'images').exists())

    def test_main_cleans_before_console_and_preserves_completed_outputs_on_reset(self):
        import main
        from preprocess_pipeline import parse_args
        args=parse_args(['--root',str(self.root),'--video-only','--skip-colmap-prepare'])
        workspace=self.workspace('generated/colmap_workspace')
        with redirect_stdout(io.StringIO()):
            main.reset_stopped_outputs(args)
        archives=list((self.root/'generated/results').glob('*/dense/final.ply'))
        self.assertEqual(len(archives),1)
        self.assertEqual(archives[0].read_bytes(),b'important model')
        self.assertFalse((workspace/'dense/final.ply').exists())
        events=[]
        @contextmanager
        def console(_workspace):
            events.append('console');yield {}
        with patch('sys.argv',['main.py','--root',str(self.root),'--video-only','--skip-colmap-prepare']), \
             patch('startup_cleanup.cleanup_project',side_effect=lambda *a,**kw:events.append('cleanup')), \
             patch.object(main,'console_run',side_effect=console), \
             patch.object(main,'run_locked_pipeline',return_value=0):
            self.assertEqual(main.main(),0)
        self.assertEqual(events,['cleanup','console'])

    def test_main_follows_active_run_without_cleanup(self):
        import main
        with patch('sys.argv',['main.py','--root',str(self.root),'--video-only','--skip-colmap-prepare']), \
             patch.object(main,'workspace_activity',return_value={'active':True,'owner':{}}), \
             patch.object(main,'follow_active_run',return_value=0) as follow, \
             patch('startup_cleanup.cleanup_project') as clean:
            self.assertEqual(main.main(),0)
        clean.assert_not_called();follow.assert_called_once()

    @unittest.skipIf(os.name=='nt','Symlink creation requires special Windows privileges')
    def test_redirected_cache_not_deleted(self):
        workspace=self.workspace()
        outside=self.root/'outside';outside.mkdir();(outside/'important.bin').write_bytes(b'keep')
        path=workspace/'dense/stereo/normal_maps';(path/'view.bin').unlink();path.rmdir()
        path.symlink_to(outside,target_is_directory=True)
        self.clean()
        self.assertEqual((outside/'important.bin').read_bytes(),b'keep')


if __name__=='__main__':
    unittest.main()
