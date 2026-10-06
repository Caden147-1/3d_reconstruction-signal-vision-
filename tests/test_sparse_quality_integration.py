"""New quality checks must not rewrite the provenance of completed models."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from colmap_bridge import write_sparse_batch
from colmap_workspace_check import (
    OPTIMIZED_PIPELINE_CODE, RECOVERY_PIPELINE_CODE, dense_readiness_failures,
    sparse_input_identity,
)
from portable_stage import parse_recipe


class SparseQualityIntegrationTests(unittest.TestCase):
    def test_bad_registration_is_blocked_even_in_legacy_summary(self):
        report = {'input': {'unique_listed_count': 1703},
                  'dense_input_sparse': {'exists': True, 'registered_image_count': 983}}
        self.assertTrue(dense_readiness_failures(report))
        report['dense_input_sparse']['registered_image_count'] = 1600
        self.assertEqual(dense_readiness_failures(report), [])

    def test_missing_model_is_left_to_structural_validation(self):
        self.assertEqual(dense_readiness_failures({'dense_input_sparse': {'exists': False}}), [])

    def test_new_recovery_dependencies_do_not_invalidate_old_recipes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'image_list.txt').write_text('video/f.jpg\n')
            script = root / 'run_colmap_sparse.bat'
            for extra, expected in [('', OPTIMIZED_PIPELINE_CODE),
                                    ('\npython sparse_recovery.py', RECOVERY_PIPELINE_CODE)]:
                script.write_text('python sparse_optimization.py' + extra)
                with patch('colmap_workspace_check.file_identity', return_value={'size': 1, 'sha256': 'test'}):
                    identity = sparse_input_identity(root, script.name)
                names = {name.removeprefix('pipeline_code/') for name in identity['files']
                         if name.startswith('pipeline_code/')}
                self.assertEqual(names, set(expected))

    def test_recovery_recipe_runs_before_alignment_and_certification(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            write_sparse_batch(workspace, workspace, workspace/'colmap.exe', None,
                               'SIMPLE_RADIAL', False, True, True, 'constrained', True,
                               3200, 24000, 32768, 32)
            text = (workspace/'run_colmap_sparse.bat').read_text(encoding='utf-8')
            self.assertLess(text.index('sparse_optimization.py'), text.index('sparse_recovery.py'))
            self.assertLess(text.index('sparse_recovery.py'), text.index('pdr_alignment.py'))
            self.assertLess(text.index('pdr_alignment.py'), text.index('--record-sparse-complete'))
            self.assertTrue(parse_recipe(text))


if __name__ == '__main__':
    unittest.main()
