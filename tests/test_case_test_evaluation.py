import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.run_case import evaluate_heldout_test

class TestEvaluationTests(unittest.TestCase):
    def test_uses_heldout_test_and_validation_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config={'corpus_index':str(root/'index.json'),'name':'cases/run'}
            with patch('scripts.evaluate_case.main') as evaluate:
                evaluate_heldout_test(config,root/'resolved.yaml','cuda:1','bf16',1)
            args=evaluate.call_args.args[0]
            self.assertIn(str(root/'test/mix.scp'),args)
            self.assertIn('chkpt/cases/run/temp_best.pth.tar',args)
            self.assertIn(str(root/'test/metadata.json'),args)
