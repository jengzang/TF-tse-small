"""CPU checks for the baseline inference interface (no trained weights required)."""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def module(self):
        self.assertTrue((ROOT / 'infer.py').is_file(), 'Missing baseline inference entry')
        import infer
        return infer

    def test_roundtrip_checkpoint_and_cli(self):
        import numpy as np
        import soundfile as sf
        import torch
        mod = self.module()
        model = mod.build_model(ROOT / 'config/config_tfgridnet_all.yaml')
        self.assertEqual(sum(p.numel() for p in model.parameters()), 298238)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = folder / 'model.pt'
            torch.save({'model_state_dict': {'module.' + k: v for k, v in model.state_dict().items()}}, checkpoint)
            rng = np.random.default_rng(34)
            sf.write(folder / 'mix.wav', rng.normal(0, .05, (3200, 2)), 16000, subtype='FLOAT')
            np.save(folder / 'embedding.npy', rng.normal(size=(1, 256)).astype('float32'))
            result = subprocess.run([
                sys.executable, str(ROOT / 'infer.py'), '--checkpoint', str(checkpoint),
                '--mix', str(folder / 'mix.wav'), '--embedding', str(folder / 'embedding.npy'),
                '--output', str(folder / 'target.wav'), '--threads', '1',
            ], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            audio, sr = sf.read(folder / 'target.wav')
            self.assertEqual(sr, 16000)
            self.assertEqual(audio.shape, (3200,))
            self.assertTrue(np.isfinite(audio).all())

    def test_mono_is_rejected(self):
        import numpy as np
        import soundfile as sf
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mono.wav'
            sf.write(path, np.zeros(3200), 16000)
            with self.assertRaisesRegex(ValueError, 'two channels'):
                mod.read_mixture(path)

    def test_embedding_shape_and_nonfinite_are_rejected(self):
        import numpy as np
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'embedding.npy'
            for value in (np.zeros(255), np.full(256, np.nan)):
                np.save(path, value)
                with self.assertRaises(ValueError):
                    mod.read_embedding(path)

    def test_checkpoint_mismatch_is_rejected(self):
        import torch
        mod = self.module()
        model = mod.build_model(ROOT / 'config/config_tfgridnet_all.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'wrong.pt'
            torch.save({'model_state_dict': {'wrong': torch.zeros(1)}}, path)
            with self.assertRaises(RuntimeError):
                mod.load_weights(model, path)


if __name__ == '__main__':
    unittest.main()
