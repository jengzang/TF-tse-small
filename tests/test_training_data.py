"""Optional synthetic-data integration check; requires training dependencies."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(importlib.util.find_spec('pyroomacoustics'), 'Install training dependencies for room simulation')
class TrainingDataTests(unittest.TestCase):
    def test_online_mixture_and_supervision_shapes(self):
        import numpy as np
        import pandas as pd
        import soundfile as sf
        import torch
        from dataset.data_wsj0_rir_pyroom_v7 import tr_wsj_mix_dataset

        torch.set_num_threads(1)
        rng = np.random.default_rng(3407)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            speakers = {}
            for speaker in ('one', 'two'):
                speakers[speaker] = {}
                for i in range(4):
                    path = folder / f'{speaker}_{i}.wav'
                    sf.write(path, rng.normal(0, .05, 16000), 16000, subtype='FLOAT')
                    speakers[speaker][str(path)] = torch.randn(1, 256)
            torch.save(speakers, folder / 'emb.pt')
            torch.save({k: list(v) for k, v in speakers.items()}, folder / 'utts.pt')
            noises = []
            for i in range(3):
                path = folder / f'noise_{i}.wav'
                sf.write(path, rng.normal(0, .02, 16000), 16000, subtype='FLOAT')
                noises.append(f'n{i} {path}\n')
            (folder / 'noise.scp').write_text(''.join(noises))
            pd.DataFrame([dict(
                room_x=4, room_y=5, room_z=3, micL_x=2, micL_y=2, mic_z=1.5,
                s1_x=1, s1_y=1, s1_z=1.5, s2_x=3, s2_y=3, s2_z=1.5, T60=.2,
            )]).to_csv(folder / 'rooms.csv', index=False)
            dataset = tr_wsj_mix_dataset(
                {'synthetic': folder / 'utts.pt'}, {'synthetic': folder / 'emb.pt'},
                folder / 'noise.scp', folder / 'rooms.csv', 'unused_rir.scp',
                ['synthetic'], 1, 10, 16000, 'max',
            )
            mix, source1, emb1, vad1, source2, emb2, vad2 = dataset[0]
            self.assertEqual(mix.shape, (1, 2, 160000))
            for value in (source1, source2, vad1, vad2):
                self.assertEqual(value.shape, (160000,))
                self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.isfinite(mix).all())
            self.assertEqual(emb1.shape, (1, 256))
            self.assertEqual(emb2.shape, (1, 256))


if __name__ == '__main__':
    unittest.main()
