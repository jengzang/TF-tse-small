import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch


class BundleDatasetTests(unittest.TestCase):
    def test_zero_mix_rejected_but_absent_reference_allowed(self):
        from dataset.bundle_data import BundleDataset
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('mix','ref','aux'):
                path=root/(name+'.wav')
                sf.write(path,np.zeros(64000,dtype=np.float32),16000,subtype='FLOAT')
                (root/(name+'.scp')).write_text(f'a {path}\n')
            cache=root/'cache.pt';torch.save({'embeddings':{'a':torch.ones(1,256)}},cache)
            ds=BundleDataset(*(str(root/(x+'.scp')) for x in ('mix','ref','aux')),
                             mono_policy='mono',embedding_cache=str(cache))
            with self.assertRaisesRegex(ValueError,'silent mix'):
                ds[0]
            sf.write(root/'mix.wav',np.ones(64000,dtype=np.float32)*.1,16000,subtype='FLOAT')
            _,target,_=ds[0]
            self.assertEqual(torch.count_nonzero(target).item(),0)

    def test_keyed_pairing_relocation_crop_and_mono(self):
        from dataset.bundle_data import BundleDataset
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            wavdir = root / 'data/.tse_aishell2_train_dev_whamr_strict_shards/train/shard_00/wav'
            wavdir.mkdir(parents=True)
            for name, data in [('mix', np.arange(20) / 20), ('ref', np.arange(20) / 40), ('aux', np.ones(20) * .2)]:
                sf.write(wavdir / f'a_{name}.wav', data, 16000, subtype='FLOAT')
                path = '/old/root/data/.tse_aishell2_train_dev_whamr_strict_shards/train/shard_00/wav/a_' + name + '.wav'
                (root / f'{name}.scp').write_text(f'a {path}\n')
            with patch('dataset.bundle_data.make_speaker_encoder', return_value=lambda x: torch.ones(1, 256)):
                ds = BundleDataset(*(str(root / (x + '.scp')) for x in ('mix','ref','aux')), bundle_root=str(root), duration=10/16000, mono_policy='duplicate', training=True)
                mix, ref, emb, mask = ds[0]
            self.assertEqual(tuple(mix.shape), (1, 2, 10))
            self.assertTrue(torch.equal(mix[0, 0], mix[0, 1]))
            self.assertTrue(torch.allclose(mix[0, 0], ref * 2))
            self.assertEqual(tuple(emb.shape), (1, 256))
            self.assertEqual(tuple(mask.shape), (10,))

    def test_mismatched_keys_rejected(self):
        from dataset.bundle_data import BundleDataset
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, key in [('mix','a'), ('ref','b'), ('aux','a')]:
                (root / f'{name}.scp').write_text(f'{key} /unused.wav\n')
            with self.assertRaisesRegex(ValueError, 'IDs'):
                BundleDataset(*(str(root / (x + '.scp')) for x in ('mix','ref','aux')))

    def test_rejects_mono_without_opt_in(self):
        from dataset.bundle_data import BundleDataset
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            wav = root / 'a.wav'
            sf.write(wav, np.ones(100, dtype=np.float32), 16000, subtype='FLOAT')
            for name in ('mix', 'ref', 'aux'):
                (root / f'{name}.scp').write_text(f'a {wav}\n')
            ds = BundleDataset(*(str(root / (x + '.scp')) for x in ('mix', 'ref', 'aux')))
            with self.assertRaisesRegex(ValueError, 'mono_policy'):
                ds[0]

    def test_cached_embedding_matches_id_and_avoids_encoder(self):
        from dataset.bundle_data import BundleDataset
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            wav = root / 'a.wav'
            sf.write(wav, np.arange(100, dtype=np.float32)/100, 16000, subtype='FLOAT')
            for name in ('mix', 'ref', 'aux'):
                (root / f'{name}.scp').write_text(f'a {wav}\n')
            cache = root / 'cache.pt'
            torch.save({'embeddings': {'a': torch.arange(256).reshape(1,256).float()}}, cache)
            ds = BundleDataset(*(str(root / (x + '.scp')) for x in ('mix','ref','aux')),
                               duration=100/16000, mono_policy='duplicate', embedding_cache=str(cache))
            with patch('dataset.bundle_data.make_speaker_encoder', side_effect=AssertionError('online encoder used')):
                _, _, emb = ds[0]
            self.assertTrue(torch.equal(emb, torch.arange(256).reshape(1,256).float()))

    def test_cache_missing_id_rejected(self):
        from dataset.bundle_data import BundleDataset
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('mix','ref','aux'):
                (root / f'{name}.scp').write_text('a /unused.wav\n')
            torch.save({'embeddings': {}}, root / 'cache.pt')
            with self.assertRaisesRegex(ValueError, 'missing'):
                BundleDataset(*(str(root / (x + '.scp')) for x in ('mix','ref','aux')), embedding_cache=str(root / 'cache.pt'))

    def test_io_error_from_worker_contains_sample_and_path(self):
        from dataset.bundle_data import BundleDataset
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('mix', 'ref', 'aux'):
                (root / f'{name}.scp').write_text(f'utt123 {root}/missing_{name}.wav\n')
            ds = BundleDataset(*(str(root / (x + '.scp')) for x in ('mix','ref','aux')),
                               mono_policy='duplicate', read_attempts=1)
            with self.assertRaises(RuntimeError) as error:
                next(iter(DataLoader(ds, batch_size=1, num_workers=1)))
            self.assertIn('utt123', str(error.exception))
            self.assertIn('missing_mix.wav', str(error.exception))

    def test_retries_transient_read_failure_without_skipping(self):
        from dataset.bundle_data import BundleDataset
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('mix', 'ref', 'aux'):
                (root / f'{name}.scp').write_text(f'utt123 {root}/{name}.wav\n')
            ds = BundleDataset(*(str(root / (x + '.scp')) for x in ('mix','ref','aux')),
                               duration=10/16000, mono_policy='duplicate', read_attempts=2,
                               retry_delay=0)
            audio=(np.ones((10,1),dtype=np.float32),16000)
            with patch('dataset.bundle_data.sf.read', side_effect=[OSError('I/O glitch'), audio, audio, audio]) as reader, \
                 patch('dataset.bundle_data.make_speaker_encoder',return_value=lambda x: torch.ones(1,256)):
                ds[0]
            self.assertEqual(reader.call_count,4)
