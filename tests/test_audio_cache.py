import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from utils.audio_cache import local_audio_path

class AudioCacheTests(unittest.TestCase):
    def test_opt_in_exact_copy_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.wav';source.write_bytes(b'waveform')
            with patch.dict(os.environ,{'TFTSE_AUDIO_CACHE':str(Path(folder)/'cache')}):
                cached=local_audio_path(str(source))
                self.assertEqual(Path(cached).read_bytes(),source.read_bytes())
                with patch('utils.audio_cache.shutil.copyfile',side_effect=AssertionError('must reuse')):
                    self.assertEqual(local_audio_path(str(source)),cached)
    def test_disabled_returns_original(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(local_audio_path('/read/only.wav'),'/read/only.wav')
