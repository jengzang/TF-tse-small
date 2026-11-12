"""Original speaker architecture/preprocessing with configurable weights."""
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from emb_extract.resnet34_yk.model_spk import ResNet34_based
from emb_extract.resnet34_yk.feat import logFbankCal

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / 'weights/model_19.pkl'


class ResNet34_yk:
    def __init__(self, checkpoint_path=None):
        path = Path(checkpoint_path or os.environ.get('TSE_SPEAKER_CHECKPOINT') or DEFAULT_CHECKPOINT)
        if not path.is_file():
            raise FileNotFoundError(f'Missing ResNet34_yk checkpoint: {path}. Set TSE_SPEAKER_CHECKPOINT or --speaker-checkpoint.')
        self.extractor = ResNet34_based(in_planes=64, block_type='base', pooling_layer='GSP', embd_dim=256, acoustic_dim=80)
        # Only load the trusted project speaker checkpoint.
        state = torch.load(path, map_location='cpu', weights_only=False)['model']
        self.extractor.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        self.extractor.eval()
        self.featCal = logFbankCal(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80)

    def embd_extract(self, wav, sr=16000):
        if sr != 16000:
            raise ValueError('Speaker enrollment must use 16000 Hz audio')
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1 or wav.size < 512 or not np.isfinite(wav).all():
            raise ValueError('Enrollment must be finite mono audio with at least 512 samples')
        std = np.std(wav)
        if std <= 1e-8:
            raise ValueError('Enrollment audio is silent or constant')
        signal = (wav - np.mean(wav)) / std
        with torch.inference_mode():
            feat = self.featCal(torch.from_numpy(signal).unsqueeze(0))
            return F.normalize(self.extractor(feat), dim=1)
