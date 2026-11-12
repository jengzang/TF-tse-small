"""Read the AISHELL-2 bundle's ID-aligned mix, direct-path target and enrollment."""
from pathlib import Path
import random
import time
import logging

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

MARKER = '/data/.tse_aishell2_train_dev_whamr_strict_shards/'


def resolve_bundle_path(path, bundle_root=None):
    path = str(path)
    if bundle_root is not None and MARKER in path:
        return Path(bundle_root) / 'data/.tse_aishell2_train_dev_whamr_strict_shards' / path.split(MARKER, 1)[1]
    return Path(path)


def read_scp(path):
    result = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            key, value = line.strip().split(maxsplit=1)
            if key in result:
                raise ValueError(f'Duplicate ID {key} in {path}')
            result[key] = value
    if not result:
        raise ValueError(f'Empty SCP: {path}')
    return result


def make_speaker_encoder():
    from emb_extract.resnet34_yk.inference import ResNet34_yk
    return ResNet34_yk().embd_extract


class BundleDataset(Dataset):
    def __init__(self, mix_scp, ref_scp, aux_scp, bundle_root=None, duration=3.9,
                 sample_rate=16000, mono_policy='reject', training=False, ref_duration=2.9, max_items=None, embedding_cache=None,
                 read_attempts=3, retry_delay=0.5, return_valid_length=False):
        self.mix, self.ref, self.aux = map(read_scp, (mix_scp, ref_scp, aux_scp))
        if set(self.mix) != set(self.ref) or set(self.mix) != set(self.aux):
            raise ValueError('SCP IDs must match exactly between mix, ref and aux')
        self.ids = list(self.mix)
        if max_items is not None:
            if int(max_items) <= 0:
                raise ValueError('max_items must be positive')
            self.ids = self.ids[:int(max_items)]
        self.root = Path(bundle_root) if bundle_root else None
        self.length = int(duration * sample_rate)
        self.ref_length = int(ref_duration * sample_rate)
        if self.length <= 0 or self.ref_length <= 0:
            raise ValueError('Durations must be positive')
        if sample_rate != 16000:
            raise ValueError('The speaker encoder requires 16000 Hz')
        if mono_policy not in ('reject', 'duplicate', 'mono'):
            raise ValueError('mono_policy must be reject, duplicate or mono')
        self.sr, self.mono_policy, self.training = sample_rate, mono_policy, training
        if not isinstance(read_attempts, int) or read_attempts < 1 or retry_delay < 0:
            raise ValueError('read_attempts must be positive and retry_delay nonnegative')
        self.read_attempts, self.retry_delay = read_attempts, float(retry_delay)
        self.return_valid_length = return_valid_length
        self.encoder = None
        self.embeddings = None
        if embedding_cache:
            cache = torch.load(embedding_cache, map_location='cpu', weights_only=True)
            self.embeddings = cache['embeddings']
            missing = set(self.ids) - set(self.embeddings)
            if missing:
                raise ValueError(f'Embedding cache missing {len(missing)} IDs, e.g. {next(iter(missing))}')

    def __len__(self):
        return len(self.ids)

    def _path(self, path):
        return resolve_bundle_path(path, self.root)

    def _read(self, path, *, uid=None, kind=None):
        path = self._path(path)
        for attempt in range(1, self.read_attempts + 1):
            try:
                wav, sr = sf.read(path, dtype='float32', always_2d=True)
                break
            except (OSError, RuntimeError) as exc:
                detail = f'sample={uid!r} kind={kind!r} path={str(path)!r}: {type(exc).__name__}: {exc}'
                if attempt == self.read_attempts:
                    raise RuntimeError(f'Audio read failed after {attempt} attempt(s): {detail}') from exc
                logging.warning('Audio read failed (attempt %d/%d), retrying: %s',
                                attempt, self.read_attempts, detail)
                time.sleep(self.retry_delay * attempt)
        if sr != self.sr or not len(wav) or not np.isfinite(wav).all():
            raise ValueError(f'Invalid audio or sample rate in {path}: {sr}')
        if kind in ('mix', 'aux') and np.sqrt(np.mean(wav.astype(np.float64)**2)) <= 1e-8:
            raise ValueError(f'{uid}: silent {kind} audio: {path}')
        return wav.T

    def __getitem__(self, index):
        uid = self.ids[index]
        mix = self._read(self.mix[uid], uid=uid, kind='mix')
        ref = self._read(self.ref[uid], uid=uid, kind='ref')
        aux = self._read(self.aux[uid], uid=uid, kind='aux') if self.embeddings is None else None
        if mix.shape[0] == 1 and self.mono_policy == 'duplicate':
            mix = np.repeat(mix, 2, axis=0)
        expected_channels = 1 if self.mono_policy == 'mono' else 2
        if mix.shape[0] != expected_channels or ref.shape[0] != 1 or (aux is not None and aux.shape[0] != 1):
            raise ValueError(f'{uid}: expected configured mix channels and mono ref/aux; check mono_policy')
        if mix.shape[-1] != ref.shape[-1]:
            raise ValueError(f'{uid}: mix and ref lengths differ')
        n = mix.shape[-1]
        start = random.randint(0, max(0, n - self.length)) if self.training else 0
        mix, ref = mix[:, start:start+self.length], ref[0, start:start+self.length]
        valid = len(ref)
        mix = np.pad(mix, ((0, 0), (0, self.length-valid)))
        ref = np.pad(ref, (0, self.length-valid))
        mask = np.zeros(self.length, dtype=np.float32)
        mask[:valid] = 1
        if self.embeddings is None:
            aux = aux[0, -self.ref_length:]
            if self.encoder is None:
                self.encoder = make_speaker_encoder()
            emb = self.encoder(aux).detach().to(dtype=torch.float32, device='cpu').clone()
        else:
            emb = self.embeddings[uid]
            if emb.shape != (1, 256) or not torch.isfinite(emb).all():
                raise ValueError(f'{uid}: invalid cached embedding')
        result = ((torch.from_numpy(mix) if self.mono_policy == 'mono' else torch.from_numpy(mix).unsqueeze(0)), torch.from_numpy(ref), emb)
        if self.training:
            return (*result, torch.from_numpy(mask))
        return (*result, valid) if self.return_valid_length else result
