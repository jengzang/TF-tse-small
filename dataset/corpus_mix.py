"""Deterministic, dry, mono corpus-balanced target speech mixtures."""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler
from utils.audio_cache import local_audio_path

CORPORA = ('librispeech', 'aishell1', 'aishell2')


class CorpusMixDataset(Dataset):
    def __init__(self, index_path, embedding_cache, split='train', epoch_size=32000,
                 duration=4, seed=3407, training=True, enrollment_mode='embedding', ref_duration=2.9, corpora=None, sir_range=None):
        if enrollment_mode not in ('embedding','waveform'):raise ValueError('Invalid enrollment mode')
        self.corpora = tuple(corpora) if corpora is not None else CORPORA
        self.sir_range = sir_range
        if not self.corpora or set(self.corpora)-set(CORPORA):raise ValueError('Invalid corpus selection')
        self.enrollment_mode=enrollment_mode;self.ref_length=int(ref_duration*16000)
        raw_index = Path(index_path).read_bytes()
        index_hash = hashlib.sha256(raw_index).hexdigest()
        self.index = json.loads(raw_index)
        del raw_index
        if self.index.get('sample_rate') != 16000:
            raise ValueError('index sample_rate must be 16000')
        seen = set()
        for split_name, corpora in self.index['splits'].items():
            current = {f'{c}:{s}' for c, speakers in corpora.items() for s in speakers}
            if seen & current:
                raise ValueError('speaker overlap between splits: ' + str(seen & current))
            seen.update(current)
        self.split, self.seed, self.training = split, seed, training and split == 'train'
        self.epoch_size, self.n = int(epoch_size), int(16000 * duration)
        if self.n < 16000:
            raise ValueError('duration must be at least one second')
        cache = (torch.load(embedding_cache, map_location='cpu', weights_only=False)
                 if isinstance(embedding_cache, (str, Path)) else embedding_cache) if enrollment_mode=='embedding' else {'embeddings':{}}
        expected_hash = cache.get('metadata', {}).get('index_sha256')
        if expected_hash and expected_hash != index_hash:
            raise ValueError('embedding cache index hash does not match enrollment index')
        self.embeddings = cache['embeddings']
        self.enrollments = self.index['enrollments']
        self.speakers = self.index['splits'][split]
        self.keys = []
        self.sources = {}
        for c in self.corpora:
            if not self.speakers.get(c):
                raise ValueError(f'missing speakers for {split}/{c}')
            for s, paths in sorted(self.speakers[c].items()):
                key = f'{c}:{s}'
                enrollment = str(Path(self.enrollments[key]['path']))
                if not Path(self.enrollments[key]['path']).is_absolute():
                    raise ValueError('enrollment path must be absolute')
                usable = []
                for p in paths:
                    if not Path(p).is_absolute():
                        raise ValueError('source path must be absolute')
                    if str(Path(p)) != enrollment:
                        usable.append(p)
                if not usable:
                    raise ValueError(f'no independent source for {key}')
                emb = torch.as_tensor(self.embeddings[key]) if enrollment_mode=='embedding' else torch.ones(1,256)
                if emb.shape != (1,256) or not torch.isfinite(emb).all() or not torch.any(emb != 0):
                    raise ValueError(f'embedding shape for {key} must be finite, nonzero [1,256]')
                self.sources[key] = sorted(usable)
                self.keys.append(key)
        if len(self.keys) < 3:
            raise ValueError('at least three speakers required')

    def __len__(self):
        return self.epoch_size

    def _rng(self, epoch, index):
        payload = f'{self.seed}|{self.split}|{epoch}|{index}'.encode()
        return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little'))

    def _read(self, key, rng):
        paths = self.sources[key]
        choices = rng.choice(len(paths), size=min(8, len(paths)), replace=False)
        errors = []

        def reject(path, reason):
            errors.append(f'{path}: {reason}')
            logging.warning('Skipping unusable source audio %s: %s', path, reason)

        def probe(path):
            try:
                info = sf.info(local_audio_path(path))
                if info.samplerate != 16000 or info.channels != 1:
                    reject(path, 'expected 16k mono'); return None
                if info.frames <= 0:
                    reject(path, 'zero frames'); return None
                return info.frames
            except Exception as exc:
                reject(path, str(exc)); return None

        def read(path, frames):
            # A long recording can have a silent crop without being a bad file.
            for attempt in range(3 if frames > self.n else 1):
                start = int(rng.integers(max(1, frames-self.n+1)))
                try:
                    audio, sr = sf.read(local_audio_path(path), start=start,
                                        frames=self.n, dtype='float32')
                except Exception as exc:
                    reject(path, str(exc)); return None
                if sr != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
                    reject(path, 'invalid decoded samples'); return None
                if not len(audio) or self._rms(audio) <= 1e-8:
                    continue
                actual = len(audio)
                return np.pad(audio, (0,self.n-actual)), {'path':path,
                    'crop_start':start, 'original_frames':frames,
                    'zero_pad_samples':self.n-actual}
            reject(path, 'empty or silent sampled audio'); return None

        short = []; tried = set()
        for idx in choices:
            path = paths[int(idx)]; tried.add(path)
            frames = probe(path)
            if frames is None: continue
            if frames < self.n:
                short.append((path,frames)); continue
            result = read(path,frames)
            if result is not None: return result
        for path,frames in sorted(short,key=lambda item:-item[1]):
            result = read(path,frames)
            if result is not None: return result
        # Exceptional fallback: keep the registered identity if the first eight
        # candidates were unusable. Never substitute another speaker or zeros.
        for path in paths:
            if path in tried: continue
            frames = probe(path)
            if frames is None: continue
            result = read(path,frames)
            if result is not None: return result
        raise RuntimeError(f'no valid non-silent audio for {key}: '+ '; '.join(errors))

    @staticmethod
    def _rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64)**2)))

    def __getitem__(self, index):
        return self.make_example(index)[0]

    def make_example(self, index):
        epoch, index = index if isinstance(index, tuple) else (0,index)
        epoch, index = int(epoch), int(index)
        if not self.training:
            epoch = 0
        pair = self.sir_range is None and not self.training and index % 8 < 2
        base = index - index % 8 if pair else index
        rng = self._rng(epoch, base)
        corpus = self.corpora[base % len(self.corpora)]
        candidates = sorted(f'{corpus}:{s}' for s in self.speakers[corpus])
        a = candidates[int(rng.integers(len(candidates)))]
        others = [s for s in self.keys if s != a]
        b = others[int(rng.integers(len(others)))]
        scenario = ('overlap' if pair else rng.choice(
            ['overlap','target-only','target-pause','target-absent'], p=[.7,.1,.1,.1]))
        xa, ia = self._read(a, rng)
        xb, ib = self._read(b, rng)
        pauses = []
        if scenario == 'target-pause':
            length = min(self.n, int(rng.uniform(.5,1.)*16000))
            start = int(rng.integers(self.n-length+1))
            xa[start:start+length] = 0
            pauses = [[start,start+length]]
        sir = (float(rng.uniform(*self.sir_range)) if self.training else float(base % 6)) if self.sir_range is not None else (max(float(rng.normal(5,10)), -5) if self.training else float([-5,0,5,10,20][base%5]))
        ar, br = self._rms(xa), self._rms(xb)
        gain = ar / br * 10**(-sir/20) if br > 0 else 0.
        xb = xb * gain
        active = [a,b]
        if scenario == 'target-only':
            mix = xa.copy(); active = [a]
        elif scenario == 'target-absent':
            # Enrollment speaker a is excluded from every audible source.
            c = [k for k in self.keys if k not in (a,b)]
            c = c[int(rng.integers(len(c)))]
            xc, ic = self._read(c,rng)
            cr = self._rms(xc)
            xc *= ar/cr*10**(-sir/20) if cr > 0 else 0
            mix = xb + xc; active = [b,c]
            ia, ib = ib, ic
        else:
            mix = xa + xb
        peak = float(np.max(np.abs(mix)))
        scale = min(1., .999 / peak) if peak else 1.
        reverse = pair and index % 8 == 1
        target_key = b if reverse else a
        ref = xb if reverse else xa
        if scenario == 'target-absent':
            ref = np.zeros(self.n,dtype=np.float32)
        mix = torch.from_numpy((mix*scale).astype(np.float32)).unsqueeze(0)
        ref = torch.from_numpy((ref*scale).astype(np.float32))
        if self.enrollment_mode=='waveform':
            from utils.enrollment import load_enrollment_waveform
            emb=load_enrollment_waveform(self.enrollments[target_key]['path'],self.ref_length)
        else:
            emb = torch.as_tensor(self.embeddings[target_key],dtype=torch.float32).clone()
        metadata = {
            'sample_id': f'{self.split}:{epoch}:{index}', 'epoch':epoch, 'index':index,
            'corpus':target_key.split(':',1)[0], 'target_speaker':target_key,
            'interferer_speakers':[k for k in active if k != target_key],
            'enrollment_path':self.enrollments[target_key]['path'],
            'source_paths':([ia['path']] if scenario == 'target-only' else [ia['path'],ib['path']]),
            'sources':([ia] if scenario == 'target-only' else [ia,ib]),
            'active_speakers':active,
            'target_source_path':None if scenario == 'target-absent' else (ib['path'] if reverse else ia['path']),
            'sir_db':-sir if reverse else sir, 'base_sir_db':sir,
            'sir_distribution':('uniform_0_5' if self.training else 'fixed_0_5_grid') if self.sir_range is not None else ('clipped_normal_5_10' if self.training else 'fixed_grid'),
            'scenario':str(scenario), 'pair_id':f'{self.split}:{base}' if pair else None,
            'pause_intervals':pauses,
            'target_active_intervals_samples': ([] if scenario == 'target-absent' else
                ([interval for interval in [[0,pauses[0][0]],[pauses[0][1],self.n]] if interval[1]>interval[0]] if pauses else [[0,self.n]])),
            'target_rms':ar, 'interferer_rms':br,
            'interferer_gain':gain, 'peak_scale':scale, 'valid_length':self.n,
        }
        last = torch.ones(self.n) if self.training else self.n
        return (mix,ref,emb,last),metadata


class EpochIndexSampler(Sampler):
    """Carry epoch into worker requests, including persistent workers."""
    def __init__(self, data_source):
        self.data_source, self.epoch = data_source, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        return iter((self.epoch,i) for i in range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
