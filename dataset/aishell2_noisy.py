"""AISHELL-2-only dry mixtures, target/interferer 0..5 dB, target/noise 5..20 dB."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from dataset.corpus_mix import CorpusMixDataset
from utils.audio_cache import local_audio_path

class AIShell2NoisyDataset(CorpusMixDataset):
    def __init__(self,*args,noise_index,**kwargs):
        super().__init__(*args,corpora=('aishell2',),sir_range=(0.,5.),**kwargs)
        noise=json.loads(Path(noise_index).read_text())['splits']
        seen=set()
        for split,paths in noise.items():
            resolved={str(Path(p).resolve()) for p in paths}
            if seen & resolved:raise ValueError('noise file overlap between splits')
            seen.update(resolved)
            if not resolved:raise ValueError('empty noise split '+split)
        self.noise_paths=noise[self.split]

    def _noise(self,rng):
        for i in rng.permutation(len(self.noise_paths))[:32]:
            path=self.noise_paths[int(i)]
            try:
                actual=local_audio_path(path);info=sf.info(actual)
                if info.samplerate!=16000 or info.channels!=1 or info.frames<=0:continue
                start=int(rng.integers(max(1,info.frames-self.n+1)))
                x,sr=sf.read(actual,start=start,frames=self.n,dtype='float32')
                if not len(x) or not np.isfinite(x).all() or self._rms(x)<=1e-8:continue
                if len(x)<self.n:x=np.tile(x,int(np.ceil(self.n/len(x))))[:self.n]
                if self._rms(x)<=1e-8:continue
                return x,dict(path=path,crop_start=start)
            except (OSError,RuntimeError):continue
        raise RuntimeError('No usable non-silent 16k mono noise recording')

    def make_example(self,index):
        (mix,ref,emb,last),meta=super().make_example(index)
        rng=self._rng(meta['epoch'],meta['index']+10_000_000)
        noise,source=self._noise(rng)
        snr=float(rng.uniform(5,20)) if self.training else float([5,10,15,20][(meta['index']//6)%4])
        target_rms=meta['target_rms']*meta['peak_scale']
        gain=target_rms/self._rms(noise)*10**(-snr/20)
        noise=torch.from_numpy(noise*gain)
        mixed=mix+noise[None]
        scale=min(1.,.95/max(float(mixed.abs().max()),1e-8))
        meta.update(noise_source=source,noise_snr_db=snr,noise_gain=gain,
                    noise_reference='clean_target_rms_before_absence',reverberation=False,
                    peak_scale=meta['peak_scale']*scale,protocol='aishell2_sir0_5_noise5_20_v1')
        return (mixed*scale,ref*scale,emb,last),meta
