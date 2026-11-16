"""Identical raw enrollment cropping for online data, fixed bundles and inference."""
import numpy as np
import soundfile as sf
import torch
from utils.audio_cache import local_audio_path

def load_enrollment_waveform(path,length=46400):
    with sf.SoundFile(str(local_audio_path(path))) as stream:
        if stream.samplerate!=16000 or stream.channels!=1 or stream.frames==0:raise ValueError(f'Invalid enrollment: {path}')
        stream.seek(max(0,stream.frames-length));x=stream.read(length,dtype='float32',always_2d=True)[:,0]
    if not np.isfinite(x).all() or np.sqrt(np.mean(x.astype(np.float64)**2))<=1e-8:raise ValueError(f'Silent/nonfinite enrollment: {path}')
    return torch.from_numpy(np.pad(x,(0,max(0,length-len(x))))).unsqueeze(0)
