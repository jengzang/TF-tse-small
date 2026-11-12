import os

import librosa
import numpy as np
import torch
import torch.nn as nn

class STFT(nn.Module):
    def __init__(self, n_fft=256, hop_length=128, win_length=256):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    
    def forward(self, y):
        num_dims = y.dim()
        assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

        batch_size = y.shape[0]
        num_samples = y.shape[-1]

        if num_dims == 3:
            y = y.reshape(-1, num_samples)  # [B * C ,T]

        complex_stft = torch.stft(
            y,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.n_fft, device=y.device),
            return_complex=True,
        )
        _, num_freqs, num_frames = complex_stft.shape

        if num_dims == 3:
            complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)
        
        # print(complex_stft)

        # mag = torch.abs(complex_stft)
        # phase = torch.angle(complex_stft)
        # real = complex_stft.real
        # imag = complex_stft.imag
        return complex_stft
        # return mag, phase, real, imag, complex_stft


class iSTFT(nn.Module):
    def __init__(self, n_fft=256, hop_length=128, win_length=256, length=None):
        super(iSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = length
    
    def forward(self, features, input_type):
        if input_type == "real_imag":
        # the feature is (real, imag) or [real, imag]
            assert isinstance(features, tuple) or isinstance(features, list)
            real, imag = features
            if real.dtype == torch.bfloat16 or imag.dtype == torch.bfloat16:
                real, imag = real.float(), imag.float()
            features = torch.complex(real, imag)
        elif input_type == "complex":
            assert torch.is_complex(features), "The input feature is not complex."
        elif input_type == "mag_phase":
            # the feature is (mag, phase) or [mag, phase]
            assert isinstance(features, tuple) or isinstance(features, list)
            mag, phase = features
            if mag.dtype == torch.bfloat16 or phase.dtype == torch.bfloat16:
                mag, phase = mag.float(), phase.float()
            features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        else:
            raise NotImplementedError(
                "Only 'real_imag', 'complex', and 'mag_phase' are supported."
            )

        return torch.istft(
            features,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.n_fft, device=features.device),
            length=self.length,
        )


def stft(y, n_fft=512, hop_length=256, win_length=512):
    """Wrapper of the official torch.stft for single-channel and multi-channel.

    Args:
        y: single- or multi-channel speech with shape of [B, C, T] or [B, T]
        n_fft: number of FFT
        hop_length: hop length
        win_length: hanning window size

    Shapes:
        mag: [B, F, T] if dims of input is [B, T], whereas [B, C, F, T] if dims of input is [B, C, T]

    Returns:
        mag, phase, real and imag with the same shape of [B, F, T] (**complex-valued** STFT coefficients)
    """
    num_dims = y.dim()
    assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

    batch_size = y.shape[0]
    num_samples = y.shape[-1]

    if num_dims == 3:
        y = y.reshape(-1, num_samples)  # [B * C ,T]

    complex_stft = torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=y.device),
        return_complex=True,
    )
    _, num_freqs, num_frames = complex_stft.shape

    if num_dims == 3:
        complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)
    
    # print(complex_stft)

    mag = torch.abs(complex_stft)
    phase = torch.angle(complex_stft)
    real = complex_stft.real
    imag = complex_stft.imag
    return mag, phase, real, imag, complex_stft


def istft(features, n_fft=512, hop_length=256, win_length=512, length=None, input_type="mag_phase"):
    """Wrapper of the official torch.istft.

    Args:
        features: [B, F, T] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
        n_fft: num of FFT
        hop_length: hop length
        win_length: hanning window size
        length: expected length of istft
        use_mag_phase: use mag and phase as the input ("features")

    Returns:
        single-channel speech of shape [B, T]
    """
    if input_type == "real_imag":
        # the feature is (real, imag) or [real, imag]
        assert isinstance(features, tuple) or isinstance(features, list)
        real, imag = features
        features = torch.complex(real, imag)
    elif input_type == "complex":
        assert torch.is_complex(features), "The input feature is not complex."
    elif input_type == "mag_phase":
        # the feature is (mag, phase) or [mag, phase]
        assert isinstance(features, tuple) or isinstance(features, list)
        mag, phase = features
        features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    else:
        raise NotImplementedError(
            "Only 'real_imag', 'complex', and 'mag_phase' are supported."
        )

    return torch.istft(
        features,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=features.device),
        length=length,
    )