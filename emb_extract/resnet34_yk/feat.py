import torch, torch.nn as nn, random
from torchaudio import transforms

class logFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels, spec_mask_time=[5,10], spec_mask_freq=[5,10]):
        super(logFbankCal, self).__init__()
        self.fbankCal = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=n_fft,
                                                  win_length=win_length,
                                                  hop_length=hop_length,
                                                  n_mels=n_mels)
        self.spec_mask_time = spec_mask_time
        self.spec_mask_freq = spec_mask_freq
#         self.dropout_f = SharedDimScaleDropout(alpha=0.1,dim=0)
#         self.dropout_t = SharedDimScaleDropout(alpha=0.1,dim=1)
    def forward(self, x, is_spec_aug=[]):
        out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        return out
