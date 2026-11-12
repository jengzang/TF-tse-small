"""Independent mono research cases; frontend and unified loss are controlled.

Only the time attention primitive is vendored. The adapter below is new code,
with GridNet's pre-normalization and exactly one outer residual retained.
"""
import torch
from torch import nn
from models.model_tfgridnet_mono import Tar_Model as MonoModel
from models.model_TFGridNet_k1_stage2_multi_v5 import GridNetV2Block
from models.vendor.fla.linear_attn import FocusedLinearAttention


class GatedFLATime(nn.Module):
    """Return an increment in the LSTM tuple interface, never a residual sum.

    Input has already passed GridNet inter_norm. This adapter uses a gate of
    that normalized input, no extra pre-norm, FFN, or retained temporal LSTM.
    Full-sequence attention and centered depthwise convolution see the future.
    """
    def __init__(self, channels, attention_dim, heads):
        super().__init__()
        if attention_dim <= 0 or heads <= 0 or attention_dim % heads:
            raise ValueError('attention_dim must be positive and divisible by heads')
        self.attention = FocusedLinearAttention(
            dim=attention_dim, input_dim=channels, num_heads=heads,
            kernel_size=7, focusing_factor=3, ropeEmb=None)
        self.gate = nn.Linear(channels, channels)

    def forward(self, x):
        # Focused power/norm and inverse normalization need FP32 under bf16.
        # Projection work also stays FP32 here; report this mixed-precision choice.
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.float()
            increment = torch.sigmoid(self.gate(value)) * self.attention(value)
        return increment.to(x.dtype), None


class FiLMGridNetBlock(GridNetV2Block):
    """ks=hs=1 GridNet with separate F/T FiLM after each existing LayerNorm."""
    def __init__(self, channels, hidden, condition_dim):
        super().__init__(channels, 1, 1, hidden)
        self.film_f = nn.Linear(condition_dim, channels * 2)
        self.film_t = nn.Linear(condition_dim, channels * 2)
        for head in (self.film_f, self.film_t):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def condition(x, z, head):
        gamma, beta = head(z).chunk(2, dim=-1)
        return x * (1 + gamma[:,None,None,:]) + beta[:,None,None,:]

    def forward(self, x, z):
        b,c,t,f = x.shape
        residual = x.permute(0,2,3,1).contiguous()
        y = self.condition(self.intra_norm(residual), z, self.film_f)
        y = self.intra_linear(self.intra_rnn(y.reshape(b*t,f,c))[0])
        y = y.reshape(b,t,f,c) + residual
        residual = y.transpose(1,2).contiguous()
        y = self.condition(self.inter_norm(residual), z, self.film_t)
        y = self.inter_linear(self.inter_rnn(y.reshape(b*f,t,c))[0])
        return (y.reshape(b,f,t,c) + residual).permute(0,3,2,1).contiguous()


class Tar_Model(MonoModel):
    def __init__(self, stft, istft, feature_dim=257, embd_dim=256, fc_dim=16,
                 emb_ks=1, emb_hs=1, num_layers=2, lstm_hidden_units=32,
                 time_layout='fixed', variant='gridnet', condition_dim=32,
                 fla_dim=64, fla_heads=2, time_bilstm_hidden=20, eps=1e-5):
        if time_layout != 'fixed' or emb_ks != 1 or emb_hs != 1:
            raise ValueError('Research cases require fixed time layout and ks=hs=1')
        if variant not in ('gridnet','mix_skip','film','film_bilstm','t_fla','t_bilstm'):
            raise ValueError('Unknown case variant')
        if feature_dim != 257 or embd_dim != 256 or min(fc_dim,num_layers,lstm_hidden_units) < 1:
            raise ValueError('Invalid case dimensions')
        super().__init__(stft,istft,feature_dim,embd_dim,fc_dim,emb_ks,emb_hs,
                         num_layers,lstm_hidden_units,eps, time_layout=time_layout)
        self.variant = variant
        self.case_arch = dict(variant=variant, fc_dim=fc_dim, layers=num_layers,
                              frequency_hidden=lstm_hidden_units)
        if variant == 'mix_skip':
            self.raw_projection = nn.Conv2d(2,fc_dim,1)
            nn.init.zeros_(self.raw_projection.weight)
            nn.init.zeros_(self.raw_projection.bias)
        elif variant in ('film','film_bilstm'):
            if condition_dim <= 0:
                raise ValueError('condition_dim must be positive')
            self.speaker_projection = nn.Sequential(nn.Linear(256,condition_dim),nn.Tanh())
            conditioned = nn.ModuleList([])
            for original in self.dual_mdl:
                block = FiLMGridNetBlock(fc_dim,lstm_hidden_units,condition_dim)
                state = block.state_dict()
                state.update(original.state_dict())
                block.load_state_dict(state, strict=True)
                conditioned.append(block)
            self.dual_mdl = conditioned
            self.case_arch['condition_dim'] = condition_dim
        elif variant == 't_fla':
            for block in self.dual_mdl:
                block.inter_rnn = GatedFLATime(fc_dim,fla_dim,fla_heads)
                block.inter_linear = nn.Identity()
            self.case_arch.update(fla_dim=fla_dim,fla_heads=fla_heads,
                                  fla_kernel=7,focusing_factor=3)
        elif variant == 't_bilstm':
            if time_bilstm_hidden <= 0:
                raise ValueError('time_bilstm_hidden must be positive')
            for block in self.dual_mdl:
                block.inter_rnn = nn.LSTM(fc_dim,time_bilstm_hidden,batch_first=True,
                                           bidirectional=True)
                block.inter_linear = nn.Linear(time_bilstm_hidden*2,fc_dim)
            self.case_arch['time_bilstm_hidden'] = time_bilstm_hidden
        if variant == 'film_bilstm':
            for block in self.dual_mdl:
                block.inter_rnn = nn.LSTM(fc_dim,time_bilstm_hidden,batch_first=True,bidirectional=True)
                block.inter_linear = nn.Linear(time_bilstm_hidden*2,fc_dim)
            self.case_arch['time_bilstm_hidden'] = time_bilstm_hidden
        if sum(p.numel() for p in self.parameters()) >= 1_000_000:
            raise ValueError('Full model exceeds strict <1,000,000 parameter budget')

    @property
    def checkpoint_contract(self):
        return {**super().checkpoint_contract,'case_arch':self.case_arch,'case_version':1}

    def fuse_backend_input(self, features, mixture_spectrum):
        if self.variant == 'mix_skip':
            raw = torch.cat((mixture_spectrum.real,mixture_spectrum.imag),dim=1)
            return features + self.raw_projection(raw)
        return features

    def run_backend(self, features, aux):
        if self.variant in ('film','film_bilstm'):
            condition = self.speaker_projection(aux[:,0,:])
            for block in self.dual_mdl:
                features = block(features,condition)
            return features
        return super().run_backend(features,aux)
