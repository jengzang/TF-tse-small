"""True mono T0 variants. Original dual-microphone implementation stays unchanged.

Forward retains source behavior except explicit mono validation, length handling,
removal of a discarded iSTFT, and opt-in corrected bottleneck time layout.
"""
import torch
from torch import nn
import torch.nn.functional as functional
from models.model_TFGridNet_k1_stage2_multi_v5 import Tar_Model as DualModel

EPS = 1e-8


def restore_time_layout(x, batch, frequencies, frames, channels, layout):
    if layout == 'fixed':
        return x.reshape(batch, frequencies, frames, channels).permute(0,3,1,2).contiguous()
    if layout == 'legacy':
        return x.reshape(batch, frames, frequencies, channels).permute(0,3,2,1).contiguous()
    raise ValueError('time_layout must be legacy or fixed')


def map_dual_checkpoint(model, state):
    """Sum duplicated audio kernels; preserve condition kernel; validate all keys.

    This mapping is numerically equivalent only for identical dual input channels
    with the legacy time layout, not for physical two-microphone recordings.
    """
    normalized = {k.removeprefix('module.'): v for k, v in state.items()}
    if len(normalized) != len(state):
        raise ValueError('Duplicate checkpoint keys after module prefix removal')
    state = normalized
    target = model.state_dict()
    if set(state) != set(target):
        raise ValueError(f'Checkpoint key mismatch: missing={sorted(set(target)-set(state))}, extra={sorted(set(state)-set(target))}')
    key = 'downsample0.cconv.wav_conv.weight'
    result = dict(state)
    if tuple(state[key].shape) != (20,3,5,1):
        raise ValueError('Expected dual first convolution [20,3,5,1]')
    result[key] = torch.cat((state[key][:,:2].sum(dim=1,keepdim=True), state[key][:,2:3]),dim=1)
    for name, value in result.items():
        if value.shape != target[name].shape:
            raise ValueError(f'Checkpoint shape mismatch: {name}')
    return result, {'transformed':[key], 'missing':[], 'extra':[], 'mapping':'sum_audio_kernels_preserve_condition'}


class Tar_Model(DualModel):
    def __init__(self, *args, time_layout='legacy', **kwargs):
        if time_layout not in ('legacy','fixed'):
            raise ValueError('time_layout must be legacy or fixed')
        super().__init__(*args, **kwargs)
        self.time_layout = time_layout
        for transform in (self.stft, self.istft):
            if (transform.n_fft, transform.hop_length, transform.win_length) != (512,320,512):
                raise ValueError('T0 requires STFT/iSTFT 512/320/512')
        self.downsample0.cconv.wav_conv = nn.Conv2d(2,20,(5,1),stride=(2,1),padding=(2,0))
        nn.init.xavier_uniform_(self.downsample0.cconv.wav_conv.weight)

    @property
    def checkpoint_contract(self):
        return {'input_channels': 1, 'time_layout': self.time_layout}

    def validate_checkpoint(self, checkpoint):
        if checkpoint.get('model_contract') != self.checkpoint_contract:
            raise ValueError('Checkpoint model contract missing or mismatched; use matching layout or explicit dual migration')

    def fuse_backend_input(self, features, mixture_spectrum):
        """Extension point; T0 does not bypass the frontend."""
        return features

    def run_backend(self, features, aux):
        for block in self.dual_mdl:
            features = block(features)
        return features

    def forward(self, input, aux):
        if input.ndim != 3 or input.shape[1] != 1 or input.shape[-1] == 0:
            raise ValueError('Expected mono waveform [B,1,N] with N > 0')
        if aux.shape != (input.shape[0], 1, 256):
            raise ValueError('Expected embedding [B,1,256]')
        if not input.isfinite().all() or not aux.isfinite().all() or (aux.norm(dim=-1) <= EPS).any():
            raise ValueError('Waveform/embedding must be finite and embedding nonzero')
        original_length = input.shape[-1]
        padded_length = max(640, ((original_length + 319) // 320) * 320)
        input = functional.pad(input, (0, padded_length - original_length))
        std = input[:,0,:].unsqueeze(1).std(dim=(1, 2), keepdim=True)
        input = input / (std+EPS)
        input_c = self.stft(input)
        xs_mag = torch.abs(input_c).clamp_min(EPS).sqrt() # [B,1,F,T]; finite silent gradients
        xs_phase = torch.angle(input_c)
        emb_mag_ori = self.l1(aux)
        emb_mag_ori = emb_mag_ori.unsqueeze(-1).repeat(1,1,1,xs_mag.shape[-1])
        emb_mag = emb_mag_ori
        d0 = self.downsample0(torch.cat([xs_mag, emb_mag], dim = 1))#B,C,F,T
        emb_mag = self.l2(emb_mag.squeeze(1).permute(0,2,1).contiguous())#B,F,T
        emb_mag = emb_mag.permute(0,2,1).contiguous().unsqueeze(1)#B,1,F,T
        d1 = self.downsample1(torch.cat([d0, emb_mag], dim = 1))
        emb_mag = self.l3(emb_mag.squeeze(1).permute(0,2,1).contiguous())
        emb_mag = emb_mag.permute(0,2,1).contiguous().unsqueeze(1)
        d2 = self.downsample2(torch.cat([d1, emb_mag], dim = 1))
        emb_mag = self.l4(emb_mag.squeeze(1).permute(0,2,1).contiguous())
        emb_mag = emb_mag.permute(0,2,1).contiguous().unsqueeze(1)
        d3 = self.downsample3(torch.cat([d2, emb_mag], dim = 1))
        emb_mag = self.l5(emb_mag.squeeze(1).permute(0,2,1).contiguous())
        emb_mag = emb_mag.permute(0,2,1).contiguous().unsqueeze(1) #B,1,F,T
        d3_bottleneckF = torch.cat([d3,emb_mag],dim=1)
        B,C,F,T = d3_bottleneckF.shape
        d3_bottleneckF = d3_bottleneckF.permute(0,3,2,1).contiguous().view(B*T, F, C)
        d3_bottleneckF = self.fgru(d3_bottleneckF)[0]
        d3_bottleneckF = self.flinear(d3_bottleneckF)
        d3_bottleneckF = d3_bottleneckF.view(B,T,F,C-1).permute(0,3,2,1).contiguous() #B,C-1,F,T
        d3_bottleneckF = d3_bottleneckF + d3
        d3_bottleneckT = torch.cat([d3_bottleneckF,emb_mag],dim=1)
        d3_bottleneckT = d3_bottleneckT.permute(0,2,3,1).contiguous().view(B*F, T, C) ###
        d3_bottleneckT = self.tgru(d3_bottleneckT)[0]
        d3_bottleneckT = self.tlinear(d3_bottleneckT)
        d3_bottleneckT = restore_time_layout(d3_bottleneckT, B, F, T, C-1, self.time_layout)
        d3_bottleneck = d3_bottleneckT + d3_bottleneckF
        u0 = self.upsample0(d3_bottleneck)
        c0 = u0 + d2
        u1 = self.upsample1(c0)
        c1 = u1 + d1
        u2 = self.upsample2(c1)
        c2 = u2 + d0
        u3 = self.upsample3(c2)
        real_part = u3 * torch.cos(xs_phase[:,0,:,:].unsqueeze(1))
        imag_part = u3 * torch.sin(xs_phase[:,0,:,:].unsqueeze(1))
        xs = torch.cat([real_part, imag_part], dim = 1)
        xs = self.conv(xs)
        emb_ri= aux.unsqueeze(-1).transpose(1, 2).repeat(1,1,xs.shape[2],xs.shape[3])
        xs = torch.cat([xs, emb_ri], dim=1)
        xs = self.fusion_conv(xs)
        xs = self.fuse_backend_input(xs, input_c)
        xs = xs.transpose(2,3)
        xs = self.run_backend(xs, aux)
        outs = self.deconv(xs) # B,2,T,F
        outs_R = outs[:,0,:,:].permute(0,2,1).contiguous()
        outs_I = outs[:,1,:,:].permute(0,2,1).contiguous() #B,F,T

        out_put = self.istft((outs_R, outs_I), input_type="real_imag") # B,1,T
        out_put = out_put.unsqueeze(1) * std
        return out_put.squeeze(1)[..., :original_length]
