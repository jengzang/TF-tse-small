"""Independent TF-TSE adaptations inspired by S-LTSS, SEF-PNet and CIGAN.

Not paper reproductions: see docs/superpowers/specs/2026-11-13-paper-ablation-design.md.
All auxiliary heads are registered and included in the strict parameter budget.
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.model_tfgridnet_cases import Tar_Model as CaseModel

class ChannelNorm(nn.Module):
    def __init__(self,channels):
        super().__init__();self.norm=nn.LayerNorm(channels)
    def forward(self,x):return self.norm(x.movedim(1,-1)).movedim(-1,1)

class LocalGlobalContext(nn.Module):
    """Pointwise local + pooled global channel gates, identity initialized.

    Inspired by SEF-PNet LCA. LN and 2*sigmoid replace its BN and sigmoid.
    """
    def __init__(self,channels,mode='both'):
        super().__init__()
        if mode not in ('local','global','both'):raise ValueError('Invalid context mode')
        self.mode=mode;hidden=max(4,channels//4)
        for name in ('local','global'):
            if mode in (name,'both'):
                branch=nn.Sequential(nn.Conv2d(channels,hidden,1),ChannelNorm(hidden),nn.ReLU(),nn.Conv2d(hidden,channels,1))
                nn.init.zeros_(branch[-1].weight);nn.init.zeros_(branch[-1].bias)
                setattr(self,name,branch)
    def forward(self,x):
        logits=0
        if hasattr(self,'local'):logits=logits+self.local(x)
        if hasattr(self,'global'):logits=logits+getattr(self,'global')(x.mean((-2,-1),keepdim=True))
        return x*(2*torch.sigmoid(logits))

class HierarchicalTime(nn.Module):
    """Three small time LSTMs; hierarchical skip injection vs plain control."""
    def __init__(self,channels,hidden=24,hierarchical=True):
        super().__init__();self.hierarchical=hierarchical
        self.fc1=nn.Linear(channels,hidden);self.fc2=nn.Linear(hidden,hidden)
        self.layers=nn.ModuleList([nn.LSTM(hidden,hidden,batch_first=True) for _ in range(3)])
        self.out=nn.Linear(hidden,channels)
    def forward(self,x):
        a=F.gelu(self.fc1(x));b=F.gelu(self.fc2(a))
        if self.hierarchical:
            x=self.layers[0](a)[0];x=self.layers[1](x+b)[0];x=self.layers[2](x+a+b)[0]
        else:
            x=b
            for layer in self.layers:x=layer(x)[0]
        return self.out(x),None

class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):ctx.weight=weight;return x.view_as(x)
    @staticmethod
    def backward(ctx,grad):return -ctx.weight*grad,None

def reverse_gradient(x,weight):return _Reverse.apply(x,weight)

def spectrum(wave):
    with torch.autocast(device_type=wave.device.type,enabled=False):
        wave=wave.float();wave=wave/(wave.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-5))
        if wave.shape[-1]<512:wave=F.pad(wave,(0,512-wave.shape[-1]))
        return torch.stft(wave,512,320,512,window=torch.hann_window(512,device=wave.device),return_complex=True)

class RawContext(nn.Module):
    """Shared acoustic projection, raw enrollment context interaction and IFI.

    External speaker encoder is unused. This is a compact adaptation, not the
    original SEF-PNet speaker-encoder-free architecture.
    """
    def __init__(self,mode='context',width=32):
        super().__init__()
        if mode not in ('pool','context','iterative'):raise ValueError('Invalid raw mode')
        self.mode=mode
        self.shared=nn.Sequential(nn.Linear(514,width),nn.LayerNorm(width),nn.GELU())
        self.out=nn.Linear(width,256)
        if mode=='iterative':self.gates=nn.ModuleList([nn.Linear(width,width) for _ in range(2)])
    def features(self,x):
        z=spectrum(x[:,0]);scale=z.abs().clamp_min(1e-8).sqrt()
        z=z/z.abs().clamp_min(1e-8)*scale
        return self.shared(torch.cat((z.real,z.imag),dim=1).transpose(1,2))
    def forward(self,mixture,enrollment):
        if enrollment.ndim!=3 or enrollment.shape[:2]!=(mixture.shape[0],1):raise ValueError('Expected raw enrollment [B,1,N]')
        if not enrollment.isfinite().all() or (enrollment.square().mean(-1)<=1e-16).any():raise ValueError('Invalid/silent raw enrollment')
        e=self.features(enrollment);pooled=e.mean(1,keepdim=True)
        if self.mode=='pool':value=pooled
        else:
            q=self.features(mixture)
            attention=(q.float()@e.float().transpose(1,2)/q.shape[-1]**.5).softmax(-1).to(e.dtype)
            value=attention@e
            if self.mode=='iterative':
                original=value
                for gate in self.gates:
                    p=torch.sigmoid(gate(value+pooled));value=p*original+(1-p)*pooled
        return self.out(value.mean(1,keepdim=True))

class AcousticIdentity(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(257,32,1),nn.GELU(),nn.Conv1d(32,32,5,padding=2),nn.GELU())
        self.out=nn.Linear(64,256)
    def forward(self,wave):
        x=self.net(torch.log1p(spectrum(wave).abs()))
        return F.normalize(self.out(torch.cat((x.mean(-1),x.var(-1,unbiased=False).clamp_min(1e-8).sqrt()),-1)),dim=-1)

class ConditionalCritic(nn.Module):
    def __init__(self):
        super().__init__();self.audio=nn.Sequential(nn.Conv1d(257,16,1),nn.LeakyReLU(.2),nn.Conv1d(16,16,5,padding=2),nn.LeakyReLU(.2))
        self.condition=nn.Linear(256,16);self.out=nn.Conv1d(32,1,1)
    def forward(self,features,embedding):
        audio=self.audio(features);z=self.condition(embedding)[:,:,None].expand(-1,-1,audio.shape[-1])
        return self.out(torch.cat((audio,z),1))

class Tar_Model(CaseModel):
    def __init__(self,*args,context_mode='none',condition_mode='film',temporal_mode='standard',
                 raw_mode='none',identity_weight=0.,adversarial_weight=0.,**kwargs):
        if condition_mode not in ('film','post_cln'):raise ValueError('Invalid condition mode')
        kwargs['variant']='film' if condition_mode=='film' else 'gridnet'
        super().__init__(*args,**kwargs)
        self.condition_mode=condition_mode;self.identity_weight=identity_weight;self.adversarial_weight=adversarial_weight
        if context_mode!='none':
            for i,c in enumerate((20,20,20,40)):
                setattr(self,f'downsample{i}',nn.Sequential(getattr(self,f'downsample{i}'),LocalGlobalContext(c,context_mode)))
        if condition_mode=='post_cln':
            channels=self.case_arch['fc_dim'];self.cln_condition=nn.Linear(256,32)
            self.cln_norms=nn.ModuleList([nn.LayerNorm(channels) for _ in self.dual_mdl])
            self.cln_heads=nn.ModuleList([nn.Linear(32,2*channels) for _ in self.dual_mdl])
        if temporal_mode not in ('standard','plain','hierarchical'):raise ValueError('Invalid temporal mode')
        if temporal_mode!='standard':
            for block in self.dual_mdl:
                block.inter_rnn=HierarchicalTime(self.case_arch['fc_dim'],24,temporal_mode=='hierarchical');block.inter_linear=nn.Identity()
        if raw_mode!='none':self.raw_context=RawContext(raw_mode)
        if identity_weight:self.identity_head=AcousticIdentity()
        if adversarial_weight:self.critic=ConditionalCritic()
        if raw_mode!='none' and (identity_weight or adversarial_weight):raise ValueError('Identity teacher requires cached embeddings')
        self.case_arch['paper']=dict(context=context_mode,condition=condition_mode,temporal=temporal_mode,raw=raw_mode,
                                     identity_weight=identity_weight,adversarial_weight=adversarial_weight)
        if sum(p.numel() for p in self.parameters())>=1_000_000:raise ValueError('Full extraction model including auxiliary heads must be <1M')
    def forward(self,input,aux):
        if hasattr(self,'raw_context'):aux=self.raw_context(input,aux)
        return super().forward(input,aux)
    def run_backend(self,features,aux):
        if self.condition_mode=='film':return super().run_backend(features,aux)
        z=torch.tanh(self.cln_condition(aux[:,0]))
        for block,norm,head in zip(self.dual_mdl,self.cln_norms,self.cln_heads):
            features=block(features);x=features.movedim(1,-1);gamma,beta=head(z).chunk(2,-1)
            features=(norm(x)*(1+gamma[:,None,None,:])+beta[:,None,None,:]).movedim(-1,1)
        return features
    def auxiliary_training_loss(self,estimate,target,aux,lengths):
        loss=estimate.sum()*0
        if not (self.identity_weight or self.adversarial_weight):return loss
        mask=torch.arange(target.shape[-1],device=target.device)[None,:]<lengths[:,None]
        present=((target!=0)&mask).any(-1)
        if not present.any():
            for module in (getattr(self,'identity_head',None),getattr(self,'critic',None)):
                if module is not None:loss=loss+sum(p.sum()*0 for p in module.parameters())
            return loss
        x=(estimate*mask)[present];s=(target*mask)[present];z=F.normalize(aux[present,0].float(),dim=-1)
        with torch.autocast(device_type=estimate.device.type,enabled=False):
            if self.identity_weight:
                output=self.identity_head(x);clean=self.identity_head(s)
                loss=loss+self.identity_weight*((1-(output*z).sum(-1)).mean()+(1-(clean*z).sum(-1)).mean())
            if self.adversarial_weight:
                fake=torch.log1p(spectrum(x).abs());real=torch.log1p(spectrum(s).abs())
                fake_logits=self.critic(reverse_gradient(fake,self.adversarial_weight),z)
                real_logits=self.critic(real,z)
                loss=loss+.5*(F.binary_cross_entropy_with_logits(real_logits,torch.ones_like(real_logits))+F.binary_cross_entropy_with_logits(fake_logits,torch.zeros_like(fake_logits)))
        return loss
