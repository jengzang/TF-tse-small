"""Unchanged original v5 network behind a mono-to-duplicate-dual I/O adapter.

No network modules or original forward operations are replaced. This preserves
its legacy temporal layout and normalization. Training uses the shared protocol.
"""
import hashlib
from pathlib import Path
import torch.nn.functional as F
from models.model_TFGridNet_k1_stage2_multi_v5 import Tar_Model as Original
SOURCE_SHA256=hashlib.sha256((Path(__file__).parent/'model_TFGridNet_k1_stage2_multi_v5.py').read_bytes()).hexdigest()

class Tar_Model(Original):
    @property
    def checkpoint_contract(self):
        return dict(input_channels=1,internal_channels=2,time_layout='original_legacy',adapter='duplicate_mono_only',original_source_sha256=SOURCE_SHA256)

    def validate_checkpoint(self,checkpoint):
        if checkpoint.get('model_contract')!=self.checkpoint_contract:
            raise ValueError('Original reference checkpoint contract mismatch')

    def forward(self,mixture,enrollment):
        if mixture.ndim!=3 or mixture.shape[1]!=1 or mixture.shape[-1]<1:
            raise ValueError('Expected mono [B,1,N]')
        n=mixture.shape[-1]
        padded=max(640,((n+319)//320)*320)
        dual=F.pad(mixture,(0,padded-n))[:,None].expand(-1,1,2,-1).contiguous()
        return super().forward(dual,enrollment)[:,:n]
