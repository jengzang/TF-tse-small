"""Original mono-duplicated reference with only its two GroupNorms replaced."""
import torch.nn as nn
from models.model_senior_reference import Tar_Model as Reference

class Tar_Model(Reference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in (self.conv, self.fusion_conv):
            previous = block[1]
            if not isinstance(previous, nn.GroupNorm):
                raise TypeError('Expected original GroupNorm')
            block[1] = nn.BatchNorm2d(previous.num_channels, eps=previous.eps,
                                      affine=previous.affine)

    @property
    def checkpoint_contract(self):
        return {**super().checkpoint_contract, 'normalization': 'BatchNorm2d'}
