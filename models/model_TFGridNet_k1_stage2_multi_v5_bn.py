"""BatchNorm2d variant of the v5 target-speaker extraction model."""
import torch.nn as nn
from models.model_TFGridNet_k1_stage2_multi_v5 import Tar_Model as GroupNormTarModel


class Tar_Model(GroupNormTarModel):
    """Replace the two 4D GroupNorm layers; leave all other layers unchanged."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in (self.conv, self.fusion_conv):
            previous = block[1]
            if not isinstance(previous, nn.GroupNorm):
                raise TypeError('Expected GroupNorm at the v5 convolution output')
            block[1] = nn.BatchNorm2d(previous.num_channels, eps=previous.eps,
                                      affine=previous.affine)
