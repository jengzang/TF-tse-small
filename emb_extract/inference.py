"""Baseline speaker extractor; unrelated extractor families were removed."""
from emb_extract.resnet34_yk.inference import ResNet34_yk


class sv_model_select:
    def __init__(self, sv_model):
        if sv_model != 'ResNet34_yk':
            raise ValueError('This baseline supports only ResNet34_yk')
        self.sv_model = ResNet34_yk()
