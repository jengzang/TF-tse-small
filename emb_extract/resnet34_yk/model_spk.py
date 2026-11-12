import torch, torch.nn as nn, numpy as np
import sys
sys.path.append(".")
from emb_extract.resnet34_yk.front_resnet import ResNet34
import emb_extract.resnet34_yk.pooling as pooling_func
import torch.nn.functional as F   
from emb_extract.resnet34_yk.feat import logFbankCal

class ResNet34_based(nn.Module):
    def __init__(self, in_planes, block_type, pooling_layer, embd_dim, acoustic_dim, dropout=0,**kwargs):
        super(ResNet34_based, self).__init__()
        print('ResNet34 based model with %s and %s ' %(block_type, pooling_layer))
        self.front = ResNet34(in_planes, block_type)
        self.pooling = getattr(pooling_func, pooling_layer)(in_planes,acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x

if __name__ == '__main__':
    in_planes = 64
    block_type = 'base'
    pooling_layer = 'GSP'
    embd_dim = 256
    acoustic_dim = 80
    model = ResNet34_based(in_planes,block_type,pooling_layer,embd_dim,acoustic_dim)
    ckpt_dict = torch.load('model_19.pkl',map_location='cpu')['model']
    model.load_state_dict({k.replace('module.',''):v for k,v in ckpt_dict.items()})

    # signal, fs = librosa.load(path, sr=self.fs)
    # signal = (signal - np.mean(signal)) / np.std(signal)
    # featCal = logFbankCal(sample_rate=16000,
    #                       n_fft=512,
    #                       win_length=int(0.025*16000),
    #                       hop_length=int(0.01*16000),
    #                       n_mels=80)

    pass