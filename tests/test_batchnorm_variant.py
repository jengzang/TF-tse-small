import unittest
import torch
from utils.feature import STFT, iSTFT


class BatchNormVariantTests(unittest.TestCase):
    def test_variant_replaces_groupnorm_and_trains_with_running_stats(self):
        from models.model_TFGridNet_k1_stage2_multi_v5_bn import Tar_Model
        torch.set_num_threads(1)
        model=Tar_Model(stft=STFT(512,320,512),istft=iSTFT(512,320,512),
                        feature_dim=257,embd_dim=256,fc_dim=16,emb_ks=1,
                        emb_hs=1,num_layers=2,lstm_hidden_units=32)
        self.assertEqual(sum(isinstance(m,torch.nn.GroupNorm) for m in model.modules()),0)
        self.assertIsInstance(model.conv[1],torch.nn.BatchNorm2d)
        self.assertIsInstance(model.fusion_conv[1],torch.nn.BatchNorm2d)
        self.assertEqual(sum(p.numel() for p in model.parameters()),298238)
        x=torch.randn(2,1,2,3200); e=torch.randn(2,1,256)
        model.train(); y=model(x,e); y.square().mean().backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertGreater(model.conv[1].num_batches_tracked.item(),0)
        model.eval()
        with torch.no_grad(): out=model(x,e)
        self.assertEqual(out.shape,y.shape)
        self.assertTrue(torch.isfinite(out).all())
