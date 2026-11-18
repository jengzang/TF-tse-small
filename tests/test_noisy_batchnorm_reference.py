import unittest
import torch
from utils.feature import STFT,iSTFT

class NoisyBatchNormTests(unittest.TestCase):
    def test_only_norms_change_and_mono_adapter_matches_direct_batchnorm(self):
        from models.model_senior_batchnorm_reference import Tar_Model
        from models.model_TFGridNet_k1_stage2_multi_v5_bn import Tar_Model as Direct
        torch.set_num_threads(1)
        kw=dict(stft=STFT(512,320,512),istft=iSTFT(512,320,512),feature_dim=257,embd_dim=256,fc_dim=16,emb_ks=1,emb_hs=1,num_layers=2,lstm_hidden_units=32)
        m=Tar_Model(**kw);direct=Direct(**kw)
        self.assertEqual(sum(isinstance(x,torch.nn.GroupNorm) for x in m.modules()),0)
        self.assertEqual(sum(isinstance(x,torch.nn.BatchNorm2d) for x in m.modules()),
                         sum(isinstance(x,torch.nn.BatchNorm2d) for x in direct.modules()))
        self.assertIsInstance(m.conv[1],torch.nn.BatchNorm2d)
        self.assertIsInstance(m.fusion_conv[1],torch.nn.BatchNorm2d)
        self.assertEqual(sum(p.numel() for p in m.parameters()),298238)
        direct.load_state_dict(m.state_dict(),strict=True)
        m.eval();direct.eval();x=torch.randn(2,1,1600);e=torch.randn(2,1,256)
        with torch.no_grad():
            expected=direct(x[:,None].expand(-1,1,2,-1).contiguous(),e)
            actual=m(x,e)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(m.checkpoint_contract['normalization'],'BatchNorm2d')
        m.train();m(x,e).square().mean().backward()
        self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in m.parameters()))
