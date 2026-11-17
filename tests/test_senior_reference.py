import unittest,torch
from utils.feature import STFT,iSTFT
from models.model_TFGridNet_k1_stage2_multi_v5 import Tar_Model as Original
class SeniorTests(unittest.TestCase):
    def test_identical_state_and_output_to_untouched_original(self):
        from models.model_senior_reference import Tar_Model
        torch.set_num_threads(1)
        kwargs=dict(stft=STFT(512,320,512),istft=iSTFT(512,320,512),feature_dim=257,embd_dim=256,fc_dim=16,emb_ks=1,emb_hs=1,num_layers=2,lstm_hidden_units=32)
        original=Original(**kwargs).eval();reference=Tar_Model(**kwargs).eval()
        reference.load_state_dict(original.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in reference.parameters()),298238)
        self.assertEqual(set(original.state_dict()),set(reference.state_dict()))
        x=torch.randn(2,1,1600);e=torch.randn(2,1,256)
        with torch.no_grad():
            expected=original(x[:,None].expand(-1,1,2,-1).contiguous(),e)
            actual=reference(x,e)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        reference.train();out=reference(x,e);out.square().mean().backward()
        self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in reference.parameters()))
