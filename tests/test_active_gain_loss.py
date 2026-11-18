import logging
import tempfile
import unittest
import torch
from utils.feature import STFT

class ActiveGainLossTests(unittest.TestCase):
    def test_log_rms_penalizes_attenuation_and_amplification(self):
        from utils.loss.losses import active_target_log_rms_loss
        y=torch.tensor([[.1,-.1,.1,-.1]])
        n=torch.tensor([4])
        self.assertAlmostEqual(active_target_log_rms_loss(y,y,n).item(),0.,places=6)
        a=active_target_log_rms_loss(.01*y,y,n)
        b=active_target_log_rms_loss(100*y,y,n)
        self.assertAlmostEqual(a.item(),b.item(),places=3)
        self.assertGreater(a.item(),4.5)

    def test_quiet_target_has_restoring_gradient_absence_and_padding_ignored(self):
        from utils.loss.losses import active_target_log_rms_loss
        y=torch.tensor([[.1,-.1,0.,0.],[0.,0.,0.,0.]])
        x=torch.tensor([[.001,-.001,9.,9.],[.3,.3,.3,.3]],requires_grad=True)
        loss=active_target_log_rms_loss(x,y,torch.tensor([2,4]));loss.backward()
        self.assertLess(x.grad[0,0].item(),0.)
        self.assertGreater(x.grad[0,1].item(),0.)
        self.assertTrue(torch.equal(x.grad[0,2:],torch.zeros(2)))
        self.assertTrue(torch.equal(x.grad[1],torch.zeros(4)))
        z=torch.ones(1,4,requires_grad=True)
        l=active_target_log_rms_loss(z,torch.zeros_like(z),torch.tensor([4]));l.backward()
        self.assertEqual(l.item(),0.)
        self.assertTrue(torch.equal(z.grad,torch.zeros_like(z)))

    def test_gain_penalty_participates_in_new_validation_selection(self):
        from trainer.trainer_dp_v1 import Trainer
        class Gain(torch.nn.Module):
            def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.01))
            def forward(self,x,e):return x[:,0,:]*self.scale
        torch.manual_seed(7)
        y=torch.randn(1,960)*.01; batch=(y[:,None],y,torch.ones(1,1,256),torch.tensor([960]))
        c=dict(device='cpu',precision='fp32',legacy_positive_clamp=False,padding_aware_loss=True,
            target_absent_loss=True,epochs=1,max_norm=5,checkpoint=0,continue_from='',print_freq=1,
            STFT512=STFT(512,256,512),STFT256=STFT(256,128,256),STFT128=STFT(128,64,128))
        with tempfile.TemporaryDirectory() as folder:
            m=Gain();opt=torch.optim.Adam(m.parameters())
            t=Trainer(folder,dict(tr_loader=[],cv_loader=[batch]),m,opt,None,logging.getLogger('gain_test'),c)
            old=t._run_valid_epoch(0)
            c['active_log_rms_weight']=1.
            t=Trainer(folder,dict(tr_loader=[],cv_loader=[batch]),m,opt,None,logging.getLogger('gain_test'),c)
            new=t._run_valid_epoch(0)
        self.assertGreater(new-old,4.5)
        self.assertAlmostEqual(t.valid_metrics['active_gain_db'],-40.,places=2)
