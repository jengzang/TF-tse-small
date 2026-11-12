import unittest
import torch
from utils.loss import losses

class NextCasesTests(unittest.TestCase):
    def test_scale_sensitive_loss_and_padding(self):
        fn = losses.normalized_waveform_loss
        s=torch.tensor([[1.,-1.,0.,0.],[0.,0.,0.,0.]])
        m=torch.tensor([[1.,-1.,0.,0.],[2.,-2.,0.,0.]])
        lengths=torch.tensor([2,2])
        x=s.clone().requires_grad_()
        self.assertEqual(fn(x,s,m,lengths).item(),0.)
        self.assertGreater(fn(s*.1,s,m,lengths).item(),.3)
        a=s.clone();a[:,2:]=999
        self.assertEqual(fn(a,s,m,lengths).item(),0.)
        x=(s+.1).requires_grad_();fn(x,s,m,lengths).backward()
        self.assertTrue(x.grad.isfinite().all())
        torch.testing.assert_close(fn(x*10,s*10,m*10,lengths),fn(x,s,m,lengths))

    def test_film_bilstm_gradients(self):
        from hyperpyyaml import load_hyperpyyaml
        from pathlib import Path
        torch.set_num_threads(1)
        text=(Path(__file__).resolve().parents[1]/'config/cases/case04_small_film.yaml').read_text()
        cfg=load_hyperpyyaml(text.replace('variant: film','variant: film_bilstm'))
        model=cfg['MaskNet'];self.assertLess(sum(p.numel() for p in model.parameters()),1000000)
        self.assertTrue(model.dual_mdl[0].inter_rnn.bidirectional)
        y=model(torch.randn(2,1,960),torch.randn(2,1,256));y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))

    def test_queue_waits_for_old_jobs(self):
        from scripts.run_next_cases import free_gpus
        self.assertEqual(free_gpus({'jobs':[{'status':'pending'}]}),[])
        self.assertEqual(free_gpus({'jobs':[{'status':'running','attempts':[{'gpu':'2'}]}]}),['0','1','3'])
        self.assertEqual(free_gpus({'jobs':[]}),['0','1','2','3'])

    def test_shared_slots_exclude_fla_and_require_headroom(self):
        from scripts.run_next_cases import shared_gpus
        old={'jobs':[{'case':'case05_wide_film','status':'running','attempts':[{'gpu':'2'}]},
                     {'case':'case07_wide_t_fla','status':'running','attempts':[{'gpu':'0'}]}]}
        self.assertEqual(shared_gpus(old,{'0':24000,'2':24000}),['2'])
        self.assertEqual(shared_gpus(old,{'2':10000}),[])
