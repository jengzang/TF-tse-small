import unittest,tempfile
from pathlib import Path
import torch

class PaperAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_context_branches_identity_init_and_gradients(self):
        from models.model_paper_ablation import LocalGlobalContext
        for mode in ('local','global','both'):
            m=LocalGlobalContext(8,mode);x=torch.randn(2,8,4,5,requires_grad=True)
            y=m(x);torch.testing.assert_close(x,y)
            y.square().mean().backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters()))

    def test_grl_changes_only_input_gradient(self):
        from models.model_paper_ablation import reverse_gradient
        x=torch.tensor([2.],requires_grad=True);w=torch.tensor([3.],requires_grad=True)
        (reverse_gradient(x,.1)*w).sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([-.3]));torch.testing.assert_close(w.grad,torch.tensor([2.]))

    def test_raw_context_independent_reference_and_validation(self):
        from models.model_paper_ablation import RawContext
        for mode in ('pool','context','iterative'):
            m=RawContext(mode);x=torch.randn(2,1,960);e=torch.randn(2,1,960)
            y=m(x,e);self.assertEqual(y.shape,(2,1,256))
            self.assertFalse(torch.allclose(y,m(x,torch.randn_like(e))))
            y.square().mean().backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters()))
            with self.assertRaises(ValueError):m(x,torch.zeros_like(e))

    def test_hierarchical_and_plain_temporal_gradients(self):
        from models.model_paper_ablation import HierarchicalTime
        for hierarchical in (False,True):
            m=HierarchicalTime(8,12,hierarchical);y=m(torch.randn(2,9,8))[0]
            self.assertEqual(y.shape,(2,9,8));y.square().mean().backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters()))

    def test_raw_dataset_uses_enrollment_not_target(self):
        from test_corpus_mix import corpus
        from dataset.corpus_mix import CorpusMixDataset
        with tempfile.TemporaryDirectory() as t:
            p,c,_=corpus(Path(t));ds=CorpusMixDataset(p,c,enrollment_mode='waveform',epoch_size=2)
            (x,y,e,_),meta=ds.make_example(0)
            self.assertEqual(e.shape,(1,46400));self.assertNotEqual(meta['enrollment_path'],meta['target_source_path'])
            self.assertGreater(float(e.square().mean()),0)

    def test_empty_enrollment_rejected(self):
        from utils.enrollment import load_enrollment_waveform
        import soundfile as sf,numpy as np
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'empty.wav';sf.write(p,np.zeros(0),16000)
            with self.assertRaises(ValueError):load_enrollment_waveform(p)

    def test_queue_slots_and_oom_budget(self):
        from scripts.run_paper_queue import available_slots,reduce_batch
        self.assertEqual(available_slots({'0':2,'1':1,'2':0,'3':1},{'0':40000,'1':10000,'2':50000,'3':30000}),['2'])
        j=dict(microbatch=16,accumulation=4);self.assertTrue(reduce_batch(j));self.assertEqual(j['microbatch']*j['accumulation'],64)
