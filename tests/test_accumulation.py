import unittest
from trainer.trainer_dp_v1 import accumulation_group_size

class AccumulationTests(unittest.TestCase):
    def test_partial_final_group(self):
        self.assertEqual([accumulation_group_size(i, 5, 2) for i in range(5)], [2,2,2,2,1])
    def test_full_batch(self):
        self.assertEqual(accumulation_group_size(0, 8, 1), 1)

    def test_updates_match_large_batch_including_partial_group(self):
        import torch
        from unittest.mock import patch
        from trainer.trainer_dp_v1 import Trainer
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(0.7))
            def forward(self,x,emb): return x[:,0]*self.weight
        class Logger:
            def info(self,*args): pass
        def run(batch, accumulation):
            model=Model()
            trainer=Trainer.__new__(Trainer)
            trainer.model=model; trainer.device=torch.device('cpu'); trainer.precision='fp32'
            trainer.optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
            x=torch.arange(1.,6.).reshape(5,1,1).repeat(1,1,8)
            dataset=torch.utils.data.TensorDataset(x, x[:,0]*2, torch.ones(5,1,256),torch.ones(5,8))
            trainer.tr_loader=torch.utils.data.DataLoader(dataset,batch_size=batch)
            trainer.accumulation_steps=accumulation;trainer.max_steps=None;trainer.global_step=0
            trainer.padding_aware_loss=True;trainer.target_absent_loss=True
            trainer.max_norm=10000;trainer.print_freq=100;trainer.logger=Logger()
            trainer.stft512=trainer.stft256=trainer.stft128=None
            with patch('trainer.trainer_dp_v1.presence_aware_sisnr_loss', side_effect=lambda a,b,l: (a-b).square().mean()), patch('trainer.trainer_dp_v1.masked_multires_spectral_loss',return_value=0.):
                trainer._run_train_epoch(0)
            return model.weight.detach(),trainer.global_step
        a, steps_a=run(1,2);b,steps_b=run(2,1)
        self.assertTrue(torch.allclose(a,b,atol=1e-6));self.assertEqual((steps_a,steps_b),(3,3))
