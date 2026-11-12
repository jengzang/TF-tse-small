import unittest
import torch
from utils.feature import STFT
from utils.loss.losses import sisnr


class PaddingAwareTests(unittest.TestCase):
    def test_sisnr_matches_individual_unpadded_and_ignores_tail(self):
        from utils.loss.losses import masked_sisnr
        torch.manual_seed(2)
        target=torch.randn(2,1200); estimate=torch.randn(2,1200,requires_grad=True)
        lengths=torch.tensor([720,1000])
        scores=masked_sisnr(estimate,target,lengths)
        expected=torch.cat([sisnr(estimate[i:i+1,:n],target[i:i+1,:n]) for i,n in enumerate(lengths.tolist())])
        self.assertTrue(torch.allclose(scores,expected,atol=1e-5))
        scores.mean().backward()
        self.assertTrue(torch.equal(estimate.grad[0,720:],torch.zeros(480)))
        self.assertTrue(torch.any(estimate.grad[0,:720][estimate.detach()[0,:720]<0] != 0))

    def test_spectral_loss_ignores_padded_values(self):
        from utils.loss.losses import masked_multires_spectral_loss
        torch.manual_seed(3); lengths=torch.tensor([720,1000])
        target=torch.randn(2,1200);estimate=torch.randn(2,1200,requires_grad=True)
        stfts=[STFT(128,64,128), STFT(256,128,256)]
        a=masked_multires_spectral_loss(estimate,target,lengths,stfts)
        changed=estimate.detach().clone();changed[0,720:]=1000;changed[1,1000:]=-1000
        b=masked_multires_spectral_loss(changed,target,lengths,stfts)
        self.assertTrue(torch.allclose(a,b,atol=1e-6))
        a.backward();self.assertTrue(torch.equal(estimate.grad[0,720:],torch.zeros(480)))

    def test_validation_dataset_returns_valid_length(self):
        from dataset.bundle_data import BundleDataset
        import tempfile, pathlib, soundfile as sf, numpy as np
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);wav=root/'audio.wav';sf.write(wav,np.ones(700,dtype='float32'),16000)
            for key in ('mix','ref','aux'): (root/(key+'.scp')).write_text(f'item {wav}\n')
            torch.save({'embeddings': {'item':torch.ones(1,256)}},root/'cache.pt')
            ds=BundleDataset(*(str(root/(key+'.scp')) for key in ('mix','ref','aux')),
                duration=0.1,mono_policy='duplicate',embedding_cache=str(root/'cache.pt'),return_valid_length=True)
            mix, ref, emb, valid=ds[0]
            self.assertEqual(valid,700);self.assertEqual(ref.numel(),1600)

    def test_validation_scores_valid_length_and_input_baseline(self):
        import logging, tempfile
        from trainer.trainer_dp_v1 import Trainer
        class Identity(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.gain=torch.nn.Parameter(torch.ones(()))
            def forward(self, x, emb):
                self.last_input_length=x.shape[-1]
                return x[:,0,0,:]*self.gain
        model=Identity(); optimizer=torch.optim.Adam(model.parameters())
        c=dict(device='cpu',precision='fp32',legacy_positive_clamp=False,padding_aware_loss=True,
               epochs=1,max_norm=5,checkpoint=0,continue_from='',print_freq=1,
               STFT512=STFT(512,256,512),STFT256=STFT(256,128,256),STFT128=STFT(128,64,128))
        torch.manual_seed(5); target=torch.randn(1,1600); inp=target+0.1*torch.randn_like(target); mixture=inp[:,None,None,:].repeat(1,1,2,1)
        mixture[...,700:]=0; target[:,700:]=0
        batch=(mixture,target,torch.ones(1,1,256),torch.tensor([700]))
        with tempfile.TemporaryDirectory() as d:
            trainer=Trainer(d,{'tr_loader':[],'cv_loader':[batch]},model,optimizer,None,logging.getLogger('test_padding_metrics'),c)
            with self.assertLogs('test_padding_metrics',level='INFO') as logs:
                loss=trainer._run_valid_epoch(0)
        self.assertEqual(model.last_input_length,960)
        self.assertAlmostEqual(loss,-sisnr(inp[:,:700],target[:,:700]).item(),places=4)
        self.assertTrue(any('SI-SNRi 0.000 dB' in line for line in logs.output))

    def test_corrected_train_and_validation_save_checkpoint(self):
        import logging, tempfile
        from hyperpyyaml import load_hyperpyyaml
        from trainer.trainer_dp_v1 import Trainer
        with open('config/config_batchnorm2d_corrected_20261104.yaml') as f: c=load_hyperpyyaml(f)
        c.update(device='cpu',precision='fp32',epochs=1,print_freq=1,checkpoint=1,continue_from='')
        model=c['modules']['masknet'];torch.set_num_threads(1)
        torch.manual_seed(4); x=torch.randn(2,1,2,3200)*.01; y=torch.randn(2,3200)*.01; e=torch.randn(2,1,256)
        mask=torch.ones(2,3200);mask[0,2400:]=0;x[0,:,:,2400:]=0;y[0,2400:]=0
        tr=(x,y,e,mask);val=(x[:1],y[:1],e[:1],torch.tensor([2400]))
        optimizer=c['optimizer'](params=model.parameters());scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        with tempfile.TemporaryDirectory() as d:
            trainer=Trainer(d,{'tr_loader':[tr],'cv_loader':[val]},model,optimizer,scheduler,logging.getLogger('test_corrected'),c)
            with self.assertLogs('test_corrected',level='INFO') as logs:trainer.train()
            self.assertTrue((__import__('pathlib').Path(d)/'temp_best.pth.tar').is_file())
            self.assertTrue(any('SI-SNRi' in line for line in logs.output))
