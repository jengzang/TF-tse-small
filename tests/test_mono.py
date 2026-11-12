import unittest
import tempfile
from pathlib import Path
import torch

class MonoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def build(self, layout='legacy'):
        from models.model_tfgridnet_mono import Tar_Model
        from utils.feature import STFT, iSTFT
        return Tar_Model(STFT(512,320,512),iSTFT(512,320,512),257,256,16,1,1,2,32,time_layout=layout)

    def test_mono_count_lengths_gradient_and_rejection(self):
        m=self.build().eval()
        self.assertEqual(sum(p.numel() for p in m.parameters()),298138)
        for b,n in ((1,1),(1,701),(2,1601)):
            x=torch.randn(b,1,n);e=torch.randn(b,1,256)
            y=m(x,e)
            self.assertEqual(y.shape,(b,n));self.assertTrue(y.isfinite().all())
        m(torch.randn(2,1,960),torch.randn(2,1,256)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters()))
        self.assertTrue(m(torch.zeros(1,1,701),torch.ones(1,1,256)).isfinite().all())
        for x,e in ((torch.zeros(1,1,2,960),torch.ones(1,1,256)),(torch.zeros(1,2,960),torch.ones(1,1,256)),(torch.zeros(1,1,960),torch.zeros(1,1,256))):
            with self.assertRaises(ValueError):m(x,e)

    def test_layout_restoration(self):
        from models.model_tfgridnet_mono import restore_time_layout
        x=torch.arange(6).reshape(2,3,1)
        y=restore_time_layout(x,1,2,3,1,'fixed')
        self.assertEqual(y[0,0].tolist(),[[0,1,2],[3,4,5]])
        self.assertFalse(torch.equal(y,restore_time_layout(x,1,2,3,1,'legacy')))

    def test_explicit_mapping_matches_duplicate_input(self):
        from infer import build_model
        from models.model_tfgridnet_mono import map_dual_checkpoint
        dual=build_model().eval();mono=self.build().eval()
        mapped,report=map_dual_checkpoint(mono,dual.state_dict())
        mono.load_state_dict(mapped,strict=True)
        self.assertEqual(report['transformed'],['downsample0.cconv.wav_conv.weight'])
        x=torch.randn(2,1,960);e=torch.randn(2,1,256)
        with torch.no_grad():
            expected=dual(x.repeat(1,2,1).unsqueeze(1),e)
            actual=mono(x,e)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
        bad=dict(dual.state_dict());bad.pop('l1.bias')
        with self.assertRaises(ValueError):map_dual_checkpoint(mono,bad)

    def test_mono_dataset_and_configs(self):
        import numpy as np
        import soundfile as sf
        from dataset.bundle_data import BundleDataset
        from hyperpyyaml import load_hyperpyyaml
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);sf.write(p/'a.wav',np.ones(701),16000)
            for k in ('mix','ref','aux'):(p/(k+'.scp')).write_text(f'u {p / "a.wav"}\n')
            torch.save({'embeddings':{'u':torch.ones(1,256)}},p/'e.pt')
            ds=BundleDataset(*(p/(k+'.scp') for k in ('mix','ref','aux')),mono_policy='mono',duration=.1,embedding_cache=p/'e.pt',return_valid_length=True)
            x,y,e,n=ds[0];self.assertEqual(x.shape,(1,1600));self.assertEqual(n,701)
        for name,layout,clamp in [('legacy','legacy',True),('fixed','fixed',True),('unified','fixed',False)]:
            with open(f'config/config_mono_{name}.yaml') as f:c=load_hyperpyyaml(f)
            self.assertEqual(c['MaskNet'].time_layout,layout)
            self.assertEqual(c['legacy_positive_clamp'],clamp)
            self.assertEqual(c['mono_policy'],'mono')

    def test_absent_loss_has_gradient_and_ignores_padding(self):
        from utils.loss.losses import presence_aware_sisnr_loss
        x=torch.ones(2,960,requires_grad=True);y=torch.zeros_like(x);y[0,:700]=torch.randn(700)
        lengths=torch.tensor([700,800])
        loss=presence_aware_sisnr_loss(x,y,lengths)
        self.assertTrue(loss.isfinite());loss.backward()
        self.assertTrue((x.grad[1,:800]>0).all());self.assertEqual(x.grad[1,800:].abs().sum(),0)

    def test_cli_and_train_validation_mono(self):
        import numpy as np
        import soundfile as sf
        import subprocess,sys,logging
        from hyperpyyaml import load_hyperpyyaml
        from trainer.trainer_dp_v1 import Trainer
        with open('config/config_mono_unified.yaml') as f:c=load_hyperpyyaml(f)
        c.update(device='cpu',epochs=1,checkpoint=1,print_freq=1)
        m=c['MaskNet'];x=torch.randn(2,1,1600)*.01;y=torch.randn(2,1600)*.01
        y[1]=0;e=torch.randn(2,1,256);mask=torch.ones(2,1600)
        opt=torch.optim.Adam(m.parameters());scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            t=Trainer(d,{'tr_loader':[(x,y,e,mask)],'cv_loader':[(x,y,e,torch.tensor([1600,1600]))]},m,opt,scheduler,logging.getLogger('mono'),c)
            t.train();self.assertTrue((p/'temp_best.pth.tar').exists())
            sf.write(p/'mix.wav',np.zeros(701),16000);np.save(p/'e.npy',np.ones(256,dtype='float32'))
            root=Path(__file__).resolve().parents[1]
            result=subprocess.run([sys.executable,str(root/'infer.py'),'--config',str(root/'config/config_mono_unified.yaml'),'--checkpoint',str(p/'temp_best.pth.tar'),'--mix',str(p/'mix.wav'),'--embedding',str(p/'e.npy'),'--output',str(p/'out.wav')],capture_output=True,text=True,cwd=d)
            self.assertEqual(result.returncode,0,result.stderr)
            a,sr=sf.read(p/'out.wav');self.assertEqual(a.shape,(701,));self.assertTrue(np.isfinite(a).all())

    def test_checkpoint_layout_contract(self):
        from infer import load_weights
        legacy=self.build();fixed=self.build('fixed')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.pt'
            torch.save({'model_state_dict':legacy.state_dict(),'model_contract':{'input_channels':1,'time_layout':'legacy'}},p)
            with self.assertRaisesRegex(ValueError,'contract'):load_weights(fixed,p)
            load_weights(legacy,p)

    def test_silent_training_gradient_finite(self):
        m=self.build('fixed')
        m(torch.zeros(2,1,960),torch.ones(2,1,256)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters()))

    def test_migration_cli_and_strict_shapes(self):
        import subprocess,sys
        from infer import build_model,load_weights
        from models.model_tfgridnet_mono import map_dual_checkpoint
        mono=self.build();dual=build_model()
        state=dual.state_dict();bad=dict(state);bad['l1.weight']=torch.zeros(1)
        with self.assertRaisesRegex(ValueError,'shape'):map_dual_checkpoint(mono,bad)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);torch.save({'model_state_dict':state},p/'dual.pt')
            root=Path(__file__).resolve().parents[1]
            r=subprocess.run([sys.executable,str(root/'scripts/migrate_dual_to_mono.py'),'--checkpoint',str(p/'dual.pt'),'--output',str(p/'mono.pt')],capture_output=True,text=True,cwd=d)
            self.assertEqual(r.returncode,0,r.stderr)
            load_weights(mono,p/'mono.pt')
            self.assertIn('sum_audio_kernels',r.stdout)

    def test_exact_zero_presence_and_source_snapshot(self):
        from utils.loss.losses import target_present
        self.assertEqual(target_present(torch.tensor([[1e-30,0.],[0.,1.]]),torch.tensor([2,1])).tolist(),[True,False])
        from train_dp_multi import save_source_snapshot
        with tempfile.TemporaryDirectory() as d:
            save_source_snapshot(Path(d))
            self.assertTrue((Path(d)/'source_snapshot/models/model_tfgridnet_mono.py').is_file())
            self.assertTrue((Path(d)/'source_snapshot/manifest.json').is_file())

    def test_training_entry_with_real_mono_bundle_and_warmstart(self):
        import numpy as np
        import soundfile as sf
        from hyperpyyaml import load_hyperpyyaml
        from train_dp_multi import main
        from types import SimpleNamespace
        import os
        root=Path(__file__).resolve().parents[1]
        config_path=root/'config/config_mono_unified.yaml'
        with open(config_path) as f:c=load_hyperpyyaml(f)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);sf.write(p/'a.wav',np.random.default_rng(1).normal(0,.02,800),16000,subtype='FLOAT')
            for k in ('mix','ref','aux'):(p/(k+'.scp')).write_text(f'u {p / "a.wav"}\n')
            torch.save({'embeddings':{'u':torch.ones(1,256)}},p/'e.pt')
            model=c['MaskNet']
            torch.save({'model_state_dict':model.state_dict(),'model_contract':model.checkpoint_contract},p/'initial.pt')
            c.update(device='cpu',epochs=1,batch_size=1,num_workers=0,duration=.05,print_freq=1,init_from=str(p/'initial.pt'))
            for split in ('train','dev'):
                for k in ('mix','ref','aux'):c[f'bundle_{split}_{k}']=str(p/(k+'.scp'))
                c[f'bundle_{split}_cache']=str(p/'e.pt')
            old=os.getcwd()
            try:
                os.chdir(d);main(c,SimpleNamespace(config=str(config_path)))
            finally:os.chdir(old)
            checkpoint=p/'chkpt'/c['name']
            self.assertTrue((checkpoint/'temp_best.pth.tar').exists())
            self.assertTrue((checkpoint/'source_snapshot/manifest.json').exists())
