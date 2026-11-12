"""Behavioral checks for independent server experiment cases."""
import tempfile
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

class CaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def configs(self):
        from utils.case_registry import CASES
        from hyperpyyaml import load_hyperpyyaml
        result = {}
        for case, path in CASES.items():
            with open(ROOT / path) as f:
                result[case] = load_hyperpyyaml(f)
        return result

    def test_all_cases_shapes_gradients_and_budget(self):
        configs = self.configs()
        self.assertEqual(len(configs), 10)
        for case, config in configs.items():
            with self.subTest(case=case):
                model = config['MaskNet']
                n = sum(p.numel() for p in model.parameters())
                self.assertLess(n, 1_000_000)
                anchor = 812794 if 'wide' in case else 298138
                self.assertLess(abs(n-anchor)/anchor, .1)
                y = model(torch.randn(2,1,701)*.01, torch.randn(2,1,256))
                self.assertEqual(y.shape, (2,701))
                y.square().mean().backward()
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
                self.assertEqual(config['sample_rate'],16000)
                self.assertTrue(config['padding_aware_loss'])
                self.assertFalse(config['legacy_positive_clamp'])
                self.assertEqual(model.time_layout,'fixed')

    def test_case_baseline_matches_t0_and_only_one_change(self):
        from infer import build_model
        configs=self.configs()
        base=build_model(ROOT/'config/config_mono_unified.yaml').eval()
        case=configs['case00_small_gridnet']['MaskNet'].eval()
        case.load_state_dict(base.state_dict(),strict=True)
        x=torch.randn(1,1,960);e=torch.randn(1,1,256)
        with torch.no_grad():torch.testing.assert_close(case(x,e),base(x,e),rtol=0,atol=0)
        for name in ('case02_small_mix_skip','case04_small_film'):
            model=configs[name]['MaskNet'].eval()
            state=model.state_dict();state.update(base.state_dict());model.load_state_dict(state,strict=True)
            with torch.no_grad():torch.testing.assert_close(model(x,e),base(x,e),rtol=0,atol=0)

    def test_raw_bypass_uses_real_original_spectrum(self):
        model=self.configs()['case02_small_mix_skip']['MaskNet']
        x=torch.zeros(1,16,257,4)
        raw=torch.complex(torch.ones(1,1,257,4),torch.full((1,1,257,4),2.))
        with torch.no_grad():model.raw_projection.weight.fill_(1);model.raw_projection.bias.zero_()
        torch.testing.assert_close(model.fuse_backend_input(x,raw),torch.full_like(x,3.))

    def test_film_is_after_norm_and_identity_initialized(self):
        model=self.configs()['case04_small_film']['MaskNet']
        block=model.dual_mdl[0]
        x=torch.randn(1,16,3,5);z=torch.randn(1,32)
        with torch.no_grad():
            initial=block(x,z)
            block.film_f.bias[16:]=1
            changed=block(x,z)
        self.assertFalse(torch.allclose(initial,changed))
        self.assertEqual(block.film_f.out_features,32)

    def test_fla_single_residual_and_future_context(self):
        from models.model_tfgridnet_cases import GatedFLATime
        layer=GatedFLATime(16,64,2).eval()
        x=torch.randn(1,9,16)
        with torch.no_grad():
            y=layer(x)[0];modified=x.clone();modified[:,5:]+=torch.randn_like(modified[:,5:])*4
            after=layer(modified)[0]
            self.assertFalse(torch.allclose(y[:,:2],after[:,:2]))
            for p in layer.parameters():p.zero_()
            torch.testing.assert_close(layer(x)[0],torch.zeros_like(x),rtol=0,atol=0)

    def test_contract_prevents_same_shape_different_behavior(self):
        from infer import load_weights
        configs=self.configs()
        a=configs['case00_small_gridnet']['MaskNet'];b=configs['case08_small_t_bilstm']['MaskNet']
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.pt';torch.save({'model_state_dict':a.state_dict(),'model_contract':a.checkpoint_contract},p)
            load_weights(a,p)
            with self.assertRaises(ValueError):load_weights(b,p)

    def test_step_budget_and_server_roundtrip(self):
        import subprocess,sys,json
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);rng=np.random.default_rng(3)
            audio_dir=p/'data/.tse_aishell2_train_dev_whamr_strict_shards/eval'
            audio_dir.mkdir(parents=True)
            sf.write(audio_dir/'mix.wav',rng.normal(0,.02,701),16000,subtype='FLOAT')
            reference=rng.normal(0,.02,701);reference[600:]=0
            sf.write(audio_dir/'ref.wav',reference,16000,subtype='FLOAT')
            sf.write(audio_dir/'zero.wav',np.zeros(701),16000,subtype='FLOAT')
            for key in ('mix','ref','aux'):
                entries=[]
                for uid in ('active','absent'):
                    name='mix.wav' if key!='ref' else ('ref.wav' if uid=='active' else 'zero.wav')
                    entries.append(f'{uid} /old/root/data/.tse_aishell2_train_dev_whamr_strict_shards/eval/{name}\n')
                (p/(key+'.scp')).write_text(''.join(entries))
            torch.save({'embeddings':{uid:torch.ones(1,256) for uid in ('active','absent')}},p/'emb.pt')
            import yaml
            overrides={'bundle_root':str(p)}
            for split in ('train','dev'):
                for key in ('mix','ref','aux'):overrides[f'bundle_{split}_{key}']=str(p/(key+'.scp'))
                overrides[f'bundle_{split}_cache']=str(p/'emb.pt')
            (p/'data.yaml').write_text(yaml.safe_dump(overrides))
            command=[sys.executable,str(ROOT/'scripts/run_case.py'),'--case','case06','--data-config',str(p/'data.yaml'),'--device','cpu','--epochs','3','--max-steps','1','--batch-size','1','--num-workers','0','--duration','.05','--threads','1']
            result=subprocess.run(command,cwd=d,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr[-4000:])
            folder=p/'chkpt/cases/case06_small_t_fla/seed3407'
            ckpt=folder/'temp_best.pth.tar'
            state=torch.load(ckpt,weights_only=False)
            self.assertEqual(state['global_step'],1)
            self.assertFalse((folder/'epoch2.pth.tar').exists())
            self.assertTrue((folder/'source_snapshot/models/vendor/fla/LICENSE').exists())
            (p/'metadata.json').write_text(json.dumps({'active':{'scenario':'target_pause','sir_db':-5,'target_active_intervals_samples':[[0,600]]},'absent':{'scenario':'absent'}}))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_case.py'),'--config',str(folder/'config.yaml'),'--checkpoint',str(ckpt),'--mix-scp',str(p/'mix.scp'),'--ref-scp',str(p/'ref.scp'),'--aux-scp',str(p/'aux.scp'),'--embedding-cache',str(p/'emb.pt'),'--metadata',str(p/'metadata.json'),'--output',str(p/'metrics.json'),'--device','cpu'],cwd=d,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr[-4000:])
            metrics=json.loads((p/'metrics.json').read_text())
            self.assertEqual(metrics['summary']['active_count'],1)
            self.assertEqual(metrics['summary']['absent_count'],1)
            self.assertIn('gain_db',metrics['items'][0])
            self.assertIn('pause_output_rms',metrics['items'][0])
            self.assertNotIn('output_sisnr_db',metrics['items'][1])
            self.assertEqual([x['samples'] for x in metrics['items']],[701,701])
            original_config=(folder/'config.yaml').read_bytes()
            original_snapshot=(folder/'source_snapshot/manifest.json').read_bytes()
            wrong=dict(state);wrong['model_contract']={'input_channels':2}
            torch.save(wrong,p/'wrong.pt')
            resume_command=list(command)
            resume_command[resume_command.index('--max-steps')+1]='2'
            result=subprocess.run(resume_command+['--resume',str(p/'wrong.pt')],cwd=d,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((folder/'config.yaml').read_bytes(),original_config)
            self.assertEqual((folder/'source_snapshot/manifest.json').read_bytes(),original_snapshot)
            self.assertFalse((folder/'resumes').exists())
            result=subprocess.run(resume_command+['--resume',str(ckpt)],cwd=d,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr[-4000:])
            self.assertEqual((folder/'config.yaml').read_bytes(),original_config)
            self.assertEqual((folder/'source_snapshot/manifest.json').read_bytes(),original_snapshot)
            self.assertEqual(len(list((folder/'resumes').glob('*/config.yaml'))),1)
            self.assertEqual(torch.load(folder/'epoch2.pth.tar',weights_only=False)['global_step'],2)


    def test_inspector_cli_and_shared_override_guard(self):
        import subprocess,sys,json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/inspect_cases.py'),'--case','case00','--seconds','.05','--repeats','1','--output',str(p/'stats.json')],cwd=d,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            rows=json.loads((p/'stats.json').read_text())['cases']
            self.assertEqual(rows[0]['total_parameters'],298138)
            self.assertGreater(rows[0]['estimated_matmul_conv_macs'],0)
            (p/'bad.yaml').write_text('sample_rate: 8000\n')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_case.py'),'--case','case00','--data-config',str(p/'bad.yaml'),'--device','cpu'],cwd=d,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Unsupported shared override',result.stderr)

    def test_same_seed_keeps_training_order_across_architectures(self):
        from unittest.mock import patch
        from scripts.run_case import main
        import numpy as np
        import random
        captured=[]
        def capture(config,args):
            captured.append((torch.randperm(30).tolist(),np.random.rand(3).tolist(),random.random()))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.yaml';p.write_text('{}\n')
            with patch('scripts.run_case.train_main',side_effect=capture):
                for case in ('case00','case04','case06'):
                    main(['--case',case,'--data-config',str(p),'--device','cpu'])
        self.assertEqual(captured[0],captured[1]);self.assertEqual(captured[0],captured[2])

    def test_film_initialization_preserves_shared_backbone(self):
        from infer import build_model
        torch.manual_seed(17);a=build_model(ROOT/'config/cases/case00_small_gridnet.yaml')
        torch.manual_seed(17);b=build_model(ROOT/'config/cases/case04_small_film.yaml')
        for key,value in a.state_dict().items():
            torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)

    def test_all_cases_bf16_and_silence_are_finite(self):
        for name,config in self.configs().items():
            with self.subTest(case=name):
                model=config['MaskNet']
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    output=model(torch.randn(1,1,701)*.01,torch.randn(1,1,256))
                output.float().square().mean().backward()
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
                model.zero_grad(set_to_none=True)
                model(torch.zeros(1,1,701),torch.randn(1,1,256)).square().mean().backward()
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))

    def test_resume_restores_best_and_scheduler(self):
        import logging
        from trainer.trainer_dp_v1 import Trainer
        c=self.configs()['case00_small_gridnet'];c.update(device='cpu',epochs=1,max_steps=1)
        x=torch.randn(1,1,701)*.01;e=torch.randn(1,1,256);y=torch.randn(1,701)*.01
        tr=(x,y,e,torch.ones(1,701));cv=(x,y,e,torch.tensor([701]))
        model=c['MaskNet'];opt=torch.optim.Adam(model.parameters());scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=2)
        with tempfile.TemporaryDirectory() as d:
            t=Trainer(d,{'tr_loader':[tr],'cv_loader':[cv]},model,opt,scheduler,logging.getLogger('resume_case'),c)
            t.train()
            ckpt=Path(d)/'temp_best.pth.tar'
            c['continue_from']=str(ckpt);c['max_steps']=2
            other=self.configs()['case00_small_gridnet']['MaskNet'];o=torch.optim.Adam(other.parameters());s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,patience=2)
            resumed=Trainer(d,{'tr_loader':[tr],'cv_loader':[cv]},other,o,s,logging.getLogger('resume_case'),c)
            self.assertEqual(resumed.best_val_loss,t.best_val_loss)
            self.assertEqual(s.state_dict(),scheduler.state_dict())
            self.assertEqual(resumed.global_step,1)
