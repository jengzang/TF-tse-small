import json,tempfile,unittest
from pathlib import Path
import numpy as np,soundfile as sf,torch
from test_corpus_mix import corpus

class NoisyTests(unittest.TestCase):
    def test_protocol_and_noise_calibration(self):
        from dataset.aishell2_noisy import AIShell2NoisyDataset
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p,cache,_=corpus(root);index=json.loads(p.read_text())
            for split in ('train','dev','test'):
                key='aishell2:'+split+'2';paths=[]
                for i in range(2):
                    f=root/(split+str(i)+'.wav');sf.write(f,np.random.default_rng(i).normal(0,.03,64000),16000,subtype='FLOAT');paths.append(str(f))
                index['splits'][split]['aishell2'][split+'2']=paths;index['enrollments'][key]={'path':paths[0]};cache['embeddings'][key]=torch.ones(1,256)
            p.write_text(json.dumps(index));noise={}
            for split in ('train','dev','test'):
                f=root/(split+'_noise.wav');sf.write(f,np.random.default_rng(8).normal(0,.1,80000),16000,subtype='FLOAT');noise[split]=[str(f)]
            nf=root/'noise.json';nf.write_text(json.dumps({'splits':noise}))
            ds=AIShell2NoisyDataset(p,cache,noise_index=nf,epoch_size=100)
            seen=set()
            for i in range(60):
                (x,y,e,_),m=ds.make_example((2,i));seen.add(m['scenario'])
                self.assertEqual(m['corpus'],'aishell2');self.assertTrue(all(s.startswith('aishell2:') for s in m['active_speakers']))
                self.assertTrue(0<=m['sir_db']<=5);self.assertTrue(5<=m['noise_snr_db']<=20)
                self.assertLessEqual(float(x.abs().max()),.950001)
                self.assertNotIn(m['enrollment_path'],m['source_paths'])
                if m['scenario']=='target-only':
                    error=x[0]-y;snr=20*torch.log10(y.square().mean().sqrt()/error.square().mean().sqrt())
                    self.assertAlmostEqual(float(snr),m['noise_snr_db'],places=3)
                if m['scenario']=='target-absent':self.assertEqual(float(y.abs().sum()),0)
                self.assertTrue(torch.equal(x,ds.make_example((2,i))[0][0]))
            self.assertEqual(len(seen),4)
            dev=AIShell2NoisyDataset(p,cache,noise_index=nf,split='dev',training=False)
            self.assertEqual([dev.make_example(i)[1]['sir_db'] for i in range(6)],list(range(6)))
            noise['test']=noise['train'];nf.write_text(json.dumps({'splits':noise}))
            with self.assertRaisesRegex(ValueError,'overlap'):AIShell2NoisyDataset(p,cache,noise_index=nf)

    def test_raw_enrollment_preserved_in_validation_loader(self):
        from train_dp_multi import build_data_loaders
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p,cache,_=corpus(root)
            aux=json.loads(p.read_text())['enrollments']['aishell2:dev0']['path']
            audio=root/'dev.wav';sf.write(audio,np.ones(16000,dtype=np.float32)*.01,16000)
            paths={}
            for k,f in [('mix',audio),('ref',audio),('aux',aux)]:
                scp=root/(k+'.scp');scp.write_text('id '+str(f)+'\n');paths['bundle_dev_'+k]=str(scp)
            ec=root/'emb.pt';torch.save({'embeddings':{'id':torch.ones(1,256)}},ec)
            config=dict(paths,dataset_type='corpus_mix',corpus_index=str(p),corpus_embedding_cache=cache,train_epoch_size=2,duration=1,sample_rate=16000,mono_policy='mono',ref_duration=2.9,num_workers=0,batch_size=1,bundle_dev_cache=str(ec),enrollment_mode='waveform')
            loaders=build_data_loaders(config,torch.device('cpu'))
            batch=next(iter(loaders['cv_loader']))
            self.assertEqual(batch[2].shape[-1],46400)
