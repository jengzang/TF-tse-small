"""Corpus training loader selection and unchanged bundle compatibility."""
import unittest
from unittest.mock import patch
import torch


class TrainingLoaderTests(unittest.TestCase):
    def test_corpus_epoch_sampler_and_worker_options(self):
        from train_dp_multi import build_data_loaders
        from dataset.corpus_mix import EpochIndexSampler
        dataset = [(torch.zeros(1,16000),torch.zeros(16000),torch.ones(1,256),torch.ones(16000))]
        config = dict(dataset_type='corpus_mix',corpus_index='index.json',corpus_embedding_cache='emb.pt',
                      train_epoch_size=100,duration=1,sample_rate=16000,seed=7,
                      batch_size=2,num_workers=2,persistent_workers=True,prefetch_factor=3,
                      validation_batch_size=4,bundle_root=None,mono_policy='mono',
                      bundle_dev_mix='mix',bundle_dev_ref='ref',bundle_dev_aux='aux',padding_aware_loss=True)
        with patch('dataset.corpus_mix.CorpusMixDataset',return_value=dataset) as corpus, \
             patch('dataset.bundle_data.BundleDataset',return_value=dataset):
            loaders=build_data_loaders(config,torch.device('cpu'))
        self.assertIsInstance(loaders['tr_loader'].sampler,EpochIndexSampler)
        self.assertTrue(loaders['tr_loader'].persistent_workers)
        self.assertEqual(loaders['tr_loader'].prefetch_factor,3)
        self.assertEqual(loaders['cv_loader'].batch_size,4)
        self.assertEqual(corpus.call_args.kwargs['epoch_size'],100)

    def test_bundle_zero_workers_omits_worker_only_options(self):
        from train_dp_multi import build_data_loaders
        dataset=[(torch.zeros(1,800),torch.zeros(800),torch.ones(1,256),torch.ones(800))]
        config=dict(dataset_type='bundle',duration=.05,sample_rate=16000,batch_size=1,num_workers=0,
                    persistent_workers=True,prefetch_factor=2,bundle_root=None,mono_policy='mono',
                    bundle_train_mix='mix',bundle_train_ref='ref',bundle_train_aux='aux',
                    bundle_dev_mix='mix',bundle_dev_ref='ref',bundle_dev_aux='aux',padding_aware_loss=True)
        with patch('dataset.bundle_data.BundleDataset',return_value=dataset):
            loaders=build_data_loaders(config,torch.device('cpu'))
        self.assertFalse(loaders['tr_loader'].persistent_workers)
        self.assertIsNone(loaders['tr_loader'].prefetch_factor)
        self.assertEqual(next(iter(loaders['tr_loader']))[0].shape,(1,1,800))

    def test_case_cli_corpus_and_performance_overrides(self):
        import tempfile
        from pathlib import Path
        from scripts.run_case import main
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=root/'data.yaml'
            config.write_text('dataset_type: corpus_mix\ncorpus_index: index.json\ncorpus_embedding_cache: embeddings.pt\ntrain_epoch_size: 30\n')
            with patch('scripts.run_case.train_main') as train:
                main(['--case','case00','--data-config',str(config),'--device','cpu',
                      '--accumulation-steps','4','--samples-per-epoch','60','--backend-acceleration',
                      '--validation-batch-size','3','--init-from',str(root/'initial.pt')])
            actual=train.call_args.args[0]
            self.assertEqual(actual['dataset_type'],'corpus_mix')
            self.assertEqual(actual['corpus_index'],str((root/'index.json').resolve()))
            self.assertEqual(actual['train_epoch_size'],60)
            self.assertEqual(actual['gradient_accumulation_steps'],4)
            self.assertEqual(actual['validation_batch_size'],3)
            self.assertTrue(actual['backend_acceleration'])
            self.assertEqual(actual['init_from'],str((root/'initial.pt').resolve()))
