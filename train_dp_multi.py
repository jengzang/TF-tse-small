import os
import torch
import numpy as np
import logging
import argparse
import time
import random
import shutil
import inspect
import tempfile
import hashlib
import json
from pathlib import Path

from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainer.trainer_dp_v1 import Trainer, PrecisionDataParallel
# from trainer.trainer_se import Trainer


# fix random seeds for reproducibility
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

torch.cuda.manual_seed_all(SEED)


def save_source_snapshot(checkpoint_dir):
    """Freeze the local mono implementation and its training dependencies."""
    root = Path(__file__).resolve().parent
    files = ['train_dp_multi.py', 'infer.py', 'models/model_tfgridnet_mono.py',
             'models/model_TFGridNet_k1_stage2_multi_v5.py', 'models/local/dcunet.py',
             'models/local/get_layer_from_string.py', 'utils/feature.py',
             'trainer/trainer_dp_v1.py', 'utils/loss/losses.py', 'dataset/bundle_data.py']
    files += ['models/model_tfgridnet_cases.py', 'models/model_senior_reference.py', 'models/model_senior_batchnorm_reference.py', 'models/vendor/fla/linear_attn.py',
              'models/vendor/fla/LICENSE', 'models/vendor/fla/SOURCE.json',
              'utils/case_registry.py', 'dataset/corpus_mix.py', 'scripts/run_case.py',
              'scripts/inspect_cases.py', 'scripts/evaluate_case.py', 'scripts/run_next_cases.py', 'models/model_paper_ablation.py', 'utils/enrollment.py', 'scripts/run_paper_queue.py', 'utils/paper_cases.py', 'scripts/run_noisy_followup.py', 'scripts/prepare_aishell2_noisy.py',
              'dataset/corpus_mix.py', 'dataset/aishell2_noisy.py', 'utils/audio_cache.py', 'scripts/prepare_corpora.py', 'scripts/run_server_queue.py']
    folder = Path(checkpoint_dir) / 'source_snapshot'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for relative in files:
        data = (root / relative).read_bytes()
        dest = folder / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        manifest[relative] = hashlib.sha256(data).hexdigest()
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def configure_backend(config):
    options = dict(enabled=config.get('backend_acceleration', False),
                   benchmark=config.get('backend_benchmark', False),
                   deterministic=config.get('backend_deterministic', True))
    if any(not isinstance(value,bool) for value in options.values()):
        raise ValueError('backend_acceleration, backend_benchmark and backend_deterministic must be boolean')
    for key,value in options.items(): setattr(torch.backends.cudnn,key,value)


def build_data_loaders(config, device):
    """Select on-the-fly corpus training or historical bundle/legacy loading.

    Corpus validation always uses a fixed manifest bundle so every case is
    scored on identical mixtures. Worker-only options are omitted for workers=0.
    """
    kind = config.get('dataset_type', 'legacy')
    if kind not in ('bundle', 'corpus_mix', 'aishell2_noisy', 'legacy'):
        raise ValueError('dataset_type must be bundle, corpus_mix or legacy')
    sampler = None
    options = dict(bundle_root=config.get('bundle_root'), duration=config['duration'],
                   sample_rate=config['sample_rate'], mono_policy=config.get('mono_policy','reject'),
                   ref_duration=config.get('ref_duration', 2.9))
    if kind in ('corpus_mix','aishell2_noisy'):
        from dataset.corpus_mix import CorpusMixDataset, EpochIndexSampler
        if config['sample_rate'] != 16000 or config.get('mono_policy') != 'mono':
            raise ValueError('corpus_mix requires 16k mono')
        dataset_cls=CorpusMixDataset
        noisy_options={}
        if kind=='aishell2_noisy':
            from dataset.aishell2_noisy import AIShell2NoisyDataset
            dataset_cls=AIShell2NoisyDataset
            noisy_options['noise_index']=config['noise_index']
        trainset = dataset_cls(config['corpus_index'], config['corpus_embedding_cache'],
                                   split='train',epoch_size=config['train_epoch_size'],
                                   duration=config['duration'],seed=config.get('seed',3407),training=True,enrollment_mode=config.get('enrollment_mode','embedding'),ref_duration=config['ref_duration'],**noisy_options)
        sampler = EpochIndexSampler(trainset)
    elif kind == 'bundle':
        from dataset.bundle_data import BundleDataset
        trainset = BundleDataset(config['bundle_train_mix'], config['bundle_train_ref'],
                                 config['bundle_train_aux'], training=True,
                                 max_items=config.get('bundle_train_limit'),
                                 embedding_cache=config.get('bundle_train_cache'), enrollment_mode=config.get('enrollment_mode','embedding'), **options)
    else:
        from dataset.data_wsj0_rir_pyroom_v7 import tr_wsj_mix_dataset
        trainset = tr_wsj_mix_dataset(
            spk2utts_path=config['train_spk2utts_path'], embd_path=config['train_embd_path'],
            noise_scp=os.path.join(config['noise_scp'], config['train_noise']),
            reverb_param_path=config['train_reverb_param_root'], rir_scp=config['rir_scp'],
            data_list=config['data_list'], iter=config['train_iter'], dur=config['duration'],
            fs=config['sample_rate'], min_max=config['min_max'])
    workers = config['num_workers']
    if type(workers) is not int or workers < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    worker_options = {}
    if workers > 0:
        prefetch = config.get('prefetch_factor', 2)
        if type(prefetch) is not int or prefetch < 1:
            raise ValueError('prefetch_factor must be a positive integer')
        persistent = config.get('persistent_workers', False)
        if not isinstance(persistent,bool):
            raise ValueError('persistent_workers must be boolean')
        worker_options.update(persistent_workers=persistent,prefetch_factor=prefetch)
    trainloader = DataLoader(trainset,batch_size=config['batch_size'],shuffle=sampler is None,
                             sampler=sampler,num_workers=workers,pin_memory=device.type=='cuda',
                             **worker_options)
    if kind in ('bundle','corpus_mix','aishell2_noisy'):
        from dataset.bundle_data import BundleDataset
        valid_options = dict(options, duration=config.get('bundle_dev_duration',config['duration']),enrollment_mode=config.get('enrollment_mode','embedding'))
        validset = BundleDataset(config['bundle_dev_mix'],config['bundle_dev_ref'],
                                 config['bundle_dev_aux'],training=False,
                                 max_items=config.get('bundle_dev_limit'),
                                 embedding_cache=config.get('bundle_dev_cache'),
                                 return_valid_length=config.get('padding_aware_loss',False),**valid_options)
    else:
        from dataset.data_wsj0_rir_pyroom_v7 import te_dataset
        validset = te_dataset(mix_scp=config['dev_mix_scp'],ref_scp=config['dev_ref_scp'],
                              aux_scp=config['dev_aux_scp'],embd_path=config['dev_embd_path'],
                              data_list=config['data_list'],fs=config['sample_rate'])
    validation_batch_size = config.get('validation_batch_size',1)
    if type(validation_batch_size) is not int or validation_batch_size < 1:
        raise ValueError('validation_batch_size must be a positive integer')
    validloader = DataLoader(validset,batch_size=validation_batch_size,shuffle=False,
                             num_workers=workers,pin_memory=device.type=='cuda',**worker_options)
    return {'tr_loader':trainloader,'cv_loader':validloader}


def main(config, args):
    configure_backend(config)
    if 'seed' in config:
        seed = config['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device(config.get('device', 'cuda'))
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Training currently supports cpu or cuda/ROCm')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA/ROCm is unavailable; set device: cpu for local training.')
    experiment = os.path.join('chkpt', config['name'])
    if os.path.exists(os.path.join(experiment, 'config.yaml')) and not config['continue_from']:
        raise FileExistsError('Experiment already exists; change name or set continue_from: ' + experiment)

    log_dir = os.path.join('logs', config['name'])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, time.strftime('%Y-%m-%d-%H%M.log',time.localtime(time.time()))
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info("Backend acceleration: %s; algorithm search: %s; deterministic kernels: %s; precision: %s", torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, config.get("precision", "fp32"))

    
    data = build_data_loaders(config, device)

    model = config['modules']['masknet']
    if config.get('init_from'):
        if config.get('continue_from'):
            raise ValueError('init_from and continue_from are mutually exclusive')
        from infer import load_weights
        load_weights(model, config['init_from'])
    
    model.to(device)
    
    logger.info(model)
    logger.info('-' * 50)

    if device.type == 'cuda':
        model = PrecisionDataParallel(model, precision=config.get('precision', 'fp32'),
                                      device_ids=[device.index] if device.index is not None else None)

    chkpt_dir = os.path.join('chkpt', config['name'])
    os.makedirs(chkpt_dir, exist_ok=True)

    model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = config['optimizer'](params=model_params)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    trainer = Trainer(chkpt_dir = chkpt_dir,
                      data = data,
                      model = model,
                      optimizer = optimizer,
                      scheduler= lr_scheduler,
                      logger = logger,
                      config = config,
                      
    )
    # Trainer validates resume architecture/state before experiment records change.
    snapshot_dir = Path(chkpt_dir)
    if config.get('continue_from'):
        resumes = snapshot_dir / 'resumes'
        resumes.mkdir(exist_ok=True)
        snapshot_dir = Path(tempfile.mkdtemp(prefix='run_', dir=resumes))
    if config.get('dataset_type') == 'corpus_mix':
        paths = {key:config[key] for key in ('corpus_index','corpus_embedding_cache',
                 'bundle_dev_mix','bundle_dev_ref','bundle_dev_aux','bundle_dev_cache')}
        provenance = {}
        for key, path in paths.items():
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda:stream.read(1024*1024), b''):
                    digest.update(block)
            provenance[key] = dict(path=str(path),sha256=digest.hexdigest())
        (snapshot_dir / 'data_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    shutil.copyfile(args.config, snapshot_dir / 'config.yaml')
    shutil.copyfile(inspect.getmodule(config['MaskNet'].__class__).__file__, snapshot_dir / 'model.py')
    if hasattr(config['MaskNet'], 'checkpoint_contract'):
        save_source_snapshot(snapshot_dir)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speech Separation: transformer')
    parser.add_argument('--config', default='config/config_tfgridnet_all.yaml', type=str,
                        help='config file path (default: None)')
    
    args = parser.parse_args()

    # Read config of the whole system.
    assert os.path.isfile(args.config), "No such file: %s" % args.config
    with open(args.config, 'r') as f:
        config_strings = f.read()
    config = load_hyperpyyaml(config_strings)
    print('INFO: Loaded hparams from: {}'.format(args.config))

    main(config, args)
