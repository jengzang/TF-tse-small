"""Launch one controlled case with shared data paths and an explicit run budget."""
import argparse
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from hyperpyyaml import load_hyperpyyaml, resolve_references
from train_dp_multi import main as train_main
from utils.case_registry import CASES, resolve_case

DATA_KEYS = {
    'bundle_root','bundle_train_mix','bundle_train_ref','bundle_train_aux',
    'bundle_dev_mix','bundle_dev_ref','bundle_dev_aux',
    'bundle_train_cache','bundle_dev_cache','ref_duration',
    'dataset_type','corpus_index','corpus_embedding_cache','train_epoch_size',
}


def evaluate_heldout_test(config, config_path, device, precision, threads):
    from scripts.evaluate_case import main as evaluate
    folder=Path(config['corpus_index']).parent/'test'
    checkpoint=Path('chkpt')/config['name']
    evaluate(['--config',str(config_path),'--checkpoint',str(checkpoint/'temp_best.pth.tar'),
              '--mix-scp',str(folder/'mix.scp'),'--ref-scp',str(folder/'ref.scp'),
              '--aux-scp',str(folder/'aux.scp'),'--embedding-cache',str(folder/'embeddings.pt'),
              '--metadata',str(folder/'metadata.json'),'--output',str(checkpoint/'test_metrics.json'),
              '--device',device,'--precision',precision,'--threads',str(threads)])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-config',type=Path,help='Explicit experimental configuration; leaves original queue registry unchanged')
    p.add_argument('--case', required=True, help='case00..case09 or full case name')
    p.add_argument('--data-config', type=Path, required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--max-steps',type=int)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--num-workers',type=int,default=8)
    p.add_argument('--duration',type=float,default=10.)
    p.add_argument('--threads',type=int,default=1,help='torch CPU threads per process')
    p.add_argument('--name',help='Experiment suffix; default case_name/seedSEED')
    p.add_argument('--resume',type=Path)
    p.add_argument('--evaluate-test',action='store_true',help='Evaluate corpus heldout test using validation best after training')
    p.add_argument('--init-from',type=Path,help='Warm-start model only; optimizer starts fresh')
    p.add_argument('--accumulation-steps',type=int,default=1)
    p.add_argument('--samples-per-epoch',type=int,help='Online corpus mixtures per epoch')
    p.add_argument('--backend-autotune',action='store_true',help='Search fast backend algorithms; permit nondeterministic kernels')
    p.add_argument('--backend-acceleration',action='store_true',help='Enable cuDNN for CUDA/ROCm')
    p.add_argument('--validation-batch-size',type=int,default=1)
    p.add_argument('--persistent-workers',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--prefetch-factor',type=int,default=2)
    args=p.parse_args(argv)
    try:
        case=resolve_case(args.case)
    except ValueError as e:
        p.error(str(e))
    if min(args.epochs,args.batch_size,args.threads)<=0 or args.duration<=0 or args.num_workers<0:
        p.error('epochs, batch-size, threads, duration must be positive; workers nonnegative')
    if min(args.accumulation_steps,args.validation_batch_size,args.prefetch_factor)<=0:
        p.error('accumulation, validation batch size and prefetch must be positive')
    if args.samples_per_epoch is not None and args.samples_per_epoch<=0:
        p.error('samples-per-epoch must be positive')
    if args.resume and args.init_from:
        p.error('resume and init-from are mutually exclusive')
    if args.max_steps is not None and args.max_steps<=0:
        p.error('max-steps must be positive')
    if not 0 <= args.seed < 2**32:
        p.error('seed must be in [0, 2**32)')
    with args.data_config.open() as f:
        shared=yaml.safe_load(f)
    if not isinstance(shared,dict) or set(shared)-DATA_KEYS:
        p.error('Unsupported shared override; data-config accepts only bundle paths/ref_duration')
    # Relative server paths resolve against the shared data config, not shell cwd.
    for key,value in shared.items():
        if key not in ('ref_duration','dataset_type','train_epoch_size') and value is not None:
            path=Path(value).expanduser()
            shared[key]=str((args.data_config.resolve().parent/path).resolve()) if not path.is_absolute() else str(path)
    overrides={**shared,'device':args.device,'precision':args.precision,'seed':args.seed,
               'epochs':args.epochs,'max_steps':args.max_steps,'batch_size':args.batch_size,
               'num_workers':args.num_workers,'duration':args.duration,
               'name':'cases/'+(args.name or f'{case}/seed{args.seed}'),
               'continue_from':str(args.resume.resolve()) if args.resume else '',
               'init_from':str(args.init_from.resolve()) if args.init_from else '',
               'gradient_accumulation_steps':args.accumulation_steps,
               'backend_benchmark':args.backend_autotune,'backend_deterministic':not args.backend_autotune,
               'backend_acceleration':args.backend_acceleration,
               'validation_batch_size':args.validation_batch_size,
               'persistent_workers':args.persistent_workers,'prefetch_factor':args.prefetch_factor}
    if args.samples_per_epoch is not None:
        overrides['train_epoch_size']=args.samples_per_epoch
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    text=(args.case_config or ROOT/CASES[case]).read_text()
    resolved=resolve_references(text,overrides=overrides,overrides_must_match=True).read()
    config=load_hyperpyyaml(resolved)
    # Model variants consume different RNG amounts during construction. Reset
    # before creating data loaders so same seed also means same sampling stream.
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    print(f'{case}: total_parameters={sum(v.numel() for v in config["MaskNet"].parameters())}; output=chkpt/{config["name"]}',flush=True)
    # train_main copies the resolved, actual config before this temp file is removed.
    with tempfile.TemporaryDirectory(prefix='tf_tse_case_') as folder:
        config_path=Path(folder)/'config.yaml';config_path.write_text(resolved)
        train_main(config,argparse.Namespace(config=str(config_path)))
        if args.evaluate_test:
            evaluate_heldout_test(config,config_path,args.device,args.precision,args.threads)


if __name__=='__main__':
    main()
