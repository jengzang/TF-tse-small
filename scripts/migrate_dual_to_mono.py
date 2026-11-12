"""Explicit dual -> legacy-layout mono migration; no optimizer state is reused."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from infer import build_model
from models.model_tfgridnet_mono import map_dual_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.checkpoint.resolve():
        parser.error('Output must be a new file distinct from the input')
    torch.set_num_threads(1)
    model = build_model(ROOT / 'config/config_mono_legacy.yaml')
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state, report = map_dual_checkpoint(model, original.get('model_state_dict', original))
    model.load_state_dict(state, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': state, 'model_contract': model.checkpoint_contract,
                'migration': report, 'source_checkpoint': str(args.checkpoint.resolve())}, args.output)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
