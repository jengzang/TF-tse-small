"""Cache the bundle's registration embeddings without changing enrollment preprocessing."""
import argparse
import hashlib
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from dataset.bundle_data import BundleDataset
from emb_extract.resnet34_yk.inference import ResNet34_yk, DEFAULT_CHECKPOINT
from hyperpyyaml import load_hyperpyyaml


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    with open(args.config) as f:
        config = load_hyperpyyaml(f)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device unavailable')
    checkpoint = Path(os.environ.get('TSE_SPEAKER_CHECKPOINT', DEFAULT_CHECKPOINT))
    cpu_encoder = ResNet34_yk(str(checkpoint))
    encoder = ResNet34_yk(str(checkpoint))
    encoder.extractor.to(device).eval()
    encoder.featCal.to(device).eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'dev'):
        output = output_dir / f'{split}_embeddings.pt'
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')
        ds = BundleDataset(config[f'bundle_{split}_mix'], config[f'bundle_{split}_ref'],
                           config[f'bundle_{split}_aux'], bundle_root=config['bundle_root'],
                           sample_rate=config['sample_rate'], duration=config['duration'],
                           ref_duration=config.get('ref_duration', 2.9), mono_policy='duplicate')
        buckets = defaultdict(list)
        embeddings = {}
        verified = 0
        def flush(batch):
            nonlocal verified
            x = torch.from_numpy(np.stack([entry[2] for entry in batch])).to(device)
            with torch.inference_mode():
                values = F.normalize(encoder.extractor(encoder.featCal(x)), dim=1).cpu().float()
            for (uid, raw, _), value in zip(batch, values):
                if verified < 3:
                    expected = cpu_encoder.embd_extract(raw).squeeze(0)
                    diff = (expected - value).abs().max().item()
                    if not torch.allclose(expected, value, atol=1e-4, rtol=1e-3):
                        raise RuntimeError(f'{uid}: CPU/GPU embedding mismatch, max diff {diff}')
                    print(f'{split} CPU/GPU check {uid}: max_abs_diff={diff:.8g}', flush=True)
                    verified += 1
                embeddings[uid] = value.reshape(1, 256).clone()
            if len(embeddings) % 160 < args.batch_size:
                print(f'{split}: {len(embeddings)}/{len(ds)}', flush=True)
        for uid in ds.ids:
            aux = ds._read(ds.aux[uid])
            if aux.shape[0] != 1:
                raise ValueError(f'{uid}: enrollment must be mono')
            raw = aux[0, -ds.ref_length:]
            std = np.std(raw)
            if raw.size < 512 or std <= 1e-8:
                raise ValueError(f'{uid}: invalid enrollment')
            signal = (raw - np.mean(raw)) / std
            bucket = buckets[len(signal)]
            bucket.append((uid, raw, signal))
            if len(bucket) >= args.batch_size:
                flush(bucket)
                buckets[len(signal)] = []
        for bucket in buckets.values():
            if bucket:
                flush(bucket)
        if len(embeddings) != len(ds):
            raise RuntimeError('Missing cached embeddings')
        cache = {'embeddings': embeddings, 'metadata': {'speaker_checkpoint_sha256': sha256(checkpoint),
                  'aux_scp_sha256': sha256(config[f'bundle_{split}_aux']),
                  'sample_rate': ds.sr, 'ref_duration': config.get('ref_duration', 2.9)}}
        tmp = output.with_suffix('.pt.tmp')
        torch.save(cache, tmp)
        os.replace(tmp, output)
        print(f'{split}: saved {len(embeddings)} embeddings to {output}', flush=True)


if __name__ == '__main__':
    main()
