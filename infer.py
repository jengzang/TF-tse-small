"""Offline mono or two-channel inference for the verified DCUNet → TFGridNet v5."""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from hyperpyyaml import load_hyperpyyaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'config/config_tfgridnet_all.yaml'


def build_model(config_path=DEFAULT_CONFIG):
    with open(config_path, encoding='utf-8') as handle:
        config = load_hyperpyyaml(handle)
    if config['sample_rate'] != 16000:
        raise ValueError('This baseline requires sample_rate=16000')
    return config['modules']['masknet']


def load_weights(model, checkpoint_path):
    # Historical trusted project checkpoints contain optimizer/NumPy RNG state.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if hasattr(model, 'validate_checkpoint'):
        model.validate_checkpoint(checkpoint)
    state = checkpoint.get('model_state_dict', checkpoint)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def read_mixture(path, channels=2):
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if rate != 16000:
        raise ValueError(f'Mixture must be 16000 Hz; got {rate}')
    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")
    if audio.shape[1] != channels:
        raise ValueError('Mixture must have exactly two channels in the intended microphone order' if channels == 2 else 'Mixture must be mono')
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Mixture is empty or contains non-finite samples')
    tensor = torch.from_numpy(audio.T.copy()).unsqueeze(0)
    return tensor.unsqueeze(0) if channels == 2 else tensor


def read_embedding(path):
    embedding = np.load(path, allow_pickle=False)
    if embedding.shape not in ((256,), (1, 256), (1, 1, 256)):
        raise ValueError(f'Expected 256-D embedding, got {embedding.shape}')
    if not np.isfinite(embedding).all() or np.linalg.norm(embedding) <= 1e-8:
        raise ValueError('Embedding must be finite and nonzero')
    return torch.from_numpy(embedding.astype(np.float32).reshape(1, 1, 256))


def read_enrollment(path, speaker_checkpoint):
    from emb_extract.resnet34_yk.inference import ResNet34_yk
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if rate != 16000 or audio.shape[1] != 1:
        raise ValueError('Enrollment must be mono 16000 Hz audio')
    return ResNet34_yk(speaker_checkpoint).embd_extract(audio[:, 0]).unsqueeze(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mix', type=Path, required=True)
    enrollment = parser.add_mutually_exclusive_group(required=True)
    enrollment.add_argument('--embedding', type=Path, help='Matching ResNet34_yk .npy embedding')
    enrollment.add_argument('--enroll', type=Path, help='Target speaker mono 16 kHz WAV')
    parser.add_argument('--speaker-checkpoint', type=Path, help='ResNet34_yk model_19.pkl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', help='cpu or cuda[:index] (also ROCm torch API)')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    inputs = [args.mix, args.checkpoint, args.config, args.embedding, args.enroll, args.speaker_checkpoint]
    if any(p and p.resolve() == args.output.resolve() for p in inputs):
        parser.error('--output must not overwrite an input file')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('Only cpu and cuda/ROCm are supported by this entry')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA/ROCm is unavailable in this Python environment')
    with open(args.config, encoding='utf-8') as handle:
        settings = load_hyperpyyaml(handle)
    mixture = read_mixture(args.mix, channels=settings.get('input_channels', 2))
    embedding = (read_embedding(args.embedding) if args.embedding
                 else read_enrollment(args.enroll, args.speaker_checkpoint))
    model = settings['modules']['masknet']
    if settings['sample_rate'] != 16000:
        raise ValueError('This baseline requires sample_rate=16000')
    load_weights(model, args.checkpoint)
    model.to(device).eval()
    length = mixture.shape[-1]
    # Pad to a hop boundary, then retain every original sample in the output.
    padded_length = max(640, ((length + 319) // 320) * 320)
    mixture = torch.nn.functional.pad(mixture, (0, padded_length - length))
    with torch.inference_mode():
        target = model(mixture.to(device), embedding.to(device))
    if target.shape[-1] < length or not torch.isfinite(target).all():
        raise RuntimeError('Invalid model output length or non-finite samples')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, target[0, :length].cpu().numpy(), 16000, subtype='FLOAT')
    print(f'Saved {args.output} ({length} samples, 16000 Hz, mono)')


if __name__ == '__main__':
    main()
