"""Original active SI-SNR loss, retained without perceptual-loss imports."""
import torch


def sisnr(x, s, eps=1e-6):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def batchMean_sisnrLoss(est, clean, eps=1e-6):
    batch_sisnr = sisnr(est, clean, eps)
    return -torch.mean(batch_sisnr)


def _valid_mask(signal, lengths):
    if signal.ndim != 2 or lengths.ndim != 1 or signal.size(0) != lengths.numel():
        raise ValueError('Expected waveform [B,T] and lengths [B]')
    lengths = lengths.to(device=signal.device, dtype=torch.long)
    if bool(((lengths <= 0) | (lengths > signal.size(1))).any()):
        raise ValueError('Invalid waveform lengths')
    return torch.arange(signal.size(1), device=signal.device)[None, :] < lengths[:, None]


def masked_sisnr(estimate, target, lengths, eps=1e-6):
    """Per-sample SI-SNR over valid samples, excluding padding."""
    if estimate.shape != target.shape:
        raise ValueError('Waveform shapes must match')
    mask = _valid_mask(estimate, lengths).to(estimate.dtype)
    count = mask.sum(dim=-1, keepdim=True)
    x = (estimate - (estimate * mask).sum(dim=-1, keepdim=True) / count) * mask
    s = (target - (target * mask).sum(dim=-1, keepdim=True) / count) * mask
    projection = (x * s).sum(dim=-1, keepdim=True) * s / (s.square().sum(dim=-1, keepdim=True) + eps)
    return 20 * torch.log10(eps + torch.linalg.vector_norm(projection, dim=-1) /
                            (torch.linalg.vector_norm(x - projection, dim=-1) + eps))


def masked_multires_spectral_loss(estimate, target, lengths, stfts):
    """Sum per-resolution real, imaginary and magnitude L1 losses on valid frames."""
    mask = _valid_mask(estimate, lengths).to(estimate.dtype)
    estimate, target = estimate * mask, target * mask
    total = estimate.new_zeros(())
    for stft in stfts:
        pred, ref = stft(estimate), stft(target)
        frame_positions = torch.arange(pred.size(-1), device=pred.device) * stft.hop_length
        frames = (frame_positions[None, :] <= lengths.to(pred.device)[:, None]).to(pred.real.dtype)
        weight = frames[:, None, :]
        denominator = frames.sum(dim=-1) * pred.size(-2)
        error = ((pred.real-ref.real).abs() + (pred.imag-ref.imag).abs() +
                 (pred.abs()-ref.abs()).abs())
        total = total + ((error * weight).sum(dim=(-2,-1)) / denominator).mean()
    return total


def target_present(target, lengths):
    """Exact-zero reference denotes absent target; quiet speech remains active."""
    mask = _valid_mask(target, lengths)
    return ((target != 0) & mask).any(dim=-1)


def presence_aware_sisnr_loss(estimate, target, lengths):
    """Active -SI-SNR, absent waveform L1, averaged over all examples."""
    mask = _valid_mask(estimate, lengths)
    present = target_present(target, lengths)
    values = (estimate.abs() * mask).sum(dim=-1) / lengths.to(estimate)
    if present.any():
        values = values.clone()
        values[present] = -masked_sisnr(estimate[present], target[present], lengths[present])
    return values.mean()


def normalized_waveform_loss(estimate, target, mixture, lengths):
    """Scale-sensitive valid-sample MSE; absent examples use mixture power."""
    mask = _valid_mask(estimate, lengths).to(estimate.dtype)
    count = lengths.to(estimate)
    present = target_present(target, lengths)
    reference = torch.where(present[:, None], target, mixture)
    power = (reference.square() * mask).sum(-1) / count
    error = ((estimate - target).square() * mask).sum(-1) / count
    return (error / power.clamp_min(1e-8)).mean()


def normalized_absence_loss(estimate,target,mixture,lengths):
    mask=_valid_mask(estimate,lengths).to(estimate.dtype)
    absent=~target_present(target,lengths)
    if not absent.any():return estimate.sum()*0
    output=(estimate.square()*mask).sum(-1)/lengths.to(estimate)
    reference=(mixture.square()*mask).sum(-1)/lengths.to(estimate)
    return (output[absent]/reference[absent].clamp_min(1e-8)).mean()
