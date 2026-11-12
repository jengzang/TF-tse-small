import os
import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from utils.loss.losses import batchMean_sisnrLoss, masked_sisnr, masked_multires_spectral_loss, presence_aware_sisnr_loss, target_present, normalized_waveform_loss

def accumulation_group_size(index, batches, steps):
    """Normalize a final partial accumulation group by its actual length."""
    return min(steps, batches - (index // steps) * steps)


def precision_context(device, precision):
    if precision == 'fp32':
        return nullcontext()
    if precision == 'bf16' and device.type in ('cpu', 'cuda'):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f'Unsupported precision {precision!r} on {device}')

class AutocastReplica(nn.Module):
    def __init__(self, module, precision):
        super().__init__()
        self.module, self.precision = module, precision

    def forward(self, *args, **kwargs):
        with precision_context(args[0].device, self.precision):
            return self.module(*args, **kwargs)


class PrecisionDataParallel(nn.DataParallel):
    def __init__(self, module, precision, **kwargs):
        super().__init__(module, **kwargs)
        self.precision = precision

    def parallel_apply(self, replicas, inputs, kwargs):
        wrapped = [AutocastReplica(replica, self.precision) for replica in replicas]
        return super().parallel_apply(wrapped, inputs, kwargs)


def pw(x,s, fs = 16000,eps=1e-8):
    """
    Calculate the energy-based loss.
    
    Args:
        x: Separated signal, a tensor of shape (N, S)
        eps: Small constant for numerical stability
    
    Returns:
        Energy-based loss
    """
    
    def l2norm(mat, keepdim=False):
        # Calculate the L2 norm (Euclidean norm) of the input tensor
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    
    # Zero-mean the input signal (remove the mean)
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    
    # Compute the squared L2 norm of the zero-mean signal
    # squared_norm = l2norm(x_zm)**2  # Compute norm and then square it
    x_squared = l2norm(x_zm, keepdim=True) ** 2
    s_squared = l2norm(s_zm, keepdim=True) ** 2
    # squared_norm = torch.clamp(squared_norm, min=eps)
    
    # Return the log10 of the squared norm as part of energy loss
    energy_loss = 10 * torch.log10(torch.abs(x_squared-s_squared) + eps)
    
    return torch.mean(energy_loss)

class Trainer(object):
    
    def __init__(self, chkpt_dir, data, model, optimizer, scheduler, logger, config):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        
        self.model = model
        self.device = torch.device(config.get('device', 'cuda'))
        self.precision = config.get('precision', 'fp32')
        self.runtime = {key:config.get(key) for key in ('backend_acceleration','backend_benchmark','backend_deterministic','batch_size','gradient_accumulation_steps','num_workers')}
        precision_context(self.device, self.precision)
        self.model.to(self.device)
        # Preserve historical experiments unless a new config opts out.
        self.legacy_positive_clamp = config.get('legacy_positive_clamp', True)
        self.config_waveform_weight = float(config.get('normalized_waveform_weight', 0.))
        self.padding_aware_loss = config.get('padding_aware_loss', False)
        self.target_absent_loss = config.get('target_absent_loss', False)
        if self.target_absent_loss and not self.padding_aware_loss:
            raise ValueError('target_absent_loss requires padding_aware_loss')
        if self.padding_aware_loss and self.legacy_positive_clamp:
            raise ValueError('padding_aware_loss requires legacy_positive_clamp: false')
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training config
        self.epochs = config['epochs']
        self.max_norm = config['max_norm']
        self.logger = logger
        # save and load model
        self.save_folder = chkpt_dir
        self.checkpoint = config['checkpoint']
        self.continue_from = config['continue_from']
        # logging
        self.print_freq = config['print_freq']
        os.makedirs(self.save_folder, exist_ok=True)
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.global_step = 0
        self.max_steps = config.get('max_steps')
        self.accumulation_steps = config.get('gradient_accumulation_steps', 1)
        if type(self.accumulation_steps) is not int or self.accumulation_steps < 1:
            raise ValueError('gradient_accumulation_steps must be a positive integer')
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps <= 0):
            raise ValueError('max_steps must be a positive integer or null')
        # self.tr_loss = 0.0
        # self.val_loss = 0.0

        # self.sepc_loss = nn.MSELoss()
        self.sepcI_loss = nn.L1Loss()
        self.sepcR_loss = nn.L1Loss()
        self.sepcM_loss = nn.L1Loss()
        # self.MSE = nn.MSELoss()

        self.stft512 = config['STFT512']
        self.stft256 = config['STFT256']
        self.stft128 = config['STFT128']
        # self.stft64 = config['STFT64']

        # Active loss is computed explicitly in _run_train_epoch.

        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            cont = torch.load(self.continue_from, map_location='cpu', weights_only=False)
            if self.max_steps is not None and 'global_step' not in cont:
                raise ValueError('Budgeted resume requires a checkpoint with global_step')
            self.global_step = cont.get('global_step', 0)
            self.start_epoch = cont['epoch']
            # DataParallel is used on CUDA, but local CPU runs are unwrapped.
            # Normalize names on both paths while keeping architecture strict.
            target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            if hasattr(target_model, 'validate_checkpoint'):
                target_model.validate_checkpoint(cont)
            state = {key.removeprefix('module.'): value
                     for key, value in cont['model_state_dict'].items()}
            target_model.load_state_dict(state, strict=True)
            self.optimizer.load_state_dict(cont['optimizer_state'])
            self.best_val_loss = cont.get('best_val_loss', float('inf'))
            if cont.get('scheduler_state') is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(cont['scheduler_state'])
            if 'prandom_state' in cont:
                random.setstate(cont['prandom_state'])
            if self.device.type == 'cuda' and cont.get('cuda_random_state_all') is not None:
                torch.cuda.set_rng_state_all(cont['cuda_random_state_all'])
            torch.set_rng_state((cont['trandom_state']))
            np.random.set_state((cont['nrandom_state']))

            # print('Loading checkpoint model %s' % self.continue_from)
            # cont = torch.load(self.continue_from, map_location='cpu', weights_only=False)
            # # self.start_epoch = cont['epoch']
            # dict = {key: value for key, value in cont['model_state_dict'].items() if (key != 'module.conv2d.weight' and key != 'module.conv2d.bias')}
            # model_dict = self.model.state_dict()
            # model_dict.update(dict)
            # self.model.load_state_dict(model_dict)

    def _run_train_epoch(self, epoch):

        start = time.time()
        total_loss = 0
        total_sisnr_loss = 0
        total_mse_loss = 0
        total_specR_loss = 0
        total_specI_loss = 0
        data_loader = self.tr_loader
        batches = 0
        samples = 0
        self.optimizer.zero_grad(set_to_none=True)
        if hasattr(getattr(data_loader, "sampler", None), "set_epoch"):
            data_loader.sampler.set_epoch(epoch)

        for i, (data) in enumerate(data_loader):
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

            # if i > 10:
            #     break

            if len(data) == 4:
                mixture, source, emb, vad_sq = (item.to(self.device, non_blocking=True) for item in data)
            elif len(data) == 7:
                mixture, source1, emb1, vad_sq1, source2, emb2, vad_sq2 = (
                    item.to(self.device, non_blocking=True) for item in data)
                mixture = torch.cat((mixture, mixture), dim=0)
                source = torch.cat((source1, source2), dim=0)
                emb = torch.cat((emb1, emb2), dim=0)
                vad_sq = torch.cat((vad_sq1, vad_sq2), dim=0)
            else:
                raise ValueError(f'Unexpected training batch of {len(data)} elements')

            with precision_context(self.device, self.precision):
                estimate_source = self.model(mixture, emb)
            estimate_source = estimate_source.float()

            if self.padding_aware_loss:
                if len(data) != 4:
                    raise ValueError('padding-aware loss requires bundle valid-sample mask')
                lengths = vad_sq.sum(dim=-1).to(dtype=torch.long)
                sisnr_loss = (presence_aware_sisnr_loss(estimate_source, source, lengths)
                              if self.target_absent_loss else -masked_sisnr(estimate_source, source, lengths).mean())
                spectral_loss = masked_multires_spectral_loss(
                    estimate_source, source, lengths, (self.stft512, self.stft256, self.stft128))
                # Bundle mask describes padding, not speech activity: do not treat
                # padding as a VAD-derived non-target region for the old pw loss.
                loss = sisnr_loss + spectral_loss
                if self.config_waveform_weight:
                    reference_mix = mixture[:, 0, :source.shape[-1]] if mixture.ndim == 3 else mixture[:, 0, 0, :source.shape[-1]]
                    loss = loss + self.config_waveform_weight * normalized_waveform_loss(
                        estimate_source, source, reference_mix, lengths)
            else:
                if self.legacy_positive_clamp:
                    estimate_source = torch.clamp(estimate_source, min=1e-10, max=1e10)
                    source = torch.clamp(source, min=1e-10, max=1e10)
            

                # estimate_source_tar = torch.where(vad_sq == 0, torch.tensor(0), estimate_source)
                # source_tar = torch.where(vad_sq == 0, torch.tensor(0), source)
                estimate_source_inter = torch.where(vad_sq == 1, torch.tensor(0), estimate_source)
                source_inter = torch.where(vad_sq == 1, torch.tensor(0), source)

                # print(estimate_source_tar.shape, source_tar.shape, estimate_source_inter.shape, source_inter.shape)
            

                # loss = batchMean_sisnrLoss(estimate_source, source) + 0.5*self.sepcI_loss(stft(estimate_source)[0], stft(source)[0]) + \
                #       0.5*self.sepcR_loss(stft(estimate_source)[3], stft(source)[3])
                # print(batchMean_sisnrLoss(estimate_source, source), self.sepcI_loss(stft(estimate_source)[2], stft(source)[2]), self.sepcR_loss(stft(estimate_source)[3], stft(source)[3]))
                 #+ 0.5*self.sepcI_loss(stft(estimate_source)[0], stft(source)[0]) + \
                       #0.5*self.sepcR_loss(stft(estimate_source)[1], stft(source)[1])
                # print('sisnr:',batchMean_sisnrLoss(estimate_source, source))
                # print('real:', 0.5*self.sepcI_loss(stft(estimate_source)[0], stft(source)[0]))
                # print('image', 0.5*self.sepcR_loss(stft(estimate_source)[1], stft(source)[1]))
                # loss = self.loss_func(estimate_source, source.squeeze(1)).float()
                # print(loss.dtype)
                sisnr_loss = batchMean_sisnrLoss(estimate_source, source)
                # mse_loss = self.MSE(stft(estimate_source)[0].float(), stft(source)[0].float()).float()
                specR_loss512 = self.sepcR_loss(self.stft512(estimate_source).real, self.stft512(source).real).float()
                specI_loss512 = self.sepcI_loss(self.stft512(estimate_source).imag, self.stft512(source).imag).float()
                specM_loss512 = self.sepcM_loss(torch.abs(self.stft512(estimate_source)), torch.abs(self.stft512(source))).float()

                specR_loss256 = self.sepcR_loss(self.stft256(estimate_source).real, self.stft256(source).real).float()
                specI_loss256 = self.sepcI_loss(self.stft256(estimate_source).imag, self.stft256(source).imag).float()
                specM_loss256 = self.sepcM_loss(torch.abs(self.stft256(estimate_source)), torch.abs(self.stft256(source))).float()

                specR_loss128 = self.sepcR_loss(self.stft128(estimate_source).real, self.stft128(source).real).float()
                specI_loss128 = self.sepcI_loss(self.stft128(estimate_source).imag, self.stft128(source).imag).float()
                specM_loss128 = self.sepcM_loss(torch.abs(self.stft128(estimate_source)), torch.abs(self.stft128(source))).float()

                pw_loss = pw(estimate_source_inter,source_inter)

                # specR_loss64 = self.sepcR_loss(self.stft64(estimate_source)[2], self.stft64(source)[2]).float()
                # specI_loss64 = self.sepcI_loss(self.stft64(estimate_source)[3], self.stft64(source)[3]).float()

                loss = sisnr_loss + 1*(specR_loss512 + specI_loss512 + specR_loss256 + specI_loss256 + specR_loss128 + specI_loss128) + (specM_loss512 + specM_loss256 + specM_loss128) + 0.01*pw_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            # loss = sisnr_loss + specR_loss512 + specI_loss512 + 0.01*pw_loss
            # loss = batchMean_sisnrLoss(estimate_source, source)
            # loss = loss.float()

            group_size = accumulation_group_size(i, len(data_loader), self.accumulation_steps)
            (loss / group_size).backward()
            if (i + 1) % self.accumulation_steps == 0 or i + 1 == len(data_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                              self.max_norm, error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            samples += mixture.shape[0]
            batches += 1

            total_loss += loss.item()
            total_sisnr_loss += sisnr_loss.item()
            # total_mse_loss += mse_loss.item()
            # total_specR_loss += specR_loss.item()
            # total_specI_loss += specI_loss.item()

            # if i % self.print_freq == 0:
            #     self.logger.info('Epoch {0:3d} | Iter {1:5d} | Avg Loss {2:3.3f} | '
            #         'sisnr {3:3.3f} | mse {4:3.3f} | sepcR {5:3.3f} | sepcI {6:3.3f} | {7:5.1f} ms/batch'.format(
            #             epoch + 1, i + 1, total_loss / (i + 1), total_sisnr_loss / (i + 1),  total_mse_loss / (i + 1),
            #             total_specR_loss / (i + 1), total_specI_loss / (i + 1), 1000 * (time.time() - start) / (i + 1)))
            
            if i % self.print_freq == 0:
                self.logger.info('Epoch {0:3d} | Iter {1:5d} | Avg Loss {2:3.3f} | '
                    'sisnr {3:3.3f}  | {4:5.1f} ms/batch'.format(
                        epoch + 1, i + 1, total_loss / (i + 1), total_sisnr_loss / (i + 1), 1000 * (time.time() - start) / (i + 1)))

        torch.cuda.empty_cache()

        if not batches:
            raise ValueError('Training loader produced no optimizer steps')
        self.train_samples = samples
        return total_loss / batches
    
    def _run_valid_epoch(self, epoch):

        start = time.time()
        total_loss = 0
        total_sisnr_loss = 0
        total_mse_loss = 0
        total_specR_loss = 0
        total_specI_loss = 0
        data_loader = self.cv_loader
        total_input_sisnr = total_output_sisnr = 0.0
        metric_count = 0
        valid_samples = 0
        absent_energy_sum = 0.0
        absent_count = 0

        for i, (data) in enumerate(data_loader):

            # if i > 10:
            #     break

            if len(data) == 4:
                mixture, source, emb, lengths = data
                lengths = lengths.to(self.device, dtype=torch.long)
            elif len(data) == 3:
                if self.padding_aware_loss:
                    raise ValueError('padding-aware validation requires valid lengths')
                mixture, source, emb = data
                lengths = None
            else:
                raise ValueError(f'Unexpected validation batch of {len(data)} elements')
            mixture = mixture.to(self.device)
            source = source.to(self.device)
            emb = emb.to(self.device)
            if lengths is not None:
                # Match the inference entry: pad only to an STFT hop boundary.
                max_length = int(lengths.max().item())
                inference_length = max(640, ((max_length + 319) // 320) * 320)
                mixture = mixture[..., :inference_length]
            with precision_context(self.device, self.precision):
                estimate_source = self.model(mixture, emb)
            estimate_source = estimate_source.float()
            if estimate_source.ndim == 3:
                estimate_source = estimate_source.squeeze(1)
            if source.ndim == 3:
                source = source.squeeze(1)
            min_len = min(estimate_source.shape[-1], source.shape[-1])
            estimate_source = estimate_source[:, :min_len]
            source = source[:, :min_len]
            if lengths is not None:
                if bool((lengths > min_len).any()):
                    raise ValueError('Model output shorter than valid audio')
                present = target_present(source, lengths) if self.target_absent_loss else torch.ones_like(lengths, dtype=torch.bool)
                if present.any():
                    out_sisnr = masked_sisnr(estimate_source[present], source[present], lengths[present])
                    reference_mix = mixture[:, 0, :min_len] if mixture.ndim == 3 else mixture[:, 0, 0, :min_len]
                    in_sisnr = masked_sisnr(reference_mix[present], source[present], lengths[present])
                    total_input_sisnr += in_sisnr.sum().item()
                    total_output_sisnr += out_sisnr.sum().item()
                    metric_count += int(present.sum())
                loss = (presence_aware_sisnr_loss(estimate_source, source, lengths)
                        if self.target_absent_loss else -out_sisnr.mean())
                if (~present).any():
                    absent = estimate_source[~present]
                    absent_lengths = lengths[~present]
                    valid = torch.arange(absent.shape[-1], device=absent.device)[None, :] < absent_lengths[:, None]
                    rms = ((absent.square() * valid).sum(-1) / absent_lengths).sqrt().mean()
                    absent_energy_sum += rms.item() * len(absent_lengths)
                    absent_count += len(absent_lengths)
                    self.logger.info('Valid absent target RMS %.6f | samples %d', rms.item(), len(absent_lengths))

            else:
                loss = batchMean_sisnrLoss(estimate_source, source)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite validation loss')
            # sisnr_loss = batchMean_sisnrLoss(estimate_source, source).float()
            # mse_loss = self.MSE(stft(estimate_source)[0].float(), stft(source)[0].float()).float()
            # specR_loss = self.sepcR_loss(stft(estimate_source)[2], stft(source)[2]).float()
            # specI_loss = self.sepcI_loss(stft(estimate_source)[3], stft(source)[3]).float()

            # loss = sisnr_loss + mse_loss + 10*specR_loss + 10*specI_loss
            # loss = self.loss_func(estimate_source[:,:min_len], source[:,:min_len])
            # loss = self.sepc_loss(estimate_source, source_mag)

            total_loss += loss.item() * mixture.shape[0]
            valid_samples += mixture.shape[0]
            # total_sisnr_loss += sisnr_loss.item()
            # total_mse_loss += mse_loss.item()
            # total_specR_loss += specR_loss.item()
            # total_specI_loss += specI_loss.item()

            # if i % self.print_freq == 0:
            #     self.logger.info('Epoch {0:3d} | Iter {1:5d} | Avg Loss {2:3.3f} | '
            #         'sisnr {3:3.3f} | mse {4:3.3f} | sepcR {5:3.3f} | sepcI {6:3.3f} | {7:5.1f} ms/batch'.format(
            #             epoch + 1, i + 1, total_loss / (i + 1), total_sisnr_loss / (i + 1),  total_mse_loss / (i + 1),
            #             total_specR_loss / (i + 1), total_specI_loss / (i + 1), 1000 * (time.time() - start) / (i + 1)))
            

            # total_loss += loss.item()

            if i % self.print_freq == 0:
                self.logger.info('Epoch {0:3d} | Iter {1:5d} | Average Valid Loss {2:3.3f} | '
                    'Current Valid Loss {3:3.6f} | {4:5.1f} ms/batch'.format(
                        epoch + 1, i + 1, total_loss / valid_samples,
                        loss.item(), 1000 * (time.time() - start) / (i + 1)))
        
        if metric_count:
            input_mean = total_input_sisnr / metric_count
            output_mean = total_output_sisnr / metric_count
            self.logger.info('Valid SI-SNR input %.3f dB | output %.3f dB | SI-SNRi %.3f dB | samples %d',
                             input_mean, output_mean, output_mean - input_mean, metric_count)
        torch.cuda.empty_cache()

        if not valid_samples:
            raise ValueError('Validation loader produced no samples')
        self.valid_metrics = dict(active_samples=metric_count, absent_samples=absent_count,
            input_sisnr=total_input_sisnr / metric_count if metric_count else None,
            output_sisnr=total_output_sisnr / metric_count if metric_count else None,
            absent_rms=absent_energy_sum / absent_count if absent_count else None)
        return total_loss / valid_samples
    
    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

            optim_state = self.optimizer.state_dict()
            self.logger.info('epoch start Learning rate: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
            self.logger.info("Training...")
            
            # train stage
            self.model.train()
            start = time.time()
            tr_loss = self._run_train_epoch(epoch)
            train_seconds = time.time() - start
            valid_start = time.time()

            # train log
            self.logger.info('-' * 85)
            self.logger.info('Train Summary | End of Epoch {0:5d} | Time {1:.2f}s | '
                'Train Loss {2:.3f}'.format(
                    epoch + 1, time.time() - start, tr_loss))
            self.logger.info('-' * 85)

            # validation stage
            self.logger.info('Cross validation...')
            
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_valid_epoch(epoch)
                
            # val log
            self.logger.info('-' * 85)
            self.logger.info('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                'Valid Loss {2:.3f}'.format(
                    epoch + 1, time.time() - start, val_loss))
            self.logger.info('-' * 85)

            metrics = dict(epoch=epoch + 1, global_step=self.global_step,
                           train_loss=tr_loss, valid_loss=val_loss,
                           train_seconds=train_seconds, valid_seconds=time.time()-valid_start,
                           train_samples=self.train_samples,
                           samples_per_second=self.train_samples / train_seconds,
                           precision=self.precision, validation=self.valid_metrics, runtime=self.runtime,
                           peak_memory_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0)
            with open(os.path.join(self.save_folder, 'metrics.jsonl'), 'a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            # Save after validation/scheduler so resumed state matches this epoch.
            checkpoint = {
                'epoch': epoch + 1,
                'global_step': self.global_step,
                'best_val_loss': self.best_val_loss,
                'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
                'model_contract': getattr(self.model.module if isinstance(self.model, nn.DataParallel) else self.model, 'checkpoint_contract', None),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'trandom_state': torch.get_rng_state(),
                'nrandom_state': np.random.get_state(),
                'prandom_state': random.getstate(),
                'cuda_random_state_all': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
            }
            if self.checkpoint:
                file_path = os.path.join(self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(checkpoint, file_path)
                self.logger.info('Saving checkpoint model to %s', file_path)
            if improved:
                best_file_path = os.path.join(self.save_folder, 'temp_best.pth.tar')
                torch.save(checkpoint, best_file_path)
                self.logger.info('Find better validated model, saving to %s', best_file_path)
