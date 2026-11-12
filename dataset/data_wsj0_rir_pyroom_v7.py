import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import librosa
from emb_extract.inference import sv_model_select
from utils.wham_room import WhamRoom
from utils.whamr_utils import quantize, fix_length

# 
RATE = 16000
eps = 1e-6


class tr_wsj_mix_dataset(Dataset):
    def __init__(self, spk2utts_path, embd_path, noise_scp, reverb_param_path, rir_scp, data_list, iter, dur, fs, min_max):

        self.noise = {x.split()[0] : x.split()[1] for x in open(noise_scp)}
        # self.rir = {x.split()[0] : x.split()[1] for x in open(rir_scp)}
        spk2utts = {}
        for dataset in data_list:
            spk2utts_data = torch.load(spk2utts_path[dataset], map_location="cpu", weights_only=False)
            spk2utts[dataset] = spk2utts_data
        self.spk2utts = spk2utts
        spk2embd = {}
        for dataset in data_list:
            spk2embd_data = torch.load(embd_path[dataset], map_location="cpu", weights_only=False)
            spk2embd[dataset] = spk2embd_data
        self.spk2embd = spk2embd
        self.data_list = data_list

        self.reverb_param_df = pd.read_csv(reverb_param_path)
        

        self.tlen = dur * fs
        
        self.fs = fs
        self.min_max = min_max
        self.len = iter
    
    def _trun_wav(self, y, tlen, offset=0):
        if y.shape[0] < tlen:
            npad = tlen - y.shape[0]
            y = np.pad(y, (0, npad), mode='constant', constant_values=0)
        else:
            y = y[offset:offset+tlen]
        return y
    
    def _norm_amplitude(self, y, scalar = None, eps=1e-6):
        if not scalar:
            scalar = np.max(np.abs(y)) + eps

        return y / scalar, scalar
    
    def _tailor_dB_FS(self, y, target_dB_FS=-25, eps=1e-6):
        rms = np.sqrt(np.mean(y**2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y, rms, scalar

    def _extend_noise(self, noise, max_length):
        noise_ex = noise
        window = np.hanning(RATE + 1)
        # Increasing window
        i_w = window[:len(window) // 2 + 1]
        # Decreasing window
        d_w = window[len(window) // 2::-1]
        # Extend until max_length is reached
        while len(noise_ex) < max_length:
            noise_ex = np.concatenate((noise_ex[:len(noise_ex) - len(d_w)],
                                    np.multiply(
                                        noise_ex[len(noise_ex) - len(d_w):],
                                        d_w) + np.multiply(
                                        noise[:len(i_w)], i_w),
                                    noise[len(i_w):]))
        noise_ex = noise_ex[:max_length]
        return noise_ex
    
    def _check_valid_audio(self, audio):
        """检查音频是否有效"""
        if audio is None:
            return False
        if np.all(audio == 0):
            print("Audio is all zeros.")
            return False
        if np.isnan(audio).any():
            print("Audio contains NaN values.")
            return False
        if np.isinf(audio).any():
            print("Audio contains Inf values.")
            return False
        return True
    
    def __getitem__(self, sample_idx):
        while True:
            if isinstance(sample_idx, int):
                index, tlen = sample_idx, None
            elif len(sample_idx) == 2:
                index, tlen = sample_idx
            else:
                raise AssertionError
            
            dataset = random.sample(self.data_list, 1)[0]
            spk_tar, spk_inter = random.sample(list(self.spk2embd[dataset].keys()), 2)
            tar_wav_path1, target_wav_path2, tar_wav_path3, tar_aux_wav_path = random.sample(list(self.spk2embd[dataset][spk_tar].keys()), 4)
            inter_wav_path1, inter_wav_path2, inter_wav_path3,inter_aux_wav_path = random.sample(list(self.spk2embd[dataset][spk_inter].keys()), 4)
            noise_id1,noise_id2,noise_id3 = random.sample(list(self.noise.keys()), 3)
            tar_embd = self.spk2embd[dataset][spk_tar][tar_aux_wav_path]
            inter_embd = self.spk2embd[dataset][spk_inter][inter_aux_wav_path]
            
            
            # aux = librosa.load(aux_wav_path, sr = self.fs)[0]
            # embd = self.sv_model.embd_extract(aux)

            tar = []

            for file in [target_wav_path2, tar_wav_path1, tar_wav_path3]:
                audio = librosa.load(file, sr = self.fs)[0]
                tar.append(audio)
            tar = np.concatenate(tar)

            # vad_sq = np.ones(len(tar))
            
            inter = []
            for file in [inter_wav_path1, inter_wav_path2, inter_wav_path3]:
                audio = librosa.load(file, sr = self.fs)[0]
                inter.append(audio)
            inter = np.concatenate(inter)

            noise = []
            for file in [self.noise[noise_id1], self.noise[noise_id2], self.noise[noise_id3]]:
                audio = librosa.load(file, sr = self.fs)[0]
                noise.append(audio)
            noise = np.concatenate(noise)

            offset = random.randint(0, max(len(tar) - self.tlen, 0)) 
            if len(tar) < self.tlen:
                tar_vad_sq = np.ones(len(tar))
                tar_vad_sq = np.pad(tar_vad_sq, (0, self.tlen-len(tar)), mode='constant', constant_values=0)
            else:
                tar_vad_sq = np.ones(self.tlen)

            tar = self._trun_wav(tar, self.tlen, offset)

            offset = random.randint(0, max(len(inter) - self.tlen, 0))
            if len(inter) < self.tlen:
                inter_vad_sq = np.ones(len(inter))
                inter_vad_sq = np.pad(inter_vad_sq, (0, self.tlen-len(inter)), mode='constant', constant_values=0)
            else:
                inter_vad_sq = np.ones(self.tlen)
            inter = self._trun_wav(inter, self.tlen, offset)

            offset = random.randint(0, max(len(noise) - self.tlen, 0))
            noise = self._trun_wav(noise, self.tlen, offset)

            # vad_sq = np.ones(len(tar))

            # pad = np.zeros(int(0.5*16000))

            num_segments = random.randint(0,4)
            min_length = int(0.5 * self.fs)
            max_length = int(1.5 * self.fs)

            for _ in range(num_segments):
                start = np.random.randint(0, len(tar) - max_length)
                length = np.random.randint(min_length, max_length)
                tar[start:start + length] = 0
                tar_vad_sq[start:start + length] = 0

            num_segments = random.randint(0,4)
            min_length = int(0.5 * self.fs)
            max_length = int(1.5 * self.fs)

            for _ in range(num_segments):
                start = np.random.randint(0, len(inter) - max_length)
                length = np.random.randint(min_length, max_length)
                inter[start:start + length] = 0
                inter_vad_sq[start:start + length] = 0

            row_index = random.randint(0,len(self.reverb_param_df)-1)

            param_row = self.reverb_param_df.iloc[row_index]
            room = WhamRoom([param_row['room_x'], param_row['room_y'], param_row['room_z']],
                            [[param_row['micL_x']+0.0575, param_row['micL_y']+0.162, param_row['mic_z']],
                                # [param_row['micR_x'], param_row['micR_y'], param_row['mic_z']]],
                            [param_row['micL_x']+0.024, param_row['micL_y'], param_row['mic_z']]],
                            [param_row['s1_x'], param_row['s1_y'], param_row['s1_z']],
                            [param_row['s2_x'], param_row['s2_y'], param_row['s2_z']],
                            param_row['T60'])
            
            room.generate_rirs()

            s1_temp, _ = self._norm_amplitude(tar)
            # # dbf_s1 = random.randint(-25, -15)
            s1_temp, _, _ = self._tailor_dB_FS(s1_temp, -25)
            s1_temp_rms = (tar**2).mean() ** 0.5

            s2_temp, _ = self._norm_amplitude(inter)
            # # dbf_s2 = random.randint(-25, -15)
            dbf = random.randint(-35, -15)
            s2_temp, _, _ = self._tailor_dB_FS(s2_temp, dbf)
            s2_temp_rms = (inter**2).mean() ** 0.5

            noise_temp, _ = self._norm_amplitude(noise)
            dbf = random.randint(-35, -15)
            noise_temp, _, _ = self._tailor_dB_FS(noise_temp, dbf)
            noise_temp_rms = (noise**2).mean() ** 0.5

            snr_spk = round(random.uniform(0,2.5),2)
            snr_ns = round(random.uniform(-5,0),2)

            s2_snr = s1_temp_rms / (10 ** (snr_spk / 20)) / (s2_temp_rms + 1e-6)
            snr_ns = s1_temp_rms / (10 ** (snr_ns / 20)) / (noise_temp_rms + 1e-6)

            # # s1_snr = 10 ** (float(snr_spk)/20)
            # # s2_snr = 10 ** (float(-snr_spk)/20)
            # # snr_ns = 10 ** (float(snr_ns)/20)

            scale_s1 = tar * 1.0
            scale_s2 = inter * s2_snr
            scale_ns = noise * snr_ns
            # scale_s1 = tar
            # scale_s2 = inter

            room.add_audio(scale_s1, scale_s2)

            anechoic = room.generate_audio(anechoic=True, fs=self.fs)
            reverberant = room.generate_audio(fs=self.fs)

            audio_len = min(len(tar), len(inter))

            s1_spatial_scaling_mic1 = np.sqrt(np.sum(scale_s1 ** 2) / (np.sum(anechoic[0, 0, :] ** 2)+1e-6))
            s2_spatial_scaling_mic1 = np.sqrt(np.sum(scale_s2 ** 2) / (np.sum(anechoic[1, 0, :] ** 2)+1e-6))
            s1_spatial_scaling_mic2 = np.sqrt(np.sum(scale_s1 ** 2) / (np.sum(anechoic[0, 1, :] ** 2)+1e-6))
            s2_spatial_scaling_mic2 = np.sqrt(np.sum(scale_s2 ** 2) / (np.sum(anechoic[1, 1, :] ** 2)+1e-6))
            

            s1_anechoic_mic1,s2_anechoic_mic1   = fix_length(anechoic[0, 0, :audio_len].T * s1_spatial_scaling_mic1,
                                                            anechoic[1, 0, :audio_len].T * s2_spatial_scaling_mic1,
                                                            self.min_max)
            s1_anechoic_mic2,s2_anechoic_mic2   = fix_length(anechoic[0, 1, :audio_len].T * s1_spatial_scaling_mic2,
                                                            anechoic[1, 1, :audio_len].T * s2_spatial_scaling_mic2,
                                                            self.min_max)
                
            s1_reverb_mic1,s2_reverb_mic1= fix_length(reverberant[0, 0, :audio_len].T * s1_spatial_scaling_mic1,
                                                        reverberant[1, 0, :audio_len].T * s2_spatial_scaling_mic1,
                                                        self.min_max)
            s1_reverb_mic2,s2_reverb_mic2= fix_length(reverberant[0, 1, :audio_len].T * s1_spatial_scaling_mic2,
                                                        reverberant[1, 1, :audio_len].T * s2_spatial_scaling_mic2,
                                                        self.min_max)
            mix_2spk_both_mic1 = s1_reverb_mic1[:audio_len] + s2_reverb_mic1[:audio_len] + scale_ns[:audio_len]
            mix_2spk_both_mic2 = s1_reverb_mic2[:audio_len] + s2_reverb_mic2[:audio_len] + scale_ns[:audio_len]

            mix_2spk_noise_mic1 = s1_anechoic_mic1[:audio_len] + s2_anechoic_mic1[:audio_len] + scale_ns[:audio_len]
            mix_2spk_noise_mic2 = s1_anechoic_mic2[:audio_len] + s2_anechoic_mic2[:audio_len] + scale_ns[:audio_len]

            # clip_s = max(np.max(np.abs(mix_2spk_both_mic1)), np.max(np.abs(mix_2spk_both_mic2)), np.max(np.abs(mix_2spk_noise_mic1)), np.max(np.abs(mix_2spk_noise_mic2))) / (0.99-1e-6)
            clip_s = max(np.max(np.abs(mix_2spk_both_mic1)), np.max(np.abs(mix_2spk_both_mic2))) / (0.99-1e-6)
            # print(mix_2spk_noise_mic1.shape)

            mix_2spk_both = np.stack([mix_2spk_both_mic2 / clip_s, mix_2spk_both_mic1 / clip_s], axis=0)
            mix_2spk_noise = np.stack([mix_2spk_noise_mic2 / clip_s, mix_2spk_noise_mic1 / clip_s], axis=0)
            
            mix_both = mix_2spk_both
            tgt_reverb_mic2 = s1_reverb_mic2 / clip_s
            inter_reverb_mic2 = s2_reverb_mic2 / clip_s

            if not self._check_valid_audio(mix_both) or not self._check_valid_audio(tgt_reverb_mic2) or not self._check_valid_audio(inter_reverb_mic2):
                print(f"Warning: Found all-zero audio at index {sample_idx}. Skipping...")
                sample_idx = (sample_idx + 1) % len(self)
                continue
            # mix_single = mix_2spk_noise
            # tgt_anechoic_mic1 = s1_anechoic_mic1 / (clip_s+1e-6)
            # tgt_reverb_mic1 = s1_reverb_mic1 / (clip_s+1e-6)
            # tgt_anechoic_mic2 = s1_anechoic_mic2 / (clip_s+1e-6)
            type = random.sample(['r', 'a'],1)[0]
            if type == 'a':
                mix_both = mix_2spk_noise
                tgt_reverb_mic2 = s1_anechoic_mic2 / clip_s
                inter_reverb_mic2 = s2_anechoic_mic2 / clip_s
            elif type == 'r':
                mix_both = mix_2spk_both
                tgt_reverb_mic2 = s1_reverb_mic2 / clip_s
                inter_reverb_mic2 = s2_reverb_mic2 / clip_s
            # tgt_reverb_mic2 = s1_reverb_mic2 / clip_s

            # inter_anechoic_mic1 = s1_anechoic_mic1 / (clip_s+1e-6)
            # inter_reverb_mic1 = s1_reverb_mic1 / (clip_s+1e-6)
            # inter_anechoic_mic2 = s2_anechoic_mic2 / (clip_s+1e-6)
            # inter_reverb_mic2 = s2_reverb_mic2 / clip_s

            # offset = random.randint(0, max(len(aux) - self.tlen, 0))
            # aux_wav = self._trun_wav(aux, self.tlen, offset)


            mix_wav = torch.from_numpy(mix_both) # 1,T
            mix_wav = mix_wav.to(torch.float32)
            target_wav = torch.from_numpy(tgt_reverb_mic2)
            inter_wav = torch.from_numpy(inter_reverb_mic2)

            # mix_wav = torch.from_numpy(mix_single) # 1,T
            # mix_wav = mix_wav.to(torch.float32)
            # target_wav = torch.from_numpy(tgt_anechoic_mic2)
            tar_vad_sq = torch.from_numpy(tar_vad_sq)
            inter_vad_sq = torch.from_numpy(inter_vad_sq)

            # mix_wav = torch.from_numpy(mix_both) # 1,T
            # mix_wav = mix_wav.to(torch.float32)
            # target_wav = torch.from_numpy(tgt_anechoic_mic2)

            # del mix,tgt,s1_reverb,mix_noise_reverb,mix_noise,scale_s1,scale_s2,s1_temp,s2_temp,rir,inter,noise,tar

            return mix_wav.unsqueeze(0), target_wav, tar_embd, tar_vad_sq, inter_wav, inter_embd, inter_vad_sq
    
    def __len__(self):
        return self.len

class te_dataset(Dataset):
    def __init__(self, mix_scp, ref_scp, aux_scp, embd_path, data_list, fs):
        self.mix = {x.split()[0]:x.split()[1] for x in open(mix_scp)}
        self.ref = {x.split()[0]:x.split()[1] for x in open(ref_scp)}
        self.aux = {x.split()[0]:x.split()[1] for x in open(aux_scp)}
        assert len(self.mix) == len(self.ref)

        self.sv_model = sv_model_select('ResNet34_yk').sv_model

        # spk2embd = {}
        # for dataset in data_list:
        # spk2embd = torch.load(embd_path[data_list[-1]])
        # spk2embd[dataset] = spk2embd_data
        # self.spk2embd = spk2embd
        # self.sv_model = sv_model_select('ResNetSE34_emb256').sv_model
        # self.spk2utt = torch.load(emb_path)
        # self.classifier = EncoderClassifier.from_hparams(source="/DKUdata/2021/zb/project/oppo/TF_TSE/emb_extract/spkrec-ecapa-voxceleb")

        mix_wav_id = []
        for l in open(mix_scp):
            mix_wav_id.append(l.split()[0])
        
        aux_wav_id = []
        for l in open(aux_scp):
            aux_wav_id.append(l.split()[0])
         
        self.mix_wav_id = mix_wav_id
        self.aux_wav_id = aux_wav_id
        # self.tlen = dur * fs
        
        self.fs = fs
        self.len = len(self.mix)
    
    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            index, tlen = sample_idx, None
        elif len(sample_idx) == 2:
            index, tlen = sample_idx
        else:
            raise AssertionError
                
        mix_utt = self.mix_wav_id[index]
        aux_utt = self.aux_wav_id[index]

        mix_wav_path = self.mix[mix_utt]
        target_wav_path = self.ref[mix_utt]
        aux_wav_path = self.aux[aux_utt]

        spk = aux_wav_path.split('/')[-2]

        # emb = self.spk2embd[spk][aux_wav_path]

        mix_wav, _ = librosa.load(mix_wav_path, sr=self.fs, mono=False)
        mix_wav = mix_wav[0:2, :]
        target_wav, _ = librosa.load(target_wav_path, sr=self.fs)
        aux_wav, _ = librosa.load(aux_wav_path, sr=self.fs)

        mix_wav = torch.from_numpy(mix_wav).unsqueeze(0) # 1,T
        target_wav = torch.from_numpy(target_wav)
        embd = self.sv_model.embd_extract(aux_wav)
        # aux_wav = torch.from_numpy(aux_wav)

        return mix_wav, target_wav, embd
    
    def __len__(self):
        return self.len