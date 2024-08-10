# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_video.py

import csv
import json
import os.path

import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL
from decord import VideoReader, cpu

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data'][:10000]
        self.data = self.pro_data(self.data)
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.skip_norm = self.audio_conf.get('skip_norm', False)
        self.audio_errors = 0
        self.video_errors = 0
        self.audio_success = 0
        self.video_success = 0

        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        self.noise = self.audio_conf.get('noise', False)
        if self.noise:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.target_length = self.audio_conf.get('target_length')

        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        self.frame_use = self.audio_conf.get('frame_use', -1)
        self.total_frame = self.audio_conf.get('total_frame', 10)
        print('now use frame {:d} from total {:d} frames'.format(self.frame_use, self.total_frame))

        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])

    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['labels'], data_json[i]['video_id'], data_json[i]['video_path'], str(data_json[i]['bbox']) if 'bbox' in data_json[i] else None]
        data_np = np.array(data_json, dtype=str)
        return data_np


    def decode_data(self, np_data):
        datum = {}
        datum['labels'] = np_data[0]
        datum['video_id'] = np_data[1]
        datum['video_path'] = np_data[2]
        # datum['bbox'] = np_data[4]
        return datum


#Audio: torch.Size([bs, 1024, 128])
#Visual: torch.Size([bs, 3, 8, 224, 224])

    def get_video_frames(self, video_path, num_frames=8):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = []
            if self.mode == 'eval':
                start_frame = max(0, int(self.total_frame / 2) - num_frames // 2)
                frame_indices = [start_frame + i for i in range(num_frames)]
            else:
                start_frame = random.randint(0, total_frames - num_frames)
                frame_indices = [start_frame + i for i in range(num_frames)]

            frames = vr.get_batch(frame_indices).asnumpy()
            frames = [Image.fromarray(frame) for frame in frames]
            frames = [self.preprocess(frame) for frame in frames]

            return torch.stack(frames).permute(1,0,2,3)
        except Exception as e:
            # print(f'Error loading video {video_path}: {e}')
            return torch.zeros((3, num_frames, self.im_res, self.im_res))  # Return a tensor of zeros as a fallback 

    def get_audio(self, video_path):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            audio = vr.get_batch(range(len(vr))).asnumpy()  # Assuming decord can extract audio in this manner
            audio = torch.FloatTensor(audio).mean(dim=1).unsqueeze(0)  # Convert to mono and add channel dimension
            return audio
        except Exception as e:
            # print(f'Error loading audio from {video_path}: {e}')
            return torch.zeros((1, self.target_length))  # Return a tensor of zeros as a fallback

    def _wav2fbank(self, audio_tensor, sr=16000, mix_lambda=-1, audio_tensor2=None):
        if audio_tensor2 is None:
            waveform = audio_tensor - audio_tensor.mean()
        else:
            waveform1 = audio_tensor - audio_tensor.mean()
            waveform2 = audio_tensor2 - audio_tensor2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            # print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            mix_sample_idx = random.randint(0, self.num_samples - 1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            mix_lambda = np.random.beta(10, 10)
            try:
                audio = self.get_audio(datum['video_path'])
                mix_audio = self.get_audio(mix_datum['video_path'])
                fbank = self._wav2fbank(audio, mix_lambda=mix_lambda, audio_tensor2=mix_audio)
                self.audio_success += 1
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                self.audio_errors += 1
            try:
                image = self.get_video_frames(datum['video_path'])
                self.video_success += 1
            except:
                image = torch.zeros((3, 8, self.im_res, self.im_res))
                self.video_errors

            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (1.0 - self.label_smooth)
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (1.0 - self.label_smooth)
            label_indices = torch.FloatTensor(label_indices)

        else:
            datum = self.data[index]
            datum = self.decode_data(datum)
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            try:
                audio = self.get_audio(datum['video_path'])
                fbank = self._wav2fbank(audio)
                self.audio_success += 1
                # print('audio success', self.audio_success)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                self.audio_errors += 1
                # print('audio errors', self.audio_errors)
            try:
                image = self.get_video_frames(datum['video_path'])
                self.video_success += 1
                # print('video success', self.video_success)
            except:
                image = torch.zeros((3, 8, self.im_res, self.im_res))
                self.video_errors += 1
                # print('video errors', self.video_errors)

            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std)

        if self.noise:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)

        return fbank, image, label_indices

    def __len__(self):
        return self.num_samples            