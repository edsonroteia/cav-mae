# Import necessary libraries
import json
import numpy as np
import torch
import torchaudio
import torchvision.transforms as T
from PIL import Image
import webdataset as wds
from torch.utils.data import IterableDataset
import csv
from decord import VideoReader, cpu
import io

class AudiosetDataset:
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Initialize the dataset class with configuration settings.
        """
        # Read dataset configuration from JSON file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        self.label_smooth = audio_conf.get('label_smooth', 0.0)
        self.melbins = audio_conf.get('num_mel_bins')
        self.freqm = audio_conf.get('freqm', 0)
        self.timem = audio_conf.get('timem', 0)
        self.mixup = audio_conf.get('mixup', 0)
        self.norm_mean = audio_conf.get('mean')
        self.norm_std = audio_conf.get('std')
        self.noise = audio_conf.get('noise', False)
        self.index_dict = self.make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        self.target_length = audio_conf.get('target_length')
        self.mode = audio_conf.get('mode')
        self.frame_use = audio_conf.get('frame_use', -1)
        self.total_frame = audio_conf.get('total_frame', 10)
        self.im_res = audio_conf.get('im_res', 224)

        # Setup image preprocessing steps
        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def make_index_dict(self, label_csv):
        """Helper function to create an index lookup from CSV."""
        index_lookup = {}
        if label_csv:
            with open(label_csv, 'r') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    index_lookup[row['mid']] = row['index']
        return index_lookup

    def _wav2fbank(self, waveform):
        """Convert waveform to Mel spectrogram."""
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, 
            htk_compat=True, 
            sample_frequency=16000, 
            use_energy=False, 
            window_type='hanning', 
            num_mel_bins=self.melbins, 
            dither=0.0, 
            frame_shift=10
        )
        
        # Padding or cutting the spectrogram to fit target length
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames

        if p > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
        elif p < 0:
            fbank = fbank[:self.target_length, :]

        return fbank

    def get_image(self, filenames):
        """Get and process images from file paths."""
        images = []
        for filename in filenames:
            if filename:
                img = Image.open(filename)
                img_tensor = self.preprocess(img)
                images.append(img_tensor)
            else:
                images.append(torch.zeros(3, self.im_res, self.im_res))

        return torch.stack(images)

    def process_item(self, item):
        """
        Process a single item from the dataset.
        """
        # Decompress the tar file to access its contents
        import pdb ; pdb.set_trace()
        datum = json.loads(item)

        # Handle video file processing
        video_bytes = datum['mkv']
        video_stream = io.BytesIO(video_bytes)
        vr = VideoReader(video_stream, ctx=cpu())


        waveform, _ = torchaudio.load(datum['video_path'])

        # Convert waveform to Mel spectrogram
        fbank = self._wav2fbank(waveform)

        # Get images from frames
        image_paths = self.randselect_img(datum['video_id'], datum['video_path'], num_video_frames=8)
        images = self.get_image(image_paths)

        # Prepare labels
        label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
        if datum['labels']:
            for label in datum['labels'].split(','):
                label_indices[int(self.index_dict[label])] = 1 - self.label_smooth

        # Normalize spectrogram
        fbank = (fbank - self.norm_mean) / self.norm_std

        # Apply SpecAug for training
        if self.mode == "train":
            fbank = torchaudio.transforms.FrequencyMasking(self.freqm)(fbank)
            fbank = torchaudio.transforms.TimeMasking(self.timem)(fbank)

        return fbank, images, torch.FloatTensor(label_indices)

    def create_dataset(self):
        """
        Create and return a WebDataset object.
        """
        print(self.data[0].keys())
        tar_paths = [item['tar_file'] for item in self.data] #list of the tar paths
        return wds.WebDataset(tar_paths).decode(wds.torch_video).map(self.process_item)
