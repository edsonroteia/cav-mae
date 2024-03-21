import os
import torch

# Assuming 'dataloaderv2.py' is the filename where the provided dataloader code resides.
from dataloaderv2 import AudiosetDataset

audio_conf = {
    'model': 'cav-mae',
    'masking_ratio': 0.75,
    'mask_mode': 'unstructured',  # Can be 'time', 'freq', or 'tf'
    'contrast_loss_weight': 0.01,
    'mae_loss_weight': 1.0,
    'tr_pos': False,
    'norm_pix_loss': True,
    'bal': None,
    'lr': 5e-5,
    'epoch': 25,
    'lrscheduler_start': 10,
    'lrscheduler_decay': 0.5,
    'lrscheduler_step': 5,
    'dataset_mean': -5.081,
    'dataset_std': 4.4849,
    'target_length': 1024,
    'noise': True,
    'mixup': 0.0,
    'batch_size': 36,
    'lr_adapt': False,
    'dataset': 'audioset',
    'label_csv': '/home/edson/code/cav-mae/datafilles/class_labels_indices.csv',
    'mode': 'train',  # Assuming training mode for the Python script
}

new_audio_conf = {'num_mel_bins': 128, 'target_length': audio_conf['target_length'], 'freqm': 0, 'timem': 0, 'mixup': audio_conf['mixup'], 'dataset': audio_conf['dataset'], 'mode':'train', 'mean':audio_conf['dataset_mean'], 'std':audio_conf['dataset_std'],
              'noise':0.5, 'label_smooth': 0, 'im_res': 1024}

audio_conf.update(new_audio_conf)

print(audio_conf)

# Path to your dataset's JSON file.
dataset_json_file = '/home/edson/code/cav-mae/datafilles/audioset_20k_cleaned_newval.json'  # Replace with your JSON file path

# Create an instance of the dataset.
dataset = AudiosetDataset(dataset_json_file, audio_conf, audio_conf['label_csv'])

# Fetch a single item to inspect.
item_index = 0  # Change this index to inspect different items.
fbank, image, label_indices = dataset[item_index]

# Check and print the shapes and properties.
print('Fbank Shape:', fbank.shape)
print('Image Shape:', image.shape)
print('Label Indices Shape:', label_indices.shape)
print('Fbank Type:', type(fbank))
print('Image Type:', type(image))
print('Label Indices Type:', type(label_indices))

# Additional properties can be checked based on specific requirements.
