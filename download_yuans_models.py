import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def main():
    models = {
        'cav-mae-scale++.pth': 'https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1',
        'cav-mae-scale+.pth': 'https://www.dropbox.com/s/xu8bfie6hz86oev/audio_model.25.pth?dl=1',
        'cav-mae.pth': 'https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1'
    }

    for model_name, url in models.items():
        print(f"Downloading {model_name}...")
        download_file(url, model_name)
        print(f"{model_name} downloaded successfully.")

if __name__ == "__main__":
    main()

