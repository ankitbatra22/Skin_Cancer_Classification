import os
import requests
from tqdm import tqdm

os.makedirs('data/raw', exist_ok=True)

URLS = {
    'part1': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip',
    'part2': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip',
    'metadata': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_LesionGroupings.csv'
}

def download_file(url: str, filename: str):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    print("Downloading HAM10000 dataset files...")
    
    download_file(
        URLS['part1'],
        'data/raw/HAM10000_images_part1.zip'
    )
    
    download_file(
        URLS['part2'],
        'data/raw/HAM10000_images_part2.zip'
    )
    
    download_file(
        URLS['metadata'],
        'data/raw/HAM10000_metadata.csv'
    )

if __name__ == "__main__":
    main() 