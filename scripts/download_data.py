import os
import zipfile
import requests
from tqdm import tqdm

# Create directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/images', exist_ok=True)

URLS = {
    'part1': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip',
    'part2': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip',
    'metadata': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_LesionGroupings.csv'
}

def download_file(url: str, filename: str):
    """Download a file with a progress bar."""
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

def extract_zip(file_path: str, extract_to: str):
    """Extract a ZIP file."""
    print(f"Extracting {file_path}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted {file_path} to {extract_to}")

def main():
    print("📥 Downloading HAM10000 dataset files...")
    
    # Download files
    download_file(URLS['part1'], 'data/raw/HAM10000_images_part1.zip')
    download_file(URLS['part2'], 'data/raw/HAM10000_images_part2.zip')
    download_file(URLS['metadata'], 'data/raw/HAM10000_metadata.csv')
    print("✅ Download complete!")
    
    # Extract ZIP files
    print("🔧 Preparing dataset...")
    extract_zip('data/raw/HAM10000_images_part1.zip', 'data/images')
    extract_zip('data/raw/HAM10000_images_part2.zip', 'data/images')
    print("✅ Data preparation complete!")

if __name__ == "__main__":
    main()
