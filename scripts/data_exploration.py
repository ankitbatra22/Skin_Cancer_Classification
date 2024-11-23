import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from pathlib import Path

def explore_all_files(data_dir: str):
    """List important files in the data directory"""
    print("\n=== File Structure Exploration ===")
    
    for root, dirs, files in os.walk(data_dir):
        # Skip the large input directory listing
        if "Training_Input" in root:
            num_files = len([f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"    Training_Input/: ({num_files} images)")
            continue
            
        level = root.replace(data_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if f.endswith('.csv'):  # Only show CSV files
                print(f"{subindent}{f}")

def explore_metadata_and_groundtruth(data_dir: str):
    """Explore both metadata and ground truth files"""
    print("\n=== Data Files Exploration ===")
    
    # Original metadata
    metadata_path = os.path.join(data_dir, "metadata.csv")
    metadata_df = pd.read_csv(metadata_path)
    print("\nMetadata file info:")
    print(metadata_df.head())
    print("\nMetadata columns:", metadata_df.columns.tolist())
    
    # Look for ground truth file
    groundtruth_dir = os.path.join(data_dir, "images", "ISIC2018_Task3_Training_GroundTruth")
    for file in os.listdir(groundtruth_dir):
        if file.endswith('.csv'):
            print(f"\nFound ground truth file: {file}")
            groundtruth_path = os.path.join(groundtruth_dir, file)
            groundtruth_df = pd.read_csv(groundtruth_path)
            print("\nGround truth columns:", groundtruth_df.columns.tolist())
            print("\nGround truth sample:")
            print(groundtruth_df.head())
            return metadata_df, groundtruth_df
    
    return metadata_df, None

def explore_images(data_dir: str):
    """Explore image characteristics"""
    print("\n=== Image Exploration ===")
    
    image_dir = os.path.join(data_dir, "images", "ISIC2018_Task3_Training_Input")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nTotal images: {len(image_files)}")
    
    # Analyze just one sample image
    if image_files:
        sample_path = os.path.join(image_dir, image_files[0])
        sample_img = Image.open(sample_path)
        print(f"Image dimensions: {sample_img.size}")
        print(f"Image mode: {sample_img.mode}")

def main():
    data_dir = "data"
    print("Starting data exploration...")
    
    # List important files
    explore_all_files(data_dir)
    
    # Explore metadata and ground truth
    metadata_df, groundtruth_df = explore_metadata_and_groundtruth(data_dir)
    
    # Basic image info
    explore_images(data_dir)
    
    print("\nExploration complete!")

if __name__ == "__main__":
    main()