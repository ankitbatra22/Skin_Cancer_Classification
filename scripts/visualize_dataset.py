import matplotlib.pyplot as plt
import numpy as np
from src.data.dataset import HAM10000Dataset
from pathlib import Path
import os
from PIL import Image

def plot_class_distribution(dataset):
    """Plot distribution of classes"""
    class_counts = dataset.df[dataset.classes].sum()
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(dataset.classes)), class_counts.values)
    plt.xticks(range(len(dataset.classes)), dataset.classes, rotation=45)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig('outputs/class_distribution.png')
    plt.show()
    plt.close()

def plot_sample_images(dataset):
    """Plot sample images from each class"""
    n_samples = 5
    fig, axes = plt.subplots(len(dataset.classes), n_samples, 
                            figsize=(15, 3*len(dataset.classes)))
    
    for idx, class_name in enumerate(dataset.classes):
        # Get indices for this class
        class_indices = np.where(dataset.df[class_name] == 1)[0]
        sample_indices = np.random.choice(class_indices, n_samples, replace=False)
        
        for j, sample_idx in enumerate(sample_indices):
            img_name = dataset.df.iloc[sample_idx]['image']
            img_path = os.path.join(dataset.root_dir, 
                                  "images/ISIC2018_Task3_Training_Input",
                                  f"{img_name}.jpg")
            img = Image.open(img_path)
            axes[idx, j].imshow(img)
            axes[idx, j].axis('off')
            
            if j == 0:  # Add class label on the left
                axes[idx, j].set_ylabel(class_name, rotation=45, 
                                      labelpad=20, fontsize=10)
    
    plt.suptitle('Sample Images from Each Class', y=1.02, fontsize=12)
    plt.tight_layout()
    #plt.savefig('outputs/sample_images.png', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_image_dimensions(dataset):
    """Plot histogram of image dimensions"""
    widths = []
    heights = []
    
    # Sample 1000 random images for dimension analysis
    sample_indices = np.random.choice(len(dataset), min(1000, len(dataset)), replace=False)
    
    for idx in sample_indices:
        img_name = dataset.df.iloc[idx]['image']
        img_path = os.path.join(dataset.root_dir, 
                              "images/ISIC2018_Task3_Training_Input",
                              f"{img_name}.jpg")
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20, edgecolor='black')
    plt.title('Image Widths Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, edgecolor='black')
    plt.title('Image Heights Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    #plt.savefig('outputs/image_dimensions.png')
    plt.show()
    plt.close()

def main():
    # Reference to dataset class
    """
    startLine: 8
    endLine: 72
    """
    dataset = HAM10000Dataset("data")
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Generate visualizations
    plot_class_distribution(dataset)
    plot_sample_images(dataset)
    plot_image_dimensions(dataset)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Total images: {len(dataset)}")
    print("\nClass distribution:")
    class_counts = dataset.df[dataset.classes].sum()
    for class_name, count in class_counts.items():
        print(f"{class_name:6s}: {int(count):5d} images ({count/len(dataset)*100:6.2f}%)")

if __name__ == "__main__":
    main() 