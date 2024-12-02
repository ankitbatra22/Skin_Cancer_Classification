import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np

class HAM10000Dataset(Dataset):
    """
    HAM10000 Dataset
    
    Classes:
    - MEL: Melanoma
    - NV: Melanocytic nevus
    - BCC: Basal cell carcinoma
    - AKIEC: Actinic keratosis / Bowen's disease (intraepithelial carcinoma)
    - BKL: Benign keratosis
    - DF: Dermatofibroma
    - VASC: Vascular lesion
    """
    
    def __init__(self, root_dir, train=True, transform=None, oversample_rates=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load ground truth
        groundtruth_path = os.path.join(
            root_dir, 
            "images/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
        )
        self.df = pd.read_csv(groundtruth_path)
        
        # Define classes
        self.classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        
        # Create train/val split
        np.random.seed(42)
        indices = np.random.permutation(len(self.df))
        split = int(0.8 * len(indices))
        
        if train:
            self.df = self.df.iloc[indices[:split]]
            if oversample_rates:
                self._oversample(oversample_rates)
        else:
            self.df = self.df.iloc[indices[split:]]
            
    def _oversample(self, oversample_rates):
        """Oversample classes based on the given rates."""
        for i, rate in enumerate(oversample_rates):
            if rate > 1:
                class_df = self.df[self.df[self.classes[i]] == 1]
                oversampled_class = class_df.sample(
                    n=int(len(class_df) * (rate - 1)),
                    replace=True,
                    random_state=42
                )
                self.df = pd.concat([self.df, oversampled_class], ignore_index=True)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(
            self.root_dir, 
            "images/ISIC2018_Task3_Training_Input",
            f"{img_name}.jpg"
        )
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Get label (convert one-hot to class index)
        label = np.argmax(self.df.iloc[idx][self.classes].values)
        
        return image, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        class_counts = self.df[self.classes].sum()
        total = class_counts.sum()
        weights = total / (len(self.classes) * class_counts)
        return torch.FloatTensor(weights.values)
