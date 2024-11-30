import torch
from torch.utils.data import Sampler
import numpy as np

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, num_classes=7, samples_per_class=None):
        self.dataset = dataset
        self.num_classes = num_classes
        
        # Get labels array
        self.labels = np.array([
            dataset[i][1] for i in range(len(dataset))
        ])
        
        # Store indices for each class
        self.class_indices = [
            np.where(self.labels == i)[0] for i in range(self.num_classes)
        ]
        
        # Allow custom samples per class or use max class size
        if samples_per_class is None:
            self.samples_per_class = max(len(indices) for indices in self.class_indices)
        else:
            self.samples_per_class = samples_per_class
        
    def __iter__(self):
        indices = []
        # Oversample minority classes
        for class_indices in self.class_indices:
            indices.extend(
                np.random.choice(
                    class_indices,
                    size=self.samples_per_class,
                    replace=True
                )
            )
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.samples_per_class * self.num_classes 