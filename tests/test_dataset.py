import unittest
import torch
import os
from src.data.dataset import HAM10000Dataset
from torchvision import transforms
import numpy as np

class TestHAM10000Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up data needed for all tests"""
        cls.data_dir = "data"
        cls.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def test_dataset_initialization(self):
        """Test if dataset initializes correctly"""
        dataset = HAM10000Dataset(self.data_dir, train=True, transform=self.transform)
        self.assertTrue(len(dataset) > 0)
    
    def test_train_val_split(self):
        """Test if train/val split works correctly"""
        train_dataset = HAM10000Dataset(self.data_dir, train=True)
        val_dataset = HAM10000Dataset(self.data_dir, train=False)
        
        # Check that splits are different
        self.assertNotEqual(len(train_dataset), len(val_dataset))
        self.assertGreater(len(train_dataset), len(val_dataset))
    
    def test_getitem(self):
        """Test if __getitem__ returns correct format"""
        dataset = HAM10000Dataset(self.data_dir, transform=self.transform)
        image, label = dataset[0]
        
        # Check types
        self.assertIsInstance(image, torch.Tensor)
        self.assertTrue(isinstance(label, (int, np.integer, torch.Tensor)))
        
        # If label is a tensor, check its value
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        # Check dimensions
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 7)  # 7 classes
    
    def test_class_weights(self):
        """Test if class weights are calculated correctly"""
        dataset = HAM10000Dataset(self.data_dir)
        weights = dataset.get_class_weights()
        
        self.assertEqual(len(weights), 7)  # 7 classes
        self.assertIsInstance(weights, torch.FloatTensor)

if __name__ == '__main__':
    unittest.main() 