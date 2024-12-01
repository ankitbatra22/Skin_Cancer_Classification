import unittest
import torch
import torch.nn as nn
from src.training.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up data needed for all tests"""
        # Create dummy data
        cls.X_train = torch.randn(100, 10)
        cls.y_train = torch.randint(0, 2, (100,))
        cls.X_val = torch.randn(20, 10)
        cls.y_val = torch.randint(0, 2, (20,))
        
        # Create dataloaders
        cls.train_loader = DataLoader(
            TensorDataset(cls.X_train, cls.y_train),
            batch_size=16
        )
        cls.val_loader = DataLoader(
            TensorDataset(cls.X_val, cls.y_val),
            batch_size=16
        )
        
        # Create config
        cls.config = {
            'training': {
                'epochs': 2,
                'criterion': {
                    'name': 'CrossEntropyLoss',
                    'params': {}
                },
                'optimizer': {
                    'name': 'Adam',
                    'params': {
                        'lr': 0.001
                    }
                }
            }
        }
    
    def test_trainer_initialization(self):
        """Test if trainer initializes correctly"""
        trainer = Trainer(self.config)
        self.assertIsNotNone(trainer.criterion)
        self.assertIsInstance(trainer.criterion, nn.CrossEntropyLoss)
    
    def test_training_loop(self):
        """Test if training runs without errors"""
        trainer = Trainer(self.config)
        model = DummyModel()
        
        try:
            trained_model = trainer.train(
                model, 
                self.train_loader, 
                self.val_loader
            )
            self.assertIsInstance(trained_model, nn.Module)
        except Exception as e:
            self.fail(f"Training failed with error: {str(e)}")
    
    def test_metrics_tracking(self):
        """Test if metrics are properly tracked"""
        trainer = Trainer(self.config)
        model = DummyModel()
        
        trainer.train(model, self.train_loader, self.val_loader)
        
        self.assertEqual(len(trainer.train_losses), 2)  # 2 epochs
        self.assertEqual(len(trainer.val_losses), 2)
        self.assertEqual(len(trainer.val_accuracies), 2)

if __name__ == '__main__':
    unittest.main() 