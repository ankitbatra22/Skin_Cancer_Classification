import unittest
import torch
import torch.nn as nn
from src.training.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from pathlib import Path

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
        
        # Create base config WITHOUT scheduler
        cls.base_config = {
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
        
        # Create separate config WITH scheduler
        cls.scheduler_config = {
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
                },
                'scheduler': {
                    'name': 'CosineAnnealingWarmRestarts',
                    'params': {
                        'T_0': 10,
                        'eta_min': 1e-6
                    }
                }
            }
        }
    
    def test_trainer_initialization(self):
        """Test if trainer initializes correctly"""
        trainer = Trainer(self.base_config)
        self.assertIsNotNone(trainer.criterion)
        self.assertIsInstance(trainer.criterion, nn.CrossEntropyLoss)
    
    def test_training_loop(self):
        """Test if training runs without errors"""
        trainer = Trainer(self.base_config)
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
        trainer = Trainer(self.base_config)
        model = DummyModel()
        
        trainer.train(model, self.train_loader, self.val_loader)
        
        self.assertEqual(len(trainer.train_losses), 2)  # 2 epochs
        self.assertEqual(len(trainer.val_losses), 2)
        self.assertEqual(len(trainer.val_accuracies), 2)
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization"""
        # Test without scheduler
        trainer_no_scheduler = Trainer(self.base_config)
        trainer_no_scheduler._setup_optimizer(DummyModel())
        trainer_no_scheduler._setup_scheduler()
        self.assertIsNone(trainer_no_scheduler.scheduler)
        
        # Test with scheduler
        trainer_with_scheduler = Trainer(self.scheduler_config)
        trainer_with_scheduler._setup_optimizer(DummyModel())
        trainer_with_scheduler._setup_scheduler()
        self.assertIsInstance(
            trainer_with_scheduler.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        )
    
    def test_scheduler_step(self):
        """Test if scheduler properly updates learning rate"""
        trainer = Trainer(self.scheduler_config)
        model = DummyModel()
        
        # Get initial learning rate
        initial_lr = self.scheduler_config['training']['optimizer']['params']['lr']
        
        # Train for one epoch
        trainer.train(model, self.train_loader, self.val_loader)
        
        # Check that learning rate has changed
        final_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertNotEqual(initial_lr, final_lr)
    
    def test_scheduler_state_saving(self):
        """Test if scheduler state is properly saved"""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(self.scheduler_config)
            model = DummyModel()
            
            # Train and save checkpoint
            trainer.train(
                model,
                self.train_loader,
                self.val_loader,
                output_dir=tmpdir
            )
            
            # Check if checkpoint contains scheduler state
            checkpoint = torch.load(Path(tmpdir) / 'best_model.pth')
            self.assertIn('scheduler_state_dict', checkpoint)
            self.assertIsNotNone(checkpoint['scheduler_state_dict'])

if __name__ == '__main__':
    unittest.main() 