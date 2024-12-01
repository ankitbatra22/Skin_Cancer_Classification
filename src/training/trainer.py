import torch
from typing import Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path
import json
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_criterion()
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def _setup_criterion(self):
        criterion_config = self.config['training']['criterion']
        criterion_class = getattr(torch.nn, criterion_config['name'])
        self.criterion = criterion_class(**criterion_config.get('params', {}))
    
    def _setup_optimizer(self, model):
        optim_config = self.config['training']['optimizer']
        optimizer_class = getattr(torch.optim, optim_config['name'])
        self.optimizer = optimizer_class(
            model.parameters(),
            **optim_config.get('params', {})
        )
    
    def _log_epoch(self, epoch: int, train_loss: float, val_metrics: Dict[str, float], epoch_time: float):
        """Print epoch results in a clean format"""
        print(f"\nEpoch [{epoch+1}/{self.config['training']['epochs']}] - {epoch_time:.2f}s")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Validation Accuracy: {val_metrics['val_acc']:.2f}%")
        
        # Store metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_metrics['val_loss'])
        self.val_accuracies.append(val_metrics['val_acc'])
    
    def _save_metrics(self, output_dir: str):
        """Save training history to JSON"""
        metrics = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_accuracy': self.val_accuracies
        }
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str = None
    ) -> torch.nn.Module:
        """Main training loop with improved logging"""
        print(f"\nStarting training on device: {self.device}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
        model = model.to(self.device)
        self._setup_optimizer(model)
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = self._train_epoch(model, train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate(model, val_loader)
            
            epoch_time = time.time() - epoch_start
            self._log_epoch(epoch, train_loss, val_metrics, epoch_time)
            
            # Save best model
            if output_dir and val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                output_path = Path(output_dir) / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': best_val_acc,
                }, str(output_path))
                print(f"  Saved new best model with accuracy: {best_val_acc:.2f}%")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        if output_dir:
            self._save_metrics(output_dir)
            print(f"Training metrics saved to {output_dir}/metrics.json")
        
        return model
    
    def _train_epoch(self, model, train_loader, epoch):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / len(train_loader)
    
    def _validate(self, model, val_loader):
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        y_true = []  # True labels
        y_pred = []  # Predicted labels

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Store predictions and true labels
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # Calculate precision, recall, F1
        precision = precision_score(y_true, y_pred, average=None)  # For each class
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)

        # Print or log metrics
        print("Precision (per class):", precision)
        print("Recall (per class):", recall)
        print("F1 Score (per class):", f1)

        # Classification report (optional, for a detailed view)
        print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']))

        return {
            'val_loss': val_loss / len(val_loader),
            'val_acc': 100. * correct / total
        }
