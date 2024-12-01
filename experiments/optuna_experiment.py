import optuna
import argparse
import yaml
import torch
from experiments.cnn_experiment import main as train_cnn
from src.models.cnn import CNN
from src.training.trainer import Trainer
from src.data.dataset import HAM10000Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run Optuna on HAM10000 dataset')
    parser.add_argument('--config', type=str, default='configs/cnn_experiment.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='outputs/optuna_experiment',
                        help='Directory to save outputs and best model')
    return parser.parse_args()  

def objective(trial, config, output_dir): 
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    
    # Update configuration
    config['training']['optimizer']['params']['lr'] = learning_rate
    config['data']['batch_size'] = batch_size
    
    # Update model dropout 
    model = CNN(num_classes=7) 
    for layer in model.classifier: 
        if isinstance(layer, torch.nn.Dropout):
            layer.p = dropout_rate

    # Prepare dataset and dataloaders
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
      
    data_aug_rate = [3, 0.5, 6, 11, 3, 32, 23] 

    train_dataset = HAM10000Dataset(config['data']['path'], train=True, transform=transform, oversample_rates=data_aug_rate)
    val_dataset = HAM10000Dataset(config['data']['path'], train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train and validate
    trainer = Trainer(config)
    model = trainer.train(model, train_loader, val_loader, output_dir=output_dir)
    val_metrics = trainer._validate(model, val_loader)
    
    return val_metrics['val_loss']  

if __name__ == '__main__':
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, config, args.output_dir), n_trials=10)

    # Print best hyperparameters
    print("Best hyperparameters:", study.best_params)
    print("Best loss:", study.best_value)


# import optuna
# import argparse
# import yaml
# import torch
# from experiments.cnn_experiment import main as train_cnn
# from src.models.cnn import CNN
# from src.training.trainer import Trainer
# from src.data.dataset import HAM10000Dataset
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import os

# def parse_args():
#     parser = argparse.ArgumentParser(description='Run Optuna on HAM10000 dataset')
#     parser.add_argument('--config', type=str, default='configs/cnn_experiment.yaml',
#                         help='Path to config YAML file')
#     parser.add_argument('--output_dir', type=str, default='outputs/optuna_experiment',
#                         help='Directory to save outputs and best model')
#     return parser.parse_args()

# def objective(trial, config, output_dir):
#     # Hyperparameters to tune
#     learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-1)
#     batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
#     dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    
#     # Update configuration
#     config['training']['optimizer']['params']['lr'] = learning_rate
#     config['data']['batch_size'] = batch_size
    
#     # Update model dropout
#     model = CNN(num_classes=7)
#     for layer in model.classifier:
#         if isinstance(layer, torch.nn.Dropout):
#             layer.p = dropout_rate

#     # Prepare dataset and dataloaders
#     transform = transforms.Compose([
#         transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(20),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     train_dataset = HAM10000Dataset(config['data']['path'], train=True, transform=transform)
#     val_dataset = HAM10000Dataset(config['data']['path'], train=False, transform=transform)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
#     # Train and validate
#     trainer = Trainer(config)
#     model = trainer.train(model, train_loader, val_loader, output_dir=output_dir)
#     val_metrics = trainer._validate(model, val_loader)
    
#     return val_metrics['val_loss']

# if __name__ == '__main__':
#     args = parse_args()
    
#     # Load configuration
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Run Optuna
#     study = optuna.create_study(direction='minimize')
#     study.optimize(lambda trial: objective(trial, config, args.output_dir), n_trials=50)

#     # Print best hyperparameters
#     print("Best hyperparameters:", study.best_params)
#     print("Best loss:", study.best_value)


# OPTUNA RUN:
# import optuna
# import argparse
# import yaml
# import torch
# from experiments.cnn_experiment import main as train_cnn
# from src.models.cnn import CNN
# from src.training.trainer import Trainer
# from src.data.dataset import HAM10000Dataset
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import os

# def parse_args():
#     parser = argparse.ArgumentParser(description='Run Optuna on HAM10000 dataset')
#     parser.add_argument('--config', type=str, default='configs/cnn_experiment.yaml',
#                         help='Path to config YAML file')
#     parser.add_argument('--output_dir', type=str, default='outputs/optuna_experiment',
#                         help='Directory to save outputs and best model')
#     return parser.parse_args()

# def objective(trial, config, output_dir):
#     # Hyperparameters to tune
#     learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-1)
#     batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
#     dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    
#     # Update configuration
#     config['training']['optimizer']['params']['lr'] = learning_rate
#     config['data']['batch_size'] = batch_size
    
#     # Update model dropout
#     model = CNN(num_classes=7)
#     for layer in model.classifier:
#         if isinstance(layer, torch.nn.Dropout):
#             layer.p = dropout_rate

#     # Prepare dataset and dataloaders
#     transform = transforms.Compose([
#         transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(20),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     train_dataset = HAM10000Dataset(config['data']['path'], train=True, transform=transform)
#     val_dataset = HAM10000Dataset(config['data']['path'], train=False, transform=transform)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
#     # Train and validate 
#     trainer = Trainer(config)
#     model = trainer.train(model, train_loader, val_loader, output_dir=output_dir)
#     val_metrics = trainer._validate(model, val_loader)
     
#     return val_metrics['val_loss']

# if __name__ == '__main__':
#     args = parse_args()
    
#     # Load configuration
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Run Optuna
#     study = optuna.create_study(direction='minimize')
#     study.optimize(lambda trial: objective(trial, config, args.output_dir), n_trials=5)

#     # Print best hyperparameters
#     print("Best hyperparameters:", study.best_params)
#     print("Best loss:", study.best_value)
