import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.cnn import CNN 
from src.data.dataset import HAM10000Dataset
from src.training.trainer import Trainer
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on HAM10000 dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save model and results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config from specified path
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # data_aug_rate = [20, 10, 5, 50, 1, 40, 5] 

    # Create datasets
    train_dataset = HAM10000Dataset(
        config['data']['path'], 
        train=True, 
        transform=transform,
        # oversample_rates=data_aug_rate
    )

    val_dataset = HAM10000Dataset(
        config['data']['path'], 
        train=False, 
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # CHANGE MODEL HERE
    model = CNN(num_classes=7)
    trainer = Trainer(config)

    trained_model = trainer.train(model, train_loader, val_loader)

    # Save model to output directory
    torch.save(trained_model.state_dict(), 
              os.path.join(args.output_dir, 'model.pth'))

if __name__ == "__main__":
    main()