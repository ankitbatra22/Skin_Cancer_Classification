import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.vit import ViT
from src.models.pretrained_vit import PretrainedViT
from src.data.dataset import HAM10000Dataset
from src.training.trainer import Trainer
import argparse
import os
from src.data.samplers import BalancedBatchSampler

class VitTransforms:
    def __init__(self, img_size=224):
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomEqualize(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.2)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

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
    
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_aug_rate = [5, 1, 15, 20, 5, 65, 45] 

    #augmenter = VitTransforms()

    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = HAM10000Dataset("data", train=True, transform=transform, oversample_rates=data_aug_rate)
    val_dataset = HAM10000Dataset("data", train=False, transform=transform)

    # Get class weights from dataset
    class_weights = train_dataset.get_class_weights().to(device)

    # Add weights to criterion params
    if 'params' not in config['training']['criterion']:
        config['training']['criterion']['params'] = {}
    config['training']['criterion']['params']['weight'] = class_weights

    # Create data loaders
    train_sampler = BalancedBatchSampler(dataset=train_dataset)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    #model = ViT(**config['model']['params'])
    model = PretrainedViT(**config['model']['params'])
    model = model.to(device)
    trainer = Trainer(config=config)
    trained_model = trainer.train(model=model, train_loader=train_loader, val_loader=val_loader)

    # Save model to output directory
    torch.save(trained_model.state_dict(), 
              os.path.join(args.output_dir, 'model.pth'))

if __name__ == "__main__":
    main()