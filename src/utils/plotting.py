import matplotlib.pyplot as plt
import json
from pathlib import Path

def plot_training_metrics(metrics_path: str, output_dir: str):
    """Plot training metrics from JSON file."""
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Plot training and validation loss
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training')
    ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(epochs, metrics['val_accuracy'], 'g-')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    # Plot learning rate if available
    if 'learning_rates' in metrics:
        ax3.plot(epochs, metrics['learning_rates'], 'm-')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
    
    # Save plot
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(output_path)
    plt.close() 