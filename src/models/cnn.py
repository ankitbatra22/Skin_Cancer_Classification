import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
          # First block
          nn.Conv2d(3, 64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),

          # Second block
          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),

          # Third block
          nn.Conv2d(128, 256, kernel_size=3, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),

          # Fourth block
          nn.Conv2d(256, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
          # Adaptive pooling for input size flexibility
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Dropout(0.5),  # Dropout to reduce overfitting
          nn.Linear(512, 256),
          nn.ReLU(inplace=True),
          nn.Dropout(0.5),
          nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
      x = self.features(x)
      x = self.classifier(x)
      return x