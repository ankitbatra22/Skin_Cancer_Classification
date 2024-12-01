import timm
import torch.nn as nn

class PretrainedViT(nn.Module):
    def __init__(self, num_classes=7, unfreeze_layers=0):
        super().__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            drop_path_rate=0.2,  # Stochastic depth
            drop_rate=0.1        # Dropout
        )
        
        # Replace classification head with better architecture
        self.model.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Freeze all layers except the last few transformer blocks
        self._freeze_layers(unfreeze_layers)
    
    def _freeze_layers(self, unfreeze_layers):
        # Freeze patch embedding
        for param in self.model.patch_embed.parameters():
            param.requires_grad = False
            
        # Freeze all transformer blocks except last n
        total_blocks = len(self.model.blocks)
        for i, block in enumerate(self.model.blocks):
            if i < total_blocks - unfreeze_layers:
                for param in block.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
