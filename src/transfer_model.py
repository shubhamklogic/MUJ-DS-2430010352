import torch
import torch.nn as nn
from torchvision import models

class RetinopathyResNet50(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True):
        super(RetinopathyResNet50, self).__init__()
        
        # 1. Load pre-trained ResNet50 weights
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        
        # 2. Freeze backbone layers initially if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # 3. Replace final fully-connected head
        in_features = self.backbone.fc.in_features  # 2048 for ResNet50
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),  # Regularization to reduce variance
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

    def unfreeze_top_layers(self):
        """
        Unfreeze the deeper layer4 residual blocks for progressive fine-tuning.
        """
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        print("-> Layer4 and Classifier head unfrozen for fine-tuning.")

if __name__ == "__main__":
    model = RetinopathyResNet50(num_classes=5, freeze_backbone=True)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(f"Transfer Learning model initialized successfully!")
    print(f"Output shape: {out.shape} (Expected: [2, 5])")