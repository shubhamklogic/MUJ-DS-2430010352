import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model

class RetinopathyFoundationPEFT(nn.Module):
    """
    Parameter-Efficient Retinal Foundation Architecture.
    Integrates Swin Transformer backbone with Low-Rank Adaptation (LoRA)
    and a 4-boundary Ordinal Cumulative Head.
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=5, lora_r=16, lora_alpha=32, pretrained=False):
        super().__init__()
        # Initialize backbone feature extractor
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        
        # Inject LoRA adapters into multi-head attention projections
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        
        # 4 monotonic ordinal cuts for 5-stage clinical classification
        self.ordinal_head = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.ordinal_head(features)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PEFT Foundation Engine] Auditing model architecture on device: {device}")
    
    model = RetinopathyFoundationPEFT(num_classes=5, lora_r=16, lora_alpha=32).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("=" * 65)
    print("     PARAMETER-EFFICIENT FOUNDATION ARCHITECTURE AUDIT")
    print("=" * 65)
    print(f"Total Backbone & Adapter Parameters: {total_params:,}")
    print(f"Trainable LoRA & Head Parameters:    {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Frozen Parameters (Retained):        {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print("-" * 65)

if __name__ == "__main__":
    main()