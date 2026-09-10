import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model

# ==============================================================================
# 1. ORDINAL CUMULATIVE CLASSIFIER HEAD
# Resolves the clinical flaw where cross-entropy treats adjacent errors
# (Stage 2 vs 3) identical to catastrophic errors (Stage 0 vs 4).
# ==============================================================================
class OrdinalClassificationHead(nn.Module):
    """
    Ordinal Logistic / Cumulative Head for 5 DR Stages (0 to 4).
    Outputs 4 binary cumulative thresholds: P(Y > k) for k in [0, 1, 2, 3].
    """
    def __init__(self, in_features: int, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        # Output (num_classes - 1) binary decision boundaries
        self.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits for cumulative thresholds
        return self.fc(x)

    def predict_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Converts cumulative threshold logits into discrete 5-class probabilities.
        """
        sigmoids = torch.sigmoid(logits)
        # Pad with 1.0 (P(Y >= 0)) on the left and 0.0 (P(Y >= 5)) on the right
        ones = torch.ones((sigmoids.size(0), 1), device=logits.device)
        zeros = torch.zeros((sigmoids.size(0), 1), device=logits.device)
        cum_probs = torch.cat([ones, sigmoids, zeros], dim=1)
        
        # P(Y = k) = P(Y >= k) - P(Y >= k+1)
        probs = cum_probs[:, :-1] - cum_probs[:, 1:]
        return torch.clamp(probs, min=1e-7, max=1.0)


class OrdinalCumulativeLoss(nn.Module):
    """
    Loss function for ordinal classification: binary cross-entropy on cumulative cuts.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        batch_size = targets.size(0)
        
        # Construct binary matrix: 1 if target > k else 0 for k in [0, 1, 2, 3]
        cuts = torch.arange(self.num_classes - 1, device=device).unsqueeze(0).expand(batch_size, -1)
        binary_targets = (targets.unsqueeze(1) > cuts).float()
        
        return F.binary_cross_entropy_with_logits(logits, binary_targets)


# ==============================================================================
# 2. FOUNDATION ADAPTER MODEL (PEFT / LoRA)
# ==============================================================================
class RetinopathyFoundationPEFT(nn.Module):
    """
    Vision Foundation Model adapted with PEFT / LoRA (r=16, alpha=32).
    Addresses RETFound / ViT parameter bloat on clinical edge devices.
    """
    def __init__(self, model_name: str = "swin_base_patch4_window7_224", num_classes: int = 5, r: int = 16, lora_alpha: int = 32):
        super().__init__()
        
        # 1. Instantiate Vision Foundation Backbone
        # Note: pretrained=False loads instantly for parameter/architecture audit.
        # In full production, pretrained=True pulls weights directly.
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        
        # 2. Configure PEFT LoRA Adapter Config
        self.peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],            # Injects low-rank adapters into attention projections
            lora_dropout=0.1,
            bias="none"
        )
        
        # 3. Wrap backbone with PEFT low-rank matrices
        self.adapted_backbone = get_peft_model(self.backbone, self.peft_config)
        
        # 4. Attach Ordinal Cumulative Classification Head
        self.ordinal_head = OrdinalClassificationHead(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.adapted_backbone(x)
        logits = self.ordinal_head(features)
        return logits

    def print_parameter_efficiency(self):
        """
        Prints trainable vs frozen parameters to prove parameter efficiency for CO5.
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        percentage = (trainable_params / total_params) * 100
        
        print("\n" + "=" * 65)
        print("  CO5 DELIVERABLE: PARAMETER-EFFICIENT FOUNDATION AUDIT")
        print("=" * 65)
        print(f"Total Base Model Parameters:      {total_params:,}")
        print(f"Trainable Adapter Parameters:     {trainable_params:,}")
        print(f"Frozen Parameters (Retained):     {total_params - trainable_params:,}")
        print(f"Trainable Ratio:                  {percentage:.2f}% (Frozen: {100 - percentage:.2f}%)")
        print("=" * 65)


# ==============================================================================
# 3. VERIFICATION & AUDIT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Initializing Parameter-Efficient Foundation Adapter...")
    
    # 1. Instantiate Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinopathyFoundationPEFT(r=16, lora_alpha=32).to(device)
    
    # 2. Print parameter efficiency metrics for viva evaluation
    model.print_parameter_efficiency()
    
    # 3. Run Forward Pass Test with dummy batch
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    dummy_labels = torch.tensor([0, 2, 4, 3], dtype=torch.long).to(device)
    
    logits = model(dummy_input)
    
    # 4. Test Loss Function & Probabilities
    criterion = OrdinalCumulativeLoss(num_classes=5)
    loss = criterion(logits, dummy_labels)
    probs = model.ordinal_head.predict_probabilities(logits)
    predicted_stages = torch.argmax(probs, dim=1)
    
    print("\n[Audit Test: Forward Pass & Ordinal Probabilities]")
    print(f"Input Shape:             {dummy_input.shape}")
    print(f"Cumulative Logits Shape: {logits.shape} (Expected: [4, 4])")
    print(f"Calculated Ordinal Loss: {loss.item():.4f}")
    print(f"Predicted Stages:        {predicted_stages.tolist()}")
    print("PEFT Foundation Adapter script verified successfully with zero errors!")