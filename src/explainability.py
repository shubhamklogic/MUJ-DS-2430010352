import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.innovation_peft import RetinopathyFoundationPEFT
from src.preprocess import ben_graham_preprocessing

class GradCAM:
    """
    Hook-safe Grad-CAM extractor adaptive to both Vision Transformers (Swin / ViT)
    and Convolutional (ResNet) backbones.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._hook_forward()

    def _hook_forward(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations = output
            if output.requires_grad:
                output.retain_grad()
        self.target_layer.register_forward_hook(forward_hook)

    def generate_heatmap(self, input_tensor):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        
        # Ordinal progression score: sum of cumulative boundary sigmoids
        score = torch.sigmoid(logits).sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("Grad-CAM could not capture gradients. Ensure requires_grad=True is set on backbone weights.")

        act = self.activations.detach().cpu().numpy()[0]
        grad = self.activations.grad.detach().cpu().numpy()[0]

        # Universal spatial aggregation handling both Transformer and CNN shapes:
        # Case A: Channels-last (H, W, C) -> Swin Transformer feature map
        if act.ndim == 3 and act.shape[2] > act.shape[0] and act.shape[2] > act.shape[1]:
            weights = np.mean(grad, axis=(0, 1))
            cam = np.sum(weights[None, None, :] * act, axis=2)

        # Case B: Channels-first (C, H, W) -> Standard CNN feature map
        elif act.ndim == 3 and act.shape[0] > act.shape[1] and act.shape[0] > act.shape[2]:
            weights = np.mean(grad, axis=(1, 2))
            cam = np.sum(weights[:, None, None] * act, axis=0)

        # Case C: Token sequence (L, C) -> Flattened Transformer tokens
        elif act.ndim == 2:
            L, C = act.shape
            weights = np.mean(grad, axis=0)
            cam_tokens = np.sum(weights * act, axis=1)
            side = int(np.sqrt(L))
            if side * side == L:
                cam = cam_tokens.reshape((side, side))
            else:
                cam = cv2.resize(cam_tokens.reshape((1, L)), (224, 224))
        else:
            raise ValueError(f"Unexpected activation shape: {act.shape}")

        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val
            
        cam = cv2.resize(cam, (224, 224))
        return cam

def find_target_layer(model):
    """
    Dynamically finds the optimal representation layer across Swin Transformer and CNNs.
    """
    target = None
    target_name = ""

    # Priority 1: Swin Transformer final block norm2 / attention layer
    for name, module in model.named_modules():
        if "layers" in name and "blocks" in name and "norm2" in name:
            target = module
            target_name = name

    if target is not None:
        print(f"[XAI Module] Located Swin Transformer target layer: '{target_name}'")
        return target

    # Priority 2: Swin Transformer final stage block
    for name, module in model.named_modules():
        if "layers" in name and "blocks" in name:
            target = module
            target_name = name

    if target is not None:
        print(f"[XAI Module] Located Swin Block target layer: '{target_name}'")
        return target

    # Priority 3: Final LayerNorm or Conv2d before classifier head
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.LayerNorm)) and "ordinal_head" not in name and "head" not in name:
            print(f"[XAI Module] Fallback located target layer: '{name}'")
            return module

    raise AttributeError("Could not identify target feature layer for Grad-CAM explainability.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[XAI Module] Running Grad-CAM visual explainability on: {device}")

    checkpoint_path = "models/best_peft_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    # Instantiate model
    model = RetinopathyFoundationPEFT(num_classes=5, lora_r=16, lora_alpha=32, pretrained=False).to(device)
    
    # Load state dict safely
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if (isinstance(checkpoint, dict) and "state_dict" in checkpoint) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    print(f"[XAI Module] Successfully loaded model weights from: '{checkpoint_path}'")

    for p in model.parameters():
        p.requires_grad = True
    model.eval()

    target_layer = find_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    test_df = pd.read_csv("data/processed/test.csv")
    img_dir = "data/raw/train_images"

    stage_names = ['Stage 0: No DR', 'Stage 1: Mild', 'Stage 2: Moderate', 'Stage 3: Severe', 'Stage 4: Proliferative']
    selected_samples = []

    for stage in range(5):
        samples = test_df[test_df['diagnosis'] == stage]
        if not samples.empty:
            row = samples.iloc[0]
            selected_samples.append((stage, row['id_code']))

    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(5, 3, figsize=(12, 18))
    plt.subplots_adjust(hspace=0.35, wspace=0.1)

    print("\nExtracting spatial attribution maps across clinical stages:")
    for row_idx, (gt_stage, id_code) in enumerate(selected_samples):
        img_path = os.path.join(img_dir, f"{id_code}.png")
        print(f"  -> Processing {stage_names[gt_stage]} (ID: {id_code})")

        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            raise FileNotFoundError(f"Retinal image could not be read: {img_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        raw_resized = cv2.resize(raw_rgb, (224, 224))

        prep_rgb = ben_graham_preprocessing(img_path, img_size=224)

        tensor_in = prep_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        tensor_in = (tensor_in - mean) / std
        tensor_in = torch.tensor(tensor_in, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        cam = grad_cam.generate_heatmap(tensor_in)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = np.float32(heatmap_colored) / 255.0 + np.float32(prep_rgb) / 255.0
        overlay = overlay / np.max(overlay)
        overlay = np.uint8(255 * overlay)

        axes[row_idx, 0].imshow(raw_resized)
        axes[row_idx, 0].set_title(f"Raw Fundus ({id_code})\n{stage_names[gt_stage]}", fontsize=9, fontweight='bold')
        axes[row_idx, 0].axis('off')

        axes[row_idx, 1].imshow(prep_rgb)
        axes[row_idx, 1].set_title("Vascular Contrast Enhanced", fontsize=9, fontweight='bold')
        axes[row_idx, 1].axis('off')

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("Grad-CAM Lesion Localization", fontsize=9, fontweight='bold')
        axes[row_idx, 2].axis('off')

    out_file = "figures/xai_gradcam_stages.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[✓] Successfully generated and saved figure to: '{out_file}'\n")

if __name__ == "__main__":
    main()