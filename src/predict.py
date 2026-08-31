import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.preprocess import ben_graham_preprocessing
from src.transfer_model import RetinopathyResNet50

def predict_single_image(image_path, model_path="models/best_finetuned_model.pth", img_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Class labels mapping
    class_names = ['No DR (Stage 0)', 'Mild (Stage 1)', 'Moderate (Stage 2)', 'Severe (Stage 3)', 'Proliferative DR (Stage 4)']
    
    # 2. Check if image and weights exist
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    # 3. Load Model
    model = RetinopathyResNet50(num_classes=5, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Preprocess Image (Ben Graham + ImageNet Normalization)
    image = ben_graham_preprocessing(image_path, img_size=img_size)
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    # 5. Forward Pass
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
        pred_class = probs.argmax()

    # 6. Display Result
    print("\n" + "=" * 45)
    print(f"Input Image: {os.path.basename(image_path)}")
    print(f"Predicted Diagnosis: {class_names[pred_class]}")
    print(f"Confidence Score:   {probs[pred_class] * 100:.2f}%")
    print("=" * 45)
    print("Class Probabilities:")
    for name, prob in zip(class_names, probs):
        print(f" - {name:28s}: {prob * 100:5.2f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_input = sys.argv[1]
    else:
        # Default fallback image from test images
        img_input = "data/raw/train_images/000c1434d8d7.png"
    
    predict_single_image(img_input)