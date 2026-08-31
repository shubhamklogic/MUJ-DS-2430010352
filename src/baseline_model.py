import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineRetinopathyCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(BaselineRetinopathyCNN, self).__init__()
        
        # Block 1: Input (3, 224, 224) -> (32, 112, 112)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: (32, 112, 112) -> (64, 56, 56)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: (64, 56, 56) -> (128, 28, 28)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling to reduce dimensions to (128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Classifier Head
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Pooling & Flattening
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Test model shape with dummy tensor
    model = BaselineRetinopathyCNN(num_classes=5)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model instantiated successfully!")
    print(f"Output shape: {output.shape} (Expected: [4, 5])")