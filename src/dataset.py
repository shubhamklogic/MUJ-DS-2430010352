import os
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.preprocess import ben_graham_preprocessing

class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, img_size=224):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row['id_code']}.png"
        img_path = os.path.join(self.img_dir, img_name)

        # Apply preprocessing
        image = ben_graham_preprocessing(img_path, img_size=self.img_size)

        # Apply Albumentations augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Fallback normalization
            transform_default = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            image = transform_default(image=image)['image']

        label = torch.tensor(row['diagnosis'], dtype=torch.long)
        return image, label

def get_transforms(img_size=224):
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_test_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_test_transform

def get_dataloaders(train_df, val_df, test_df, img_dir, batch_size=32, img_size=224, num_workers=2):
    train_transform, val_test_transform = get_transforms(img_size)

    train_dataset = RetinopathyDataset(train_df, img_dir, transform=train_transform, img_size=img_size)
    val_dataset = RetinopathyDataset(val_df, img_dir, transform=val_test_transform, img_size=img_size)
    test_dataset = RetinopathyDataset(test_df, img_dir, transform=val_test_transform, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import pandas as pd
    
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    img_dir = "data/raw/train_images"

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, img_dir, batch_size=4, num_workers=0
    )

    images, labels = next(iter(train_loader))
    print(f"Data pipeline test successful!")
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Batch labels: {labels}")