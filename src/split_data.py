import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. Load Data
RAW_DATA_PATH = "data/raw"
df = pd.read_csv(os.path.join(RAW_DATA_PATH, "train.csv"))

print(f"Total samples: {len(df)}")
print(df.head())

# Label Mapping for Viva Reference:
# 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# 2. Check Class Imbalance
plt.figure(figsize=(8, 5))
sns.countplot(x='diagnosis', data=df, palette='viridis')
plt.title('APTOS 2019 - Class Distribution')
plt.xticks(ticks=range(5), labels=class_names)
plt.xlabel('Diagnosis Stage')
plt.ylabel('Count')
plt.savefig('data/processed/class_distribution.png')
# plt.show()

# 3. Stratified Split (70% Train, 15% Val, 15% Test)
# First split: 70% train, 30% temp (val + test)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.30, 
    stratify=df['diagnosis'], 
    random_state=42
)

# Second split: Split the 30% into 15% val and 15% test
val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50, 
    stratify=temp_df['diagnosis'], 
    random_state=42
)

print(f"Train samples: {len(train_df)} ({len(train_df)/len(df):.1%})")
print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df):.1%})")
print(f"Test samples: {len(test_df)} ({len(test_df)/len(df):.1%})")

# 4. Save the splits
os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)
print("Splits saved successfully in data/processed/")