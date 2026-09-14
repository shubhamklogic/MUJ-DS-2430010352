# Parameter-Efficient Retinal Foundation Adaptation for Diabetic Retinopathy Classification

## 1. Project Title
Automated Detection and Classification of Diabetic Retinopathy Stages Using Fine-Tuned ResNet-50 and Parameter-Efficient Foundation Adaptation (LoRA + Ordinal Cumulative Head)

## 2. Student Details
- **Student Name:** Shubham Kumar
- **Registration Number:** 2430010352
- **Section:** D
- **Course Component Target:** CO3 (10 Marks) + CO4 (5 Marks) + CO5 (5 Marks) = 20 Marks

## 3. Dataset
- **Dataset Name:** APTOS 2019 Blindness Detection
- **Source:** Kaggle (Aravind Eye Hospital, India)
- **Modality:** Digital Retinal Fundus Photography (Topcon, Zeiss, Canon)
- **Total Cohort Size:** 3,662 retinal fundus images (PNG format)
- **Clinical Staging (5 Ordinal Classes):**
  - Stage 0 (No DR): 1,805 samples (49.29%)
  - Stage 1 (Mild): 370 samples (10.10%)
  - Stage 2 (Moderate): 999 samples (27.28%)
  - Stage 3 (Severe): 193 samples (5.27%)
  - Stage 4 (Proliferative DR): 295 samples (8.06%)
- **Partition Strategy (Zero-Leakage Stratified Split):**
  - Train Set: 2,563 images (69.99%)
  - Validation Set: 549 images (14.99%)
  - Hold-Out Test Set: 550 images (15.02%)
  - Independence Audit: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅ (Confirmed via `src/audit_splits.py`)
- **Image Preprocessing Pipeline:**
  - Ben Graham's method: Circular threshold cropping (`tol=7`), local Gaussian contrast subtraction ($\sigma=10$), resizing to $224 \times 224 \times 3$, and ImageNet normalization.

## 4. Model Architectures
1. **Scratch Baseline CNN:** Custom 3-block convolutional neural network (Conv2d -> BatchNorm -> ReLU -> MaxPool -> AdaptiveAvgPool -> Dense). Total Parameters: 102,277. Diagnosed underfitting and optimization variance (val loss spiked to 1.4187, accuracy plateaued at ~71%).
2. **CO3 Transfer Learning Backbone:** ResNet-50 initialized with pre-trained ImageNet-1K weights. Replaced final FC layer with: `AdaptiveAvgPool2d` -> `Linear(2048, 256)` -> `BatchNorm1d` -> `ReLU` -> `Dropout(0.4)` -> `Linear(256, 5)`. Optimized using 2-Stage progressive unfreezing (Stage 1: frozen backbone; Stage 2: `layer4` unfrozen with differential learning rates).
3. **CO5 Innovation Architecture:** Vision Transformer foundation model adapted via Parameter-Efficient Fine-Tuning (PEFT / LoRA, rank $r=16$, $\alpha=32$, dropout $0.1$) targeting attention projections. The standard flat classification head is replaced with a **4-cut Ordinal Cumulative Logistic Head** modeling cumulative stage probabilities $P(Y > k)$ trained with cumulative binary cross-entropy loss.

## 5. Hyperparameters
| Hyperparameter | Baseline CNN | ResNet-50 (CO3 Baseline) | PEFT Foundation Adapter (CO5) |
| :--- | :--- | :--- | :--- |
| **Input Resolution** | 224 × 224 × 3 | 224 × 224 × 3 | 224 × 224 × 3 |
| **Batch Size** | 32 | 32 | 16 |
| **Total Epochs** | 10 | 10 (Stage 1: 3, Stage 2: 7) | 3 (Empirical Convergence) |
| **Optimizer** | Adam | AdamW | AdamW |
| **Head Learning Rate** | 1e-3 | 1e-3 (Stage 1) / 3e-4 (Stage 2) | 2e-4 |
| **Backbone Learning Rate** | N/A | 1e-4 (`layer4` in Stage 2) | Frozen (>99% weights) |
| **Weight Decay ($L_2$)** | 0.0 | 1e-4 | 1e-4 |
| **Dropout Rate** | 0.0 | 0.4 | 0.1 (Adapter Dropout) |
| **Loss Function** | Categorical Cross-Entropy | Categorical Cross-Entropy | Cumulative Ordinal Loss |
| **Scheduler** | None | ReduceLROnPlateau (factor=0.5, patience=2) | Cosine Annealing |

## 6. Hyperparameter Tuning
Empirical hyperparameter search was conducted (`src/tune_hyperparameters.py`) across candidate configurations:
- Learning rates tested: `[1e-3, 3e-4, 1e-4]`
- Dropout rates tested: `[0.3, 0.4, 0.5]`
- Weight decay tested: `[1e-4, 1e-3]`
- **Selected Configuration (Candidate H2):** Head LR = 1e-3, Dropout = 0.4, Weight Decay = 1e-4.
- **Selection Justification:** While lower dropout (0.3) achieved slightly higher short-term accuracy, Dropout = 0.4 provided stronger variance regularization and prevented over-memorization on minority disease stages during progressive fine-tuning. Full log saved in `results/hyperparameter_tuning.csv`.

## 7. Results Summary
- **Overall Test Accuracy:** 80.73% (444 / 550 correct test predictions)
- **Quadratic Weighted Kappa (QWK):** 0.8616 (Strong Ordinal Agreement)
- **Macro Average Specificity:** 0.9483 (94.83% true negative rate)
- **Multiclass ROC-AUC:** Macro = 0.9286, Weighted = 0.9551
- **Stratified 5-Fold Cross-Validation:** Mean Accuracy = 86.09% (±1.92%), Mean QWK = 0.9047 (`results/kfold_results.csv`)
- **Key Result Artifacts:**
  - `results/confusion_matrix.png`: Test set confusion matrix showing misclassifications constrained to adjacent ordinal stages
  - `results/training_curve.png`: Dual loss and accuracy convergence curves marking the unfreeze transition point at Epoch 3.5
  - `results/test_medical_metrics.csv`: Per-class precision, sensitivity, specificity, and F1 scores
  - `data/processed/6_Evaluation_Metrics_Classification_Report.txt`: Comprehensive metrics and methodological verdict

## 8. Innovation (CO5 Contribution vs. SOTA)
- **Problem Identified:** Modern medical vision foundation models (such as RETFound with 304M parameters) suffer from severe parameter bloat requiring >22 GB VRAM for full fine-tuning, making clinical deployment on edge hardware or standard laptops infeasible. Furthermore, standard categorical cross-entropy treats all classification errors equally, creating an empirical sensitivity bottleneck on Stage 3 (Severe DR recall = 13.79%).
- **Novel Contribution:** We developed a parameter-efficient ordinal framework (`src/innovation_peft.py`):
  1. Injected Low-Rank Adaptation (LoRA $r=16$, $\alpha=32$) into attention projections, freezing 99.12% of parameters and training only 774K adapter weights (0.88% trainable ratio).
  2. Implemented a 4-cut Ordinal Cumulative Logistic Head modeling cumulative disease boundaries $P(Y > k)$, penalizing distant stage errors and eliminating the Stage 3 diagnostic drag.
  3. Validated empirically through dedicated training (`src/train_peft.py`, saved checkpoint `models/best_peft_model.pth`) and a 6-experiment Factorial Design of Experiments matrix (`results/ablation.csv`, `figures/doe_ablation_architecture.png`), achieving a peak theoretical QWK of 0.9215 on a 4.3 GB VRAM footprint.

### Benchmark Protocol Transparency & Literature Grounding
- **In-Domain Evaluated Models:** Both our Baseline ResNet-50 (CO3) and proposed PEFT Foundation Architecture (CO5) were trained and evaluated directly on our verified zero-leakage stratified split (Train: 2,563; Val: 549; Test: 550) on local consumer hardware.
- **Published SOTA Baselines:** External benchmark numbers for DenseNet-121 (Kassani et al., 2019), EfficientNet-B4 (Tan & Le, 2019), Swin Transformer (Liu et al., 2021), and RETFound (Zhou et al., Nature 2023) reflect reported evaluation performances on the identical APTOS 2019 dataset cited directly. Their cross-validation and test partitions vary per respective published literature protocols.

## 9. Folder Structure
```text
2430010352/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── dataset_information.txt
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── 1_Class_Distribution_and_Imbalance.png
│       ├── 2_Baseline_Loss_Accuracy_Curves_Bias_Variance.png
│       └── 6_Evaluation_Metrics_Classification_Report.txt
│
├── src/
│   ├── __init__.py
│   ├── reproducibility.py
│   ├── audit_splits.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── split_data.py
│   ├── baseline_model.py
│   ├── transfer_model.py
│   ├── train_baseline.py
│   ├── train_finetune.py
│   ├── tune_hyperparameters.py
│   ├── cross_val.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── generate_sota_figure.py
│   ├── generate_module_c_figures.py
│   ├── innovation_peft.py
│   ├── train_peft.py
│   └── generate_peft_curves.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curve.png
│   ├── 05_kfold_cross_validation.png
│   ├── kfold_results.csv
│   ├── hyperparameter_tuning.csv
│   ├── test_medical_metrics.csv
│   ├── comparison.csv
│   ├── ablation.csv
│   └── peft_training_history.csv
│
├── models/
│   ├── model_description.txt
│   ├── baseline_cnn.pth
│   ├── best_finetuned_model.pth
│   └── best_peft_model.pth
│
└── figures/
    ├── sota_comparison_chart.png
    ├── doe_ablation_architecture.png
    └── peft_training_curves.png
```

## 10. Reproduction Commands
```bash
# 1. Audit data partitions and verify zero leakage
python -m src.audit_splits

# 2. Run empirical hyperparameter tuning search
python -m src.tune_hyperparameters

# 3. Run empirical 5-fold stratified cross-validation
python -m src.cross_val

# 4. Evaluate complete test metrics suite (Accuracy, QWK, Specificity, ROC-AUC)
python -m src.evaluate

# 5. Generate CO4 SOTA comparison figure
python -m src.generate_sota_figure

# 6. Audit CO5 PEFT parameter efficiency and run DoE ablation plot
python -m src.innovation_peft
python -m src.generate_module_c_figures

# 7. Execute empirical PEFT foundation training and plot curves
python -m src.train_peft
python -m src.generate_peft_curves
```