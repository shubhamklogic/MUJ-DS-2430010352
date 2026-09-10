# Parameter-Efficient Foundation Adaptation for Diabetic Retinopathy Classification

## 1. Student Identification
- **Name:** Shubham Kumar
- **Registration Number:** 2430010352
- **Section:** D
- **Submission Scope:** CO3 (10 Marks) + CO4 (5 Marks) + CO5 (5 Marks) = 20 Marks

## 2. Dataset Description
- **Dataset Name:** APTOS 2019 Blindness Detection
- **Acquisition Source:** Digital Retinal Fundus Cameras (Zeiss, Topcon, Canon) via Aravind Eye Hospital
- **Total Samples:** 3,662 fundus photographs
- **Clinical Staging:** 5 Ordinal Stages (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR)
- **Split Strategy:** Stratified 70% Train (2,563), 15% Val (549), 15% Test (550)
- **Preprocessing:** Ben Graham's method (uninformative margin cropping + local Gaussian blur contrast enhancement at 224x224)

## 3. Model Architecture & Pipeline
1. **Empirical Baseline CNN:** Custom 3-block convolutional neural network trained from scratch to diagnose high bias and high variance.
2. **Transfer Learning Baseline (CO3):** 2-stage fine-tuned ResNet50 (Stage 1: frozen backbone; Stage 2: unfrozen Layer 4 + classification head).
3. **Proposed Innovation (CO5):** Parameter-Efficient Foundation Adapter (`src/innovation_peft.py`) utilizing Low-Rank Adaptation (LoRA, $r=16, \alpha=32$) with an Ordinal Cumulative Logistic Head.

## 4. Hyperparameters
| Hyperparameter | Baseline CNN | ResNet50 (CO3) | PEFT Foundation Adapter (CO5) |
| :--- | :--- | :--- | :--- |
| **Input Dimensions** | 224 x 224 | 224 x 224 | 224 x 224 |
| **Batch Size** | 32 | 32 | 16 (Gradient Accumulation = 2) |
| **Epochs** | 10 | 10 (Stage 1: 3, Stage 2: 7) | 10 |
| **Optimizer** | Adam ($lr=1\times 10^{-3}$) | AdamW (Head: $1\times 10^{-3}$, L4: $1\times 10^{-4}$) | AdamW ($lr=2\times 10^{-4}$) |
| **Loss Function** | Categorical Cross-Entropy | Categorical Cross-Entropy | Ordinal Cumulative Binary Cross-Entropy |
| **Regularization** | None | Dropout ($p=0.4$) + Weight Decay ($1\times 10^{-4}$) | Adapter Dropout ($p=0.1$) |
| **Learning Rate Scheduler** | None | ReduceLROnPlateau (factor=0.5, patience=2) | Cosine Annealing |

## 5. Hyperparameter Tuning Rationale
- **Batch Size (32):** Maximizes GPU throughput while fitting within standard laptop VRAM limits.
- **2-Stage Unfreeze Strategy:** Freezing the backbone in Stage 1 protects pre-trained ImageNet representations from disruptive initial gradients. Unfreezing Layer 4 in Stage 2 with a lower learning rate ($1\times 10^{-4}$) allows high-level microvascular feature adaptation.
- **LoRA Configuration ($r=16, \alpha=32$):** Rank 16 provides the capacity needed for subtle microvascular lesions (microaneurysms, hemorrhages) without parameter bloat.

## 6. Evaluation Results & Deliverables
- **Overall Test Accuracy:** 80.73% (444 / 550 correct on hold-out test set)
- **Quadratic Weighted Kappa (QWK):** 0.8616 (Substantial clinical agreement)
- **5-Fold Stratified Cross-Validation:** 81.03% Mean Validation Accuracy ($\pm 0.6\%$)
- **Artifacts Generated:**
  - Class Distribution: `data/processed/1_Class_Distribution_and_Imbalance.png`
  - Baseline Curves: `data/processed/2_Baseline_Loss_Accuracy_Curves_Bias_Variance.png`
  - Fine-Tuning Curves: `data/processed/3_FineTuned_ResNet50_Loss_Accuracy_Curves.png`
  - Confusion Matrix & QWK: `data/processed/4_Test_Confusion_Matrix_and_QWK.png`
  - K-Fold Validation: `data/processed/5_kfold_cross_validation.png`
  - Metrics Report: `data/processed/06_Evaluation_Metrics_Classification_Report.txt`

## 7. Comparative Analysis (CO4 Deliverable)
- **Benchmark Source:** Tabulated against published literature on the identical APTOS 2019 dataset (`results/comparison.csv` and `figures/sota_comparison_chart.png`).
- **Key Findings:** Our baseline ResNet50 (0.8616 QWK) outperforms DenseNet121 (0.8410). The proposed foundation PEFT adapter (0.9215 QWK) surpasses full RETFound fine-tuning (0.9120) while remaining deployable on edge devices.

## 8. Novel Innovation & Ablation Study (CO5 Deliverable)
- **Clinical Problem:** Full foundation transformers (e.g., RETFound, 304M parameters) require $\ge 22\text{ GB}$ VRAM and suffer from class-imbalance boundary misclassifications on minority stages like Stage 3.
- **Methodology:** Implemented in `src/innovation_peft.py`. Freezes $99.12\%$ of weights ($86.7\text{M}$ parameters) and tunes only $0.88\%$ ($774\text{K}$ parameters) using LoRA. Replaces flat cross-entropy with a 4-boundary Ordinal Cumulative Head.
- **Design of Experiments (DoE):** Documented across 6 configurations (E1 to E6) in `results/ablation.csv` and visualized in `figures/doe_ablation_architecture.png`.

## 9. Folder Structure
```text
<RegistrationNumber>/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── split_data.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── baseline_model.py
│   ├── train_baseline.py
│   ├── transfer_model.py
│   ├── train_finetune.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── cross_val.py
│   ├── export_metrics.py
│   ├── generate_module_c_figures.py
│   └── innovation_peft.py
│
├── results/
│   ├── comparison.csv
│   └── ablation.csv
│
├── models/
│   ├── baseline_cnn.pth
│   └── best_finetuned_model.pth
│
└── figures/
    ├── sota_comparison_chart.png
    └── doe_ablation_architecture.png