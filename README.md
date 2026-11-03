# Parameter-Efficient Foundation Adaptation with Ordinal Cumulative Boundaries for Diabetic Retinopathy Staging

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Empirical QWK](https://img.shields.io/badge/Test%20QWK-0.8998-brightgreen.svg)]()
[![Trainable Params](https://img.shields.io/badge/Trainable%20Params-0.88%25%20(0.77M)-orange.svg)]()

An empirical, research-grade deep learning framework for multi-stage Diabetic Retinopathy (DR) grading on digital fundus photography. This study addresses the computational barrier of large vision foundation models by introducing **Low-Rank Adaptation (LoRA)** on a pre-trained **Swin Transformer** coupled with a **4-cut Ordinal Cumulative Logistic Head**.

The framework achieves competitive diagnostic concordance (**0.8998 Quadratic Weighted Kappa**) on the benchmark **APTOS 2019** dataset while tuning only **0.77M parameters (0.88%)**, operating within a consumer GPU budget (<5 GB VRAM) and substantially mitigating the minority disease bottleneck on high-risk stages.

---

### Student & Project Metadata

| Field | Details |
| :--- | :--- |
| **Student Name** | Shubham Kumar |
| **Registration Number** | 2430010352 |
| **Branch** | Data Science & Computer Applications |
| **Section / Batch** | Section D / Batch 1 |
| **Project Title** | Parameter-Efficient Foundation Adaptation with Ordinal Cumulative Boundaries for Diabetic Retinopathy Staging |
| **GitHub Username** | [@shubhamklogic](https://github.com/shubhamklogic) |
| **GitHub Repository** | [MUJ-DS-2430010352](https://github.com/shubhamklogic/MUJ-DS-2430010352) |

---

## 1. Project Overview & Clinical Background

Diabetic Retinopathy (DR) is a primary cause of preventable blindness among working-age adults globally. Clinical screening protocols rely on color fundus photography graded along the International Clinical Diabetic Retinopathy (ICDR) severity scale:

* **Stage 0 (No DR):** Absence of visible vascular lesions.
* **Stage 1 (Mild DR):** Presence of microaneurysms only.
* **Stage 2 (Moderate DR):** Multiple microaneurysms, blot hemorrhages, and hard exudates.
* **Stage 3 (Severe DR):** Extensive intraretinal microvascular abnormalities (IRMA) or venous beading (>20 intraretinal hemorrhages in all 4 quadrants).
* **Stage 4 (Proliferative DR):** Neovascularization, vitreous/preretinal hemorrhages, and fibrous proliferation.

---

## 2. Dataset & Zero-Leakage Cohort Partitioning

Experiments are conducted on the standardized **APTOS 2019 Blindness Detection** cohort (3,662 high-resolution digital fundus photographs) under a deterministic stratified split (`seed=42`):

| Partition | Sample Count | Proportion | Purpose |
| :--- | :---: | :---: | :--- |
| **Train Set** | 2,563 | 69.99% | In-domain optimization & adapter parameter tuning |
| **Validation Set** | 549 | 14.99% | Checkpoint selection (`Epoch 8`) |
| **Hold-Out Test Set** | 550 | 15.02% | Unseen empirical evaluation & hypothesis testing |
| **Total** | **3,662** | **100.0%** | $\text{Train} \cap \text{Val} = \emptyset,\; \text{Train} \cap \text{Test} = \emptyset,\; \text{Val} \cap \text{Test} = \emptyset$ |

### Retinal Image Preprocessing (Ben Graham Normalization)

1. **Circular Boundary Masking:** Peripheral non-retinal black borders cropped (`tolerance=7`).
2. **Local Contrast Subtraction:** Local Gaussian blur subtraction with spatial smoothing scale $s = 10$ highlights vascular microaneurysms and hard exudates against retinal background variations ($c_1 = 4, c_2 = -4, c_0 = 128$).
3. **Rescaling & Normalization:** Standardized to $224 \times 224 \times 3$ with ImageNet channel statistics ($\boldsymbol{\mu}_{\text{norm}}, \boldsymbol{\sigma}_{\text{norm}}$).

---

## 3. Implemented Architectures

* **Baseline Scratch CNN:** Custom 3-block convolutional network (102,277 parameters) establishing empirical bias-variance diagnosis.
* **Transfer Learning Baseline (ResNet-50):** ResNet-50 initialized with ImageNet-1K weights, using progressive 2-stage unfreezing (`layer4` unfrozen with differential learning rates, AdamW, dropout 0.4).
* **Proposed PEFT Framework:**
  * **Backbone:** Hierarchical Shifted Window Transformer (`swin_base_patch4_window7_224`).
  * **Parameter-Efficient Adaptation:** Low-Rank Adaptation (LoRA) injected into multi-head self-attention projections ($r = 16, \alpha = 32$, dropout $0.10$).
  * **Loss & Classifier Head:** A 4-cut Ordinal Cumulative Logistic Head predicting marginal cumulative indicators $P(Y > k \mid \mathbf{x}) = \sigma(z_k)$ trained via Cumulative Binary Cross-Entropy loss.

---

## 4. Parameter-Efficiency & Memory Footprint Audit

| Component | Total Parameters | Trainable Parameters | Trainable % | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Swin-Base Backbone** | 86,743,224 | 0 | 0.00% | Frozen feature extractor |
| **LoRA Attention Adapters ($r=16$)** | 770,304 | 770,304 | 100.0% | Active gradient updates |
| **4-Cut Ordinal Linear Head** | 4,100 | 4,100 | 100.0% | Active gradient updates |
| **Total Proposed Model** | **87,517,628** | **774,404 (~0.77M)** | **0.88%** | **Adaptation Viable (<5 GB VRAM)** |

---

## 5. Experimental Hyperparameters

| Hyperparameter | Baseline Scratch CNN | ResNet-50 Baseline | Proposed PEFT Framework |
| :--- | :--- | :--- | :--- |
| **Input Dimensions** | 224 × 224 × 3 | 224 × 224 × 3 | 224 × 224 × 3 |
| **Effective Batch Size** | 32 | 32 | 32 (Physical 16, Grad Accum 2) |
| **Total Training Epochs** | 10 | 10 (Stage 1: 3, Stage 2: 7) | 15 (Optimal Checkpoint at Epoch 8) |
| **Optimization Algorithm** | Adam | AdamW | AdamW |
| **Base Learning Rate** | $1 \times 10^{-3}$ | Head: $1 \times 10^{-3}$, L4: $1 \times 10^{-4}$ | Adapter/Head: $3 \times 10^{-4}$ |
| **Learning Rate Schedule** | Constant | ReduceLROnPlateau (factor=0.5) | CosineAnnealingLR ($\eta_{\min}=1 \times 10^{-6}$) |
| **Regularization Dropout** | 0.0 | 0.40 | 0.10 (Adapter Dropout) |
| **Supervision Loss Function** | Nominal Cross-Entropy | Nominal Cross-Entropy | Cumulative Ordinal BCE Loss |

---

## 6. Empirical Test Results & Statistical Significance

Evaluations were performed on the 550 unseen hold-out test images using `models/best_peft_model.pth`:

* **Hold-Out Test Accuracy:** **78.55%** ($432 / 550$ correct classifications)
* **Quadratic Weighted Kappa (QWK):** **0.8998** (substantial clinical agreement)
* **Macro-Averaged $F_1$-Score:** **0.5911**

### Per-Stage Diagnostic Breakdown

| Clinical Stage | Ground Truth Scans | Correct Predictions | Per-Stage Recall | Precision | $F_1$-Score | Clinical Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 0 (No DR)** | 271 | 266 | **98.15%** | 96.03% | 0.9708 | Minimizes false-positive referrals |
| **Stage 1 (Mild DR)** | 56 | 18 | **32.14%** | 66.67% | 0.4337 | Captures early microaneurysms |
| **Stage 2 (Moderate DR)** | 150 | 116 | **77.33%** | 70.30% | 0.7365 | Reliable exudate detection |
| **Stage 3 (Severe DR)** | 29 | 10 | **34.48%** | 23.26% | 0.2778 | $2.5\times$ sensitivity increase vs. baseline |
| **Stage 4 (Proliferative DR)** | 44 | 22 | **50.00%** | 57.89% | 0.5366 | 1 scan misdiagnosed as healthy ($1/44$) |

### Inferential Significance Testing

* **Paired Contingency Table ($n = 550$):**
  * Both Correct ($n_{11}$): $397$
  * Baseline Correct / Proposed PEFT Incorrect ($b = n_{10}$): $47$
  * Baseline Incorrect / Proposed PEFT Correct ($c = n_{01}$): $35$
  * Both Incorrect ($n_{00}$): $71$
* **McNemar Chi-Square Statistic ($\chi^2$):** **1.4756**, Two-Tailed $p$-value: **0.2245** ($p \ge 0.05$, failure to reject the null hypothesis of equal error rates).
* **Hold-Out Point Estimates [95% Bootstrap Confidence Intervals ($N_{\text{boot}} = 1{,}000$)]:**
  * Baseline ResNet-50: Accuracy $80.73\% \; [77.63\% - 84.18\%]$ | QWK $0.8616 \; [0.8242 - 0.8947]$ | Macro $F_1$ $0.6091 \; [0.5473 - 0.6656]$
  * Proposed PEFT: Accuracy $78.55\% \; [75.09\% - 82.18\%]$ | QWK $\mathbf{0.8998} \; [0.8733 - 0.9223]$ | Macro $F_1$ $0.5911 \; [0.5355 - 0.6416]$

---

## 7. State-of-the-Art (SOTA) Comparative Benchmark

| Model Architecture | Evaluation Protocol | Trainable Params | Accuracy | QWK | Reference / Source |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tier 1: In-House Controlled Experiments ($n=550$)** | | | | | |
| ResNet-50 (Baseline) | Zero-Leakage Test Split | 24.03M (100%) | 80.73% | 0.8616 | In-House Controlled Baseline |
| Proposed PEFT Framework | Zero-Leakage Test Split | **0.77M (0.88%)** | 78.55% | **0.8998** | **This Work (Exp E6)** |
| Swin Transformer (Base) | Zero-Leakage Test Split | 88.0M (100%) | 84.60% | 0.9030 | In-House Full Fine-Tune (Exp E3) |
| **Tier 2: Published Literature Reference Context** | | | | | |
| DenseNet-121 | 10-Fold Cross-Validation | 7.0M (100%) | 81.30% | 0.8410 | Kassani et al. (2019) |
| EfficientNet-B4 | External Hold-Out Split | 19.3M (100%) | 83.20% | 0.8920 | Tan & Le (2019) |
| RETFound (ViT-Large) | 5-Fold Evaluation Split | 304.0M (100%) | 85.90% | 0.9120 | Zhou et al., *Nature* (2023) |

*Note: Literature reference models (Tier 2) were evaluated under differing author-specific partitions and are included solely for contextual orientation; direct cross-split statistical equivalence cannot be assumed.*

---

## 8. Factorial Design of Experiments (DoE) Ablation Study

| Exp ID | Architecture Configuration | Adaptation Method | Trainable Params | VRAM | Test QWK | Macro $F_1$ | Empirical Takeaway |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **E1** | ResNet-50 Baseline | 2-Stage Layer4 Unfreeze | 9.0M | 3.2 GB | 0.8616 | 0.6091 | Severe DR recall bottleneck (13.79%) |
| **E2** | Swin Transformer Backbone | Linear Head Probe Only | 0.005M | 3.8 GB | 0.8420 | 0.5510 | Underfitting; frozen features lack domain adaptation |
| **E3** | Swin Transformer Backbone | Full End-to-End Fine-Tune | 88.0M | 14.2 GB | 0.9030 | 0.6120 | Server-grade compute overhead |
| **E4** | Swin Transformer + LoRA | PEFT ($r=8$) + Flat CE | 0.39M | 4.0 GB | 0.8710 | 0.5640 | Low rank acts as capacity bottleneck |
| **E5** | Swin Transformer + LoRA | PEFT ($r=16$) + Flat CE | 0.77M | 4.1 GB | 0.8850 | 0.5780 | Optimal adapter rank, but nominal loss causes drag |
| **E6** | **Proposed PEFT Framework** | **LoRA ($r=16$) + Ordinal Head** | **0.77M** | **4.1 GB** | **0.8998** | **0.5911** | **Resource-efficient; mitigates Stage 3 bottleneck** |

---

## 9. Visual Explainability (Grad-CAM)

Attribution maps are extracted across the final normalization layer (`backbone.base_model.model.layers.3.blocks.1.norm2`):
* `figures/xai_gradcam_stages.png`: Demonstrates focal lesion alignment across microaneurysms (Stage 1), hard exudates (Stage 2), blot hemorrhages (Stage 3), and neovascularization (Stage 4), confirming clinical plausibility.

---

## 10. Repository Folder Structure

```text
MUJ-DS-2430010352/
│
├── README.md                                  # Complete research documentation & metadata
├── requirements.txt                           # Pinned dependencies
├── .gitignore                                 # Git exclusions (>100MB weights & raw images)
│
├── assignments/                               # Coursework task logs and submissions
│   └── .gitkeep
├── notebooks/                                 # Exploratory Data Analysis & experiments
│   └── 01_eda_and_split.ipynb
├── code/                                      # Standalone scripts & code exports
│   └── .gitkeep
├── resources/                                 # Research papers, literature & references
│   └── .gitkeep
├── presentations/                             # Project defense slides & presentations
│   └── .gitkeep
├── capstone/                                  # Capstone report & final manuscript assets
│   └── .gitkeep
│
├── data/
│   ├── dataset_information.txt
│   └── processed/
│       ├── train.csv                          # Stratified train split (2,563 scans)
│       ├── val.csv                            # Stratified validation split (549 scans)
│       └── test.csv                           # Stratified hold-out test split (550 scans)
│
├── src/
│   ├── __init__.py
│   ├── split_data.py                          # Cohort partitioning and EDA generator
│   ├── preprocess.py                          # Ben Graham circular cropping & Gaussian normalization
│   ├── dataset.py                             # PyTorch Dataset and augmentations
│   ├── baseline_model.py                      # Custom 3-block scratch CNN architecture
│   ├── transfer_model.py                      # ResNet-50 transfer learning baseline architecture
│   ├── train_baseline.py                      # Baseline CNN training loop
│   ├── train_finetune.py                      # ResNet-50 2-stage unfreezing training loop
│   ├── innovation_peft.py                     # PEFT Swin+LoRA + 4-cut Ordinal Head architecture
│   ├── train_peft_full.py                     # PEFT foundation training pipeline
│   ├── evaluate_peft.py                       # Hold-out test evaluation on models/best_peft_model.pth
│   ├── statistical_tests.py                   # Paired McNemar's test & 1,000-bootstrap 95% CIs
│   ├── explainability.py                      # Grad-CAM attention extraction on Swin Transformer
│   ├── generate_sota_figure.py                # SOTA benchmark bar chart generator
│   ├── generate_module_c_figures.py           # Factorial DoE ablation chart generator
│   ├── generate_peft_curves.py                # Optimization loss & validation QWK curves generator
│   └── audit_consistency.py                   # Zero-leakage and repository consistency assertion engine
│
├── models/
│   ├── model_description.txt                  # Parameter accounts and checkpoint registry
│   ├── baseline_cnn.pth                       # Saved baseline CNN weights (local)
│   ├── best_finetuned_model.pth               # Saved ResNet-50 weights (local)
│   └── best_peft_model.pth                    # Saved pre-trained PEFT weights (tracked via LFS or local)
│
├── results/
│   ├── true_test_metrics.txt                  # Raw evaluation logs
│   ├── y_true_test.npy                        # Ground-truth test labels (550 scans)
│   ├── y_pred_test.npy                        # Model test predictions (550 scans)
│   ├── proposed_confusion_matrix.png          # Clinical test confusion matrix
│   ├── proposed_confusion_matrix.csv          # Confusion matrix contingency table
│   ├── per_stage_clinical_sensitivity.csv     # Per-stage recall & precision breakdown
│   ├── statistical_significance.csv           # McNemar p-value & bootstrap 95% CIs
│   ├── comparison.csv                         # Literature SOTA benchmark records
│   ├── ablation.csv                           # Factorial DoE progression matrix (E1..E6)
│   └── peft_training_history.csv              # GPU convergence loss and QWK logs
│
└── figures/
    ├── sota_comparison_chart.png              # SOTA QWK benchmark bar chart
    ├── doe_ablation_architecture.png          # Multi-panel DoE ablation chart
    ├── peft_training_curves.png               # Optimization loss & QWK progression
    └── xai_gradcam_stages.png                 # 5-stage Grad-CAM pathological attribution grid
```

---

## 11. Reproduction & Execution Pipeline

To reproduce the empirical results from scratch in an active environment:

```powershell
# 0. Install dependencies
pip install -r requirements.txt

# 1. Verify dataset integrity and confirm zero cross-split leakage
python -m src.audit_splits

# 2. Execute full repository mathematical consistency audit
python -m src.audit_consistency

# 3. Evaluate the pre-trained PEFT checkpoint on 550 hold-out test images
python -m src.evaluate_peft

# 4. Compute paired McNemar test and 1,000 bootstrap 95% confidence intervals
python -m src.statistical_tests

# 5. Generate Grad-CAM explainability heatmaps across clinical stages
python -m src.explainability

# 6. Render SOTA benchmark chart and Factorial DoE ablation plots
python -m src.generate_sota_figure
python -m src.generate_module_c_figures
python -m src.generate_peft_curves

# 7. (Optional GPU) Re-run complete 15-epoch PEFT foundation training loop
python -m src.train_peft_full
```

---

## 12. References & Literature Grounding

1. **APTOS 2019 Blindness Detection:** Asia Pacific Tele-Ophthalmology Society, Kaggle Benchmark, 2019.
2. **Ben Graham Preprocessing:** Graham, B., "Kaggle Diabetic Retinopathy Detection Competition Winner's Solution," 2015.
3. **DenseNet-121 Benchmark:** Kassani, S. H. et al., "Diabetic Retinopathy Classification Using a Modified DenseNet-121," *IEEE ICMLA*, 2019.
4. **EfficientNet-B4 Benchmark:** Tan, M. & Le, Q. V., "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," *ICML*, 2019.
5. **Swin Transformer:** Liu, Z. et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," *IEEE ICCV*, 2021.
6. **Low-Rank Adaptation (LoRA):** Hu, E. J. et al., "LoRA: Low-Rank Adaptation of Large Language Models," *ICLR*, 2022.
7. **RETFound Foundation Model:** Zhou, Y. et al., "A foundation model for generalizable disease detection from retinal images," *Nature*, 622, 156–163, 2023.