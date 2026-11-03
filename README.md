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
| **Section** | D |
| **Project Title** | Parameter-Efficient Foundation Adaptation with Ordinal Cumulative Boundaries for Diabetic Retinopathy Staging |
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
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
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