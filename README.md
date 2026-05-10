# 5329_ass2
# MAE Frequency-Aware Masking Analysis
## Project Overview

This project investigates how different masking strategies influence the frequency characteristics of representations learned by Masked Autoencoders (MAE).

Instead of focusing solely on downstream accuracy, this project analyzes MAE from a frequency-domain perspective and studies how masking shape affects:

* Low-frequency semantic representations
* High-frequency local details
* Layer-wise representation behavior
* Downstream task performance

The project aims to provide deeper insights into the representation learning mechanism of self-supervised vision models.

---

# Research Questions

1. How do different masking shapes affect frequency representations in MAE?

2. How does masking ratio influence low-frequency and high-frequency learning behavior?

3. Do frequency biases correlate with downstream task performance?

---

# Masking Strategies

The project compares multiple masking methods:

* Random Masking
* Block Masking
* Edge-Prioritized Masking
* Structure-Prioritized Masking

---

# Dataset

Primary Dataset:

* CIFAR-10

Optional Additional Dataset:

* CIFAR-100

---

# Project Structure

```text
mae-frequency-analysis/
│
├── datasets/
├── models/
├── masks/
├── analysis/
├── notebooks/
├── figures/
├── paper/
└── README.md
```

---

# Methodology

The project follows the workflow below:

1. Train MAE encoder using different masking strategies
2. Extract encoder representations
3. Apply Fourier Transform (FFT)
4. Analyze low/high-frequency energy distributions
5. Perform layer-wise representation analysis
6. Evaluate downstream task performance

---

# Frequency Analysis

The project uses Fourier Transform (FFT) to analyze encoder embeddings.

Main metrics include:

* High-frequency energy ratio
* Low-frequency energy ratio
* Layer-wise frequency evolution
* Frequency spectrum visualization

---

# Experimental Setup

Model:

* ViT-small Patch16

Framework:

* PyTorch
* timm

Training:

* CIFAR-10
* 224×224 resized input
* Multiple masking ratios

---

# Team Responsibilities

Member A:

* Baseline MAE training
* Dataset pipeline
* Feature extraction

Member B:

* Masking strategy implementation
* Mask visualization
* Ablation study

Member C:

* FFT analysis
* Frequency visualization
* Representation analysis

---

# Environment

Recommended environment:

* Python 3.10+
* PyTorch
* CUDA-enabled GPU
* Google Colab

---

# Installation

```bash
pip install torch torchvision timm matplotlib
```

---

# Future Work

Potential future extensions include:

* Adaptive masking policies
* Frequency-aware dynamic masking
* Medical image validation
* Segmentation downstream tasks

---

# License

This repository is for academic and educational purposes only.
