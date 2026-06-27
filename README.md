# Diagnosing Cultural Bias in AVA Trained Aesthetic Scoring

<p align="center">

### A Ground Truth Free Framework for Cross Cultural Evaluation of Deep Learning Based Image Aesthetic Assessment

</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![IEEE](https://img.shields.io/badge/Research-IEEE-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)]()

</p>

---

<p align="center">

**Pipeline Overview**

Replace the image below with your pipeline figure.

```markdown
<p align="center">
<img src="outputs/figures/paper_figures/fig_pipeline.png" width="900">
</p>
```

</p>

---

## Abstract

Image aesthetic assessment models are almost exclusively trained on Western centric datasets such as AVA. Although these models achieve strong benchmark performance, their behaviour across different cultural contexts remains largely unexplored due to the absence of ground truth aesthetic labels for non Western imagery.

This repository accompanies the paper **Diagnosing Cultural Bias in AVA Trained Aesthetic Scoring** and presents a fully reproducible framework for diagnosing behavioural differences without requiring labeled evaluation datasets.

Instead of measuring accuracy, the framework studies changes in model behaviour through statistical analysis, uncertainty estimation, and explainable artificial intelligence.

---

# Table of Contents

* Overview
* Motivation
* Contributions
* Repository Structure
* Experimental Pipeline
* Dataset
* Results
* Reproducing the Paper
* Model
* Citation
* License
* Contact

---

# Why This Research?

Modern aesthetic assessment models learn visual preferences primarily from Western photography communities.

This naturally raises an important research question.

> **How does an AVA trained model behave when evaluating photographs originating from different cultural regions?**

Since no large scale aesthetic benchmark exists for most regions of the world, conventional accuracy based evaluation becomes impossible.

This project introduces a behavioural diagnostic framework capable of studying model behaviour without requiring any ground truth ratings.

---

# Contributions

✅ Introduces a novel ground truth free evaluation framework.

✅ Fine tunes a ResNet18 aesthetic regressor following the NIMA formulation.

✅ Independently collects three culturally diverse photographic evaluation datasets.

✅ Develops a fully automated Wikimedia Commons scraping and filtering pipeline.

✅ Releases provenance records for complete reproducibility.

✅ Performs explainability analysis using Grad CAM.

✅ Demonstrates statistically significant behavioural shifts across multiple cultural regions.

---

# Repository Structure

```text
cultural-bias-aesthetic-scoring
│
├── assets
├── data
│   ├── ava
│   └── cultural_bias
│
├── docs
├── models
│   └── checkpoints
│
├── notebooks
├── outputs
│   ├── figures
│   └── logs
│
├── paper
├── src
│   ├── training
│   ├── analysis
│   └── visualization
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Experimental Pipeline

```text
AVA Dataset

↓

Data Preparation

↓

ResNet18 Fine Tuning

↓

Model Selection

↓

Frozen Model

↓

Inference on Non Western Datasets

↓

Distribution Shift Analysis

↓

Grad CAM Analysis

↓

Statistical Evaluation
```

---

# Dataset

## Training Dataset

| Dataset | Images |
| ------- | -----: |
| AVA     | 23,198 |

The repository contains AVA metadata only.

The original AVA image dataset is not redistributed due to licensing and storage limitations.

---

## Curated Evaluation Dataset

The non Western evaluation dataset was independently collected for this research.

The complete collection pipeline includes

* Recursive category traversal
* Automatic category filtering
* Resolution filtering
* License verification
* Duplicate removal
* Manual quality inspection
* Provenance logging

### Final Dataset

| Region                       |  Images |
| ---------------------------- | ------: |
| East Asia                    |     186 |
| South Asia                   |     289 |
| Middle East and North Africa |     212 |
| **Total**                    | **687** |

Approximately **188,000 candidate image entries** were processed before producing the final benchmark.

---

# Results

## Distribution Shift

Replace with:

```markdown
<p align="center">
<img src="outputs/figures/paper_figures/fig_distributions.png" width="850">
</p>
```

---

## Attention Entropy

Replace with:

```markdown
<p align="center">
<img src="outputs/figures/paper_figures/fig_entropy.png" width="850">
</p>
```

---

## Grad CAM Visualization

Replace with:

```markdown
<p align="center">
<img src="outputs/figures/paper_figures/fig_gradcam.png" width="850">
</p>
```

---

# Key Findings

| Observation            | Finding                               |
| ---------------------- | ------------------------------------- |
| Score Distribution     | Significant regional shifts           |
| Prediction Range       | Compressed on all evaluation datasets |
| Prediction Uncertainty | Largely unchanged                     |
| Grad CAM Entropy       | Consistently more diffuse             |
| Border to Centre Ratio | Significant only for MENA             |

---

# Running the Project

Train the model

```bash
python src/training/train_ava.py
```

Evaluate AVA

```bash
python src/training/eval_ava_test.py
```

Run inference

```bash
python src/analysis/inference_nonwestern.py
```

Distribution analysis

```bash
python src/analysis/dist_shift_analysis.py
```

Grad CAM analysis

```bash
python src/analysis/gradcam_analysis.py
```

---

# Trained Model

The repository includes the best performing checkpoint

```text
models/checkpoints/resnet18_ava_best.pt
```

---

# Citation

```bibtex
@inproceedings{aghayeva2026,
title={Diagnosing Cultural Bias in AVA Trained Aesthetic Scoring: A Ground Truth Free Framework},
author={Aghayeva, Nargiz},
year={2026}
}
```

---

# Acknowledgements

This work was conducted at ADA University.

The author thanks the CeDAR Laboratory for computational resources and technical support.

---

# Contact

**Nargiz Aghayeva**

School of Information Technologies and Engineering

ADA University

Baku, Azerbaijan

Email: [nargizkaghayeva@gmail.com](mailto:nargizkaghayeva@gmail.com)
