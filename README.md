# 🐱 CatMeows-Recognition

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/ML-SVM-orange.svg" alt="SVM">
  <img src="https://img.shields.io/badge/Audio-MFCC-purple.svg" alt="MFCC">
</p>

<p align="center">
  <b>Cat Meow Emotion Recognition System Based on MFCC Features and SVM Classifier</b>
</p>

<p align="center">
  <a href="README-zh.md">🇨🇳 中文文档</a> | <a href="README.md">🇬🇧 English</a>
</p>

---

## 📖 Overview

This project implements an audio classification system that can recognize different types of cat meows. By extracting MFCC (Mel-Frequency Cepstral Coefficients) features from audio signals and using SVM (Support Vector Machine) classifier, the system can distinguish between three different cat behavioral states:

| Category | Description |
|----------|-------------|
| **Brushing** | Cat meows during brushing/grooming |
| **UnfamiliarSurroundings** | Cat meows in unfamiliar environments |
| **WaitForFood** | Cat meows when waiting for food |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/CatMeows-Recognition.git
cd CatMeows-Recognition
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

**Run the training pipeline:**
```bash
python run.py
```

This will:
1. Load the audio dataset from `dataset/` folder
2. Extract MFCC features from all audio files
3. Perform 10-fold cross-validation
4. Generate visualizations and save to `outputs/`
5. Save the trained model

## 📁 Project Structure

```
CatMeows-Recognition/
├── dataset/                       # Audio dataset
│   ├── Brushing/                  # Brushing category samples
│   ├── UnfamiliarSurroundings/    # Unfamiliar surroundings samples
│   └── WaitForFood/               # Wait for food samples
├── src/                           # Source code
│   ├── data/
│   │   └── loader.py              # Dataset loading module
│   ├── features/
│   │   └── mfcc.py                # MFCC feature extraction
│   └── models/
│       └── svm.py                 # SVM classifier
├── tools/                         # Utility tools
│   ├── logger.py                  # Logging utilities
│   └── visualization.py           # Visualization functions
├── img/                           # Sample visualization images
├── outputs/                       # Training outputs
├── run.py                         # Main entry point
└── requirements.txt               # Dependencies
```

## 🔧 Technical Details

### MFCC Feature Extraction

The system extracts the following features from each audio sample:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_mfcc` | 20 | Number of MFCC coefficients |
| `n_mels` | 26 | Number of mel filterbanks |
| `win_length` | 30ms | Window length |
| `hop_length` | 20ms | Hop length |
| `fmin` | 0 Hz | Minimum frequency |
| `fmax` | 4000 Hz | Maximum frequency |

**Feature composition:**
- MFCC coefficients (20)
- Delta coefficients (1st order derivative)
- Delta-delta coefficients (2nd order derivative)
- Statistical pooling: mean, std, min, max

**Total feature dimension:** 20 × 3 (MFCC + Δ + ΔΔ) × 4 (stats) = **240 features**

### SVM Classifier

| Parameter | Value |
|-----------|-------|
| Kernel | RBF (Radial Basis Function) |
| C | 10 |
| Gamma | scale |

### Evaluation Metrics

- Accuracy: Overall classification accuracy
- F1 Score: Macro and weighted F1 scores
- Precision/Recall: Per-class and macro averages
- AUC-ROC: Area under ROC curve (One-vs-Rest)
- Confusion Matrix: Detailed classification breakdown

## 🖼️ Visualizations

### Spectrogram Comparison
<p align="center">
  <img src="img/spectrogram_comparison.png" alt="Spectrogram Comparison" width="80%">
</p>
<p align="center"><i>Spectrogram comparison of different cat meow categories</i></p>

### t-SNE Feature Visualization
<p align="center">
  <img src="img/tsne.png" alt="t-SNE Visualization" width="60%">
</p>
<p align="center"><i>t-SNE visualization showing feature clustering by category</i></p>

### SVM Decision Boundary
<p align="center">
  <img src="img/svm_decision_boundary.png" alt="SVM Decision Boundary" width="60%">
</p>
<p align="center"><i>SVM decision boundary visualization in PCA-reduced 2D space</i></p>

### Confusion Matrix
<p align="center">
  <img src="img/confusion_matrix.png" alt="Confusion Matrix" width="50%">
</p>
<p align="center"><i>Confusion matrix showing classification performance</i></p>

### ROC Curve
<p align="center">
  <img src="img/roc_curve.png" alt="ROC Curve" width="60%">
</p>
<p align="center"><i>ROC curves for each category (One-vs-Rest)</i></p>

### Precision-Recall Curve
<p align="center">
  <img src="img/pr_curve.png" alt="PR Curve" width="60%">
</p>
<p align="center"><i>Precision-Recall curves for each category</i></p>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  ⭐ If this project helps you, please give it a Star! ⭐
</p>
