#  Forest Fire Detection — EfficientNetB0

> Binary image classifier that detects **fire vs. no-fire** in forest scenes, built with transfer learning on EfficientNetB0 and trained on the *DeepFire* dataset.

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This project aims to automatically detect forest fires from aerial or ground-level images. Early fire detection can drastically reduce response time and limit ecological damage.

The classifier is trained end-to-end using **EfficientNetB0** as a backbone (pre-trained on ImageNet), with a two-phase transfer learning approach: head training followed by fine-tuning of the top layers.

---

## Dataset

**DeepFire — Forest Fire Dataset**

- Two classes: `fire` / `nofire`
- Split: `Training/` and `Testing/` directories, each containing `fire/` and `nofire/` subfolders
- Images resized to **224 × 224** pixels
- 15% of training data is reserved as a validation set

---

## Model Architecture

| Component | Detail |
|---|---|
| Backbone | EfficientNetB0 (ImageNet weights) |
| Input size | 224 × 224 × 3 |
| Head | GlobalAveragePooling2D → Dropout → Dense(1, sigmoid) |
| Loss | Binary Cross-Entropy |
| Optimizer | Adam |
| Output | Probability score ∈ [0, 1] — threshold 0.5 |

---

## Training Strategy

Training is split into two phases:

**Phase 1 — Head training** (backbone frozen)
- Epochs: 10
- Learning rate: `1e-3`
- Only the classification head is trained

**Phase 2 — Fine-tuning** (top 25% of backbone unfrozen)
- Epochs: 20
- Learning rate: `1e-5`
- The best checkpoint is saved automatically via `ModelCheckpoint`

Callbacks used: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`.

---

## Results

Metrics computed on the held-out test set:

| Metric | Value |
|---|---|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |
| ROC-AUC | — |

> *(Fill in with your actual scores after running `version1.ipynb`)*

Training curves (accuracy, loss, precision, recall) and confusion matrix are generated automatically and saved to `fire_wow_output/reports/`.

---

## Project Structure

```
.
├── version1.ipynb              # Full training pipeline (data prep → training → evaluation)
├── version1__1_.ipynb          # Variant / iteration of the training notebook
├── test1.ipynb                 # Internal test / exploration notebook
├── test2.ipynb                 # External evaluation on a custom image folder
├── final_efficientnetb0.keras  # Saved final model weights
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/forest-fire-detection.git
cd forest-fire-detection
```

### 2. Install dependencies

```bash
pip install tensorflow keras scikit-learn pandas matplotlib seaborn pillow
```

### 3. Prepare the dataset

Download the [DeepFire — Forest Fire Dataset](https://www.kaggle.com/datasets/) and place it at:

```
/path/to/DeepFire/Forest Fire Dataset/
    Training/
        fire/
        nofire/
    Testing/
        fire/
        nofire/
```

Update `DATASET_ROOT` in `version1.ipynb` accordingly.

---

## Usage

### Train the model

Open and run `version1.ipynb` cell by cell (Google Colab or local Jupyter).

The final model is saved as:
```
fire_wow_output/models/final_efficientnetb0.keras
```

### Run inference on a custom image folder

Use `test2.ipynb`. Set the following paths:

```python
MODEL_PATH = "path/to/final_efficientnetb0.keras"
IMAGE_DIR  = "path/to/your/images"   # subfolders: fire/ and no_fire/
OUTPUT_DIR = "path/to/results"
```

The notebook outputs per-image predictions, overall metrics, and a list of misclassified images.

---

## Dependencies

| Package | Version |
|---|---|
| Python | ≥ 3.9 |
| TensorFlow / Keras | ≥ 2.12 |
| scikit-learn | ≥ 1.2 |
| NumPy | ≥ 1.23 |
| Pandas | ≥ 1.5 |
| Matplotlib | ≥ 3.6 |
| Seaborn | ≥ 0.12 |
| Pillow | ≥ 9.0 |

---

## License

This project is released under the [MIT License](LICENSE).

---
