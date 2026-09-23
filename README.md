# Brain Tumor Classification 🧠

A machine learning project that classifies brain tumor types from MRI images using Python. Compares **6 models** — 4 classical ML algorithms and 2 deep learning models (CNN and ResNet50) — with CNN achieving the best accuracy of **89.8%**.

---

## What it does

- Loads and preprocesses MRI brain scan images
- Performs EDA to understand the dataset — class distribution, image samples, pixel statistics
- Trains and compares 6 models across classical ML and deep learning
- Evaluates each model with accuracy scores, confusion matrices, and a full model comparison chart
- Saves all graphs and visualizations to the `Graphs/` folder

---

## Tumor Classes

| Label | Class |
|-------|-------|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | Pituitary Tumor |
| 3 | No Tumor |

---

## Models & Results

### Classical ML (blue)
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 81.2% |
| Decision Tree | 78.4% |
| Random Forest | 89.1% |
| SVM | 82.9% |

### Deep Learning (orange)
| Model | Accuracy |
|-------|----------|
| CNN (custom) | **89.8% ✅ Best** |
| ResNet50 (transfer learning) | 81.1% |

> CNN edges out Random Forest by 0.7% and outperforms ResNet50 — showing that a well-tuned custom CNN can beat a heavyweight pretrained model on this dataset.

---

## Why I built this

Medical image classification is one of the most impactful real-world applications of ML. Wanted to go beyond classical algorithms and compare them directly against deep learning approaches on the same dataset. The result was interesting — a custom CNN outperformed ResNet50 (transfer learning), which shows that pretrained models don't always win on domain-specific medical data.

---

## Tech Stack

| Library | Usage |
|---------|-------|
| Python | Core language |
| scikit-learn | Classical ML models, metrics, train/test split |
| TensorFlow / Keras | CNN and ResNet50 deep learning models |
| NumPy | Array operations and image data handling |
| matplotlib | Confusion matrices, accuracy comparison chart, EDA plots |
| OpenCV / PIL | Image loading and preprocessing |
| UV | Package manager (`uv.lock` for reproducible installs) |

---

## How to run it

**1. Clone the repo**
```bash
git clone https://github.com/KhizerAhmad/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification
```

**2. Install dependencies**

With pip:
```bash
pip install -r requirements.txt
```

Or with UV:
```bash
uv sync
```

**3. Add the dataset**

Download the Brain Tumor MRI dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and place it in the project directory.

**4. Run it**
```bash
python main.py
```

---

## Project Structure

```
Brain-Tumor-Classification/
│
├── main.py              # Full pipeline — preprocessing, EDA, training, evaluation
├── Graphs/
│   ├── CNN_CM.png                   # CNN confusion matrix
│   ├── ResNet50_CM.png              # ResNet50 confusion matrix
│   └── All_AI_Models_Accuracy.png  # Full model comparison chart
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── .gitignore
```

---

## Pipeline Overview

```
MRI Images → Preprocessing → EDA
     │
     ├── Classical ML ──→ Logistic Regression
     │                ──→ Decision Tree
     │                ──→ Random Forest
     │                ──→ SVM
     │
     └── Deep Learning ─→ Custom CNN         ← Best (89.8%)
                       ─→ ResNet50 (Transfer Learning)
     │
     └── Evaluate all → Confusion Matrices + Accuracy Chart → Graphs/
```

---

## Evaluation Metrics

- Accuracy score per model
- Confusion matrix heatmaps per model
- Side-by-side all-models accuracy comparison bar chart

---

## Results Graphs

**All Models Accuracy Comparison**

![All Models Accuracy](Graphs/All_AI_Models_Accuracy.png)

**CNN Confusion Matrix**

![CNN Confusion Matrix](Graphs/CNN_CM.png)

**ResNet50 Confusion Matrix**

![ResNet50 Confusion Matrix](Graphs/ResNet50_CM.png)

---

## Author

**Khizer Ahmad** — built this as part of my Research Assistant work at the University of Lahore, extending a classical ML classification project with deep learning models to compare approaches on real medical imaging data.

Feel free to fork it and try adding more augmentation or fine-tuning ResNet50 further.
