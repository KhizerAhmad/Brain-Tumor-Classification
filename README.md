# Brain Tumor Classification 🧠

A machine learning project that classifies brain tumor types from MRI images using Python. Covers the full ML pipeline — image preprocessing, exploratory data analysis (EDA), training multiple models, and evaluating them with proper metrics and visualizations.

---

## What it does

- Loads and preprocesses MRI brain scan images
- Performs EDA to understand the dataset — class distribution, image samples, pixel statistics
- Trains and compares 4 ML classification models
- Evaluates each model with accuracy scores, classification reports, and confusion matrices
- Saves all graphs and visualizations to the `Graphs/` folder

---

## Tumor Classes

The dataset classifies MRI images into tumor types:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

---

## Models Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline classifier |
| Decision Tree | Tree-based, interpretable model |
| Random Forest | Ensemble of decision trees |
| Support Vector Machine (SVM) | High-dimensional image classification |

---

## Why I built this

Medical image classification is one of the most impactful real-world applications of ML. Wanted to work with actual image data (not just tabular CSVs), go through a proper preprocessing pipeline, and compare how classical ML algorithms handle a multi-class classification problem on image features.

---

## Tech Stack

| Library | Usage |
|---------|-------|
| Python | Core language |
| scikit-learn | ML models, metrics, train/test split |
| NumPy | Array operations and image data handling |
| matplotlib | Visualizations, confusion matrices, EDA plots |
| OpenCV / PIL | Image loading and preprocessing |
| UV | Package manager (`uv.lock` for dependency locking) |

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

Or with UV (faster):
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
├── main.py              # Full ML pipeline — preprocessing, EDA, training, evaluation
├── Graphs/              # Generated plots and confusion matrices
├── requirements.txt     # pip dependencies
├── pyproject.toml       # Project metadata
├── uv.lock              # UV lockfile for reproducible installs
└── .gitignore
```

---

## Pipeline Overview

```
MRI Images → Preprocessing → Feature Extraction → Train/Test Split
     → Logistic Regression  ┐
     → Decision Tree        ├─ Evaluate → Confusion Matrix + Accuracy
     → Random Forest        │
     → SVM                  ┘
          → Compare Results → Save Graphs
```

---

## Evaluation Metrics

- Accuracy score per model
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmaps saved to `Graphs/`

---

## Sample Results

> *(Add your accuracy scores per model here once you run it)*

---

## Screenshots

<img width="798" height="563" alt="Class Distribution" src="https://github.com/user-attachments/assets/4aad7d6f-ce80-4dae-bce6-747e4d796933" />
<img width="797" height="570" alt="Accuracy Graph" src="https://github.com/user-attachments/assets/68fa1248-207d-451d-a4a9-edff4428d957" />


---

## Author

**Khizer Ahmad** — built this to get hands-on with medical image classification, multi-class ML evaluation, and working through a real-world end-to-end ML pipeline.

Feel free to fork it and try adding a CNN for better accuracy on image data.
