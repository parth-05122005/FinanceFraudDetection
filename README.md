# 🏦 Bank Transaction Fraud Detection

> Final Year Project — Comparative Study of Machine Learning and Deep Learning Models for Fraud Detection

---

## 📌 Project Overview

This project builds and compares **8 classification models** (6 ML + 2 Deep Learning) to detect fraudulent bank transactions. It addresses the real-world challenge of **class imbalance** using SMOTE and class weighting, and evaluates models using industry-standard metrics including F1 Score and ROC-AUC.

### Key Highlights
- **Dataset**: Bank Transaction Fraud Detection (CSV)
- **Models Compared**: Random Forest, Decision Tree, AdaBoost, Logistic Regression, Gradient Boosting, Naive Bayes, LSTM, Transformer
- **Imbalance Handling**: SMOTE + `class_weight='balanced'`
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Visualizations**: 10+ plots including confusion matrices, ROC curves, feature importance

---

## 📁 Project Structure

```
bank-fraud-detection/
│
├── model.ipynb       # Main project code (all steps)
├── Bank_Transaction_Fraud_Detection.csv  # Dataset (place here before running)
│
├── outputs/                       # Generated plots (auto-created on run)
│   ├── eda_fraud_distribution.png
│   ├── eda_feature_distributions.png
│   ├── eda_boxplots.png
│   ├── eda_correlation_heatmap.png
│   ├── feature_importance.png
│   ├── cm_ml_models.png
│   ├── cm_dl_models.png
│   ├── roc_ml_models.png
│   ├── dl_training_curves.png
│   ├── pr_curve_dl.png
│   └── final_comparison.png
│
└── README.md
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8 or above
- Jupyter Notebook or JupyterLab (recommended) — or run as a plain `.py` script

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/bank-fraud-detection.git
cd bank-fraud-detection
```

### Step 2 — Install Dependencies

Run this once inside your Jupyter notebook or terminal:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn tensorflow
```

> ⚠️ If you are using Jupyter and get `ModuleNotFoundError`, run this in a notebook cell instead:
> ```python
> import sys, subprocess
> subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn", "-q"])
> ```

### Step 3 — Add the Dataset

Place `Bank_Transaction_Fraud_Detection.csv` in the same folder as the script.

> Dataset source: [Kaggle — Bank Transaction Fraud Detection](https://www.kaggle.com/)

### Step 4 — Run the Project

**Option A — Jupyter Notebook (Recommended)**
```bash
jupyter notebook
```
Open `fraud_detection_final.py`, run cells top to bottom (`Shift + Enter`).

**Option B — Run as Python Script**
```bash
python fraud_detection_final.py
```

---

## 📊 Project Pipeline

```
Raw CSV Data
     │
     ▼
Dataset Exploration (head, info, describe, missing values, duplicates)
     │
     ▼
EDA (fraud distribution, boxplots, histograms, correlation heatmap)
     │
     ▼
Preprocessing
  ├── Drop irrelevant columns (IDs, PII)
  ├── Label Encoding (categorical → numeric)
  ├── Train/Test Split (75/25, stratified)
  ├── Outlier Removal (IQR, non-fraud only)
  ├── Standard Scaling
  └── SMOTE (fix class imbalance)
     │
     ▼
ML Models (6 models with class_weight='balanced')
     │
     ▼
Deep Learning Models (LSTM + Transformer with optimal threshold)
     │
     ▼
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
     │
     ▼
Visualizations + Final Comparison Table
```

---

## 🤖 Models Used

| # | Model | Type | Imbalance Handling |
|---|-------|------|--------------------|
| 1 | Random Forest | ML — Ensemble | `class_weight='balanced'` |
| 2 | Decision Tree | ML — Tree | `class_weight='balanced'` |
| 3 | AdaBoost | ML — Boosting | SMOTE |
| 4 | Logistic Regression | ML — Linear | `class_weight='balanced'` |
| 5 | Gradient Boosting | ML — Boosting | SMOTE |
| 6 | Naive Bayes | ML — Probabilistic | SMOTE |
| 7 | LSTM | Deep Learning | Class weights + Optimal threshold |
| 8 | Transformer | Deep Learning | Class weights + Optimal threshold |

---

## 📈 Evaluation Metrics

| Metric | Why It Matters for Fraud Detection |
|--------|-------------------------------------|
| **Accuracy** | Overall correctness — misleading on imbalanced data |
| **Precision** | Of all predicted frauds, how many are real? (avoids false alarms) |
| **Recall** | Of all actual frauds, how many did we catch? (most critical) |
| **F1 Score** | Harmonic mean of Precision & Recall — primary metric |
| **ROC-AUC** | Model's ability to distinguish fraud vs non-fraud overall |

> 📌 **F1 Score** is used as the primary ranking metric because accuracy alone is misleading on imbalanced fraud datasets.

---

## 🖼️ Sample Outputs

The following plots are automatically generated and saved:

- `eda_fraud_distribution.png` — Class imbalance bar chart
- `eda_boxplots.png` — Feature distributions split by fraud label
- `eda_correlation_heatmap.png` — Feature correlation matrix
- `feature_importance.png` — Top 10 features from Random Forest
- `cm_ml_models.png` — Confusion matrices for all 6 ML models
- `roc_ml_models.png` — ROC curves with AUC scores
- `dl_training_curves.png` — LSTM & Transformer accuracy/loss over epochs
- `final_comparison.png` — Side-by-side bar chart of all models on all metrics

---

## ⚙️ Key Design Decisions

**Why SMOTE?**
The dataset has severe class imbalance (fraud is a rare event). Without SMOTE, models simply predict "Not Fraud" for everything and still achieve ~95% accuracy. SMOTE generates synthetic fraud samples so models actually learn fraud patterns.

**Why class_weight='balanced' AND SMOTE?**
SMOTE rebalances the training data. `class_weight='balanced'` further penalizes the model for missing fraud cases during training. Using both together gives the best results.

**Why optimal threshold instead of 0.5?**
For LSTM and Transformer, the default 0.5 threshold may not be optimal on imbalanced data. We use the Precision-Recall curve to find the threshold that maximizes F1 Score automatically.

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
tensorflow
```

Full install command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```

---

## 👤 Author

**[Parth Khandelwal - 23BDS0043 and Sudeep Sharma - 23BDS0315]**
3rd Year B.Tech — [Computer Science with Specialization in Data Science]
[Vellore Institute of Technology], [2026]

---

## 📄 License

This project is for academic purposes only.