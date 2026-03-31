# 🏦 Bank Transaction Fraud Detection


---

## 📌 Project Overview

Financial fraud is a critical threat to banking systems worldwide. This project builds, compares, and deploys **8 classification models** (6 Machine Learning + 2 Deep Learning) to automatically detect fraudulent bank transactions in real time.

The project addresses the most difficult real-world challenge in fraud detection — **severe class imbalance** — where fraudulent transactions make up less than 5% of all transactions. Standard models simply ignore the minority class and achieve misleading ~95% accuracy by predicting everything as legitimate. This project solves that using **SMOTE oversampling** and **class weighting**.

The best performing model (Random Forest) is deployed via a **Flask web application** with a modern UI where users can enter transaction details and get an instant fraud/legitimate prediction with a probability score.

---

## ✨ Key Features

- 8 models trained and compared on the same dataset
- Handles class imbalance using SMOTE + `class_weight='balanced'`
- Evaluation using F1 Score and ROC-AUC (not just accuracy)
- Automatic optimal threshold selection for LSTM and Transformer
- 10+ visualizations: confusion matrices, ROC curves, feature importance, training curves
- Live web UI with dynamic input fields powered by Flask
- All preprocessing (encoding, scaling) saved and reused at prediction time

---

## 📁 Project Structure

```
bank-fraud-detection/
│
├── model.py        # Main training pipeline (all 20 steps)
├── save_model.py                   # Run after training to export model files
├── app.py                          # Flask web server
├── index.html                      # Frontend UI
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore
│
└── (generated after running save_model.py)
    ├── model_rf.pkl                # Trained Random Forest model
    ├── scaler.pkl                  # Fitted StandardScaler
    ├── feature_columns.json        # Ordered list of feature names
    └── label_mappings.json         # Encoding maps for categorical columns
```

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8 or above
- Jupyter Notebook or JupyterLab (recommended for training)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/parth-05122005/FinanceFraudDetection.git
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ If you are inside Jupyter and get `ModuleNotFoundError` for `imblearn`, run this in a notebook cell:
> ```python
> import sys, subprocess
> subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn", "-q"])
> ```

### Step 3 — Add the Dataset

Download `Bank_Transaction_Fraud_Detection.csv` from Kaggle and place it in the project root folder.

> 📥 Dataset: [Bank Transaction Fraud Detection — Kaggle](https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection)

### Step 4 — Train the Models

Open and run `model.ipy` in Jupyter Notebook, top to bottom.

### Step 5 — Export the Model

After training completes, run `save_model.py` (either as a notebook cell or script). This generates:
- `model_rf.pkl` — trained Random Forest
- `scaler.pkl` — fitted scaler
- `feature_columns.json` — feature order
- `label_mappings.json` — categorical encodings

### Step 6 — Launch the Web App

```bash
python app.py
```

Open your browser and navigate to → **http://localhost:5000**

---

## 📊 Project Pipeline

```
Raw CSV Data
     │
     ▼
Dataset Exploration
  └── shape, dtypes, head/tail, describe, missing values, duplicates, unique counts
     │
     ▼
Exploratory Data Analysis (EDA)
  └── Fraud distribution, feature histograms, boxplots, correlation heatmap,
      categorical fraud rates, outlier detection (IQR)
     │
     ▼
Preprocessing
  ├── Drop irrelevant columns (IDs, PII)
  ├── Label Encoding (categorical → numeric)
  ├── Stratified Train/Test Split (75% / 25%)
  ├── Outlier Removal (IQR on non-fraud training rows only)
  ├── Standard Scaling (fit on train, transform both)
  └── SMOTE (oversample minority class in training set only)
     │
     ▼
Model Training
  ├── 6 ML Models (with class_weight='balanced')
  └── 2 DL Models — LSTM + Transformer (with EarlyStopping + optimal threshold)
     │
     ▼
Evaluation
  └── Accuracy, Precision, Recall, F1 Score, ROC-AUC
      Confusion Matrices, ROC Curves, PR Curves
     │
     ▼
Deployment
  └── Best model (Random Forest) served via Flask + HTML UI
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
| 7 | LSTM | Deep Learning | Class weights + Auto threshold |
| 8 | Transformer | Deep Learning | Class weights + Auto threshold |

---

## 🌲 Why Random Forest Was Chosen for Deployment

After evaluating all 8 models on the test set, **Random Forest was selected** for the web application. Here is the detailed reasoning:

### 1. Best Balance of Precision and Recall
In fraud detection, two types of errors exist:
- **False Negative** — predicting a fraud as legitimate (very costly — real money lost)
- **False Positive** — predicting a legitimate transaction as fraud (inconvenient — customer frustration)

Random Forest achieved the highest **F1 Score** among all models, meaning it balanced both error types better than any other model. A high recall ensures real frauds are caught; a reasonable precision avoids flagging too many legitimate transactions.

### 2. Handles Class Imbalance Natively
Random Forest supports `class_weight='balanced'`, which automatically adjusts the weight of each class inversely proportional to its frequency. This means the model is penalised more for missing a fraud (minority class) during training — directly addressing the imbalance problem without requiring SMOTE alone.

### 3. No Feature Scaling Required (but still benefits from it)
Unlike Logistic Regression or SVM, Random Forest is a tree-based model that does not require features to be on the same scale. This makes it more robust to features with very different ranges (e.g., Age: 20–70 vs Transaction_Amount: 0–500,000).

### 4. Built-in Feature Importance
Random Forest provides `feature_importances_` out of the box. This allows us to understand **which transaction attributes are most predictive of fraud** — a critical requirement for any real banking fraud detection system where decisions must be explainable.

### 5. Resistant to Overfitting
By averaging the predictions of hundreds of decorrelated decision trees (ensemble learning), Random Forest generalises better than a single Decision Tree. It showed a smaller gap between train and test performance compared to Gradient Boosting and AdaBoost.

### 6. Fast Inference for Real-time Use
Once trained, Random Forest predictions are made in milliseconds. For a web application that must return a result instantly when a transaction is submitted, this is essential. Deep learning models (LSTM, Transformer) are significantly slower at inference time.

### 7. Deep Learning Models Were Unstable on This Dataset
The LSTM and Transformer models struggled with the extreme class imbalance even after SMOTE and class weighting. Their accuracy scores collapsed (predicting one class entirely) across multiple runs, making them unreliable for deployment without extensive hyperparameter tuning. Random Forest consistently produced stable, reproducible results.

### Summary Table

| Criterion | Random Forest | Logistic Regression | LSTM | Gradient Boosting |
|-----------|:---:|:---:|:---:|:---:|
| High F1 Score | ✅ | ⚠️ | ⚠️ | ❌ |
| Handles Imbalance | ✅ | ✅ | ⚠️ | ⚠️ |
| Feature Importance | ✅ | ⚠️ | ❌ | ✅ |
| Fast Inference | ✅ | ✅ | ❌ | ✅ |
| Stable Results | ✅ | ✅ | ❌ | ✅ |
| No Scaling Needed | ✅ | ❌ | ❌ | ✅ |

**Conclusion:** Random Forest provided the best overall trade-off across all deployment criteria and was therefore chosen as the production model for the web application.

---

## 📈 Evaluation Metrics

| Metric | Why It Matters |
|--------|----------------|
| **Accuracy** | Overall correctness — misleading on imbalanced data |
| **Precision** | Of all flagged frauds, how many were real? |
| **Recall** | Of all actual frauds, how many did we catch? ← most critical |
| **F1 Score** | Harmonic mean of Precision & Recall — primary metric used |
| **ROC-AUC** | Overall discrimination ability across all thresholds |

> F1 Score is used as the primary ranking metric. In fraud detection, accuracy alone is highly misleading — a model that predicts "Not Fraud" for every transaction achieves 95% accuracy but catches zero frauds.

---

## 🖥️ Web Application

The Flask web app (`app.py` + `index.html`) provides:

- **Dynamic input fields** — automatically generated from the saved model's feature list
- **Smart field types** — dropdowns for low-cardinality columns, text inputs for high-cardinality ones (City, Date, Time etc.)
- **Real-time prediction** — submits to `/predict` endpoint and returns result instantly
- **Visual result** — colour-coded FRAUD 🚨 / LEGITIMATE ✅ with fraud probability bar and confidence score

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
flask
joblib
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 👤 Author

**[Parth Khandelwal - 23BDS0043 and Sudeep Sharma - 23BDS0315]**

---

## 📄 License

This project is developed for academic purposes only.