# 💳 Credit Card Fraud Detection System

This project uses Machine Learning techniques to detect fraudulent credit card transactions using classification algorithms and imbalanced data handling methods.

---

# 📌 Project Overview

Credit card fraud is one of the major financial threats in digital transactions.  
This project aims to identify fraudulent transactions using machine learning models trained on transaction datasets.

The system performs fraud analysis and predicts whether a transaction is legitimate or fraudulent.

---

# 🧠 Objective

To build a machine learning model that detects fraudulent credit card transactions with strong recall and precision on a highly imbalanced dataset.

---

# 🧩 Features

- Fraud Detection using Machine Learning
- Imbalanced Dataset Handling using SMOTE
- Transaction Classification
- Performance Evaluation Metrics
- Real-Time Prediction System
- Streamlit Interactive Dashboard

---

# 📂 Dataset

- **Source:** [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Provided by:** Machine Learning Group – ULB (Université Libre de Bruxelles) and Worldline
- **Content:** Transactions made by European cardholders over two days in September 2013
- **Size:** 284,807 transactions, of which 492 are fraudulent (0.172%) — a highly imbalanced dataset
- **Features:**
  - `V1`–`V28`: anonymized PCA-transformed components
  - `Time`: seconds elapsed since the first transaction
  - `Amount`: transaction amount
  - `Class`: target label (`1` = fraud, `0` = normal)
- **License:** Database Contents License (DbCL) v1.0

**Setup:** Download `creditcard.csv` from the Kaggle link above and place it in the project root before running `python train_model.py`.

**Citation:** Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI.

---

# ⚙️ Technologies Used

- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn (SMOTE)
- ReportLab (PDF report)
- Streamlit

---

# 🧮 Steps Performed

## Data Collection
- Loaded and analyzed credit card transaction dataset.
- Explored fraudulent and non-fraudulent transaction distribution.

## Data Preprocessing
- Handled imbalanced classes using SMOTE on the training set.
- Performed feature scaling and normalization (`Amount` and `Time` scaled with separate RobustScalers).

## Model Building
- Trained and compared Logistic Regression, Random Forest and Isolation Forest.
- Split dataset into 70% training and 30% testing sets (stratified).
- Saved the best model (by F1-score) as `model.pkl`.

## Model Evaluation
- Evaluated model performance using:
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - PR-AUC

## Deployment
- Built a Streamlit web application for real-time fraud prediction and transaction analysis.

---

# 📊 Results

| Metric (Random Forest, test set) | Performance |
|---|---|
| F1-Score | 0.739 |
| Precision | 0.684 |
| Recall | 0.804 |
| ROC-AUC | 0.977 |
| PR-AUC | 0.811 |

Trained on the full Kaggle Credit Card Fraud dataset (284,807 transactions, 492 frauds) with a 70/30 stratified split. Accuracy is not reported because it is misleading on highly imbalanced data.

---

# 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/shreykumarsingh/Credit-Card-Fraud-Detection-System.git
cd Credit-Card-Fraud-Detection-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model (requires `creditcard.csv` in the project root):

```bash
python train_model.py
```

Run the application:

```bash
streamlit run app.py
```

To try batch prediction, upload `sample_transactions.csv` on the **Batch Prediction** page.
