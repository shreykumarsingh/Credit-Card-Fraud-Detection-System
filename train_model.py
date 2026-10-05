# ================== CREDIT CARD FRAUD DETECTION - IMPROVED ==================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
TEST_SIZE = 0.3
N_SPLITS = 5
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
print("=" * 60)
print("  CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
print("=" * 60)

print("\n[1/7] Loading dataset...")
data = pd.read_csv("creditcard.csv")
print(f"      Loaded: {len(data):,} transactions, {data.shape[1]} features")

# ---------------- DATA ANALYSIS ----------------
print("\n[2/7] Analyzing data...")
fraud_count = data["Class"].sum()
normal_count = len(data) - fraud_count
fraud_ratio = fraud_count / len(data)

print(f"      Normal transactions: {normal_count:,}")
print(f"      Fraud transactions:  {fraud_count:,}")
print(f"      Fraud ratio:         {fraud_ratio:.4%}")

# Check for missing values
missing = data.isnull().sum().sum()
print(f"      Missing values:      {missing}")

# ---------------- FEATURE ENGINEERING ----------------
print("\n[3/7] Feature engineering...")

# Scale Amount and Time using separate RobustScalers (less sensitive to outliers)
# BUG FIX: Previously a single scaler was re-fitted on Time, overwriting Amount params
amount_scaler = RobustScaler()
time_scaler = RobustScaler()
data["scaled_Amount"] = amount_scaler.fit_transform(data["Amount"].values.reshape(-1, 1))
data["scaled_Time"] = time_scaler.fit_transform(data["Time"].values.reshape(-1, 1))

# Drop original Amount and Time
data.drop(["Amount", "Time"], axis=1, inplace=True)

# Reorder columns: put scaled features first
cols = ["scaled_Amount", "scaled_Time"] + [c for c in data.columns if c not in ["scaled_Amount", "scaled_Time", "Class"]] + ["Class"]
data = data[cols]

print("      Scaled Amount and Time features using RobustScaler")
print(f"      Final feature count: {data.shape[1] - 1}")

# Save both scalers for inference
joblib.dump({"amount": amount_scaler, "time": time_scaler}, "scaler.pkl")
print("      Saved scalers as scaler.pkl")

# ---------------- TRAIN-TEST SPLIT ----------------
print("\n[4/7] Splitting data...")
X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"      Train: {X_train.shape[0]:,} samples (Fraud: {y_train.sum()})")
print(f"      Test:  {X_test.shape[0]:,} samples (Fraud: {y_test.sum()})")

# ---------------- APPLY SMOTE ----------------
print("\n[5/7] Applying SMOTE to handle class imbalance...")
minority_count = int((y_train == 1).sum())
k_neighbors = min(5, max(1, minority_count - 1))
smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"      Before SMOTE - Normal: {(y_train == 0).sum():,}, Fraud: {(y_train == 1).sum():,}")
print(f"      After SMOTE  - Normal: {(y_train_resampled == 0).sum():,}, Fraud: {(y_train_resampled == 1).sum():,}")

# ---------------- MODEL TRAINING & COMPARISON ----------------
print("\n[6/7] Training and comparing models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight="balanced"
    ),
    "Isolation Forest": IsolationForest(
        n_estimators=200, contamination=fraud_ratio,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = {}
best_model_name = None
best_f1 = 0
best_model = None

for name, model in models.items():
    print(f"\n      --- {name} ---")

    if name == "Isolation Forest":
        # Unsupervised: train on original (non-SMOTE) data
        model.fit(X_train)
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == 1, 0, 1)
        y_pred_proba = model.decision_function(X_test)
        y_pred_proba = -y_pred_proba  # Higher = more anomalous
    else:
        # Supervised: train on SMOTE-resampled data
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }

    print(f"      F1-Score:  {f1:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      ROC-AUC:   {roc_auc:.4f}")
    print(f"      PR-AUC:    {pr_auc:.4f}")

    # Track best model by F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

print(f"\n      >>> Best Model: {best_model_name} (F1: {best_f1:.4f}) <<<")

# ---------------- SAVE BEST MODEL & RESULTS ----------------
print("\n[7/7] Saving model, results, and generating visualizations...")

# Save best model
joblib.dump(best_model, "model.pkl")
print(f"      Saved best model ({best_model_name}) as model.pkl")

# Save feature names
feature_names = list(X.columns)
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)
print("      Saved feature_names.json")

# Save all results (without numpy arrays)
results_summary = {}
for name, res in results.items():
    results_summary[name] = {k: v for k, v in res.items() if k not in ["y_pred", "y_pred_proba", "confusion_matrix"]}
    results_summary[name]["confusion_matrix"] = res["confusion_matrix"]

with open("outputs/results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

# ============ VISUALIZATIONS ============

# --- 1. Confusion Matrices for all models ---
fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
if len(results) == 1:
    axes = [axes]

for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    ax.set_title(f"{name}\nF1: {res['f1']:.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved confusion_matrices.png")

# --- 2. ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_pred_proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Model Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved roc_curves.png")

# --- 3. Precision-Recall Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res["y_pred_proba"])
    ax.plot(rec, prec, label=f"{name} (AP={res['pr_auc']:.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves - Model Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved pr_curves.png")

# --- 4. Model Comparison Bar Chart ---
metrics_to_plot = ["f1", "precision", "recall", "roc_auc", "pr_auc"]
metric_labels = ["F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"]
model_names = list(results.keys())
x = np.arange(len(metric_labels))
width = 0.8 / len(model_names)

fig, ax = plt.subplots(figsize=(10, 6))
for i, name in enumerate(model_names):
    values = [results[name][m] for m in metrics_to_plot]
    ax.bar(x + i * width, values, width, label=name)

ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_title("Model Comparison - All Metrics")
ax.set_xticks(x + width * (len(model_names) - 1) / 2)
ax.set_xticklabels(metric_labels)
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis="y")
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved model_comparison.png")

# --- 5. Feature Importance (for tree-based models) ---
if best_model_name == "Random Forest":
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(range(len(indices)), importances[indices], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top 15 Features - {best_model_name}")
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved feature_importance.png")
else:
    stale_fi = f"{OUTPUT_DIR}/feature_importance.png"
    if os.path.exists(stale_fi):
        os.remove(stale_fi)

# ============ PDF REPORT ============
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    doc = SimpleDocTemplate(f"{OUTPUT_DIR}/Fraud_Report.pdf")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Credit Card Fraud Detection Report", styles["Title"]))
    story.append(Spacer(1, 12))

    summary = (
        f"Total Transactions: {len(data):,}<br/>"
        f"Fraud Cases: {fraud_count:,}<br/>"
        f"Normal Cases: {normal_count:,}<br/>"
        f"Fraud Ratio: {fraud_ratio:.4%}<br/><br/>"
        f"Best Model: {best_model_name}<br/>"
        f"Best F1-Score: {best_f1:.4f}"
    )
    story.append(Paragraph(summary, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Results table
    table_data = [["Model"] + metric_labels]
    for name in model_names:
        row = [name] + [f"{results[name][m]:.4f}" for m in metrics_to_plot]
        table_data.append(row)

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Add charts
    for chart_file in ["confusion_matrices.png", "roc_curves.png", "pr_curves.png", "model_comparison.png"]:
        path = f"{OUTPUT_DIR}/{chart_file}"
        if os.path.exists(path):
            story.append(Paragraph(chart_file.replace(".png", "").replace("_", " ").title(), styles["Heading2"]))
            story.append(Image(path, width=480, height=360))
            story.append(Spacer(1, 12))

    if os.path.exists(f"{OUTPUT_DIR}/feature_importance.png"):
        story.append(Paragraph("Feature Importance", styles["Heading2"]))
        story.append(Image(f"{OUTPUT_DIR}/feature_importance.png", width=400, height=400))

    doc.build(story)
    print(f"      Saved Fraud_Report.pdf")

except ImportError:
    print("      Skipping PDF report (reportlab not installed)")

# ============ FINAL SUMMARY ============
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"\n  Best Model:  {best_model_name}")
print(f"  F1-Score:    {best_f1:.4f}")
print(f"  Precision:   {results[best_model_name]['precision']:.4f}")
print(f"  Recall:      {results[best_model_name]['recall']:.4f}")
print(f"  ROC-AUC:     {results[best_model_name]['roc_auc']:.4f}")
print(f"  PR-AUC:      {results[best_model_name]['pr_auc']:.4f}")
print(f"\n  Outputs saved in: {OUTPUT_DIR}/")
print(f"  Model saved as:   model.pkl")
print(f"  Scaler saved as:  scaler.pkl")
print("\n  Run 'streamlit run app.py' to launch the web app!")
print("=" * 60)
