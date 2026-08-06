"""
make_roc_figure.py
==================
Reproduces the ROC curve + confusion matrix of the saved transient_model.joblib
on the held-out Deep-TAO test set (same split as train_model.py, random_state=42).

Output: report_figures/roc_curve.png  and  report_figures/confusion_matrix.png
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

from train_model import load_dataset, DATA_DIR_DEFAULT, MODEL_OUT_DEFAULT

OUT_DIR = "report_figures"
os.makedirs(OUT_DIR, exist_ok=True)

print("[1/4] Loading dataset (this can take a few minutes)...")
X, y, _ = load_dataset(str(DATA_DIR_DEFAULT))
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print("[2/4] Recreating the same train/test split (random_state=42)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("[3/4] Loading saved model + scoring test set...")
md = joblib.load(str(MODEL_OUT_DEFAULT))
clf, scaler = md["classifier"], md["scaler"]
thr = md.get("threshold", 0.5)

X_test_s = scaler.transform(X_test)
y_prob = clf.predict_proba(X_test_s)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"      AUC = {roc_auc:.3f}  (model file stores {md.get('auc'):.3f})")

# ── ROC curve ────────────────────────────────────────────────
plt.figure(figsize=(5.2, 5.0))
plt.plot(fpr, tpr, color="#2563eb", lw=2.4,
         label=f"Random Forest (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="#9aa5b1", lw=1.3, ls="--",
         label="Random baseline (AUC = 0.50)")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve — Transient Classifier", fontsize=13)
plt.legend(loc="lower right", fontsize=10, frameon=True)
plt.grid(True, ls="--", alpha=0.35)
plt.xlim(-0.01, 1.0); plt.ylim(0.0, 1.01)
plt.tight_layout()
roc_path = os.path.join(OUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=200, bbox_inches="tight")
print(f"      saved: {roc_path}")

# ── Confusion matrix (at the saved Youden-J threshold) ───────
y_pred = (y_prob >= thr).astype(int)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4.6, 4.2))
im = ax.imshow(cm, cmap="Blues")
labels = ["Non-transient", "Transient"]
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title(f"Confusion Matrix (threshold = {thr:.2f})", fontsize=12)
thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=15, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=200, bbox_inches="tight")
print(f"      saved: {cm_path}")

print("[4/4] Done.")
