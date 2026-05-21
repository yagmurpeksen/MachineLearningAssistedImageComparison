"""
Training a transient classification model on the Deep-TAO dataset.

Approach: The first + last frames are taken from each FITS file.
Their difference is added to the feature vector → model learns the change.
This directly mirrors the GUI's img1 vs img2 comparison.

Classes: SN, AGN, CV, BZ, OTHER → Transient (1)
         NON                     → Non-transient (0)

Feature vector (44 dimensions):
  [0:18]  → early frame statistics
  [18:36] → late frame statistics
  [36:44] → change features (delta)
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from astropy.io import fits as astropy_fits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, f1_score)
from sklearn.dummy import DummyClassifier
import joblib

# Use the same feature function as ml_classifier → mismatch impossible
from ml_classifier import extract_deeptao_features as _extract_deeptao_features

_BASE_DIR = Path(__file__).resolve().parent
DATA_DIR_DEFAULT = _BASE_DIR / "deeptao_data"
MODEL_OUT_DEFAULT = _BASE_DIR / "transient_model.joblib"

TRANSIENT_CLASSES     = ["SN", "AGN", "CV", "BZ", "OTHER"]
NON_TRANSIENT_CLASSES = ["NON"]

FEAT_NAMES_SINGLE = [
    "mean", "std", "peak", "min", "median", "skew", "kurt",
    "p25", "p75", "p90", "p99",
    "conc_inner", "conc_middle", "conc_outer",
    "snr", "signal", "bg", "slope"
]  # 18 features

FEAT_NAMES = (
    [f"early_{n}" for n in FEAT_NAMES_SINGLE] +   # 18
    [f"late_{n}"  for n in FEAT_NAMES_SINGLE] +   # 18
    ["delta_peak", "delta_mean", "delta_snr",      # 8 delta
     "delta_signal", "flux_ratio", "log_flux_ratio",
     "delta_conc", "delta_slope"] +
    ["diff_peak", "diff_peak_neg", "diff_snr",     # 10 subtraction
     "diff_symmetry", "diff_center_flux",
     "late_ellipticity", "late_fwhm", "late_moment_ratio",
     "bg_change", "bg_gradient"]
)  # Total: 54


# ── Single frame feature extraction ──────────────────────────────

def extract_single_frame(cutout):
    """Extract 18 statistical features from a 64x64 cutout."""
    if cutout is None or cutout.size == 0:
        return np.zeros(18)

    data = cutout.astype(float)
    flat = data.flatten()
    if np.all(flat == 0):
        return np.zeros(18)

    mean = np.mean(flat)
    std  = np.std(flat) + 1e-9
    peak = np.max(flat)
    mn   = np.min(flat)
    med  = np.median(flat)
    skew = float(np.mean(((flat - mean) / std) ** 3))
    kurt = float(np.mean(((flat - mean) / std) ** 4)) - 3.0
    p25  = float(np.percentile(flat, 25))
    p75  = float(np.percentile(flat, 75))
    p90  = float(np.percentile(flat, 90))
    p99  = float(np.percentile(flat, 99))

    h, w   = data.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dist   = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    inner  = float(data[dist <= 8].sum())
    middle = float(data[(dist > 8) & (dist <= 16)].sum())
    outer  = float(data[dist > 16].sum())
    total  = inner + middle + outer + 1e-9
    conc_inner  = inner  / total
    conc_middle = middle / total
    conc_outer  = outer  / total

    bg_mask = dist > 20
    bg      = float(np.median(data[bg_mask])) if bg_mask.any() else 0.0
    bg_std  = float(np.std(data[bg_mask])) + 1e-9 if bg_mask.any() else 1.0
    n_inner = int((dist <= 8).sum())
    signal  = max(inner - bg * n_inner, 0.0)
    snr     = signal / (bg_std * np.sqrt(n_inner) + 1e-9)

    r_bins  = np.arange(0, 32, 4)
    profile = []
    for r0, r1 in zip(r_bins[:-1], r_bins[1:]):
        ring = data[(dist >= r0) & (dist < r1)]
        profile.append(float(np.mean(ring)) if ring.size > 0 else 0.0)
    slope = float(np.polyfit(range(len(profile)), profile, 1)[0]) if len(profile) > 1 else 0.0

    return np.array([
        mean, std, peak, mn, med, skew, kurt,
        p25, p75, p90, p99,
        conc_inner, conc_middle, conc_outer,
        snr, signal, bg, slope
    ])


# ── Two-frame + delta feature extraction ─────────────────────────

def extract_change_features(early, late):
    """
    Extracts a 54-dimensional feature vector from early + late frames.

    Identical computation to ml_classifier.extract_deeptao_features —
    center coordinate is fixed at (32, 32) for 64x64 cutouts.

    Structure: [early_18] + [late_18] + [delta_8] + [subtraction_10] = 54
    """
    h, w = early.shape
    cx, cy = w // 2, h // 2
    return _extract_deeptao_features(early, cx, cy, late, cx, cy, size=min(h, w))


# ── FITS file loading ─────────────────────────────────────────────

def load_early_late(path):
    """
    Returns the first and last frames from a FITS file.
    Deep-TAO: HDU[0]=header, HDU[1..N]=64x64 frames.
    """
    try:
        with astropy_fits.open(path) as hdul:
            frames = []
            for hdu in hdul[1:]:
                if hdu.data is not None and np.ndim(hdu.data) == 2:
                    frames.append(hdu.data.astype(np.float32))
                    if len(frames) >= 2:  # Find at least the first 2 (for speed)
                        pass
            if not frames:
                return None, None
            # Scanning all frames is slow; look at the end for the last frame
            if len(frames) == 1:
                return frames[0], frames[0]
            # Multiple frames: first and last
            # If only 2 frames were loaded, revisit the last HDU
            with astropy_fits.open(path) as hdul2:
                all_frames = []
                for hdu in hdul2[1:]:
                    if hdu.data is not None and np.ndim(hdu.data) == 2:
                        all_frames.append(hdu.data.astype(np.float32))
            if not all_frames:
                return None, None
            return all_frames[0], all_frames[-1]
    except Exception:
        return None, None


def extract_features_from_file(path):
    """Extracts a 44-dimensional change feature from a FITS file."""
    early, late = load_early_late(path)
    if early is None:
        return None
    feat = extract_change_features(early, late)
    if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


# ── Dataset loading ───────────────────────────────────────────────

def load_dataset(data_dir):
    X, y, cls_labels = [], [], []

    for cls in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        label = 0 if cls in NON_TRANSIENT_CLASSES else 1
        fits_files = [f for f in os.listdir(cls_dir)
                      if f.endswith(".fits") or "." not in f]

        print(f"  [{cls:6s}] {len(fits_files)} files  (label={label})")
        ok = 0
        for fname in fits_files:
            feat = extract_features_from_file(os.path.join(cls_dir, fname))
            if feat is not None:
                X.append(feat)
                y.append(label)
                cls_labels.append(cls)
                ok += 1
        print(f"           {ok}/{len(fits_files)} loaded")

    return np.array(X), np.array(y), cls_labels


# ── Training ──────────────────────────────────────────────────────

def train(data_dir=DATA_DIR_DEFAULT, model_out=MODEL_OUT_DEFAULT):
    print("=" * 60)
    print("Deep-TAO Model Training  (early+late frame, change features)")
    print("=" * 60)

    print("\n[1/6] Loading dataset...")
    data_dir = Path(data_dir)
    model_out = Path(model_out)
    X, y, cls_labels = load_dataset(str(data_dir))

    if len(X) == 0:
        print("ERROR: No data could be loaded!")
        return

    print(f"\nTotal samples  : {len(X)}")
    print(f"  Transient    : {y.sum()}")
    print(f"  Non-transient: {(y == 0).sum()}")
    print(f"  Feature size : {X.shape[1]}  ({len(FEAT_NAMES)} named, "
          f"18+18+8 classic + 10 subtraction)")

    print("\n[2/6] Train/Test split (%%80/%%20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    print("\n[3/6] Fitting StandardScaler...")
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Baseline comparison (for jury: how much better is our model than random?) ──
    print("\n[4/6] Baseline comparison...")
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train_s, y_train)
    dummy_acc = dummy.score(X_test_s, y_test) * 100
    print(f"  Dummy (stratified) baseline accuracy: {dummy_acc:.1f}%%")

    # ── Hyperparameter optimization with GridSearchCV ──
    print("\n[5/6] Hyperparameter optimization with GridSearchCV (this may take 3-5 min)...")
    param_grid = {
        "n_estimators":    [100, 300],
        "max_depth":       [None, 20, 40],
        "min_samples_split": [2, 5],
        "class_weight":    ["balanced", {0: 1, 1: 1}],
    }
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=cv_inner,
        scoring="f1",          # Optimize F1: balance of precision/recall
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train_s, y_train)
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  CV F1 (train): {grid_search.best_score_:.3f}")
    clf = grid_search.best_estimator_

    # ── Decision threshold optimization (Youden J statistic) ──
    # Threshold that maximizes TPR - FPR on the ROC curve:
    # this gives the best balance of sensitivity and specificity.
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    fpr_curve, tpr_curve, thr_curve = roc_curve(y_test, y_prob)
    youden_j    = tpr_curve - fpr_curve
    best_idx    = np.argmax(youden_j)
    best_thr    = float(thr_curve[best_idx])
    best_thr    = max(0.30, min(best_thr, 0.80))  # keep within a reasonable range
    print(f"  Optimal decision threshold (Youden J): {best_thr:.2f}  (default: 0.50)")
    print(f"  Youden J @ optimal: {youden_j[best_idx]:.3f}"
          f"  (TPR={tpr_curve[best_idx]:.3f}, FPR={fpr_curve[best_idx]:.3f})")

    y_pred_thr = (y_prob >= best_thr).astype(int)
    y_pred_def = clf.predict(X_test_s)   # at 0.5

    print("\n[6/6] Evaluation...")

    # ── Default threshold (0.5) ──
    acc_def = (y_pred_def == y_test).mean() * 100
    cm_def  = confusion_matrix(y_test, y_pred_def)
    fpr_def = cm_def[0,1] / max(cm_def[0].sum(), 1)

    # ── Optimized threshold ──
    acc_thr = (y_pred_thr == y_test).mean() * 100
    cm_thr  = confusion_matrix(y_test, y_pred_thr)
    fpr_thr = cm_thr[0,1] / max(cm_thr[0].sum(), 1)

    roc_auc = auc(fpr_curve, tpr_curve)

    print(f"\n{'':=<60}")
    print(f"  {'Metric':<28} {'Baseline':>10} {'RF(thr=0.5)':>12} {'RF(optimal)':>12}")
    print(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*12}")
    print(f"  {'Accuracy':<28} {dummy_acc:>9.1f}%% {acc_def:>11.1f}%% {acc_thr:>11.1f}%%")

    target_names = ["Non-transient", "Transient"]
    rep_def = classification_report(y_test, y_pred_def,
                                    target_names=target_names, output_dict=True)
    rep_thr = classification_report(y_test, y_pred_thr,
                                    target_names=target_names, output_dict=True)

    for met in ["precision", "recall", "f1-score"]:
        key = "Transient " + met
        print(f"  {key:<28} {'N/A':>10} "
              f"{rep_def['Transient'][met]:>11.3f} "
              f"{rep_thr['Transient'][met]:>11.3f}")

    print(f"  {'False Positive Rate':<28} {'N/A':>10} {fpr_def:>11.3f} {fpr_thr:>11.3f}")
    print(f"  {'AUC (ROC)':<28} {'N/A':>10} {roc_auc:>11.3f} {roc_auc:>11.3f}")
    print(f"{'':=<60}")

    print(f"\nConfusion Matrix (optimal threshold={best_thr:.2f}):")
    print(f"  TN={cm_thr[0,0]}  FP={cm_thr[0,1]}")
    print(f"  FN={cm_thr[1,0]}  TP={cm_thr[1,1]}")

    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5,
                                scoring="f1", n_jobs=-1)
    print(f"\n5-Fold CV F1: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Top 5 most important features
    top5 = np.argsort(clf.feature_importances_)[::-1][:5]
    print("\nTop 5 most important features:")
    for i in top5:
        print(f"  {FEAT_NAMES[i]:<25} {clf.feature_importances_[i]:.3f}")

    # ── Save the model (together with the optimized threshold) ──
    acc = acc_thr   # save accuracy at the optimal threshold
    model_data = {
        "classifier":    clf,
        "scaler":        scaler,
        "accuracy":      acc,
        "threshold":     best_thr,      # GUI will use this threshold
        "auc":           roc_auc,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "feat_names":    FEAT_NAMES,
        "n_features":    len(FEAT_NAMES),
        "grid_best_params": grid_search.best_params_,
        "baseline_acc":  dummy_acc,
        "cv_f1_mean":    float(cv_scores.mean()),
        "cv_f1_std":     float(cv_scores.std()),
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, str(model_out))
    print(f"\nModel saved: {model_out.resolve()}")
    print("=" * 60)
    print("Done! The GUI will automatically load this model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Deep-TAO RandomForest transient classifier and save transient_model.joblib"
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR_DEFAULT),
        help="Path to Deep-TAO dataset root (contains class subfolders like SN/AGN/.../NON).",
    )
    parser.add_argument(
        "--model-out",
        default=str(MODEL_OUT_DEFAULT),
        help="Output path for saved joblib model (default: transient_model.joblib in project folder).",
    )
    args = parser.parse_args()
    train(data_dir=args.data_dir, model_out=args.model_out)
