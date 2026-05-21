"""
ml_classifier.py
================
Machine learning components:
  - Hand-crafted feature extraction (Deep-TAO / Gómez et al. 2020 approach)
  - TransientClassifier  : user-labeled supervised Random Forest
  - Isolation Forest     : unsupervised anomaly detection
  - extract_deeptao_features : 44-dimensional vector for pre-trained model

Dependencies
------------
  - detection.py   : bg_subtracted_flux, calculate_star_snr
  - scikit-learn   : optional (controlled via ML_AVAILABLE)
"""

import numpy as np
from detection import bg_subtracted_flux, calculate_star_snr

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# ════════════════════════════════════════════════════════════════
#  DEEP-TAO FEATURE EXTRACTION
#  Identical computation to train_model.py → compatible with the pre-trained model
# ════════════════════════════════════════════════════════════════

def _extract_single_frame_feats(cutout):
    """
    Extracts 18 statistical features from a 64x64 cutout.

    Features: mean, std, peak, min, median, skew, kurt,
              p25, p75, p90, p99,
              conc_inner, conc_middle, conc_outer,
              snr, signal, bg, radial_profile_slope

    Identical to train_model.py::extract_single_frame.
    """
    if cutout is None or cutout.size == 0:
        return np.zeros(18)
    data = cutout.astype(float)
    flat = data.flatten()
    if np.all(flat == 0):
        return np.zeros(18)

    mean = np.mean(flat);  std = np.std(flat) + 1e-9
    peak = np.max(flat);   mn  = np.min(flat)
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
    slope = (float(np.polyfit(range(len(profile)), profile, 1)[0])
             if len(profile) > 1 else 0.0)

    return np.array([
        mean, std, peak, mn, med, skew, kurt,
        p25, p75, p90, p99,
        conc_inner, conc_middle, conc_outer,
        snr, signal, bg, slope
    ])


def _extract_subtraction_features(cutout_early, cutout_late):
    """
    Extracts 10 features from the difference image (late - early).

    Applies the standard astronomical image subtraction:
    a real transient appears as a clean, round, Gaussian point source in the
    difference image; a variable star leaves a residual (residual/dipole).

    Features (10 dimensions):
      diff_peak        — largest positive pixel in the difference cutout
      diff_peak_neg    — most negative pixel (indicator of dipole residual)
      diff_snr         — central signal / edge noise
      diff_symmetry    — positive pixel fraction (1=entirely new source, 0=entirely negative)
      diff_center_flux — net flux within central 8px
      late_ellipticity — roundness of the late source (0=round, 1=elongated)
      late_fwhm        — estimated FWHM (image distortion / noise indicator)
      late_moment_ratio— Ixx/Iyy ratio (shape symmetry)
      bg_change        — relative change in background level
      bg_gradient      — background gradient magnitude (indicator of nearby bright source)
    """
    h, w   = cutout_early.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dist   = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    # ── Difference image ──────────────────────────────────────────
    diff = cutout_late.astype(float) - cutout_early.astype(float)

    diff_peak     = float(diff.max())
    diff_peak_neg = float(diff.min())          # negative means dipole residual present

    # Edge region noise
    edge_mask = dist > (min(h, w) // 2 - 4)
    bg_noise  = float(np.std(diff[edge_mask])) + 1e-9
    center_flux = float(diff[dist <= 8].sum())
    diff_snr    = center_flux / (bg_noise * np.sqrt(int((dist <= 8).sum())) + 1e-9)

    # Positive pixel fraction (1 = entirely new source, 0 = entirely subtraction residual)
    diff_symmetry = float((diff > 0).sum()) / max(diff.size, 1)
    diff_center_flux = center_flux

    # ── PSF shape of the late source (2nd moments) ───────────────
    late = cutout_late.astype(float)
    inner_mask = dist <= 12
    total_flux = late[inner_mask].sum() + 1e-9
    w_x   = float((late[inner_mask] * xx[inner_mask]).sum()) / total_flux
    w_y   = float((late[inner_mask] * yy[inner_mask]).sum()) / total_flux
    dx    = xx[inner_mask] - w_x
    dy    = yy[inner_mask] - w_y
    Ixx   = float((late[inner_mask] * dx**2).sum()) / total_flux + 1e-9
    Iyy   = float((late[inner_mask] * dy**2).sum()) / total_flux + 1e-9
    Ixy   = float((late[inner_mask] * dx * dy).sum()) / total_flux

    # Ellipticity: 0 = round (point source), 1 = fully elongated
    eigval_sum  = Ixx + Iyy
    eigval_diff = np.sqrt(max((Ixx - Iyy)**2 + 4 * Ixy**2, 0.0))
    lam_max     = (eigval_sum + eigval_diff) / 2.0
    lam_min     = (eigval_sum - eigval_diff) / 2.0
    ellipticity    = float(1.0 - np.sqrt(max(lam_min, 0.0) / (lam_max + 1e-9)))
    moment_ratio   = float(Ixx / (Iyy + 1e-9))

    # Estimated FWHM: 2√(2ln2) × sigma ≈ 2.355 × sqrt(Ixx+Iyy)/2
    late_fwhm = float(2.355 * np.sqrt((Ixx + Iyy) / 2.0))

    # ── Background change ─────────────────────────────────────────
    bg_mask  = dist > 20
    bg_early = float(np.median(cutout_early[bg_mask])) if bg_mask.any() else 0.0
    bg_late  = float(np.median(cutout_late[bg_mask]))  if bg_mask.any() else 0.0
    bg_change = (bg_late - bg_early) / (abs(bg_early) + 1.0)

    # Background gradient: average difference in horizontal and vertical direction
    half = h // 2
    grad_y = abs(float(late[bg_mask & (yy > half)].mean() if (bg_mask & (yy > half)).any() else 0.0) -
                 float(late[bg_mask & (yy < half)].mean() if (bg_mask & (yy < half)).any() else 0.0))
    grad_x = abs(float(late[bg_mask & (xx > half)].mean() if (bg_mask & (xx > half)).any() else 0.0) -
                 float(late[bg_mask & (xx < half)].mean() if (bg_mask & (xx < half)).any() else 0.0))
    bg_gradient = float(np.sqrt(grad_x**2 + grad_y**2))

    return np.array([
        diff_peak, diff_peak_neg, diff_snr, diff_symmetry, diff_center_flux,
        ellipticity, late_fwhm, moment_ratio,
        bg_change, bg_gradient,
    ])


def extract_deeptao_features(img_early, cx_e, cy_e,
                              img_late,  cx_l, cy_l, size=64):
    """
    Extracts a 54-dimensional change feature vector.

    Structure: [early_18] + [late_18] + [delta_8] + [subtraction_10]  = 54 dimensions

    Added subtraction features (v2):
      diff_peak, diff_peak_neg, diff_snr, diff_symmetry, diff_center_flux,
      late_ellipticity, late_fwhm, late_moment_ratio, bg_change, bg_gradient

    Identical to train_model.py::extract_change_features.
    """
    def _crop(img, cx, cy):
        h, w = img.shape
        half = size // 2
        y0 = max(0, int(cy) - half); y1 = min(h, int(cy) + half)
        x0 = max(0, int(cx) - half); x1 = min(w, int(cx) + half)
        return img[y0:y1, x0:x1]

    cutout_e = _crop(img_early, cx_e, cy_e)
    cutout_l = _crop(img_late,  cx_l, cy_l)

    # Align different crop sizes (edge case)
    min_h = min(cutout_e.shape[0], cutout_l.shape[0])
    min_w = min(cutout_e.shape[1], cutout_l.shape[1])
    cutout_e = cutout_e[:min_h, :min_w]
    cutout_l = cutout_l[:min_h, :min_w]

    f_early = _extract_single_frame_feats(cutout_e)
    f_late  = _extract_single_frame_feats(cutout_l)

    IDX_PEAK=2; IDX_MEAN=0; IDX_SNR=14; IDX_SIGNAL=15
    IDX_CONC=11; IDX_SLOPE=17

    e_signal   = f_early[IDX_SIGNAL] + 1e-9
    l_signal   = f_late[IDX_SIGNAL]
    flux_ratio = l_signal / e_signal

    delta = np.array([
        f_late[IDX_PEAK]  - f_early[IDX_PEAK],
        f_late[IDX_MEAN]  - f_early[IDX_MEAN],
        f_late[IDX_SNR]   - f_early[IDX_SNR],
        l_signal - f_early[IDX_SIGNAL],
        flux_ratio,
        float(np.log1p(abs(flux_ratio - 1.0))),
        f_late[IDX_CONC]  - f_early[IDX_CONC],
        f_late[IDX_SLOPE] - f_early[IDX_SLOPE],
    ])

    f_sub = _extract_subtraction_features(cutout_e, cutout_l)

    return np.concatenate([f_early, f_late, delta, f_sub])  # 18+18+8+10 = 54 dimensions


# ════════════════════════════════════════════════════════════════
#  SUPERVISED CLASSIFIER  (user-labeled)
# ════════════════════════════════════════════════════════════════

class TransientClassifier:
    """
    Supervised Random Forest classifier for transient vs. artifact.

    User labels events in the GUI → collects samples via add_sample()
    → model is trained with train() → predictions made with predict().

    Model is saved to user_model.joblib and persists across sessions.
    """

    def __init__(self):
        self._samples = []   # list of (feature_vector, label)
        self._model   = None

    def add_sample(self, features, label):
        """Add a labeled sample. label: 0=artifact, 1=transient."""
        self._samples.append((np.array(features, dtype=np.float32), int(label)))

    def train(self):
        """Train the Random Forest on accumulated samples."""
        if not ML_AVAILABLE:
            raise RuntimeError("scikit-learn is not installed.")
        if len(self._samples) < 4:
            raise ValueError("At least 4 labeled samples are required for training.")
        X = np.array([s[0] for s in self._samples])
        y = np.array([s[1] for s in self._samples])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self._model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._model.fit(X, y)

    def predict(self, features):
        """Returns (label, probability)."""
        if self._model is None:
            return 0, 0.0
        fv = np.nan_to_num(
            np.array(features, dtype=np.float32).reshape(1, -1),
            nan=0.0, posinf=0.0, neginf=0.0)
        label = int(self._model.predict(fv)[0])
        proba = float(self._model.predict_proba(fv)[0][label])
        return label, proba

    def cross_val_accuracy(self):
        """5-fold cross-validation accuracy (None → insufficient samples)."""
        if not ML_AVAILABLE or self._model is None:
            return None
        X = np.array([s[0] for s in self._samples])
        y = np.array([s[1] for s in self._samples])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if len(set(y)) < 2 or len(y) < 4:
            return None
        cv_folds = min(5, len(y))
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X, y, cv=cv_folds)
        return float(scores.mean())

    def save(self, path):
        """Save model + samples to a joblib file."""
        if not ML_AVAILABLE:
            raise RuntimeError("joblib/sklearn is not installed.")
        joblib.dump({"model": self._model, "samples": self._samples}, path)

    def load(self, path):
        """Load model + samples from a joblib file."""
        if not ML_AVAILABLE:
            raise RuntimeError("joblib/sklearn is not installed.")
        data = joblib.load(path)
        self._model   = data.get("model")
        self._samples = data.get("samples", [])

    @property
    def is_trained(self):
        return self._model is not None

    @property
    def n_samples(self):
        return len(self._samples)


# Application-wide shared singleton classifier instance
classifier = TransientClassifier()


# ════════════════════════════════════════════════════════════════
#  HAND-CRAFTED FEATURES  (for Isolation Forest)
# ════════════════════════════════════════════════════════════════

def extract_cutout_features(img, cx, cy, size=32):
    """
    Extracts 9 statistical features from around a star.

    (mean, std, peak, skew, concentration, inner_flux,
     outer_flux, bg, snr_factor)
    """
    h, w = img.shape
    half = size // 2
    y0 = max(0, int(cy) - half); y1 = min(h, int(cy) + half)
    x0 = max(0, int(cx) - half); x1 = min(w, int(cx) + half)
    cutout = img[y0:y1, x0:x1].astype(float)
    if cutout.size == 0:
        return np.zeros(9)

    flat = cutout.flatten()
    clx  = cx - x0; cly = cy - y0
    mean = float(np.mean(flat))
    std  = float(np.std(flat)) + 1e-9
    peak = float(np.max(flat))
    skew = float(np.mean(((flat - mean) / std) ** 3))

    yy, xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]
    dist   = np.sqrt((xx - clx)**2 + (yy - cly)**2)
    inner  = float(cutout[dist <= 4].sum())
    outer  = float(cutout[dist > 4].sum())
    concentration = inner / (inner + outer + 1e-9)
    bg    = float(np.median(flat))
    snr_f = peak / std

    return np.array([mean, std, peak, skew, concentration,
                     inner, outer, bg, snr_f])


def extract_pair_features(img1, img2, stars1, stars2, matched, idx,
                           r_inner=4, r_outer=8):
    """
    Full ML feature vector for a matched star pair.

    Vector: [9 features img1] + [9 features img2] +
            [flux_ratio, log_flux_ratio, snr1, snr2, |dx|, |dy|, dist]
    """
    i, j, dx, dy, dist = matched[idx]
    cx1, cy1 = stars1[i][0], stars1[i][1]
    cx2, cy2 = stars2[j][0], stars2[j][1]

    f1 = extract_cutout_features(img1, cx1, cy1)
    f2 = extract_cutout_features(img2, cx2, cy2)

    flux1 = bg_subtracted_flux(img1, cx1, cy1, r_inner, r_outer)
    flux2 = bg_subtracted_flux(img2, cx2, cy2, r_inner, r_outer)
    flux_ratio     = flux2 / (flux1 + 1e-9)
    log_flux_ratio = float(np.log1p(abs(flux_ratio - 1.0)))

    snr1 = calculate_star_snr(img1, cx1, cy1, r_inner, r_outer)
    snr2 = calculate_star_snr(img2, cx2, cy2, r_inner, r_outer)

    return np.concatenate([
        f1, f2,
        [flux_ratio, log_flux_ratio, snr1, snr2,
         abs(dx), abs(dy), float(dist)]
    ])


# ════════════════════════════════════════════════════════════════
#  ISOLATION FOREST  (unsupervised anomaly score)
# ════════════════════════════════════════════════════════════════

def run_isolation_forest(features, contamination=0.05):
    """
    Computes anomaly scores using Isolation Forest.

    No labels required → runs immediately after comparison.
    Returns: score array (higher = more anomalous, normalized to 0-1)

    contamination : expected anomaly fraction (default 5%)
    """
    if not ML_AVAILABLE or len(features) < 5:
        return np.zeros(len(features))

    X = np.array(features)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(contamination=contamination,
                          n_estimators=100, random_state=42)
    clf.fit(X_scaled)

    raw    = clf.score_samples(X_scaled)
    scores = 1.0 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return scores
