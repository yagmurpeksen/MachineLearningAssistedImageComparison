import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.spatial import KDTree
import os
import sys
from pathlib import Path

try:
    from astropy.io import fits as astropy_fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False


# ════════════════════════════════════════════════════════════════
#  FITS LOADER
# ════════════════════════════════════════════════════════════════

def load_fits_image(path):
    """
    FITS dosyasını yükler.
    Döner: (uint8 normalize görüntü, header dict, orijinal float32 data)
    """
    if not FITS_AVAILABLE:
        raise ImportError("astropy kurulu değil.  pip install astropy")

    with astropy_fits.open(path) as hdul:
        data, header = None, None
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) >= 2:
                data   = hdu.data.astype(np.float32)
                header = hdu.header
                break
        if data is None:
            raise ValueError("FITS dosyasında görüntü verisi bulunamadı.")

    # 3-boyutlu küp ise ilk frame'i al  (e.g. [1, H, W])
    if data.ndim == 3:
        data = data[0]

    # NaN / Inf temizle
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Sky background'u bul (medyan), yıldızları öne çıkar
    # Medyan = gökyüzü seviyesi, std = gürültü
    sky    = float(np.median(data))
    bg_pix = data[data < np.percentile(data, 90)]
    sigma  = float(np.std(bg_pix)) if len(bg_pix) > 10 else 1.0
    vmin   = sky - 1.0 * sigma          # gökyüzünün biraz altı → siyah
    vmax   = sky + 15.0 * sigma         # yıldızlar bu aralıkta → beyaz
    vmax   = max(vmax, vmin + 1.0)

    clipped = np.clip(data, vmin, vmax)
    # Sqrt stretch: hem parlak hem zayıf yıldızları görünür kılar
    shifted = clipped - vmin
    sqrt_d  = np.sqrt(shifted)
    s_max   = float(sqrt_d.max())
    if s_max > 0:
        data_u8 = (sqrt_d / s_max * 255).astype(np.uint8)
    else:
        data_u8 = np.zeros_like(data, dtype=np.uint8)

    # Header'dan faydalı bilgileri al
    info = {}
    for key in ("EXPTIME", "DATE-OBS", "TELESCOP", "INSTRUME", "FILTER",
                "OBJECT",  "NAXIS1",   "NAXIS2",  "BITPIX"):
        if key in header:
            info[key] = header[key]

    return data_u8, info, data, header


# ════════════════════════════════════════════════════════════════
#  STAR DETECTION
# ════════════════════════════════════════════════════════════════

def detect_stars(img, pct=97, min_area=3, max_area=150, min_peak=20):
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    t = max(float(np.percentile(blurred, pct)), float(min_peak))
    _, binary = cv.threshold(blurred, t, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    n, labels, stats, centroids = cv.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    stars = []
    for i in range(1, n):
        area = stats[i, cv.CC_STAT_AREA]
        cx, cy = centroids[i]
        mask = (labels == i).astype(np.uint8)
        peak = int(cv.minMaxLoc(img, mask=mask)[1])
        # Saturate yıldızlar (peak=255): alan sınırı uygulanmaz,
        # diffraction spike'lar alanı şişirir ama merkez doğrudur
        saturated = (peak >= 254)
        if not saturated and not (min_area <= area <= max_area):
            continue
        if area < min_area:
            continue
        if peak >= min_peak:
            stars.append((float(cx), float(cy), int(area), peak))
    return stars


# ════════════════════════════════════════════════════════════════
#  ALIGNMENT  (star-centroid affine + RANSAC)
# ════════════════════════════════════════════════════════════════

def star_affine_align(img1_n, img2_n, pct=97, coarse_radius=25, ransac_thresh=5.0):
    h, w = img1_n.shape[:2]
    s1 = detect_stars(img1_n, pct=pct)
    s2 = detect_stars(img2_n, pct=pct)
    if len(s1) < 4 or len(s2) < 4:
        return img2_n, None, "no_stars"
    pts1 = np.float32([[s[0], s[1]] for s in s1])
    pts2 = np.float32([[s[0], s[1]] for s in s2])
    tree2 = KDTree(pts2)
    d, idx = tree2.query(pts1, k=1)
    pairs = [(i, int(idx[i])) for i in range(len(s1)) if d[i] < coarse_radius]
    if len(pairs) < 4:
        return img2_n, None, "few_pairs"
    src = np.float32([[pts2[j][0], pts2[j][1]] for i, j in pairs]).reshape(-1, 1, 2)
    dst = np.float32([[pts1[i][0], pts1[i][1]] for i, j in pairs]).reshape(-1, 1, 2)
    H, mask = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC,
                                          ransacReprojThreshold=ransac_thresh)
    if H is None:
        return img2_n, None, "ransac_failed"
    img2_aligned = cv.warpAffine(img2_n, H, (w, h),
                                  flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    inliers = int(mask.sum()) if mask is not None else 0
    return img2_aligned, H, f"star-affine({inliers} inliers)"


# ════════════════════════════════════════════════════════════════
#  MATCHING
# ════════════════════════════════════════════════════════════════

def match_stars_mutual(stars1, stars2, max_distance):
    if not stars1 or not stars2:
        return [], list(range(len(stars1))), list(range(len(stars2)))
    pts1 = np.array([[s[0], s[1]] for s in stars1])
    pts2 = np.array([[s[0], s[1]] for s in stars2])
    tree1 = KDTree(pts1); tree2 = KDTree(pts2)
    d_fwd, i_fwd = tree2.query(pts1, k=1)
    _, i_bwd = tree1.query(pts2, k=1)
    matched, used1, used2 = [], set(), set()
    for i, (dist, j) in enumerate(zip(d_fwd, i_fwd)):
        j = int(j)
        if dist <= max_distance and i_bwd[j] == i:
            matched.append((i, j, float(stars2[j][0]-stars1[i][0]),
                             float(stars2[j][1]-stars1[i][1]), float(dist)))
            used1.add(i); used2.add(j)
    unm1 = [i for i in range(len(stars1)) if i not in used1]
    unm2 = [j for j in range(len(stars2)) if j not in used2]
    return matched, unm1, unm2


# ════════════════════════════════════════════════════════════════
#  MOVING DETECTION
# ════════════════════════════════════════════════════════════════

def detect_moving_outliers(matched, iqr_multiplier=3.0, min_residual_px=3.0,
                           max_moving_frac=0.05):
    """
    Matched yildizlar arasinda anormal pozisyon degisimi olanlari tespit eder.

    Adaptif esik: Medyan residual yuksekse (farkli teleskop/PSF mismatch),
    floor otomatik yukari cekilerek false positive bastirilir.
    max_moving_frac: esitlenmis yildizlarin en fazla bu kadar yuzdesi
                     'moving' isaretlenir (0.05 = %5).
    """
    if len(matched) < 4:
        return [], 0.0, 0.0, min_residual_px, np.array([])
    dxs = np.array([m[2] for m in matched])
    dys = np.array([m[3] for m in matched])
    med_dx, med_dy = float(np.median(dxs)), float(np.median(dys))
    residuals = np.sqrt((dxs - med_dx)**2 + (dys - med_dy)**2)
    q1, q3 = np.percentile(residuals, 25), np.percentile(residuals, 75)
    iqr = q3 - q1

    # IQR tabanli esik
    thr = max(q3 + iqr_multiplier * iqr, min_residual_px)

    # --- Adaptif PSF-mismatch duzeltmesi ---
    # Medyan residual yuksekse (>1px) farkli enstruman/epoch sinyali var.
    # Etkili taban = medyan * 4  (tipik PSF kaymasinin 4 kati olmali)
    median_res = float(np.median(residuals))
    if median_res > 1.0:
        adaptive_floor = median_res * 4.0
        thr = max(thr, adaptive_floor)

    # --- Oran siniri: en fazla %5 'moving' isaretlenir ---
    moving_idx = [i for i, r in enumerate(residuals) if r > thr]
    cap = max(1, int(len(matched) * max_moving_frac))
    if len(moving_idx) > cap:
        # En buyuk residual'dan baslayarak sadece cap kadar al
        sorted_idx = sorted(range(len(residuals)), key=lambda i: -residuals[i])
        moving_idx = sorted_idx[:cap]
        thr = float(residuals[sorted_idx[cap - 1]])  # bilgi icin

    return moving_idx, med_dx, med_dy, float(thr), residuals



# ════════════════════════════════════════════════════════════════
#  ML — FEATURE EXTRACTION + ANOMALY DETECTION
#  Makaledeki yaklaşım (Deep-TAO / Gómez et al. 2020) adapte edildi:
#  Her yıldız çifti için handcrafted özellikler çıkarılır,
#  Isolation Forest ile anomali skoru hesaplanır.
#  Etiket gerekmez → hemen çalışır.
# ════════════════════════════════════════════════════════════════

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

def _app_base_dir() -> Path:
    """
    Base directory for reading/writing app artifacts.
    - When running as a script: directory containing this file.
    - When frozen (PyInstaller): directory containing the executable.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


MODEL_PATH = _app_base_dir() / "transient_model.joblib"

# Load pre-trained Deep-TAO model at startup
pretrained_model = None
if ML_AVAILABLE:
    try:
        _pt_data = joblib.load(MODEL_PATH)
        if "classifier" in _pt_data and "scaler" in _pt_data:
            pretrained_model = _pt_data
    except Exception:
        pass


def _extract_single_frame_feats(cutout):
    """
    64×64 cutout'tan 18 istatistiksel özellik çıkar.
    train_model.py::extract_single_frame ile birebir aynı.
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
    slope = float(np.polyfit(range(len(profile)), profile, 1)[0]) if len(profile) > 1 else 0.0

    return np.array([
        mean, std, peak, mn, med, skew, kurt,
        p25, p75, p90, p99,
        conc_inner, conc_middle, conc_outer,
        snr, signal, bg, slope
    ])


def extract_deeptao_features(img_early, cx_e, cy_e,
                              img_late,  cx_l, cy_l, size=64):
    """
    44 özellik çıkarır — train_model.py::extract_change_features ile aynı.
    Erken frame (img1) + geç frame (img2) + 8 delta özelliği.
    Pre-trained Deep-TAO Random Forest modeli için kullanılır.
    """
    def _crop(img, cx, cy):
        h, w = img.shape
        half = size // 2
        y0 = max(0, int(cy)-half); y1 = min(h, int(cy)+half)
        x0 = max(0, int(cx)-half); x1 = min(w, int(cx)+half)
        return img[y0:y1, x0:x1]

    early_cut = _crop(img_early, cx_e, cy_e)
    late_cut  = _crop(img_late,  cx_l, cy_l)

    f_early = _extract_single_frame_feats(early_cut)
    f_late  = _extract_single_frame_feats(late_cut)

    IDX_PEAK=2; IDX_MEAN=0; IDX_SNR=14; IDX_SIGNAL=15; IDX_CONC=11; IDX_SLOPE=17
    e_signal = f_early[IDX_SIGNAL] + 1e-9
    l_signal = f_late[IDX_SIGNAL]
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

    return np.concatenate([f_early, f_late, delta])  # 44 boyut


# ════════════════════════════════════════════════════════════════
#  TRANSIENT CLASSIFIER  (supervised Random Forest)
#  Users label events as transient (1) or artifact (0).
#  Requires sklearn; falls back gracefully if unavailable.
# ════════════════════════════════════════════════════════════════

class TransientClassifier:
    """Supervised Random Forest classifier for transient vs. artifact."""

    def __init__(self):
        self._samples = []   # list of (feature_vector, label)
        self._model   = None

    def add_sample(self, features, label):
        """Add a labeled training sample. label: 0=artifact, 1=transient."""
        self._samples.append((np.array(features, dtype=np.float32), int(label)))

    def train(self):
        """Train the Random Forest on all collected samples."""
        if not ML_AVAILABLE:
            raise RuntimeError("scikit-learn not available.")
        if len(self._samples) < 4:
            raise ValueError("Need at least 4 labeled samples to train.")
        X = np.array([s[0] for s in self._samples])
        y = np.array([s[1] for s in self._samples])
        # Replace NaN / Inf that may appear in feature vectors
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self._model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._model.fit(X, y)

    def predict(self, features):
        """Return (label, probability) for a feature vector."""
        if self._model is None:
            return 0, 0.0
        fv = np.nan_to_num(
            np.array(features, dtype=np.float32).reshape(1, -1),
            nan=0.0, posinf=0.0, neginf=0.0)
        label = int(self._model.predict(fv)[0])
        proba = float(self._model.predict_proba(fv)[0][label])
        return label, proba

    def cross_val_accuracy(self):
        """Return mean cross-val accuracy (or None if too few samples)."""
        if not ML_AVAILABLE or self._model is None:
            return None
        X = np.array([s[0] for s in self._samples])
        y = np.array([s[1] for s in self._samples])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n_classes = len(set(y))
        if n_classes < 2 or len(y) < 4:
            return None
        cv_folds = min(5, len(y))
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X, y, cv=cv_folds)
        return float(scores.mean())

    def save(self, path):
        """Persist model + samples using joblib."""
        if not ML_AVAILABLE:
            raise RuntimeError("joblib/sklearn not available.")
        joblib.dump({"model": self._model, "samples": self._samples}, path)

    def load(self, path):
        """Load model + samples from a joblib file."""
        if not ML_AVAILABLE:
            raise RuntimeError("joblib/sklearn not available.")
        data = joblib.load(path)
        self._model   = data.get("model")
        self._samples = data.get("samples", [])

    @property
    def is_trained(self):
        return self._model is not None

    @property
    def n_samples(self):
        return len(self._samples)


# Global classifier instance
classifier = TransientClassifier()


def extract_cutout_features(img, cx, cy, size=32):
    """
    Bir yıldız çevresinden istatistiksel özellikler çıkar.
    Makale: 64x64 cutout → CNN. Biz: 32x32 cutout → handcrafted features.
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

    return np.array([mean, std, peak, skew, concentration, inner, outer, bg, snr_f])


def extract_pair_features(img1, img2, stars1, stars2, matched, idx,
                           r_inner=4, r_outer=8):
    """
    Eşleşmiş bir yıldız çifti için tüm ML özelliklerini döner.
    """
    i, j, dx, dy, dist = matched[idx]
    cx1, cy1 = stars1[i][0], stars1[i][1]
    cx2, cy2 = stars2[j][0], stars2[j][1]

    f1 = extract_cutout_features(img1, cx1, cy1)
    f2 = extract_cutout_features(img2, cx2, cy2)

    flux1 = bg_subtracted_flux(img1, cx1, cy1, r_inner, r_outer)
    flux2 = bg_subtracted_flux(img2, cx2, cy2, r_inner, r_outer)
    flux_ratio = flux2 / (flux1 + 1e-9)
    log_flux_ratio = float(np.log1p(abs(flux_ratio - 1.0)))

    snr1 = calculate_star_snr(img1, cx1, cy1, r_inner, r_outer)
    snr2 = calculate_star_snr(img2, cx2, cy2, r_inner, r_outer)

    return np.concatenate([
        f1, f2,
        [flux_ratio, log_flux_ratio, snr1, snr2,
         abs(dx), abs(dy), float(dist)]
    ])


def run_isolation_forest(features, contamination=0.05):
    """
    Isolation Forest ile anomali tespiti.
    Döner: her örnek için anomaly score (yüksek = daha anormal).
    contamination: beklenen anomali oranı
    """
    if not ML_AVAILABLE or len(features) < 5:
        return np.zeros(len(features))

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(contamination=contamination,
                          n_estimators=100, random_state=42)
    clf.fit(X_scaled)

    # score_samples: daha negatif = daha anormal → pozitife çevir
    raw = clf.score_samples(X_scaled)
    scores = 1.0 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return scores


# ════════════════════════════════════════════════════════════════
#  BRIGHTNESS DETECTION
#  Background-subtracted aperture photometry + IQR outlier filter.
#  Same statistical approach as moving detection — only stars whose
#  brightness change is a true outlier vs. all matched stars get flagged.
# ════════════════════════════════════════════════════════════════

def bg_subtracted_flux(img, cx, cy, r_inner=4, r_outer=8):
    """Aperture flux with annulus background subtraction."""
    h, w = img.shape
    r = r_outer
    y0 = max(0, int(cy) - r); y1 = min(h, int(cy) + r + 1)
    x0 = max(0, int(cx) - r); x1 = min(w, int(cx) + r + 1)
    patch = img[y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (xx - cx)**2 + (yy - cy)**2
    annulus   = (dist2 > r_inner**2) & (dist2 <= r_outer**2)
    star_mask = dist2 <= r_inner**2
    if not star_mask.any():
        return 0.0
    bg = float(np.median(patch[annulus])) if annulus.any() else 0.0
    return max(float(patch[star_mask].sum()) - bg * int(star_mask.sum()), 0.0)


# ════════════════════════════════════════════════════════════════
#  FOTOMETRİ KALİBRASYONU  (APASS + WCS)
#  ZP = sıfır nokta  →  V = -2.5*log10(flux) + ZP
# ════════════════════════════════════════════════════════════════

def _try_photometry_zp(header, stars, flux_img, r_inner=6, r_outer=12):
    """
    FITS başlığındaki WCS + APASS kataloğuyla sıfır nokta hesaplar.
    Döner: (zp: float, n_cal: int)  veya  (None, 0) başarısız olursa.
    """
    try:
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
        import astropy.units as u
        import warnings
        warnings.filterwarnings('ignore')

        if header is None:
            return None, 0
        # WCS için minimum anahtar kontrolü
        if not any(k in header for k in ('CRVAL1', 'CTYPE1', 'CD1_1')):
            return None, 0

        wcs = WCS(header)
        nx  = int(header.get('NAXIS1', flux_img.shape[1]))
        ny  = int(header.get('NAXIS2', flux_img.shape[0]))
        cen = wcs.pixel_to_world(nx / 2, ny / 2)

        # APASS sorgula
        viz = Vizier(columns=['RAJ2000', 'DEJ2000', 'Vmag'], row_limit=400)
        cat_raw = viz.query_region(cen, radius=12 * u.arcmin, catalog='APASS')
        if not cat_raw or len(cat_raw) == 0:
            return None, 0
        cat = cat_raw[0]
        try:
            cat = cat[~cat['Vmag'].data.mask]
        except Exception:
            pass
        if len(cat) < 5:
            return None, 0

        # Enstrümantal magnitudes hesapla
        fluxes = np.array([
            bg_subtracted_flux(flux_img, s[0], s[1], r_inner, r_outer)
            for s in stars
        ])
        inst_mag = -2.5 * np.log10(np.maximum(fluxes, 1.0))

        # WCS → gökyüzü koordinatları
        x_arr = np.array([s[0] for s in stars])
        y_arr = np.array([s[1] for s in stars])
        sky   = wcs.pixel_to_world(x_arr, y_arr)
        fit_sc = SkyCoord(ra=sky.ra.deg,  dec=sky.dec.deg,  unit='deg')
        cat_sc = SkyCoord(ra=np.array(cat['RAJ2000']),
                          dec=np.array(cat['DEJ2000']), unit='deg')
        idx, sep, _ = fit_sc.match_to_catalog_sky(cat_sc)
        ok = sep.arcsec < 5.0

        if ok.sum() < 5:
            return None, 0

        diff    = np.array(cat['Vmag'])[idx[ok]] - inst_mag[ok]
        med_zp  = float(np.median(diff))
        robust  = np.abs(diff - med_zp) < 0.5
        if robust.sum() < 3:
            return None, 0
        zp = float(np.median(diff[robust]))
        return zp, int(robust.sum())

    except Exception:
        return None, 0


# ════════════════════════════════════════════════════════════════
#  SNR  (Signal-to-Noise Ratio)
#  CCD denklemi: SNR = signal / sqrt(signal + n_pix * bg_sigma²)
#  signal  = arka plan çıkarılmış yıldız akısı (ADU)
#  bg_sigma = halka bölgesindeki piksel gürültüsü (standart sapma)
# ════════════════════════════════════════════════════════════════

def calculate_star_snr(img, cx, cy, r_inner=4, r_outer=8):
    h, w = img.shape
    r    = r_outer
    y0 = max(0, int(cy)-r); y1 = min(h, int(cy)+r+1)
    x0 = max(0, int(cx)-r); x1 = min(w, int(cx)+r+1)
    patch  = img[y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2  = (xx-cx)**2 + (yy-cy)**2
    annulus   = (dist2 > r_inner**2) & (dist2 <= r_outer**2)
    star_mask = dist2 <= r_inner**2
    if not star_mask.any():
        return 0.0
    n_star   = int(star_mask.sum())
    bg_med   = float(np.median(patch[annulus])) if annulus.any() else 0.0
    bg_sigma = float(np.std(patch[annulus]))    if annulus.any() else 1.0
    signal   = max(float(patch[star_mask].sum()) - bg_med * n_star, 0.0)
    noise    = np.sqrt(signal + n_star * bg_sigma**2 + 1e-9)
    return signal / noise


def filter_stars_by_snr(stars, img, min_snr=3.0, r_inner=4, r_outer=8):
    """SNR eşiğinin altındaki yıldızları eler. SNR listesini de döner."""
    snr_list    = [calculate_star_snr(img, s[0], s[1], r_inner, r_outer) for s in stars]
    filtered    = [s for s, snr in zip(stars, snr_list) if snr >= min_snr]
    snr_kept    = [snr for snr in snr_list if snr >= min_snr]
    return filtered, snr_list, snr_kept


def detect_brightness_outliers(matched, stars1, stars2, img1_n, img2_n,
                                r_inner=4, r_outer=8,
                                flux_ratio_thr=2.5, min_flux=50):
    """
    Flag stars whose flux changed by more than flux_ratio_thr after global
    exposure correction.  flux_ratio_thr=2.5 means 2.5x brighter/dimmer.
    This is robust against noise (+-20-30%) and JPEG compression.
    Typical values:
      2.0 = flag stars that doubled / halved
      3.0 = only very dramatic changes (nova / flare level)
    """
    fluxes = []
    for (i, j, dx, dy, dist) in matched:
        b1 = bg_subtracted_flux(img1_n, stars1[i][0], stars1[i][1], r_inner, r_outer)
        b2 = bg_subtracted_flux(img2_n, stars2[j][0], stars2[j][1], r_inner, r_outer)
        fluxes.append((b1, b2))

    # Median ratio corrects global exposure / sky difference between frames
    ratios = [b2 / b1 for b1, b2 in fluxes if b1 > min_flux and b2 > 0]
    global_ratio = float(np.median(ratios)) if ratios else 1.0

    brightened, dimmed, stationary = [], [], []
    changes_list = []
    for idx, (i, j, dx, dy, dist) in enumerate(matched):
        b1, b2 = fluxes[idx]
        b2_corr = b2 / global_ratio if global_ratio > 0 else b2
        if b1 > min_flux:
            ratio = b2_corr / b1
            ch = (ratio - 1.0) * 100.0
        else:
            ratio = 1.0
            ch = 0.0
        changes_list.append(ch)
        entry = {
            "idx": idx, "i": i, "j": j,
            "x1": stars1[i][0], "y1": stars1[i][1],
            "x2": stars2[j][0], "y2": stars2[j][1],
            "b1": b1, "b2": b2, "change_pct": ch,
            "peak1": stars1[i][3], "peak2": stars2[j][3],
        }
        if b1 <= min_flux:
            stationary.append(entry)
        elif ratio >= flux_ratio_thr:
            brightened.append(entry)
        elif ratio <= 1.0 / flux_ratio_thr:
            dimmed.append(entry)
        else:
            stationary.append(entry)

    brightened.sort(key=lambda x: x["change_pct"], reverse=True)
    dimmed.sort(key=lambda x: x["change_pct"])
    return brightened, dimmed, stationary, np.array(changes_list)


# ════════════════════════════════════════════════════════════════
#  APPLICATION
# ════════════════════════════════════════════════════════════════

class ImageComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stellar Analysis: Moving Objects & Brightness Changes")
        self.root.geometry("1500x900")
        self.root.configure(bg="#1a1a2e")
        self.img1 = self.img2 = None
        self.img1_path = self.img2_path = None
        self.fits_raw1 = self.fits_raw2 = None   # orijinal float32 FITS verisi
        self.fits_info1 = self.fits_info2 = {}   # header bilgisi
        self.fits_header1 = self.fits_header2 = None  # tam FITS header (WCS için)
        self.phot_zp1 = self.phot_zp2 = None    # fotometri sıfır noktası
        self.phot_n_cal1 = self.phot_n_cal2 = 0
        self.mode   = tk.StringVar(value="both")
        # Analysis result storage (for labeling and redraws)
        self.last_brightened = []
        self.last_dimmed     = []
        self.last_matched    = []
        self.last_stars1     = []
        self.last_stars2     = []
        self.img1_n = None
        self.img2_n = None
        self.cmap   = tk.StringVar(value="gray")
        # Parameters
        self.p_thr_pct    = tk.DoubleVar(value=97.0)
        self.p_min_area   = tk.IntVar(value=3)
        self.p_max_area   = tk.IntVar(value=2000)
        self.p_min_peak   = tk.IntVar(value=20)
        self.p_coarse_r   = tk.DoubleVar(value=25.0)
        self.p_match_r    = tk.DoubleVar(value=8.0)
        self.p_iqr_mult   = tk.DoubleVar(value=4.0)
        self.p_min_res    = tk.DoubleVar(value=5.0)
        self.p_max_mov_frac = tk.DoubleVar(value=0.02)

        self.p_r_inner    = tk.IntVar(value=4)
        self.p_r_outer    = tk.IntVar(value=8)
        self.p_min_flux   = tk.DoubleVar(value=50.0)
        self.p_flux_ratio_thr = tk.DoubleVar(value=2.5)
        self.p_min_snr        = tk.DoubleVar(value=3.0)
        self.create_widgets()

    def create_widgets(self):
        tf = tk.Frame(self.root, bg="#16213e", height=52)
        tf.pack(fill=tk.X); tf.pack_propagate(False)
        tk.Label(tf, text="Stellar Analysis  ·  Moving Objects & Brightness Changes",
                 font=("Arial", 15, "bold"), bg="#16213e", fg="#e0e0ff").pack(pady=12)

        bf = tk.Frame(self.root, bg="#1a1a2e")
        bf.pack(fill=tk.X, padx=18, pady=8)

        def mkbtn(text, cmd, color, bold=False, state=tk.NORMAL):
            f = ("Arial", 10, "bold") if bold else ("Arial", 10)
            b = tk.Button(bf, text=text, command=cmd, font=f,
                          bg=color, fg="white", padx=11, pady=7,
                          cursor="hand2", relief=tk.RAISED, bd=3, state=state)
            b.pack(side=tk.LEFT, padx=3); return b

        mkbtn("Image 1", self.load_image1, "#2980b9")
        mkbtn("Image 2", self.load_image2, "#2980b9")
        self.btn_compare = mkbtn("Analyze", self.compare_images,
                                  "#27ae60", bold=True, state=tk.DISABLED)
        self.btn_catalog = mkbtn("Star Catalog", self.analyze_catalog,
                                  "#16a085", state=tk.DISABLED)
        self.btn_save = mkbtn("Save", self.save_result, "#e67e22", state=tk.DISABLED)
        mkbtn("Settings", self.open_settings, "#8e44ad")
        self.btn_label = mkbtn("Label Events", self.open_label_window, "#8e44ad", state=tk.DISABLED)
        self.btn_train = mkbtn("Train Model", self.train_model, "#c0392b", state=tk.DISABLED)

        # ── Colormap seçici ──
        tk.Label(bf, text="  Colormap:", font=("Arial", 10),
                 bg="#1a1a2e", fg="#bdc3c7").pack(side=tk.LEFT, padx=(12, 2))
        cmaps = ["gray", "hot", "inferno", "plasma", "viridis", "magma", "cividis"]
        cmap_menu = tk.OptionMenu(bf, self.cmap, *cmaps)
        cmap_menu.config(font=("Arial", 10), bg="#2c3e50", fg="white",
                         activebackground="#3d5166", activeforeground="white",
                         highlightthickness=0, bd=0)
        cmap_menu["menu"].config(bg="#2c3e50", fg="white", font=("Arial", 10))
        cmap_menu.pack(side=tk.LEFT, padx=3)

        tk.Label(bf, text="  Mode:", font=("Arial", 10),
                 bg="#1a1a2e", fg="#bdc3c7").pack(side=tk.LEFT, padx=(12, 2))
        for val, lbl in [("moving", "Moving"), ("brightness", "Brightness"), ("both", "Both")]:
            tk.Radiobutton(bf, text=lbl, variable=self.mode, value=val,
                           font=("Arial", 10), bg="#1a1a2e", fg="#ecf0f1",
                           selectcolor="#2c3e50", activebackground="#1a1a2e",
                           activeforeground="white", cursor="hand2").pack(side=tk.LEFT, padx=3)

        if pretrained_model is not None:
            _pt_acc = pretrained_model.get("accuracy", 0)
            _pt_n   = pretrained_model.get("n_train", 0)
            _status_init = (f"Deep-TAO model loaded  (acc={_pt_acc:.0f}%  trained on {_pt_n} samples)"
                            f"  |  Select two images to begin.")
            _status_col  = "#2ecc71"
        else:
            _status_init = "Select two images to begin."
            _status_col  = "#95a5a6"
        self.status_label = tk.Label(bf, text=_status_init,
                                     font=("Arial", 10), bg="#1a1a2e", fg=_status_col)
        self.status_label.pack(side=tk.RIGHT, padx=10)

        cf = tk.Frame(self.root, bg="#0f0f23", relief=tk.SUNKEN, bd=2)
        cf.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 6))
        self.canvas_frame = tk.Frame(cf, bg="#0f0f23")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.stats_text = tk.Text(self.root, height=5, font=("Courier", 9),
                                  bg="#0d0d1a", fg="#00ff88",
                                  state=tk.DISABLED, relief=tk.FLAT)
        self.stats_text.pack(fill=tk.X, padx=18, pady=(0, 8))

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Analysis Parameters")
        win.configure(bg="#1a1a2e"); win.resizable(False, False)
        sections = [
            ("── Star Detection ──────────────────────────────────────────", None, ""),
            ("Detection Percentile:", self.p_thr_pct, "97 = top 3% brightest pixels"),
            ("Min Star Area (px):", self.p_min_area, "Smaller blobs ignored (noise)"),
            ("Max Star Area (px):", self.p_max_area, "Saturated/bright stars can be 500-2000px²"),
            ("Min Peak Brightness:", self.p_min_peak, "Dim detections ignored"),
            ("── Alignment ───────────────────────────────────────────────", None, ""),
            ("Coarse Alignment Radius (px):", self.p_coarse_r, "Initial pairing radius"),
            ("── Moving Detection ────────────────────────────────────────", None, ""),
            ("Final Match Radius (px):", self.p_match_r, "Max px to match same star (keep 5-10)"),
            ("Moving IQR Multiplier:", self.p_iqr_mult, "Lower = more sensitive"),
            ("Min Residual to Flag Moving (px):", self.p_min_res, "Minimum hareket esigi. 5px=belirgin hareket, 1.5px=cok hassas (false positive artar)"),
            ("Max Moving Fraction (0-1):", self.p_max_mov_frac, "En fazla bu oran 'moving' isaretlenir. 0.02=%2 onerilir, 0.05 cok hassas"),

            ("── Brightness Detection ────────────────────────────────────", None, ""),
            ("Aperture Inner Radius (px):", self.p_r_inner, "Star signal aperture"),
            ("Aperture Outer Radius (px):", self.p_r_outer, "Background annulus outer edge"),
            ("Min Flux for Brightness (ADU):", self.p_min_flux, "Ignore faint stars (noise prone)"),
            ("Brightness Flux Ratio Threshold:", self.p_flux_ratio_thr, "2.5 = flag 2.5x brighter/dimmer. Lower = more sensitive."),
            ("── SNR Filter ──────────────────────────────────────────────", None, ""),
            ("Min SNR to Keep Star:", self.p_min_snr, "3.0 = yıldız en az 3x gürültü üstünde olmalı. Yükseltirsek sadece parlak yıldızlar kalır."),
        ]
        row = 0
        for lbl, var, tip in sections:
            if var is None:
                tk.Label(win, text=lbl, font=("Arial", 9, "italic"), bg="#1a1a2e",
                         fg="#7f8c8d").grid(row=row, column=0, columnspan=3,
                                            sticky="w", padx=14, pady=(10,2))
            else:
                tk.Label(win, text=lbl, font=("Arial", 10), bg="#1a1a2e",
                         fg="#ecf0f1", anchor="w").grid(row=row, column=0,
                                                         sticky="w", padx=14, pady=3)
                tk.Entry(win, textvariable=var, width=8,
                         font=("Arial", 10)).grid(row=row, column=1, padx=8)
                tk.Label(win, text=tip, font=("Arial", 8), bg="#1a1a2e",
                         fg="#7f8c8d").grid(row=row, column=2, sticky="w", padx=6)
            row += 1
        tk.Button(win, text="Close", command=win.destroy, bg="#2c3e50", fg="white",
                  font=("Arial", 10), padx=16, pady=6, cursor="hand2").grid(
            row=row, column=0, columnspan=3, pady=14)

    def load_image1(self): self._load(1)
    def load_image2(self): self._load(2)

    def _load(self, num):
        fits_types = "*.fits *.fit *.fts *.fits.gz" if FITS_AVAILABLE else ""
        filetypes  = []
        if FITS_AVAILABLE:
            filetypes.append(("FITS files", fits_types))
        filetypes += [("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                      ("All files",   "*.*")]

        path = filedialog.askopenfilename(title=f"Select Image {num}",
                                          filetypes=filetypes)
        if not path: return

        ext = os.path.splitext(path)[1].lower()
        is_fits = ext in (".fits", ".fit", ".fts") or path.endswith(".fits.gz")

        # Uzantısız dosyalar için FITS deneme (örn. "dss_search")
        if not is_fits and ext == "" and FITS_AVAILABLE:
            try:
                from astropy.io import fits as _afits
                with _afits.open(path) as _h:
                    _h.verify('fix')
                is_fits = True   # FITS magic doğrulandı
            except Exception:
                pass

        if is_fits:
            try:
                img, info, raw, hdr = load_fits_image(path)
            except Exception as e:
                messagebox.showerror("FITS Error", str(e)); return
            setattr(self, f"fits_raw{num}", raw)
            setattr(self, f"fits_info{num}", info)
            setattr(self, f"fits_header{num}", hdr)
            # Header bilgilerini kısa özet olarak göster
            exp  = info.get("EXPTIME", "?")
            filt = info.get("FILTER",  "?")
            obj  = info.get("OBJECT",  "")
            tip  = f"FITS | exp={exp}s  filter={filt}  {obj}".strip()
        else:
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is None:
                messagebox.showerror("Error", "Could not load image!"); return
            setattr(self, f"fits_raw{num}", None)
            setattr(self, f"fits_info{num}", {})
            setattr(self, f"fits_header{num}", None)
            tip = os.path.basename(path)

        setattr(self, f"img{num}", img)
        setattr(self, f"img{num}_path", path)
        self.status_label.config(
            text=f"Image {num}: {os.path.basename(path)}  [{tip}]", fg="#2ecc71")
        self.btn_catalog.config(state=tk.NORMAL)   # tek resimle çalışır
        if self.img1 is not None and self.img2 is not None:
            self.btn_compare.config(state=tk.NORMAL)
            self.status_label.config(text="Both images loaded - click Analyze", fg="#3498db")

    def compare_images(self):
        if self.img1 is None or self.img2 is None: return
        try:
            def status(msg, color="#3498db"):
                self.status_label.config(text=msg, fg=color); self.root.update_idletasks()

            h, w = self.img1.shape[:2]
            img1_n  = cv.normalize(self.img1, None, 0, 255, cv.NORM_MINMAX)
            img2r_n = cv.normalize(cv.resize(self.img2, (w, h)), None, 0, 255, cv.NORM_MINMAX)

            # FITS ise flux ölçümü için orijinal float veriyi kullan
            flux1 = self.fits_raw1 if self.fits_raw1 is not None else img1_n.astype(np.float32)
            flux2 = self.fits_raw2 if self.fits_raw2 is not None else img2r_n.astype(np.float32)

            status("Aligning (star-centroid RANSAC)...")
            img2_aligned, H, align_method = star_affine_align(
                img1_n, img2r_n, pct=self.p_thr_pct.get(),
                coarse_radius=self.p_coarse_r.get())

            status("Detecting stars...")
            kw = dict(pct=self.p_thr_pct.get(), min_area=self.p_min_area.get(),
                      max_area=self.p_max_area.get(), min_peak=self.p_min_peak.get())
            stars1_raw = detect_stars(img1_n, **kw)
            stars2_raw = detect_stars(img2_aligned, **kw)

            status("Calculating SNR...")
            ri, ro = self.p_r_inner.get(), self.p_r_outer.get()
            min_snr = self.p_min_snr.get()
            stars1, snr_all1, snr_kept1 = filter_stars_by_snr(
                stars1_raw, img1_n, min_snr, ri, ro)
            stars2, snr_all2, snr_kept2 = filter_stars_by_snr(
                stars2_raw, img2_aligned, min_snr, ri, ro)

            diff = cv.absdiff(img1_n, img2_aligned)

            status("Matching stars...")
            matched, unm1, unm2 = match_stars_mutual(
                stars1, stars2, max_distance=self.p_match_r.get())

            status("Detecting moving objects...")
            moving_idx, med_dx, med_dy, thr_r, residuals = detect_moving_outliers(
                matched, iqr_multiplier=self.p_iqr_mult.get(),
                min_residual_px=self.p_min_res.get(),
                max_moving_frac=self.p_max_mov_frac.get())


            status("Measuring brightness changes (aperture photometry)...")
            # FITS varsa ham float veriyi, yoksa normalize uint8 kullan
            flux2_aligned = cv.warpAffine(flux2, H, (w, h),
                flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT) if H is not None else flux2
            bright_result = detect_brightness_outliers(
                matched, stars1, stars2, flux1, flux2_aligned,
                r_inner=self.p_r_inner.get(), r_outer=self.p_r_outer.get(),
                flux_ratio_thr=self.p_flux_ratio_thr.get(),
                min_flux=self.p_min_flux.get())
            brightened, dimmed, stationary, changes_arr = bright_result

            # Remove moving stars from brightness lists
            moving_set = set(moving_idx)
            matched_moving = {matched[k][0] for k in moving_set}
            brightened = [r for r in brightened if r["i"] not in matched_moving]
            dimmed     = [r for r in dimmed     if r["i"] not in matched_moving]
            stationary_b = [r for r in stationary if r["i"] not in matched_moving]

            # ── Fotometri kalibrasyonu (WCS varsa APASS ile) ──
            status("Fotometri kalibrasyonu (APASS)...")
            hdr1 = getattr(self, 'fits_header1', None)
            zp1, n_cal1 = _try_photometry_zp(hdr1, stars1, flux1,
                                              r_inner=self.p_r_inner.get(),
                                              r_outer=self.p_r_outer.get())
            self.phot_zp1   = zp1
            self.phot_n_cal1 = n_cal1
            if zp1 is not None:
                for r in brightened + dimmed + stationary_b:
                    r['v1']     = float(-2.5 * np.log10(max(r['b1'], 1.0)) + zp1)
                    r['v2']     = float(-2.5 * np.log10(max(r['b2'], 1.0)) + zp1)
                    r['delta_v'] = r['v2'] - r['v1']

            # ── ML: Isolation Forest anomali skoru ──
            status("Running ML anomaly detection (Isolation Forest)...")
            ml_scores = {}
            if ML_AVAILABLE and len(matched) >= 5:
                try:
                    feats = []
                    valid_idx = []
                    for k in range(len(matched)):
                        try:
                            fv = extract_pair_features(
                                img1_n, img2_aligned,
                                stars1, stars2, matched, k, ri, ro)
                            feats.append(fv)
                            valid_idx.append(k)
                        except Exception:
                            pass
                    if len(feats) >= 5:
                        scores = run_isolation_forest(feats)
                        for k, score in zip(valid_idx, scores):
                            ml_scores[k] = float(score)
                except Exception:
                    pass

            moving = []
            for idx in moving_idx:
                i, j, dx, dy, dist = matched[idx]
                moving.append({
                    "x1": stars1[i][0], "y1": stars1[i][1],
                    "x2": stars2[j][0], "y2": stars2[j][1],
                    "dx": dx, "dy": dy,
                    "res_dx": dx - med_dx, "res_dy": dy - med_dy,
                    "residual": float(residuals[idx]),
                    "b1": stars1[i][3], "b2": stars2[j][3],
                    "ml_score": ml_scores.get(idx, 0.0),
                })

            # ML skoru brightness listelerine de ekle
            for r in brightened + dimmed + stationary_b:
                r["ml_score"] = ml_scores.get(r["idx"], 0.0)

            self.ml_scores = ml_scores

            # ── Store analysis results for labeling ──
            self.last_matched   = matched
            self.last_stars1    = stars1
            self.last_stars2    = stars2
            self.last_brightened = brightened
            self.last_dimmed     = dimmed

            # ── Supervised classifier predictions (user-labeled in-app model) ──
            if classifier.is_trained and ML_AVAILABLE:
                for r in brightened + dimmed:
                    try:
                        fv = extract_pair_features(
                            img1_n, img2_aligned,
                            stars1, stars2, matched, r["idx"], ri, ro)
                        lbl, prob = classifier.predict(fv)
                        r["ml_label"] = lbl
                        r["ml_prob"]  = prob
                    except Exception:
                        r["ml_label"] = 0
                        r["ml_prob"]  = 0.0

            # ── Pre-trained Deep-TAO model predictions ──
            if pretrained_model is not None and ML_AVAILABLE:
                _clf_pt    = pretrained_model["classifier"]
                _scaler_pt = pretrained_model["scaler"]
                _n_feat    = pretrained_model.get("n_features", 44)
                for r in brightened + dimmed:
                    try:
                        # img1=erken, img2=geç → değişimi modele ver
                        fv = extract_deeptao_features(
                            img1_n, r["x1"], r["y1"],
                            img2_aligned, r["x2"], r["y2"])
                        fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
                        # Boyut uyumsuzluğu varsa (eski model) sıfırla
                        if len(fv) != _n_feat:
                            fv = np.zeros(_n_feat)
                        fv_s = _scaler_pt.transform(fv.reshape(1, -1))
                        lbl  = int(_clf_pt.predict(fv_s)[0])
                        prob = float(_clf_pt.predict_proba(fv_s)[0][lbl])
                        r["pt_label"] = lbl
                        r["pt_prob"]  = prob
                    except Exception:
                        r["pt_label"] = 0
                        r["pt_prob"]  = 0.0

            self.img1_n = img1_n; self.img2_n = img2_aligned; self.diff = diff

            # Store draw state so train_model() can redraw
            self._last_moving      = moving
            self._last_stationary  = stationary_b
            self._last_unm1        = unm1
            self._last_unm2        = unm2
            self._last_med_dx      = med_dx
            self._last_med_dy      = med_dy
            self._last_thr_r       = thr_r
            self._last_residuals   = residuals
            self._last_changes_arr = changes_arr

            self._draw(img1_n, img2_aligned, stars1, stars2,
                       moving, stationary_b, brightened, dimmed,
                       unm1, unm2, matched, med_dx, med_dy, thr_r, residuals,
                       changes_arr, self.mode.get())

            self._update_stats(align_method, len(stars1), len(stars2), len(matched),
                               moving, brightened, dimmed, len(unm1), len(unm2),
                               med_dx, med_dy, thr_r, residuals,
                               snr_kept1, snr_kept2)

            self.btn_save.config(state=tk.NORMAL)
            self.btn_label.config(state=tk.NORMAL)
            status(f"Done  |  {len(moving)} moving  |  "
                   f"{len(brightened)} brightened  {len(dimmed)} dimmed  |  "
                   f"Matched={len(matched)}  |  {align_method}", "#2ecc71")

        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Failed.", fg="#e74c3c")

    def _draw(self, img1_n, img2_n, stars1, stars2,
              moving, stationary, brightened, dimmed,
              unm1, unm2, matched, med_dx, med_dy, thr_r, residuals,
              changes_arr, mode):

        for w in self.canvas_frame.winfo_children(): w.destroy()
        BG = "#0f0f23"

        ncols = 3 + (1 if mode in ("moving","both") else 0) + (1 if mode in ("brightness","both") else 0)
        fig = plt.figure(figsize=(5.5*ncols, 5.4), facecolor=BG)
        axes = [fig.add_subplot(1, ncols, i+1) for i in range(ncols)]
        for ax in axes: ax.set_facecolor(BG); ax.axis("off")

        ax_i = 0

        # ── Frame 1
        axes[ax_i].imshow(img1_n, cmap=self.cmap.get(), vmin=0, vmax=255)
        axes[ax_i].set_title(f"Frame 1  ({len(stars1)} stars)", color="white", fontsize=10, fontweight="bold")
        for s in stars1:
            axes[ax_i].add_patch(plt.Circle((s[0],s[1]), min(18, max(3, np.sqrt(s[2]) * 0.5)),
                                             color="cyan", fill=False, lw=0.6, alpha=0.45))
        ax_i += 1

        # ── Frame 2
        axes[ax_i].imshow(img2_n, cmap=self.cmap.get(), vmin=0, vmax=255)
        axes[ax_i].set_title(f"Frame 2 (aligned)  ({len(stars2)} stars)", color="white", fontsize=10, fontweight="bold")
        for s in stars2:
            axes[ax_i].add_patch(plt.Circle((s[0],s[1]), min(18, max(3, np.sqrt(s[2]) * 0.5)),
                                             color="lime", fill=False, lw=0.6, alpha=0.45))
        ax_i += 1

        # ── Displacement scatter
        axes[ax_i].set_facecolor(BG); axes[ax_i].axis("on")
        all_s = stationary
        if all_s:
            axes[ax_i].scatter([e["x2"]-e["x1"] for e in all_s],
                                [e["y2"]-e["y1"] for e in all_s],
                                c="#3498db", s=16, alpha=0.55, label=f"Stationary ({len(all_s)})", zorder=2)
        for m in moving:
            axes[ax_i].scatter(m["dx"], m["dy"], c="red", s=100, marker="*", zorder=5, label="Moving")
        axes[ax_i].plot(med_dx, med_dy, "+", color="yellow", markersize=12, markeredgewidth=2, zorder=6,
                        label=f"Drift ({med_dx:+.2f},{med_dy:+.2f})")
        theta = np.linspace(0, 2*np.pi, 200)
        axes[ax_i].plot(med_dx+thr_r*np.cos(theta), med_dy+thr_r*np.sin(theta),
                        "--", color="yellow", lw=1, alpha=0.7, label=f"thr={thr_r:.2f}px")
        axes[ax_i].axvline(med_dx, color="white", lw=0.4, alpha=0.3)
        axes[ax_i].axhline(med_dy, color="white", lw=0.4, alpha=0.3)
        axes[ax_i].set_title("Displacement Map", color="white", fontsize=9, fontweight="bold")
        axes[ax_i].set_xlabel("dx (px)", color="#aaa", fontsize=8)
        axes[ax_i].set_ylabel("dy (px)", color="#aaa", fontsize=8)
        axes[ax_i].tick_params(colors="white", labelsize=7)
        for sp in axes[ax_i].spines.values(): sp.set_edgecolor("#444")
        axes[ax_i].set_aspect("equal", adjustable="datalim")
        hs, ls = axes[ax_i].get_legend_handles_labels()
        seen = {}
        for hh,ll in zip(hs,ls):
            if ll not in seen: seen[ll]=hh
        axes[ax_i].legend(seen.values(),seen.keys(), fontsize=6,
                          facecolor="#1a1a2e", labelcolor="white", edgecolor="gray")
        ax_i += 1

        # ── Moving panel
        if mode in ("moving", "both"):
            axes[ax_i].imshow(img1_n, cmap=self.cmap.get(), vmin=0, vmax=255)
            axes[ax_i].set_title(
                f"Moving Objects  ({len(moving)})",
                color="white", fontsize=10, fontweight="bold")
            for r in stationary:
                axes[ax_i].plot(r["x1"], r["y1"], ".", color="#2980b9", markersize=2, alpha=0.3)
            for k, m in enumerate(moving):
                x1,y1,x2,y2 = m["x1"],m["y1"],m["x2"],m["y2"]
                axes[ax_i].plot(x1+med_dx, y1+med_dy, "o", color="#2ecc71", markersize=5, zorder=4)
                axes[ax_i].add_patch(plt.Circle((x1,y1), 9, color="red", fill=False, lw=2.2, zorder=5))
                axes[ax_i].annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2.5, mutation_scale=18), zorder=6)
                axes[ax_i].plot(x2,y2,"o",color="red",markersize=5,zorder=7)
                axes[ax_i].text(x2+9,y2-7,f"#{k+1} {m['residual']:.2f}px",
                                color="red",fontsize=7,fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2",fc="#1a1a2e",ec="red",alpha=0.7))
            leg_m=[mpatches.Patch(color="#2980b9", label="Stationary"),
                   plt.Line2D([0],[0], color="red",    lw=2, label="Moving (RANSAC)"),
                   plt.Line2D([0],[0], marker="o", color="#2ecc71", lw=0,
                              markersize=5, label="Expected pos")]
            axes[ax_i].legend(handles=leg_m,loc="lower right",fontsize=6.5,
                              framealpha=0.75,facecolor="#1a1a2e",labelcolor="white",edgecolor="gray")
            ax_i += 1

        # ── Brightness panel
        if mode in ("brightness", "both"):
            axes[ax_i].imshow(img1_n, cmap=self.cmap.get(), vmin=0, vmax=255)
            axes[ax_i].set_title(
                f"Brightness Changes  ({len(brightened)}↑  {len(dimmed)}↓)",
                color="white", fontsize=10, fontweight="bold")

            # All stationary stars: small cyan circles (same as frame 1 panel)
            for r in stationary:
                axes[ax_i].add_patch(plt.Circle((r["x1"],r["y1"]), max(3, np.sqrt(
                    stars1[r["i"]][2])), color="cyan", fill=False, lw=0.6, alpha=0.4))

            def _clf_label_txt(r, change_prefix):
                """Return label string: pretrained model > in-app model > ML score."""
                ml  = r.get("ml_score", 0.0)
                v1  = r.get('v1', None)
                v2  = r.get('v2', None)
                dv  = r.get('delta_v', None)
                mag_str = (f"  V:{v1:.2f}→{v2:.2f} (ΔV={dv:+.2f})"
                           if v1 is not None else "")
                if "pt_label" in r:
                    lbl = "Transient" if r["pt_label"] == 1 else "Non-transient"
                    pct = int(r.get("pt_prob", 0.0) * 100)
                    return f"{change_prefix}{r['change_pct']:.0f}%{mag_str} | {lbl} {pct}%"
                elif classifier.is_trained and "ml_label" in r:
                    lbl = "Transient" if r["ml_label"] == 1 else "Artifact"
                    pct = int(r.get("ml_prob", 0.0) * 100)
                    return f"{change_prefix}{r['change_pct']:.0f}%{mag_str} | {lbl} {pct}%"
                else:
                    return f"{change_prefix}{r['change_pct']:.0f}%{mag_str}  ML:{ml:.2f}"

            # Brightened: bright orange filled glow + outer ring + label
            for k, r in enumerate(brightened):
                cx, cy = r["x1"], r["y1"]
                ml  = r.get("ml_score", 0.0)
                is_transient = r.get("pt_label", r.get("ml_label", -1)) == 1
                lw  = 3.5 if (ml > 0.7 or is_transient) else 2.2
                axes[ax_i].add_patch(plt.Circle((cx,cy), 12,
                    color="#ff6b35", fill=False, lw=lw, alpha=0.95, zorder=5))
                axes[ax_i].add_patch(plt.Circle((cx,cy), 4,
                    color="#ff6b35", fill=True, alpha=0.85, zorder=6))
                axes[ax_i].text(cx+14, cy-8,
                    _clf_label_txt(r, "+"),
                    color="#ff6b35", fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0d0d1a", ec="#ff6b35", alpha=0.8))

            # Dimmed: blue-white ring + label
            for k, r in enumerate(dimmed):
                cx, cy = r["x1"], r["y1"]
                ml  = r.get("ml_score", 0.0)
                is_transient = r.get("pt_label", r.get("ml_label", -1)) == 1
                lw  = 3.5 if (ml > 0.7 or is_transient) else 2.2
                axes[ax_i].add_patch(plt.Circle((cx,cy), 12,
                    color="#74b9ff", fill=False, lw=lw, alpha=0.95, zorder=5))
                axes[ax_i].add_patch(plt.Circle((cx,cy), 4,
                    color="#74b9ff", fill=True, alpha=0.85, zorder=6))
                axes[ax_i].text(cx+14, cy-8,
                    _clf_label_txt(r, ""),
                    color="#74b9ff", fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0d0d1a", ec="#74b9ff", alpha=0.8))

            leg_b=[mpatches.Patch(color="cyan",    label="Stationary"),
                   mpatches.Patch(color="#ff6b35",  label="Brightened ↑"),
                   mpatches.Patch(color="#74b9ff",  label="Dimmed ↓"),
                   mpatches.Patch(color="white",    label="ML score (0-1, higher=anomalous)")]
            axes[ax_i].legend(handles=leg_b, loc="lower left", fontsize=6.5,
                              framealpha=0.75, facecolor="#1a1a2e",
                              labelcolor="white", edgecolor="gray")

        plt.tight_layout(pad=0.4)
        canvas = FigureCanvasTkAgg(fig, self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig = fig

    def _update_stats(self, method, n1, n2, n_match, moving, brightened, dimmed,
                      n_unm1, n_unm2, med_dx, med_dy, thr_r, residuals,
                      snr_kept1=None, snr_kept2=None, large_movers=None):
        # SNR istatistikleri
        snr_info = ""
        if snr_kept1 and snr_kept2:
            snr_info = (
                f"  SNR  F1: min={min(snr_kept1):.1f}  median={float(np.median(snr_kept1)):.1f}"
                f"  max={max(snr_kept1):.1f}  |  "
                f"F2: min={min(snr_kept2):.1f}  median={float(np.median(snr_kept2)):.1f}"
                f"  max={max(snr_kept2):.1f}\n"
            )
        res_info = ""
        if len(residuals) > 0:
            res_info = (f"  Residuals: min={residuals.min():.3f}  median={np.median(residuals):.3f}"
                        f"  max={residuals.max():.3f}  -> moving_thr={thr_r:.3f}px\n")
        _lm = large_movers or []
        mov_lines = "".join(
            f"    [Moving #{k+1}]  ({m['x1']:.0f},{m['y1']:.0f})->({m['x2']:.0f},{m['y2']:.0f})"
            f"  |res|={m['residual']:.3f}px  dx={m['dx']:+.2f} dy={m['dy']:+.2f}\n"
            for k,m in enumerate(moving))
        mov_lines += "".join(
            f"    [FAST #{k+1}]   ({lm['x1']:.0f},{lm['y1']:.0f})->({lm['x2']:.0f},{lm['y2']:.0f})"
            f"  dist={lm['dist']:.1f}px  dx={lm['dx']:+.1f} dy={lm['dy']:+.1f}"
            f"  peak_ratio={lm['peak_ratio']:.2f}\n"
            for k,lm in enumerate(_lm))
        def _mag_str(r):
            if 'v1' in r:
                return f"  V:{r['v1']:.2f}→{r['v2']:.2f} (ΔV={r['delta_v']:+.2f})"
            return ""
        bright_lines = "".join(
            f"    [Bright #{k+1}]  ({r['x1']:.0f},{r['y1']:.0f})"
            f"  flux:{r['b1']:.0f}->{r['b2']:.0f}  {r['change_pct']:+.1f}%"
            + _mag_str(r) +
            f"  ML:{r.get('ml_score',0):.2f}\n"
            for k,r in enumerate(brightened[:5]))
        bright_lines += "".join(
            f"    [Dim   #{k+1}]  ({r['x1']:.0f},{r['y1']:.0f})"
            f"  flux:{r['b1']:.0f}->{r['b2']:.0f}  {r['change_pct']:+.1f}%"
            + _mag_str(r) +
            f"  ML:{r.get('ml_score',0):.2f}\n"
            for k,r in enumerate(dimmed[:5]))
        zp1     = getattr(self, 'phot_zp1', None)
        n_cal1  = getattr(self, 'phot_n_cal1', 0)
        phot_info = (f"  Fotometri: ZP={zp1:.3f}  ({n_cal1} kal. yıldız, APASS V-band)\n"
                     if zp1 is not None
                     else "  Fotometri: WCS bulunamadı — magnitüd hesaplanamadı\n")
        line = (
            f"  Alignment: {method}\n"
            f"  Stars: F1={n1}  F2={n2}  Matched={n_match}  "
            f"Unmatched1={n_unm1}  Unmatched2={n_unm2}  drift=({med_dx:+.3f},{med_dy:+.3f})px\n"
            + snr_info + res_info + phot_info
            + f"  Moving={len(moving)}  FastMovers={len(_lm)}  Brightened={len(brightened)}  Dimmed={len(dimmed)}\n"
            + mov_lines + bright_lines
        ).rstrip()
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, line)
        self.stats_text.config(state=tk.DISABLED)

    def analyze_catalog(self):
        """Tek görüntü için yıldız katalogu — konum, parlaklık, SNR."""
        img  = self.img1 if self.img1 is not None else self.img2
        raw  = self.fits_raw1 if self.fits_raw1 is not None else (
               self.fits_raw2 if self.fits_raw2 is not None else None)
        path = self.img1_path if self.img1 is not None else self.img2_path

        img_n    = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
        flux_img = raw if raw is not None else img_n.astype(np.float32)

        try:
            self.status_label.config(text="Building star catalog...", fg="#3498db")
            self.root.update_idletasks()

            kw = dict(pct=self.p_thr_pct.get(), min_area=self.p_min_area.get(),
                      max_area=self.p_max_area.get(), min_peak=self.p_min_peak.get())
            stars_raw = detect_stars(img_n, **kw)

            ri, ro   = self.p_r_inner.get(), self.p_r_outer.get()
            min_snr  = self.p_min_snr.get()
            stars, _, _ = filter_stars_by_snr(stars_raw, img_n, min_snr, ri, ro)

            catalog = []
            for idx, s in enumerate(stars):
                cx, cy, area, peak = s
                flux = bg_subtracted_flux(flux_img, cx, cy, ri, ro)
                snr  = calculate_star_snr(img_n, cx, cy, ri, ro)
                catalog.append({"id": idx+1, "x": round(cx, 2), "y": round(cy, 2),
                                 "area": area, "peak": peak,
                                 "flux": round(flux, 1), "snr": round(snr, 2)})

            catalog.sort(key=lambda r: r["flux"], reverse=True)
            # ID'leri flux sırasına göre yeniden ver
            for k, r in enumerate(catalog):
                r["id"] = k + 1

            fname = os.path.basename(path) if path else "image"
            self._show_catalog_window(img_n, catalog, fname)
            self.status_label.config(
                text=f"Catalog: {len(catalog)} stars found", fg="#2ecc71")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _show_catalog_window(self, img_n, catalog, title):
        from tkinter import ttk
        import csv

        win = tk.Toplevel(self.root)
        win.title(f"Star Catalog — {title}  ({len(catalog)} stars)")
        win.geometry("1150x680")
        win.configure(bg="#1a1a2e")

        # ── Sol: fotoğraf ──
        fig_frame = tk.Frame(win, bg="#1a1a2e")
        fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="#0f0f23")
        ax.imshow(img_n, cmap=self.cmap.get(), vmin=0, vmax=255)
        ax.set_facecolor("#0f0f23"); ax.axis("off")
        ax.set_title(f"{len(catalog)} stars  (SNR ≥ {self.p_min_snr.get():.1f})",
                     color="white", fontsize=10, fontweight="bold")
        for r in catalog:
            if r["peak"] >= 254:
                radius = min(20, max(8, np.sqrt(r["area"]) * 0.6))
                color  = "yellow"
                lw     = 1.5
            else:
                radius = min(15, max(5, np.sqrt(r["area"]) * 0.5))
                color  = "cyan"
                lw     = 1.0
            ax.add_patch(plt.Circle((r["x"], r["y"]), radius,
                         color=color, fill=False, lw=lw, alpha=0.85))
            ax.text(r["x"] + radius + 2, r["y"] - 5, str(r["id"]),
                    color=color, fontsize=5, alpha=0.9)
        canvas = FigureCanvasTkAgg(fig, fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

        # ── Sağ: tablo + export ──
        right = tk.Frame(win, bg="#1a1a2e", width=470)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 8), pady=8)
        right.pack_propagate(False)

        def export_csv():
            p = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All", "*.*")])
            if not p: return
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id","x","y","area","peak","flux","snr"])
                w.writeheader(); w.writerows(catalog)
            messagebox.showinfo("Exported", p)

        tk.Button(right, text="Export CSV", command=export_csv,
                  bg="#e67e22", fg="white", font=("Arial", 10, "bold"),
                  padx=12, pady=5, cursor="hand2", relief=tk.RAISED, bd=3
                  ).pack(pady=(0, 6), anchor="w")

        # Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Cat.Treeview", background="#0d0d1a", foreground="#ecf0f1",
                        fieldbackground="#0d0d1a", font=("Courier", 9), rowheight=20)
        style.configure("Cat.Treeview.Heading", background="#16213e",
                        foreground="#00ff88", font=("Arial", 9, "bold"))
        style.map("Cat.Treeview", background=[("selected", "#2c3e50")])

        cols = ("ID", "X (px)", "Y (px)", "Area", "Peak", "Flux", "SNR")
        col_w = (40, 70, 70, 50, 55, 70, 60)
        tree = ttk.Treeview(right, columns=cols, show="headings",
                            style="Cat.Treeview")
        for col, w in zip(cols, col_w):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center", stretch=False)

        for r in catalog:
            tree.insert("", "end", values=(
                r["id"], r["x"], r["y"], r["area"], r["peak"],
                f"{r['flux']:.0f}", f"{r['snr']:.1f}"))

        sb = ttk.Scrollbar(right, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def open_label_window(self):
        """Brightness değişimlerini birer birer etiketle — kart sistemi."""
        events = list(self.last_brightened) + list(self.last_dimmed)
        if not events:
            messagebox.showinfo("Label Events",
                                "Etiketlenecek olay yok.\nÖnce Analyze çalıştır.")
            return

        n_total  = len(events)
        n_bright = len(self.last_brightened)
        current  = [0]          # mutable index
        labeled  = {}           # idx → 0/1

        win = tk.Toplevel(self.root)
        win.title("Label Events for Training")
        win.geometry("700x560")
        win.configure(bg="#1a1a2e")
        win.resizable(False, False)

        # ── Başlık ──
        header = tk.Frame(win, bg="#16213e")
        header.pack(fill=tk.X)
        progress_var = tk.StringVar()
        tk.Label(header, textvariable=progress_var,
                 font=("Arial", 12, "bold"), bg="#16213e", fg="#ecf0f1"
                 ).pack(side=tk.LEFT, padx=16, pady=10)
        status_var = tk.StringVar(value="")
        tk.Label(header, textvariable=status_var,
                 font=("Arial", 10), bg="#16213e", fg="#95a5a6"
                 ).pack(side=tk.RIGHT, padx=16, pady=10)

        # ── Kart alanı ──
        card = tk.Frame(win, bg="#0f0f23", relief=tk.SUNKEN, bd=2)
        card.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        img_holder  = tk.Frame(card, bg="#0f0f23")
        img_holder.pack(pady=8)
        info_holder = tk.Frame(card, bg="#0f0f23")
        info_holder.pack(pady=4)

        # ── Alt butonlar ──
        btn_bar = tk.Frame(win, bg="#1a1a2e")
        btn_bar.pack(fill=tk.X, padx=16, pady=10)

        def _render(idx):
            """Mevcut kartı çiz."""
            for w in img_holder.winfo_children():  w.destroy()
            for w in info_holder.winfo_children(): w.destroy()

            r         = events[idx]
            is_bright = r in self.last_brightened
            star_num  = idx + 1 if is_bright else (idx - n_bright + 1)
            star_kind = "Brightened ↑" if is_bright else "Dimmed ↓"
            color     = "#ff6b35" if is_bright else "#74b9ff"
            sign      = "+" if is_bright else ""
            chg       = r.get("change_pct", 0.0)

            progress_var.set(f"Olay {idx+1} / {n_total}")
            done = sum(1 for v in labeled.values() if v == 0)
            trans = sum(1 for v in labeled.values() if v == 1)
            status_var.set(f"Transient: {trans}  |  Artifact: {done}  |  Kalan: {n_total-len(labeled)}")

            # ── Görüntüler ──
            try:
                cx, cy       = int(r["x1"]), int(r["y1"])
                h_img, w_img = self.img1_n.shape

                fig, (ax_map, ax_zoom) = plt.subplots(
                    1, 2, figsize=(6.5, 3.0),
                    facecolor="#0f0f23",
                    gridspec_kw={"width_ratios": [1.4, 1]})

                # Harita — tam resim
                ax_map.imshow(self.img1_n, cmap=self.cmap.get(), vmin=0, vmax=255)
                ax_map.plot(cx, cy, "o", color=color,
                            markersize=14, markerfacecolor="none",
                            markeredgewidth=2.5)
                ax_map.plot(cx, cy, "+", color=color,
                            markersize=16, markeredgewidth=2)
                ax_map.set_facecolor("#0f0f23"); ax_map.axis("off")
                ax_map.set_title(f"#{star_num} {star_kind}  —  tam resimdeki konum",
                                 color=color, fontsize=9, fontweight="bold", pad=4)

                # Yakın görünüm
                half = 55
                x0 = max(0, cx-half); x1 = min(w_img, cx+half)
                y0 = max(0, cy-half); y1 = min(h_img, cy+half)
                ctx      = self.img1_n[y0:y1, x0:x1]
                lcx, lcy = cx-x0, cy-y0
                ax_zoom.imshow(ctx, cmap=self.cmap.get(), vmin=0, vmax=255)
                ax_zoom.add_patch(plt.Circle((lcx, lcy), 12,
                                  color=color, fill=False, lw=2.2))
                ax_zoom.axvline(lcx, color=color, lw=0.8, alpha=0.5)
                ax_zoom.axhline(lcy, color=color, lw=0.8, alpha=0.5)
                ax_zoom.set_facecolor("#0f0f23"); ax_zoom.axis("off")
                ax_zoom.set_title("Yakın görünüm", color="#aaa",
                                  fontsize=8, pad=4)

                fig.tight_layout(pad=0.3)
                cv_widget = FigureCanvasTkAgg(fig, img_holder)
                cv_widget.draw()
                cv_widget.get_tk_widget().pack()
                plt.close(fig)
            except Exception:
                tk.Label(img_holder, text="Görüntü yüklenemedi",
                         bg="#0f0f23", fg="gray").pack()

            # ── Bilgi satırı ──
            tk.Label(info_holder,
                     text=f"Değişim: {sign}{chg:.0f}%",
                     font=("Arial", 11, "bold"),
                     bg="#0f0f23", fg=color).pack(side=tk.LEFT, padx=20)
            tk.Label(info_holder,
                     text=f"Konum: ({r['x1']:.0f}, {r['y1']:.0f})",
                     font=("Arial", 10),
                     bg="#0f0f23", fg="#7f8c8d").pack(side=tk.LEFT, padx=20)
            if "pt_label" in r:
                pt_lbl = "Transient" if r["pt_label"] == 1 else "Non-transient"
                pt_pct = int(r.get("pt_prob", 0.0) * 100)
                ml_col = "#2ecc71" if r["pt_label"] == 1 else "#95a5a6"
                tk.Label(info_holder,
                         text=f"Model: {pt_lbl} {pt_pct}%",
                         font=("Arial", 10, "bold"),
                         bg="#0f0f23", fg=ml_col).pack(side=tk.LEFT, padx=20)
            else:
                ml = r.get("ml_score", 0.0)
                tk.Label(info_holder,
                         text=f"Anomali: {ml:.2f}",
                         font=("Arial", 10),
                         bg="#0f0f23", fg="#95a5a6").pack(side=tk.LEFT, padx=20)

        def _label(lbl):
            """Etiketi kaydet ve bir sonraki karta geç."""
            idx = current[0]
            r   = events[idx]
            try:
                fv = extract_pair_features(
                    self.img1_n, self.img2_n,
                    self.last_stars1, self.last_stars2,
                    self.last_matched, r["idx"],
                    self.p_r_inner.get(), self.p_r_outer.get())
                classifier.add_sample(fv, lbl)
            except Exception as e:
                messagebox.showerror("Feature Error", str(e)); return

            labeled[idx] = lbl

            # Train Model butonu etkinleştir?
            vals = list(labeled.values())
            if len(vals) >= 4 and 0 in vals and 1 in vals:
                self.btn_train.config(state=tk.NORMAL)

            # Sonraki olaya geç
            next_idx = idx + 1
            if next_idx >= n_total:
                # Hepsi bitti
                for w in img_holder.winfo_children():  w.destroy()
                for w in info_holder.winfo_children(): w.destroy()
                t = sum(1 for v in labeled.values() if v == 1)
                a = sum(1 for v in labeled.values() if v == 0)
                progress_var.set(f"Tamamlandı!  {n_total} olay etiketlendi")
                status_var.set(f"Transient: {t}  |  Artifact: {a}")
                tk.Label(img_holder,
                         text=f"✓ Tüm olaylar etiketlendi\n\nTransient: {t}   Artifact: {a}\n\n"
                              f"{'Train Model butonuna basabilirsin.' if (t>=2 and a>=2) else 'En az 2 transient + 2 artifact gerekli.'}",
                         font=("Arial", 13), bg="#0f0f23",
                         fg="#2ecc71" if (t>=2 and a>=2) else "#e74c3c",
                         justify="center").pack(expand=True, pady=40)
                btn_transient.config(state=tk.DISABLED)
                btn_artifact.config(state=tk.DISABLED)
                return

            current[0] = next_idx
            _render(next_idx)

        # ── Butonlar ──
        btn_artifact  = tk.Button(btn_bar, text="Artifact  ✗",
                                  command=lambda: _label(0),
                                  font=("Arial", 13, "bold"),
                                  bg="#e74c3c", fg="white",
                                  padx=30, pady=12, cursor="hand2",
                                  relief=tk.RAISED, bd=3)
        btn_artifact.pack(side=tk.LEFT, padx=8)

        btn_transient = tk.Button(btn_bar, text="Transient  ✓",
                                  command=lambda: _label(1),
                                  font=("Arial", 13, "bold"),
                                  bg="#27ae60", fg="white",
                                  padx=30, pady=12, cursor="hand2",
                                  relief=tk.RAISED, bd=3)
        btn_transient.pack(side=tk.RIGHT, padx=8)

        tk.Button(btn_bar, text="Atla →",
                  command=lambda: _label_skip(),
                  font=("Arial", 11), bg="#2c3e50", fg="white",
                  padx=16, pady=12, cursor="hand2").pack(side=tk.LEFT, padx=4)

        def _label_skip():
            """Etiketsiz atla."""
            next_idx = current[0] + 1
            if next_idx >= n_total:
                win.destroy(); return
            current[0] = next_idx
            _render(next_idx)

        # İlk kartı göster
        _render(0)

    def train_model(self):
        """Train the Random Forest on all labeled samples."""
        if not ML_AVAILABLE:
            messagebox.showerror("ML Unavailable", "scikit-learn is not installed.")
            return

        vals = [s[1] for s in classifier._samples]
        n_total = len(vals)
        n_transient = vals.count(1)
        n_artifact  = vals.count(0)

        if n_transient < 2 or n_artifact < 2:
            messagebox.showwarning(
                "Not Enough Samples",
                f"Need at least 2 samples per class.\n"
                f"Current: {n_transient} transient, {n_artifact} artifact")
            return

        try:
            classifier.train()
        except Exception as e:
            messagebox.showerror("Training Error", str(e)); return

        # Cross-val accuracy
        acc = classifier.cross_val_accuracy()
        if acc is not None:
            acc_str = f"{acc*100:.1f}%"
        else:
            acc_str = "N/A (too few samples for CV)"

        # Save model
        try:
            classifier.save(str(MODEL_PATH))
            save_msg = f"\nSaved to: {MODEL_PATH}"
        except Exception as e:
            save_msg = f"\n(Save failed: {e})"

        messagebox.showinfo(
            "Model Trained",
            f"Model trained!\n"
            f"Samples: {n_total}  ({n_transient} transient, {n_artifact} artifact)\n"
            f"Cross-val accuracy: {acc_str}"
            + save_msg)

        # Re-run predictions on current data if analysis results available
        if (self.last_matched and self.last_stars1 and
                self.last_brightened is not None and self.img1_n is not None):
            ri = self.p_r_inner.get()
            ro = self.p_r_outer.get()
            for r in self.last_brightened + self.last_dimmed:
                try:
                    fv = extract_pair_features(
                        self.img1_n, self.img2_n,
                        self.last_stars1, self.last_stars2,
                        self.last_matched, r["idx"], ri, ro)
                    lbl, prob = classifier.predict(fv)
                    r["ml_label"] = lbl
                    r["ml_prob"]  = prob
                except Exception:
                    r["ml_label"] = 0
                    r["ml_prob"]  = 0.0

            # Redraw with updated predictions
            try:
                self._draw(self.img1_n, self.img2_n,
                           self.last_stars1, self.last_stars2,
                           getattr(self, "_last_moving", []),
                           getattr(self, "_last_stationary", []),
                           self.last_brightened, self.last_dimmed,
                           getattr(self, "_last_unm1", []),
                           getattr(self, "_last_unm2", []),
                           self.last_matched,
                           getattr(self, "_last_med_dx", 0.0),
                           getattr(self, "_last_med_dy", 0.0),
                           getattr(self, "_last_thr_r", 1.0),
                           getattr(self, "_last_residuals", np.array([])),
                           getattr(self, "_last_changes_arr", np.array([])),
                           self.mode.get())
            except Exception:
                pass

    def save_result(self):
        if not hasattr(self, "fig"): return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("All","*.*")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight",
                             facecolor=self.fig.get_facecolor())
            messagebox.showinfo("Saved", path)


def main():
    root = tk.Tk()
    ImageComparisonApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()