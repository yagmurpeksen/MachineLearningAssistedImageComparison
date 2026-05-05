"""
Deep-TAO dataset ile transient sınıflandırma modeli eğitimi.

Yaklaşım: Her FITS dosyasından ilk + son frame alınır.
İkisinin farkı özellik vektörüne eklenir → model değişimi öğrenir.
Bu, GUI'nin img1 vs img2 karşılaştırmasıyla birebir örtüşür.

Sınıflar: SN, AGN, CV, BZ, OTHER → Transient (1)
          NON                     → Non-transient (0)

Özellik vektörü (44 boyut):
  [0:18]  → erken frame istatistikleri
  [18:36] → geç frame istatistikleri
  [36:44] → değişim özellikleri (delta)
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from astropy.io import fits as astropy_fits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_DIR  = r"C:\Users\pekse\Desktop\cse492\deeptao_data"
MODEL_OUT = r"C:\Users\pekse\Desktop\cse492\transient_model.joblib"

TRANSIENT_CLASSES     = ["SN", "AGN", "CV", "BZ", "OTHER"]
NON_TRANSIENT_CLASSES = ["NON"]

FEAT_NAMES_SINGLE = [
    "mean", "std", "peak", "min", "median", "skew", "kurt",
    "p25", "p75", "p90", "p99",
    "conc_inner", "conc_middle", "conc_outer",
    "snr", "signal", "bg", "slope"
]  # 18 özellik

FEAT_NAMES = (
    [f"early_{n}" for n in FEAT_NAMES_SINGLE] +   # 18
    [f"late_{n}"  for n in FEAT_NAMES_SINGLE] +   # 18
    ["delta_peak", "delta_mean", "delta_snr",      # 8 delta
     "delta_signal", "flux_ratio", "log_flux_ratio",
     "delta_conc", "delta_slope"]
)  # Toplam: 44


# ── Tek frame özellik çıkarma ─────────────────────────────────────

def extract_single_frame(cutout):
    """64x64 cutout'tan 18 istatistiksel özellik çıkar."""
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


# ── İki frame + delta özellik çıkarma ────────────────────────────

def extract_change_features(early, late):
    """
    Erken + geç frame'den 44 boyutlu özellik vektörü çıkar.
    Delta özellikler değişimi kodlar: bu projenin asıl amacı.
    """
    f_early = extract_single_frame(early)
    f_late  = extract_single_frame(late)

    # İndeks referansları (FEAT_NAMES_SINGLE sırasına göre)
    IDX_PEAK   = 2   # peak
    IDX_MEAN   = 0   # mean
    IDX_SNR    = 14  # snr
    IDX_SIGNAL = 15  # signal
    IDX_CONC   = 11  # conc_inner
    IDX_SLOPE  = 17  # slope

    e_signal = f_early[IDX_SIGNAL] + 1e-9
    l_signal = f_late[IDX_SIGNAL]

    delta_peak   = f_late[IDX_PEAK]   - f_early[IDX_PEAK]
    delta_mean   = f_late[IDX_MEAN]   - f_early[IDX_MEAN]
    delta_snr    = f_late[IDX_SNR]    - f_early[IDX_SNR]
    delta_signal = l_signal - f_early[IDX_SIGNAL]
    flux_ratio   = l_signal / e_signal
    log_flux_r   = float(np.log1p(abs(flux_ratio - 1.0)))
    delta_conc   = f_late[IDX_CONC]  - f_early[IDX_CONC]
    delta_slope  = f_late[IDX_SLOPE] - f_early[IDX_SLOPE]

    delta = np.array([
        delta_peak, delta_mean, delta_snr, delta_signal,
        flux_ratio, log_flux_r, delta_conc, delta_slope
    ])

    return np.concatenate([f_early, f_late, delta])  # 18+18+8 = 44


# ── FITS dosyası yükleme ──────────────────────────────────────────

def load_early_late(path):
    """
    FITS dosyasından ilk ve son frame'i döner.
    Deep-TAO: HDU[0]=header, HDU[1..N]=64x64 frame'ler.
    """
    try:
        with astropy_fits.open(path) as hdul:
            frames = []
            for hdu in hdul[1:]:
                if hdu.data is not None and np.ndim(hdu.data) == 2:
                    frames.append(hdu.data.astype(np.float32))
                    if len(frames) >= 2:  # En az ilk 2'yi bul (hız için)
                        pass
            if not frames:
                return None, None
            # Tüm frame'leri taramak yavaş olur, son frame için sona bak
            if len(frames) == 1:
                return frames[0], frames[0]
            # Birden fazla frame: ilk ve son
            # Eğer sadece 2 frame yüklendiyse son HDU'ya tekrar git
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
    """Bir FITS dosyasından 44 boyutlu değişim özelliği çıkar."""
    early, late = load_early_late(path)
    if early is None:
        return None
    feat = extract_change_features(early, late)
    if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


# ── Dataset yükleme ───────────────────────────────────────────────

def load_dataset(data_dir):
    X, y, cls_labels = [], [], []

    for cls in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        label = 0 if cls in NON_TRANSIENT_CLASSES else 1
        fits_files = [f for f in os.listdir(cls_dir)
                      if f.endswith(".fits") or "." not in f]

        print(f"  [{cls:6s}] {len(fits_files)} dosya  (label={label})")
        ok = 0
        for fname in fits_files:
            feat = extract_features_from_file(os.path.join(cls_dir, fname))
            if feat is not None:
                X.append(feat)
                y.append(label)
                cls_labels.append(cls)
                ok += 1
        print(f"           {ok}/{len(fits_files)} yuklendi")

    return np.array(X), np.array(y), cls_labels


# ── Eğitim ────────────────────────────────────────────────────────

def train(data_dir=DATA_DIR, model_out=MODEL_OUT):
    print("=" * 60)
    print("Deep-TAO Model Egitimi  (erken+gec frame, degisim ozellikleri)")
    print("=" * 60)

    print("\n[1/4] Dataset yukleniyor...")
    X, y, cls_labels = load_dataset(data_dir)

    if len(X) == 0:
        print("HATA: Hic veri yuklenemedi!")
        return

    print(f"\nToplam ornek  : {len(X)}")
    print(f"  Transient   : {y.sum()}")
    print(f"  Non-transient: {(y == 0).sum()}")
    print(f"  Ozellik boyutu: {X.shape[1]}  ({len(FEAT_NAMES)} isimli)")

    print("\n[2/4] Train/Test split (%%80/%%20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    print("\n[3/4] StandardScaler + Random Forest egitiliyor...")
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",   # sinif dengesizligini duzeltir
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_s, y_train)

    print("\n[4/4] Degerlendirme...")
    y_pred = clf.predict(X_test_s)
    acc    = (y_pred == y_test).mean() * 100

    print(f"\nTest Accuracy: {acc:.1f}%%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Non-transient", "Transient"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5, n_jobs=-1)
    print(f"\n5-Fold CV: {cv_scores.mean()*100:.1f}%% +/- {cv_scores.std()*100:.1f}%%")

    # En onemli 5 ozellik
    top5 = np.argsort(clf.feature_importances_)[::-1][:5]
    print("\nEn onemli 5 ozellik:")
    for i in top5:
        print(f"  {FEAT_NAMES[i]:<25} {clf.feature_importances_[i]:.3f}")

    # Kaydet
    model_data = {
        "classifier": clf,
        "scaler":     scaler,
        "accuracy":   acc,
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "feat_names": FEAT_NAMES,
        "n_features": len(FEAT_NAMES),
    }
    joblib.dump(model_data, model_out)
    print(f"\nModel kaydedildi: {model_out}")
    print("=" * 60)
    print("Hazir! GUI otomatik olarak bu modeli yukleyecek.")


if __name__ == "__main__":
    train()
