"""
image_comparison_gui.py
=======================
Main application file — GUI layer.

Core logic is kept in separate modules (single responsibility principle):
  fits_loader.py    → FITS loading and normalization
  detection.py      → Star detection, alignment, matching, brightness analysis
  ml_classifier.py  → Machine learning (TransientClassifier, Deep-TAO features)
  photometry.py     → APASS catalog query, ZP calibration

This file contains only the GUI components (ImageComparisonApp).
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False
    tk = None
    filedialog = None
    messagebox = None
    FigureCanvasTkAgg = None

import os
import sys
import threading
from pathlib import Path

# ── Module imports ───────────────────────────────────────────────
from fits_loader import load_fits_image, sqrt_stretch, FITS_AVAILABLE  # noqa: F401

from detection import (                                           # noqa: F401
    detect_stars,
    detect_stars_dao,
    match_detection_depth,
    DAO_AVAILABLE,
    star_affine_align,
    match_stars_mutual,
    detect_moving_outliers,
    detect_moving_from_unmatched,
    bg_subtracted_flux,
    calculate_star_snr,
    filter_stars_by_snr,
    detect_brightness_outliers,
    normalize_pair_joint,
    filter_stars_from_border,
)

from ml_classifier import (                                       # noqa: F401
    ML_AVAILABLE,
    TransientClassifier,
    classifier,
    extract_deeptao_features,
    extract_pair_features,
    run_isolation_forest,
)

# joblib: for model loading/saving (comes with sklearn)
if ML_AVAILABLE:
    import joblib

from photometry import (                                          # noqa: F401
    _lookup_sat_mags_from_catalog,
    _try_photometry_zp,
)


# ════════════════════════════════════════════════════════════════
#  WCS PIXEL OFFSET HELPER
# ════════════════════════════════════════════════════════════════

def _wcs_pixel_offset(hdr1, hdr2):
    """
    Computes pixel offset (x2-x1, y2-y1) from the CRVAL1/CRVAL2 difference
    of two FITS headers.

    Returns
    -------
    (dx, dy) : float
        dx = x2 - x1 (x pixel difference of epoch2 stars relative to epoch1)
        Use shift_x = -dx to align epoch2 with epoch1.
    """
    import math
    try:
        ra1, dec1  = float(hdr1['CRVAL1']), float(hdr1['CRVAL2'])
        ra2, dec2  = float(hdr2['CRVAL1']), float(hdr2['CRVAL2'])
        cd11 = float(hdr1.get('CD1_1', -0.000281))  # deg/px (negative: RA increases westward)
        cd22 = float(hdr1.get('CD2_2',  0.000281))  # deg/px (positive: Dec increases northward)
        dec_mid = math.radians((dec1 + dec2) / 2.0)
        # pixel offsets: x2-x1 and y2-y1
        dx = (ra1 - ra2) * math.cos(dec_mid) / cd11
        dy = (dec1 - dec2) / cd22
        return float(dx), float(dy)
    except Exception:
        return 0.0, 0.0


# ════════════════════════════════════════════════════════════════
#  FIND APPLICATION BASE DIR  (script or PyInstaller EXE)
# ════════════════════════════════════════════════════════════════


def _app_base_dir() -> Path:
    """
    Base directory for reading/writing app artifacts.
    - When running as a script: directory containing this file.
    - When frozen (PyInstaller): directory containing the executable.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


MODEL_PATH      = _app_base_dir() / "transient_model.joblib"   # Deep-TAO pre-trained
USER_MODEL_PATH = _app_base_dir() / "user_model.joblib"          # user's own model

# Load pre-trained Deep-TAO model at startup
pretrained_model = None
if ML_AVAILABLE:
    try:
        _pt_data = joblib.load(MODEL_PATH)
        if "classifier" in _pt_data and "scaler" in _pt_data:
            pretrained_model = _pt_data
    except Exception:
        pass

# Load user's in-app model at startup (labels + model persisted from previous session)
if ML_AVAILABLE and USER_MODEL_PATH.exists():
    try:
        classifier.load(str(USER_MODEL_PATH))
    except Exception:
        pass



#  GUI — APPLICATION CLASS
# ════════════════════════════════════════════════════════════════

class ImageComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stellar Analysis")
        self.root.geometry("1500x900")
        self.root.configure(bg="#1a1a2e")
        self.img1 = self.img2 = None
        self.img1_path = self.img2_path = None
        self.fits_raw1 = self.fits_raw2 = None   # original float32 FITS data
        self.fits_info1 = self.fits_info2 = {}   # header info
        self.fits_header1 = self.fits_header2 = None  # full FITS header (for WCS)
        self.phot_zp1 = self.phot_zp2 = None    # photometry zero point
        self.phot_n_cal1 = self.phot_n_cal2 = 0
        self.mode       = tk.StringVar(value="both")
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
        self.p_flux_ratio_thr = tk.DoubleVar(value=1.5)
        self.p_min_snr        = tk.DoubleVar(value=1.0)
        # DAO detection (sigma-clipped threshold for FITS)
        self.p_use_dao        = tk.BooleanVar(value=True)
        self.p_dao_fwhm       = tk.DoubleVar(value=3.0)
        self.p_dao_sigma      = tk.DoubleVar(value=5.0)
        self.p_use_ml         = tk.BooleanVar(value=False)
        self._catalog_busy    = False
        self._analyze_busy    = False
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
        self.btn_alerce = mkbtn("ALeRCE Lookup", self.open_alerce_window, "#1abc9c", state=tk.DISABLED)

        # ── Colormap selector ──
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

        # ── ML Filter toggle (always visible) ──
        tk.Label(bf, text="  ML Filter:", font=("Arial", 10),
                 bg="#1a1a2e", fg="#bdc3c7").pack(side=tk.LEFT, padx=(12, 2))
        tk.Checkbutton(bf, variable=self.p_use_ml,
                       bg="#1a1a2e", activebackground="#1a1a2e",
                       fg="#ecf0f1", selectcolor="#2c3e50",
                       cursor="hand2").pack(side=tk.LEFT, padx=2)

        if pretrained_model is not None:
            _pt_acc = pretrained_model.get("accuracy", 0)
            _status_init = f"Deep-TAO ready (acc={_pt_acc:.0f}%)  |  Select images"
            _status_col  = "#2ecc71"
        else:
            _status_init = "Select images"
            _status_col  = "#95a5a6"
        cf = tk.Frame(self.root, bg="#0f0f23", relief=tk.SUNKEN, bd=2)
        cf.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 6))
        self.canvas_frame = tk.Frame(cf, bg="#0f0f23")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.stats_text = tk.Text(self.root, height=5, font=("Courier", 9),
                                  bg="#0d0d1a", fg="#00ff88",
                                  state=tk.DISABLED, relief=tk.FLAT)
        self.stats_text.pack(fill=tk.X, padx=18, pady=(0, 0))

        sb = tk.Frame(self.root, bg="#1a1a2e", height=28)
        sb.pack(fill=tk.X, padx=0, pady=(0, 4))
        sb.pack_propagate(False)
        self.status_label = tk.Label(sb, text=_status_init,
                                     font=("Arial", 10), bg="#1a1a2e", fg=_status_col)
        self.status_label.pack(side=tk.LEFT, padx=14, pady=4)

    def open_settings(self):
        from gui.settings_window import open_settings
        open_settings(self)

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

        # Try FITS for files without extension (e.g. "dss_search")
        if not is_fits and ext == "" and FITS_AVAILABLE:
            try:
                from astropy.io import fits as _afits
                with _afits.open(path) as _h:
                    _h.verify('fix')
                is_fits = True   # FITS magic verified
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
            # Show header info as a short summary
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
        fname = os.path.basename(path)
        fname = fname if len(fname) <= 22 else fname[:19] + "..."
        self.status_label.config(
            text=f"Image {num}: {fname}  [{tip}]", fg="#2ecc71")
        self.btn_catalog.config(state=tk.NORMAL)   # works with a single image
        if self.img1 is not None and self.img2 is not None:
            self.btn_compare.config(state=tk.NORMAL)
            self.status_label.config(text="Ready — click Analyze", fg="#3498db")

    def _compare_set_status(self, text, color="#3498db"):
        """Thread-safe status (UI does not freeze while Analyze runs in the background)."""
        self.root.after(0, lambda: self.status_label.config(text=text, fg=color))

    def _snapshot_analysis_params(self):
        """Read Tk variables in the main thread (worker thread safety)."""
        return {
            "mode": self.mode.get(),
            "use_dao": self.p_use_dao.get(),
            "dao_fwhm": self.p_dao_fwhm.get(),
            "dao_sigma": self.p_dao_sigma.get(),
            "min_area": self.p_min_area.get(),
            "max_area": self.p_max_area.get(),
            "min_peak": self.p_min_peak.get(),
            "coarse_r": self.p_coarse_r.get(),
            "thr_pct": self.p_thr_pct.get(),
            "match_r": self.p_match_r.get(),
            "iqr_mult": self.p_iqr_mult.get(),
            "min_res": self.p_min_res.get(),
            "max_mov_frac": self.p_max_mov_frac.get(),
            "r_inner": self.p_r_inner.get(),
            "r_outer": self.p_r_outer.get(),
            "min_snr": self.p_min_snr.get(),
            "use_ml": self.p_use_ml.get(),
            "flux_ratio_thr": self.p_flux_ratio_thr.get(),
            "min_flux": self.p_min_flux.get(),
        }

    def compare_images(self):
        if self.img1 is None or self.img2 is None:
            return
        if self._analyze_busy:
            return
        self._analysis_params = self._snapshot_analysis_params()
        self._analyze_busy = True
        self.btn_compare.config(state=tk.DISABLED)
        self._compare_set_status("Analyze starting (in background)...", "#3498db")
        threading.Thread(target=self._compare_worker, daemon=True).start()

    def _compare_worker(self):
        try:
            result = self._compare_compute()
            self.root.after(0, lambda: self._compare_apply(result))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=e: self._compare_on_error(err))

    def _compare_apply(self, result):
        self._analyze_busy = False
        self.btn_compare.config(state=tk.NORMAL)
        if result is None:
            return
        for key, val in result.get("attrs", {}).items():
            setattr(self, key, val)
        self._draw(**result["draw"])
        self._update_stats(**result["stats"])
        self.btn_save.config(state=tk.NORMAL)
        self.btn_label.config(state=tk.NORMAL)
        # Enable ALeRCE lookup only when WCS is truly usable (CRVAL keywords present)
        _hdr2 = getattr(self, 'fits_header2', None)
        _has_wcs = (_hdr2 is not None
                    and 'CRVAL1' in _hdr2 and 'CRVAL2' in _hdr2
                    and 'CRPIX1' in _hdr2 and 'CRPIX2' in _hdr2)
        _has_det = bool(self.last_brightened or self.last_dimmed)
        if _has_wcs and _has_det:
            self.btn_alerce.config(state=tk.NORMAL)
        self.status_label.config(text=result["status_text"], fg="#2ecc71")

    def _compare_on_error(self, err):
        self._analyze_busy = False
        self.btn_compare.config(state=tk.NORMAL)
        messagebox.showerror("Error", str(err))
        self.status_label.config(text="Failed.", fg="#e74c3c")

    def _compare_compute(self):
        def status(msg, color="#3498db"):
            self._compare_set_status(msg, color)

        p = self._analysis_params
        h, w = self.img1.shape[:2]
        img2_rs = cv.resize(self.img2, (w, h))
        # Shared scale: same percentile threshold in both frames (F1/F2 count balance)
        img1_n, img2r_n = normalize_pair_joint(self.img1, img2_rs)

        # If FITS, use the original float data for flux measurement
        flux1 = self.fits_raw1 if self.fits_raw1 is not None else img1_n.astype(np.float32)
        flux2 = self.fits_raw2 if self.fits_raw2 is not None else img2r_n.astype(np.float32)

        # DAO or percentile?
        _use_dao = (p["use_dao"] and DAO_AVAILABLE
                    and self.fits_raw1 is not None
                    and self.fits_raw2 is not None)

        if _use_dao:
            # Pre-compute DAO stars for alignment
            status("Detecting stars (DAO)...")
            _dao_fwhm  = p["dao_fwhm"]
            _dao_sigma = p["dao_sigma"]
            dao1_for_align = detect_stars_dao(
                flux1, fwhm=_dao_fwhm, sigma_thresh=_dao_sigma,
                min_area=p["min_area"], max_area=p["max_area"])
            flux2_rs = cv.resize(flux2, (w, h)) if flux2.shape[:2] != (h, w) else flux2

            # ── WCS pre-shift (pre-align epoch2 if CRVAL differs) ──
            _hdr1 = getattr(self, 'fits_header1', None)
            _hdr2 = getattr(self, 'fits_header2', None)
            _wcs_dx, _wcs_dy = 0.0, 0.0
            _wcs_H = None
            if _hdr1 is not None and _hdr2 is not None:
                try:
                    _wcs_dx, _wcs_dy = _wcs_pixel_offset(_hdr1, _hdr2)
                except Exception:
                    _wcs_dx, _wcs_dy = 0.0, 0.0
            # Offset > coarse_radius/2 means a significant WCS shift is present
            _wcs_total = (_wcs_dx**2 + _wcs_dy**2) ** 0.5
            if _wcs_total > p["coarse_r"] / 2.0:
                status("WCS pre-aligning epoch2...")
                _sx, _sy = -_wcs_dx, -_wcs_dy   # shift to apply
                _M_wcs = np.float32([[1, 0, _sx], [0, 1, _sy]])
                flux2_wcs = cv.warpAffine(
                    flux2_rs.astype(np.float32), _M_wcs, (w, h),
                    flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
                img2_wcs_n = sqrt_stretch(flux2_wcs)
                dao2_for_align = detect_stars_dao(
                    flux2_wcs, fwhm=_dao_fwhm, sigma_thresh=_dao_sigma,
                    min_area=p["min_area"], max_area=p["max_area"])
                # H_wcs to compose with RANSAC result
                _wcs_H = np.float32([[1, 0, _sx], [0, 1, _sy]])
                _ref_img2 = img2_wcs_n
                _ref_flux2_rs = flux2_wcs
            else:
                dao2_for_align = detect_stars_dao(
                    flux2_rs, fwhm=_dao_fwhm, sigma_thresh=_dao_sigma,
                    min_area=p["min_area"], max_area=p["max_area"])
                _ref_img2 = img2r_n
                _ref_flux2_rs = flux2_rs

            status("Aligning (DAO stars)...")
            img2_aligned, H_resid, align_method = star_affine_align(
                img1_n, _ref_img2, pct=p["thr_pct"],
                coarse_radius=p["coarse_r"],
                stars1=dao1_for_align if len(dao1_for_align) >= 4 else None,
                stars2=dao2_for_align if len(dao2_for_align) >= 4 else None)

            # Compose: H = H_resid ∘ H_wcs (first WCS shift, then RANSAC)
            if H_resid is not None and _wcs_H is not None:
                H_wcs_3x3 = np.array([[1, 0, _wcs_H[0, 2]],
                                       [0, 1, _wcs_H[1, 2]],
                                       [0, 0, 1]], dtype=np.float64)
                H_res_3x3 = np.vstack([H_resid.astype(np.float64), [0, 0, 1]])
                H = (H_res_3x3 @ H_wcs_3x3)[:2, :].astype(np.float32)
            else:
                H = H_resid
        else:
            status("Aligning...")
            img2_aligned, H, align_method = star_affine_align(
                img1_n, img2r_n, pct=p["thr_pct"],
                coarse_radius=p["coarse_r"])

        status("Detecting stars...")
        _border = 10
        if _use_dao:
            # DAO on the aligned flux2 — each frame computes its own threshold
            # If WCS pre-shift was applied, _ref_flux2_rs is the shifted image;
            # H already contains the WCS+RANSAC composition (applied directly to flux2_rs)
            flux2_aligned_for_dao = cv.warpAffine(
                flux2_rs, H, (w, h),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT) if H is not None else flux2_rs
            stars1_raw = detect_stars_dao(
                flux1, fwhm=_dao_fwhm, sigma_thresh=_dao_sigma,
                min_area=p["min_area"], max_area=p["max_area"])
            stars2_raw = detect_stars_dao(
                flux2_aligned_for_dao, fwhm=_dao_fwhm, sigma_thresh=_dao_sigma,
                min_area=p["min_area"], max_area=p["max_area"])
        else:
            kw = dict(pct=p["thr_pct"], min_area=p["min_area"],
                      max_area=p["max_area"], min_peak=p["min_peak"])
            stars1_raw = detect_stars(img1_n, **kw)
            stars2_raw = detect_stars(img2_aligned, **kw)

        stars1_raw = filter_stars_from_border(stars1_raw, w, h, _border)
        stars2_raw = filter_stars_from_border(stars2_raw, w, h, _border)

        # Depth balancing in DAO mode (both directions):
        # Trim the longer list from either epoch: cap max ratio at min×3.
        # - epoch1 >> epoch2 (SN: perfect vs bad seeing): epoch1 is trimmed,
        #   undetectable faint sources are removed from list → fewer false moving.
        # - epoch2 >> epoch1 (moving object: bright vs deep observation):
        #   ratio=2.0 ≤ 3.0 → no trimming, all epoch1 stars preserved.
        if _use_dao:
            stars1_raw, stars2_raw = match_detection_depth(
                stars1_raw, stars2_raw, max_ratio=3.0)

        status("Calculating SNR...")
        ri, ro = p["r_inner"], p["r_outer"]
        min_snr = p["min_snr"]
        stars1, snr_all1, snr_kept1 = filter_stars_by_snr(
            stars1_raw, img1_n, min_snr, ri, ro)
        stars2, snr_all2, snr_kept2 = filter_stars_by_snr(
            stars2_raw, img2_aligned, min_snr, ri, ro)

        diff = cv.absdiff(img1_n, img2_aligned)

        status("Matching stars...")
        matched, unm1, unm2 = match_stars_mutual(
            stars1, stars2, max_distance=p["match_r"])

        # Alignment quality: inlier count + match rate
        import re as _re
        _inlier_m = _re.search(r'(\d+) inliers', align_method)
        _n_inliers = int(_inlier_m.group(1)) if _inlier_m else 0
        _match_rate = len(matched) / max(min(len(stars1), len(stars2)), 1)
        _poor_alignment = _n_inliers < 6   # match_rate low is ok if n_inliers is high (depth mismatch)

        status("Detecting motion...")
        moving_idx, med_dx, med_dy, thr_r, residuals = detect_moving_outliers(
            matched, iqr_multiplier=p["iqr_mult"],
            min_residual_px=p["min_res"],
            max_moving_frac=p["max_mov_frac"])


        status("Measuring brightness...")
        # If FITS available use raw float data, otherwise normalized uint8
        flux2_aligned = cv.warpAffine(flux2, H, (w, h),
            flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT) if H is not None else flux2

        # ML filter: always use the threshold set by the user.
        # Previously it dropped to 1.3x when ML was on — this caused a false positive explosion.
        # Now the threshold is always taken from the settings value;
        # the ML filter only eliminates non-transients as post-processing.
        use_ml_filter = (pretrained_model is not None and ML_AVAILABLE
                         and p["use_ml"])
        detection_thr = p["flux_ratio_thr"]

        bright_result = detect_brightness_outliers(
            matched, stars1, stars2, flux1, flux2_aligned,
            r_inner=p["r_inner"], r_outer=p["r_outer"],
            flux_ratio_thr=detection_thr,
            min_flux=p["min_flux"])
        brightened, dimmed, stationary, changes_arr = bright_result

        # ── Unmatched stars: nova (new in epoch2) and disappeared (gone in epoch2) ──
        # If alignment is poor (few inliers or low match rate), unmatched detection
        # becomes unreliable — all unmatched stars produce spurious changes. Skip.
        _ri2 = p["r_inner"]
        _ro2 = p["r_outer"]
        _mf2 = p["min_flux"]
        _unm_idx = len(matched)   # sentinel idx: out-of-range → ML skips safely

        # Use a higher threshold for unmatched stars.
        # A real nova/transient is very bright (>>10x); threshold shift artifacts are 2-4x.
        # Keeping min_flux higher also filters out noise.
        _unm_thr  = max(detection_thr * 4.0, 10.0)   # at least 10x brightening required
        _unm_flux = _mf2 * 8                          # at least 8x min_flux brightness

        if not _poor_alignment:
            _unm2_frac = len(unm2) / max(len(stars2), 1)
            _unm1_frac = len(unm1) / max(len(stars1), 1)

            # Depth mismatch: many unmatched but alignment is geometrically solid.
            # Use lower flux floor so faint-in-ZTF-units transients are not missed.
            # Ratio threshold stays high (>=10x) to suppress noise false-positives.
            _depth_mismatch = _n_inliers >= 20 and (_unm2_frac >= 0.20 or _unm1_frac >= 0.20)
            _eff_unm_flux   = _mf2 * 0.5 if _depth_mismatch else _unm_flux

            # Stars only in epoch2 → brightened (nova / new transient)
            if _unm2_frac < 0.20 or _depth_mismatch:
                for j in unm2:
                    cx2, cy2 = stars2[j][0], stars2[j][1]
                    b2 = bg_subtracted_flux(flux2_aligned, cx2, cy2, _ri2, _ro2)
                    b1 = bg_subtracted_flux(flux1,         cx2, cy2, _ri2, _ro2)
                    if b2 > _eff_unm_flux and b2 / (b1 + 1.0) >= _unm_thr:
                        ch = (b2 / max(b1, 1.0) - 1.0) * 100.0
                        brightened.append({
                            "i": -1, "j": j, "idx": _unm_idx,
                            "x1": cx2, "y1": cy2, "x2": cx2, "y2": cy2,
                            "b1": b1,  "b2": b2,  "change_pct": ch,
                            "peak1": 0, "peak2": stars2[j][3],
                            "area1": 1, "area2": stars2[j][2],
                            "ml_score": 0.0,
                        })

            # Stars only in epoch1 → dimmed (disappeared / faded out)
            if _unm1_frac < 0.20 or _depth_mismatch:
                for i in unm1:
                    cx1, cy1 = stars1[i][0], stars1[i][1]
                    b1 = bg_subtracted_flux(flux1,         cx1, cy1, _ri2, _ro2)
                    b2 = bg_subtracted_flux(flux2_aligned, cx1, cy1, _ri2, _ro2)
                    if b1 > _eff_unm_flux and b2 / (b1 + 1e-9) <= 1.0 / _unm_thr:
                        ch = (b2 / (b1 + 1e-9) - 1.0) * 100.0
                        dimmed.append({
                            "i": i, "j": -1, "idx": _unm_idx,
                            "x1": cx1, "y1": cy1, "x2": cx1, "y2": cy1,
                            "b1": b1,  "b2": b2,  "change_pct": ch,
                            "peak1": stars1[i][3], "peak2": 0,
                            "area1": stars1[i][2], "area2": 1,
                            "ml_score": 0.0,
                        })

        # Remove moving stars from brightness lists
        moving_set = set(moving_idx)
        matched_moving = {matched[k][0] for k in moving_set}
        brightened = [r for r in brightened if r["i"] not in matched_moving]
        dimmed     = [r for r in dimmed     if r["i"] not in matched_moving]
        stationary_b = [r for r in stationary if r["i"] not in matched_moving]

        # ── Photometry calibration (with APASS if WCS available) ──
        # Full field: APASS/Vizier can take minutes with tens of thousands of stars
        if len(stars1) > 800:
            status("Photometry calibration (bright subset)...")
        else:
            status("Photometry calibration...")
        hdr1 = getattr(self, 'fits_header1', None)
        zp1, n_cal1 = _try_photometry_zp(hdr1, stars1, flux1,
                                          r_inner=p["r_inner"],
                                          r_outer=p["r_outer"],
                                          max_stars=400)
        if zp1 is not None:
            for r in brightened + dimmed + stationary_b:
                r['v1']     = float(-2.5 * np.log10(max(r['b1'], 1.0)) + zp1)
                r['v2']     = float(-2.5 * np.log10(max(r['b2'], 1.0)) + zp1)
                r['delta_v'] = r['v2'] - r['v1']

        # ── ML: Isolation Forest anomaly score ──
        status("ML analysis...")
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

        # ── Unmatched-pair fallback: moving objects where mutual-NN failed ──
        # When a background star falls near the asteroid's epoch2 position, the asteroid
        # drops into unm1; this pass matches it against unm2.
        #
        # Guard: If Unmatched1 is large (>10% or >20 stars) it most likely
        # comes from a quality difference (bad seeing/airmass), not a moving
        # object. Skip in that case to avoid false positives.
        _unm_frac = len(unm1) / max(len(stars1), 1)
        if len(unm1) <= 20 and _unm_frac < 0.10:
            _unm_moving = detect_moving_from_unmatched(
                stars1, stars2, unm1, unm2,
                med_dx=med_dx, med_dy=med_dy,
                min_motion=p["min_res"],
                max_motion=min(w, h) * 0.3)
            moving.extend(_unm_moving)

        # Also add ML scores to brightness lists
        for r in brightened + dimmed + stationary_b:
            r["ml_score"] = ml_scores.get(r["idx"], 0.0)

        # ── Supervised classifier predictions (user-labeled in-app model) ──
        if classifier.is_trained and ML_AVAILABLE:
            for r in brightened + dimmed:
                try:
                    match_idx = r.get("idx", -1)
                    if match_idx >= 0 and match_idx < len(matched) and stars1 and stars2:
                        i, j = matched[match_idx][0], matched[match_idx][1]
                        cx_e, cy_e = stars1[i][0], stars1[i][1]
                        cx_l, cy_l = stars2[j][0], stars2[j][1]
                    else:
                        cx_e = cx_l = int(r.get("x2", r.get("x1", 0)))
                        cy_e = cy_l = int(r.get("y2", r.get("y1", 0)))
                    fv = extract_deeptao_features(flux1, cx_e, cy_e,
                                                  flux2_aligned, cx_l, cy_l)
                    fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
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
            # Use the saved Youden-J optimal threshold (default: 0.5)
            _pt_thr    = pretrained_model.get("threshold", 0.50)

            # IMPORTANT: Feed the model with raw FITS float data (not 0-255 normalized)
            # because training was done on raw FITS data.
            # If FITS is not loaded, normalized image is used (fallback).
            _img_early = flux1          # raw float (or normalized fallback)
            _img_late  = flux2_aligned  # raw float, aligned

            for r in brightened + dimmed:
                try:
                    # img_early=early, img_late=late → pass change to model
                    fv = extract_deeptao_features(
                        _img_early, r["x1"], r["y1"],
                        _img_late,  r["x2"], r["y2"])
                    fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
                    # If feature size mismatch (old model), zero out
                    if len(fv) != _n_feat:
                        fv = np.zeros(_n_feat)
                    fv_s       = _scaler_pt.transform(fv.reshape(1, -1))
                    # Get transient probability via predict_proba, compare with saved threshold
                    prob_trans = float(_clf_pt.predict_proba(fv_s)[0][1])
                    lbl        = int(prob_trans >= _pt_thr)
                    r["pt_label"] = lbl
                    r["pt_prob"]  = prob_trans
                except Exception:
                    r["pt_label"] = 0
                    r["pt_prob"]  = 0.0

        # ── ML Filter: Remove non-transients ──
        # If user model is trained, it takes priority; otherwise pretrained Deep-TAO.
        # label = 1 → transient → keep  |  label = 0 → non-transient → remove
        if use_ml_filter:
            status("ML filtering...")
            n_before = len(brightened) + len(dimmed)
            if classifier.is_trained:
                brightened = [r for r in brightened
                              if r.get("ml_label", 1) == 1]
                dimmed     = [r for r in dimmed
                              if r.get("ml_label", 1) == 1]
            else:
                brightened = [r for r in brightened
                              if r.get("pt_label", 1) == 1]
                dimmed     = [r for r in dimmed
                              if r.get("pt_label", 1) == 1]
            n_after    = len(brightened) + len(dimmed)
            n_filtered = n_before - n_after

        draw = dict(
            img1_n=img1_n, img2_n=img2_aligned, stars1=stars1, stars2=stars2,
            moving=moving, stationary=stationary_b, brightened=brightened,
            dimmed=dimmed, unm1=unm1, unm2=unm2, matched=matched,
            med_dx=med_dx, med_dy=med_dy, thr_r=thr_r, residuals=residuals,
            changes_arr=changes_arr, mode=p["mode"],
        )
        stats = dict(
            method=align_method, n1=len(stars1), n2=len(stars2),
            n_match=len(matched), moving=moving, brightened=brightened,
            dimmed=dimmed, n_unm1=len(unm1), n_unm2=len(unm2),
            med_dx=med_dx, med_dy=med_dy, thr_r=thr_r, residuals=residuals,
            snr_kept1=snr_kept1, snr_kept2=snr_kept2,
            poor_alignment=_poor_alignment,
            n_inliers=_n_inliers, match_rate=_match_rate,
        )
        attrs = dict(
            img1_n=img1_n, img2_n=img2_aligned, diff=diff,
            phot_zp1=zp1, phot_n_cal1=n_cal1,
            ml_scores=ml_scores,
            last_matched=matched, last_stars1=stars1, last_stars2=stars2,
            last_brightened=brightened, last_dimmed=dimmed,
            _last_moving=moving, _last_stationary=stationary_b,
            _last_unm1=unm1, _last_unm2=unm2,
            _last_med_dx=med_dx, _last_med_dy=med_dy,
            _last_thr_r=thr_r, _last_residuals=residuals,
            _last_changes_arr=changes_arr,
        )
        status_text = (
            f"Done  |  {len(moving)} moving  |  "
            f"{len(brightened)} brightened  {len(dimmed)} dimmed  |  "
            f"Matched={len(matched)}  |  {align_method}"
        )
        return {"draw": draw, "stats": stats, "attrs": attrs, "status_text": status_text}

    @staticmethod
    def _stars_for_display(stars, max_n=800):
        if len(stars) <= max_n:
            return stars
        step = max(1, len(stars) // max_n)
        return stars[::step]

    @staticmethod
    def _stretch_pair_for_display(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Both panels share the same brightness scale — F2 should not appear washed-out or fully black."""
        stack = np.concatenate([img1.ravel(), img2.ravel()])
        lo, hi = np.percentile(stack, [1.0, 99.5])
        span = max(float(hi - lo), 1.0)

        def _sc(im: np.ndarray) -> np.ndarray:
            out = (im.astype(np.float32) - lo) / span * 255.0
            return np.clip(out, 0, 255).astype(np.uint8)

        return _sc(img1), _sc(img2)

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

        disp1 = self._stars_for_display(stars1)
        disp2 = self._stars_for_display(stars2)
        show1, show2 = self._stretch_pair_for_display(img1_n, img2_n)

        # ── Frame 1
        axes[ax_i].imshow(show1, cmap=self.cmap.get(), vmin=0, vmax=255)
        axes[ax_i].set_title(f"Frame 1  ({len(stars1)} stars)", color="white", fontsize=10, fontweight="bold")
        for s in disp1:
            axes[ax_i].add_patch(plt.Circle((s[0],s[1]), min(18, max(3, np.sqrt(s[2]) * 0.5)),
                                             color="cyan", fill=False, lw=0.6, alpha=0.45))
        ax_i += 1

        # ── Frame 2
        axes[ax_i].imshow(show2, cmap=self.cmap.get(), vmin=0, vmax=255)
        axes[ax_i].set_title(f"Frame 2 (aligned)  ({len(stars2)} stars)", color="white", fontsize=10, fontweight="bold")
        for s in disp2:
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
                # Area change note (nova/transient expansion)
                a1, a2 = r.get("area1", 0), r.get("area2", 0)
                area_str = (f" [area:{a1}→{a2}px²]"
                            if a2 > 0 and a2 / max(a1, 1) >= 10 else "")
                if "pt_label" in r:
                    lbl = "Transient" if r["pt_label"] == 1 else "Non-transient"
                    pct = int(r.get("pt_prob", 0.0) * 100)
                    return f"{change_prefix}{r['change_pct']:.0f}%{mag_str}{area_str} | {lbl} {pct}%"
                elif classifier.is_trained and "ml_label" in r:
                    lbl = "Transient" if r["ml_label"] == 1 else "Artifact"
                    pct = int(r.get("ml_prob", 0.0) * 100)
                    return f"{change_prefix}{r['change_pct']:.0f}%{mag_str}{area_str} | {lbl} {pct}%"
                else:
                    return f"{change_prefix}{r['change_pct']:.0f}%{mag_str}{area_str}  ML:{ml:.2f}"

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
                   mpatches.Patch(color="#ff6b35",  label="Brightened"),
                   mpatches.Patch(color="#74b9ff",  label="Dimmed"),
                   mpatches.Patch(color="white",    label="ML score")]
        else:
            leg_b = []

        # ── Single figure-level legend — bottom right of screen
        all_handles = []
        if mode in ("moving", "both"):
            all_handles += leg_m
        if mode in ("brightness", "both"):
            all_handles += leg_b
        if all_handles:
            fig.legend(handles=all_handles,
                       loc="lower right",
                       bbox_to_anchor=(0.99, 0.01),
                       ncol=1, fontsize=7,
                       framealpha=0.88, facecolor="#1a1a2e",
                       labelcolor="white", edgecolor="#555")

        plt.tight_layout(pad=0.4)
        canvas = FigureCanvasTkAgg(fig, self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig = fig

    def _update_stats(self, method, n1, n2, n_match, moving, brightened, dimmed,
                      n_unm1, n_unm2, med_dx, med_dy, thr_r, residuals,
                      snr_kept1=None, snr_kept2=None,
                      poor_alignment=False, n_inliers=0, match_rate=0.0):
        # SNR statistics
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
        mov_lines = "".join(
            f"    [Moving #{k+1}]  ({m['x1']:.0f},{m['y1']:.0f})->({m['x2']:.0f},{m['y2']:.0f})"
            f"  |res|={m['residual']:.3f}px  dx={m['dx']:+.2f} dy={m['dy']:+.2f}\n"
            for k,m in enumerate(moving))
        def _mag_str(r):
            if 'v1' in r:
                return f"  V:{r['v1']:.2f}->{r['v2']:.2f} (dV={r['delta_v']:+.2f})"
            return ""

        # WCS: pixel -> RA/Dec and distance to known transients
        import warnings as _warnings
        _wcs_obj = None
        try:
            from astropy.wcs import WCS as _WCS
            _hdr2 = getattr(self, 'fits_header2', None)
            if _hdr2 is not None:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    _wcs_obj = _WCS(_hdr2, naxis=2)
        except Exception:
            pass

        def _wcs_str(cx, cy):
            if _wcs_obj is None:
                return ""
            try:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    ra, dec = _wcs_obj.all_pix2world(float(cx), float(cy), 0)
                ra, dec = float(ra), float(dec)
                parts = [f"  RA={ra:.4f} Dec={dec:.4f}"]
                # Compare against known transients
                known = [
                    ("SN2023ixf", 210.9109, 54.3116),
                ]
                for name, kra, kdec in known:
                    cos_d = np.cos(np.radians((dec + kdec) / 2))
                    sep = np.sqrt(((ra - kra) * cos_d)**2 + (dec - kdec)**2) * 3600.0
                    if sep < 120:
                        parts.append(f"  [{name}: {sep:.1f}\"]")
                return "".join(parts)
            except Exception:
                return ""

        # Print WCS validation table to terminal
        if _wcs_obj is not None and (brightened or dimmed):
            print("\n--- Brightness Detections (WCS) ---")
            for k, r in enumerate(brightened):
                ws = _wcs_str(r['x2'], r['y2'])
                print(f"  Bright #{k+1:2d}  ({r['x2']:6.0f},{r['y2']:6.0f})"
                      f"  {r['change_pct']:+10.1f}%{ws}")
            for k, r in enumerate(dimmed):
                ws = _wcs_str(r['x1'], r['y1'])
                print(f"  Dim   #{k+1:2d}  ({r['x1']:6.0f},{r['y1']:6.0f})"
                      f"  {r['change_pct']:+10.1f}%{ws}")
            print("---------------------------------------\n")

        # Bright/dim summary lines in the stats box (scrollable)
        bright_lines = "".join(
            f"    [Bright #{k+1}]  ({r['x1']:.0f},{r['y1']:.0f})"
            f"  flux:{r['b1']:.0f}->{r['b2']:.0f}  {r['change_pct']:+.1f}%"
            + _mag_str(r) + _wcs_str(r['x2'], r['y2']) +
            f"  ML:{r.get('ml_score',0):.2f}\n"
            for k,r in enumerate(brightened[:5]))
        bright_lines += "".join(
            f"    [Dim   #{k+1}]  ({r['x1']:.0f},{r['y1']:.0f})"
            f"  flux:{r['b1']:.0f}->{r['b2']:.0f}  {r['change_pct']:+.1f}%"
            + _mag_str(r) + _wcs_str(r['x1'], r['y1']) +
            f"  ML:{r.get('ml_score',0):.2f}\n"
            for k,r in enumerate(dimmed[:5]))
        zp1     = getattr(self, 'phot_zp1', None)
        n_cal1  = getattr(self, 'phot_n_cal1', 0)
        phot_info = (f"  Photometry: ZP={zp1:.3f}  ({n_cal1} cal. stars, APASS V-band)\n"
                     if zp1 is not None
                     else "  Photometry: APASS calibration unavailable — magnitude not computed\n")
        ml_filter_info = ""
        if pretrained_model is not None:
            ml_filter_info = "  ML Detection: Deep-TAO filter active (threshold=1.3x)\n"
        align_warn = ""
        if poor_alignment:
            align_warn = (f"  !! POOR ALIGNMENT: {n_inliers} inliers — skipping unmatched brightness\n")
        elif match_rate < 0.25 and n_inliers >= 20:
            align_warn = (f"  ~~ Depth mismatch: {n_inliers} inliers, {match_rate*100:.0f}% matches "
                          f"— checking unmatched with lower threshold\n")
        line = (
            f"  Alignment: {method}\n"
            f"  Stars: F1={n1}  F2={n2}  Matched={n_match}  "
            f"Unmatched1={n_unm1}  Unmatched2={n_unm2}  drift=({med_dx:+.3f},{med_dy:+.3f})px\n"
            + align_warn + snr_info + res_info + phot_info + ml_filter_info
            + f"  Moving={len(moving)}  Brightened={len(brightened)}  Dimmed={len(dimmed)}\n"
            + mov_lines + bright_lines
        ).rstrip()
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, line)
        self.stats_text.config(state=tk.DISABLED)

    # ── Star catalog ──────────────────────────────────────────────────────
    def analyze_catalog(self):
        """Star catalog for a single image (in background — UI does not freeze)."""
        from gui.catalog_window import analyze_catalog
        analyze_catalog(self)

    # ── Label & train ─────────────────────────────────────────────────────
    def open_label_window(self):
        """Label brightness changes one by one — card system."""
        from gui.label_window import open_label_window
        open_label_window(self)

    def train_model(self):
        """Train the Random Forest on all labeled samples."""
        from gui.label_window import train_model
        train_model(self)

    # ── ALeRCE cross-match ─────────────────────────────────────────────────
    def open_alerce_window(self):
        """Opens ALeRCE ZTF broker lookup window for brightness detections."""
        from gui.alerce_window import open_alerce_window
        open_alerce_window(self)

    # ── Save result ─────────────────────────────────────────────────────
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
    if not TK_AVAILABLE:
        raise RuntimeError(
            "tkinter is not available. Install the OS package for tkinter "
            "(e.g. python3-tk on Ubuntu/Debian) or run on a system with Tk support."
        )
    root = tk.Tk()
    ImageComparisonApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()