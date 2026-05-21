"""
validate_at2020xnd.py
=====================
Validates the detection of AT2020xnd (Type Ibn, mag=16.5) using WCS.

Usage:  python validate_at2020xnd.py
"""

import sys
import re
import warnings
import numpy as np
import cv2 as cv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from fits_loader import load_fits_image, FITS_AVAILABLE
from detection import (
    detect_stars_dao, match_detection_depth, star_affine_align,
    match_stars_mutual, bg_subtracted_flux, detect_brightness_outliers,
    filter_stars_by_snr,
)

if not FITS_AVAILABLE:
    print("astropy is not installed -- pip install astropy")
    sys.exit(1)

SN_RA        = 8.050
SN_DEC       = 36.760
MATCH_ARCSEC = 30.0
MIN_FLUX     = 25.0   # calibrated ZTF units (lower than synthetic data)
UNM_THR      = 10.0   # 10x ratio for unmatched sources

PRE  = Path(__file__).parent / "sn_target" / "AT2020xnd" / "pre_sn.fits"
POST = Path(__file__).parent / "sn_target" / "AT2020xnd" / "post_sn.fits"

print("=" * 65)
print("  AT2020xnd Validation  (Type Ibn, mag=16.5)")
print(f"  Target: RA={SN_RA}  Dec={SN_DEC}")
print("=" * 65)

print("\n[1] Loading FITS files...")
img1_u8, info1, flux1, hdr1 = load_fits_image(str(PRE))
img2_u8, info2, flux2, hdr2 = load_fits_image(str(POST))
h, w = flux1.shape[:2]
print(f"  pre_sn  : {w}x{h}")
print(f"  post_sn : {w}x{h}")

print("\n[2] Computing SN pixel position using WCS...")
WCS_OK = False
sn_px, sn_py = -1.0, -1.0
wcs = None
try:
    from astropy.wcs import WCS as AstropyWCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = AstropyWCS(hdr2, naxis=2)
    sn_px, sn_py = wcs.all_world2pix(SN_RA, SN_DEC, 0)
    WCS_OK = True
    sn_px, sn_py = float(sn_px), float(sn_py)
    in_frame = (0 <= sn_px < w) and (0 <= sn_py < h)
    print(f"  AT2020xnd pixel (post_sn): ({sn_px:.1f}, {sn_py:.1f})  in_frame={in_frame}")
except Exception as e:
    print(f"  WCS error: {type(e).__name__}")

print("\n[3] Aligning images...")
img1_n = img1_u8.astype(np.float32) / 255.0
img2_n = img2_u8.astype(np.float32) / 255.0

dao_pre  = detect_stars_dao(flux1, fwhm=3.0, sigma_thresh=5.0)
dao_post = detect_stars_dao(flux2, fwhm=3.0, sigma_thresh=5.0)
print(f"  DAO: pre={len(dao_pre)}  post={len(dao_post)}")

img2_aligned_n, H, method = star_affine_align(
    img1_n, img2_n,
    stars1=dao_pre  if len(dao_pre)  >= 4 else None,
    stars2=dao_post if len(dao_post) >= 4 else None,
)
print(f"  {method}")

inlier_m  = re.search(r'(\d+) inliers', method)
n_inliers = int(inlier_m.group(1)) if inlier_m else 0

flux2_aligned = cv.warpAffine(
    flux2, H, (w, h), flags=cv.INTER_LINEAR,
    borderMode=cv.BORDER_CONSTANT) if H is not None else flux2

print("\n[4] Detecting stars (sigma=5, SNR>=3.0)...")
stars1_raw = detect_stars_dao(flux1,           fwhm=3.0, sigma_thresh=5.0)
stars2_raw = detect_stars_dao(flux2_aligned,   fwhm=3.0, sigma_thresh=5.0)
stars1_raw, stars2_raw = match_detection_depth(stars1_raw, stars2_raw, max_ratio=3.0)
stars1, snr1, _ = filter_stars_by_snr(stars1_raw, img1_n,         min_snr=3.0, r_inner=5, r_outer=15)
stars2, snr2, _ = filter_stars_by_snr(stars2_raw, img2_aligned_n, min_snr=3.0, r_inner=5, r_outer=15)
print(f"  F1={len(stars1)}  F2={len(stars2)}")

print("\n[5] Matching stars...")
matched, unm1, unm2 = match_stars_mutual(stars1, stars2, max_distance=5.0)
match_rate = len(matched) / max(min(len(stars1), len(stars2)), 1)
unm2_frac  = len(unm2) / max(len(stars2), 1)
print(f"  Matched={len(matched)}  Unm2={len(unm2)} ({unm2_frac*100:.0f}%)")
print(f"  n_inliers={n_inliers}  match_rate={match_rate:.3f}")

poor_align    = n_inliers < 6
depth_mismatch = n_inliers >= 20 and unm2_frac >= 0.20
if poor_align:
    print("  !! POOR ALIGNMENT: skipping unmatched detections")
elif depth_mismatch:
    print(f"  ~~ Depth mismatch: using low flux threshold ({MIN_FLUX})")

print("\n[6] Direct flux measurement at SN position using WCS...")
if WCS_OK:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs_pre = AstropyWCS(hdr1, naxis=2)
        sn_px_pre, sn_py_pre = [float(v) for v in wcs_pre.all_world2pix(SN_RA, SN_DEC, 0)]
        print(f"  SN pre  pixel: ({sn_px_pre:.1f}, {sn_py_pre:.1f})")
        print(f"  SN post pixel: ({sn_px:.1f}, {sn_py:.1f})")
        f_pre  = bg_subtracted_flux(flux1, sn_px_pre, sn_py_pre, 5, 15)
        f_post = bg_subtracted_flux(flux2, sn_px,     sn_py,     5, 15)
        ratio  = f_post / max(f_pre, 1.0)
        print(f"  Flux pre={f_pre:.1f}  post={f_post:.1f}  ratio={ratio:.1f}x")
        if ratio >= 5.0 and f_post >= MIN_FLUX:
            print(f"  >> DIRECT DETECTION: {ratio:.1f}x brightening at WCS position!")
        else:
            print(f"  >> No sufficient brightening at WCS position (ratio={ratio:.1f}x, flux={f_post:.1f})")
    except Exception as e:
        print(f"  Error: {e}")

print("\n[7] Unmatched brightness detection (searching for new transient)...")
ri, ro = 5, 15
brightened_u = []
if not poor_align:
    for j in unm2:
        cx2, cy2 = float(stars2[j][0]), float(stars2[j][1])
        b2 = bg_subtracted_flux(flux2_aligned, cx2, cy2, ri, ro)
        b1 = bg_subtracted_flux(flux1,         cx2, cy2, ri, ro)
        if b2 > MIN_FLUX and b2 / (b1 + 1.0) >= UNM_THR:
            brightened_u.append({"x": cx2, "y": cy2, "b1": b1, "b2": b2,
                                  "change_pct": (b2/max(b1,1.0)-1.0)*100.0})

print(f"  Unmatched brightened: {len(brightened_u)}")

print("\n[8] Distance of each detection to the SN...")
# Aligned image is in pre coordinate space — use pre WCS for RA/Dec
_wcs_for_aligned = None
try:
    from astropy.wcs import WCS as _WA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _wcs_for_aligned = _WA(hdr1, naxis=2)
except Exception:
    _wcs_for_aligned = wcs  # fallback

def px_to_radec(cx, cy):
    if not WCS_OK or _wcs_for_aligned is None:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra, dec = _wcs_for_aligned.all_pix2world(float(cx), float(cy), 0)
    return float(ra), float(dec)

# Also compute SN position in aligned (pre) coordinate system
sn_px_aligned, sn_py_aligned = sn_px_pre, sn_py_pre  # from pre WCS

def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    cos_dec = np.cos(np.radians((dec1 + dec2) / 2))
    return np.sqrt(((ra1 - ra2) * cos_dec)**2 + (dec1 - dec2)**2) * 3600.0

for r in brightened_u:
    ra, dec = px_to_radec(r["x"], r["y"])  # uses pre WCS (correct for aligned image)
    r["ra"], r["dec"] = ra, dec
    if WCS_OK and ra is not None:
        r["dist"] = angular_sep_arcsec(ra, dec, SN_RA, SN_DEC)
    else:
        # pixel distance in aligned image = pre coordinate space
        r["dist"] = float(np.sqrt((r["x"] - sn_px_aligned)**2 + (r["y"] - sn_py_aligned)**2)) * 1.01
brightened_u.sort(key=lambda r: r["dist"])

print()
print("=" * 72)
print(f"  {'#':>3}  {'x':>7}  {'y':>7}  {'Dist':>9}  {'Change':>12}")
print("-" * 72)
sn_found = False
for k, r in enumerate(brightened_u[:20]):
    flag = ""
    if r["dist"] <= MATCH_ARCSEC:
        flag = "  <-- AT2020xnd"
        sn_found = True
    elif r["dist"] <= 60:
        flag = "  (nearby)"
    print(f"  {k+1:>3}  {r['x']:>7.1f}  {r['y']:>7.1f}  "
          f"{r['dist']:>7.1f}\"  {r['change_pct']:>+12.1f}%{flag}")
if len(brightened_u) > 20:
    print(f"  ... ({len(brightened_u)-20} more)")
print("=" * 72)

if sn_found:
    hits = [r for r in brightened_u if r["dist"] <= MATCH_ARCSEC]
    best = hits[0]
    print(f"\n  AT2020xnd DETECTED! ({len(hits)} detections, r<{MATCH_ARCSEC}\")")
    print(f"  Closest: ({best['x']:.1f}, {best['y']:.1f})"
          f"  dist={best['dist']:.1f}\"  change={best['change_pct']:+.1f}%")
else:
    print(f"\n  AT2020xnd not found within {MATCH_ARCSEC}\".")
    if brightened_u:
        best = brightened_u[0]
        print(f"  Closest: dist={best['dist']:.1f}\"  ({best['x']:.1f}, {best['y']:.1f})")
    print(f"  SN post pixel: ({sn_px:.1f}, {sn_py:.1f})")

print()
