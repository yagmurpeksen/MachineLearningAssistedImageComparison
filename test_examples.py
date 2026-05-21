"""
test_examples.py
================
SN 2026fov (brightness change) ve moving object orneklerini
komut satirinda test eder -- GUI gerektirmez.

Kullanim:
  python test_examples.py
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Bagimlilik kontrolu ──────────────────────────────────────────
print("=" * 60)
print("  Bagimlilik kontrolu")
print("=" * 60)

try:
    from fits_loader import load_fits_image, FITS_AVAILABLE
    print(f"  fits_loader  : OK  (FITS_AVAILABLE={FITS_AVAILABLE})")
except Exception as e:
    print(f"  fits_loader  : HATA -- {e}"); sys.exit(1)

try:
    from detection import (
        detect_stars_dao, match_detection_depth, DAO_AVAILABLE,
        star_affine_align, match_stars_mutual,
        detect_moving_outliers, detect_brightness_outliers,
        filter_stars_from_border,
    )
    print(f"  detection    : OK  (DAO_AVAILABLE={DAO_AVAILABLE})")
except Exception as e:
    print(f"  detection    : HATA -- {e}"); sys.exit(1)

try:
    from astropy.io import fits as astropy_fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

import cv2 as cv

BASE = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════
#  Yardimci: FITS header'dan WCS pixel offset hesapla
# ════════════════════════════════════════════════════════════════

def wcs_pixel_offset(hdr1, hdr2):
    """
    iki FITS header'indaki CRVAL farkindan pixel cinsinden
    (dx2-dx1, dy2-dy1) tahmin eder.
    Pozitif dx: epoch2 yildizlari epoch1'e gore x'te daha saga.
    Pozitif dy: epoch2 yildizlari epoch1'e gore y'de daha yukari.
    """
    try:
        ra1, dec1 = hdr1['CRVAL1'], hdr1['CRVAL2']
        ra2, dec2 = hdr2['CRVAL1'], hdr2['CRVAL2']
        cd11 = hdr1.get('CD1_1', -0.000281)  # deg/px (RA, typically negative)
        cd22 = hdr1.get('CD2_2',  0.000281)  # deg/px (Dec, typically positive)
        dec_mid = math.radians((dec1 + dec2) / 2)
        # RA offset in pixels (signed: positive means epoch2 is to the right)
        dx = (ra1 - ra2) * math.cos(dec_mid) / cd11
        # Dec offset in pixels (signed: positive means epoch2 is above)
        dy = (dec1 - dec2) / cd22
        return dx, dy
    except Exception:
        return 0.0, 0.0


def get_headers(path):
    """FITS dosyasindan ilk goruntu header'ini dondurur."""
    if not ASTROPY_OK:
        return {}
    try:
        with astropy_fits.open(path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    return dict(hdu.header)
        return {}
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════════
#  Ana analiz fonksiyonu
# ════════════════════════════════════════════════════════════════

def analyze_pair(path1, path2,
                 match_radius=8.0,
                 dao_sigma=5.0,
                 coarse_radius=25,
                 use_wcs_preshift=False):
    """
    Iki FITS dosyasini yukler, hizalar, eslestirir ve transient/moving tespiti yapar.

    Parametreler
    ------------
    use_wcs_preshift : WCS CRVAL farkindan hesaplanan pixel offset'i ile
                       epoch2 yildizlarini onceden kaydir -- kaba hizalama
                       icin coarse_radius yetersiz kalmasi durumunda kullan.
    """
    print(f"\n  F1: {os.path.basename(path1)}")
    print(f"  F2: {os.path.basename(path2)}")

    for p in (path1, path2):
        if not os.path.exists(p):
            print(f"  !! Dosya yok: {p}"); return None

    # load_fits_image -> (data_u8, info, data_float32, header)
    img1_n, info1, raw1, hdr1 = load_fits_image(path1)
    img2_n, info2, raw2, hdr2 = load_fits_image(path2)
    h, w = img1_n.shape[:2]

    print(f"  DATE-OBS F1: {info1.get('DATE-OBS','?')}  F2: {info2.get('DATE-OBS','?')}")
    print(f"  MAGLIM   F1: {info1.get('MAGLIM','?')}   F2: {info2.get('MAGLIM','?')}")
    print(f"  SEEING   F1: {info1.get('SEEING','?'):.2f}\"  F2: {info2.get('SEEING','?'):.2f}\""
          if isinstance(info1.get('SEEING'), (int, float)) else
          f"  SEEING   F1: {info1.get('SEEING','?')}\"  F2: {info2.get('SEEING','?')}\"")

    if not (DAO_AVAILABLE and raw1 is not None and raw2 is not None):
        print("  DAO kullanimi eskin."); return None

    flux1 = raw1.astype(np.float64)
    flux2 = raw2.astype(np.float64)
    flux2_rs = cv.resize(flux2, (w, h)) if flux2.shape[:2] != (h, w) else flux2

    # ── DAO tespiti (hizalama icin) ─────────────────────────────
    dao1_align = detect_stars_dao(flux1,   fwhm=3.0, sigma_thresh=dao_sigma)

    # WCS-tabani on-kaydirma:
    # epoch2 goruntusunu WCS pixel offset kadar oteliyoruz, sonra
    # residual hizalamasi yapiyoruz. H_toplam = H_resid * H_wcs
    wcs_dx, wcs_dy = 0.0, 0.0
    if use_wcs_preshift:
        hdr1_raw = get_headers(path1)
        hdr2_raw = get_headers(path2)
        wcs_dx, wcs_dy = wcs_pixel_offset(hdr1_raw, hdr2_raw)
        # wcs_dx = x2 - x1 (epoch2 star pixel - epoch1 star pixel)
        # epoch2 goruntusunu epoch1 ile uyumlu hale getirmek icin:
        # shift = -(wcs_dx, wcs_dy)  yani epoch2'yi -wcs_dx kadar kaydir
        shift_x = -wcs_dx
        shift_y = -wcs_dy
        print(f"\n  WCS pixel offset (x2-x1): dx={wcs_dx:.1f}  dy={wcs_dy:.1f}px")
        print(f"  Uygulanan kayma: shift_x={shift_x:+.1f}  shift_y={shift_y:+.1f}px")
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        flux2_wcs = cv.warpAffine(flux2_rs.astype(np.float32), M_shift, (w, h),
                                   flags=cv.INTER_LINEAR,
                                   borderMode=cv.BORDER_CONSTANT)
        img2_wcs = cv.normalize(flux2_wcs, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        dao2_align = detect_stars_dao(flux2_wcs, fwhm=3.0, sigma_thresh=dao_sigma)
    else:
        flux2_wcs = flux2_rs.astype(np.float32)
        img2_wcs  = img2_n
        dao2_align = detect_stars_dao(flux2_rs, fwhm=3.0, sigma_thresh=dao_sigma)

    print(f"\n  DAO (hizalama oncesi): F1={len(dao1_align)}, F2={len(dao2_align)}")

    # ── Hizalama (WCS-kaydirmis epoch2 uzerinde) ────────────────
    img2_aligned, H_resid, method = star_affine_align(
        img1_n, img2_wcs,
        coarse_radius=coarse_radius,
        stars1=dao1_align, stars2=dao2_align)
    print(f"  Hizalama: method={method}, H={'OK' if H_resid is not None else 'None'}")

    # H_toplam = H_resid ∘ M_shift (once WCS kaydirmasi, sonra residual hizalama)
    # M_shift = [[1,0,shift_x],[0,1,shift_y]] uygulandi
    if H_resid is not None and wcs_dx != 0.0:
        shift_x = -wcs_dx;  shift_y = -wcs_dy
        H_wcs_3x3 = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]], dtype=np.float64)
        H_resid_3x3 = np.vstack([H_resid.astype(np.float64), [0, 0, 1]])
        H_total = (H_resid_3x3 @ H_wcs_3x3)[:2, :].astype(np.float32)
    else:
        H_total = H_resid

    # ── DAO tespiti (hizalama sonrasi) ──────────────────────────
    flux2_aligned = cv.warpAffine(
        flux2_rs.astype(np.float32), H_total, (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT) if H_total is not None else flux2_rs.astype(np.float32)

    stars1_raw = detect_stars_dao(flux1,         fwhm=3.0, sigma_thresh=5.0)
    stars2_raw = detect_stars_dao(flux2_aligned,  fwhm=3.0, sigma_thresh=5.0)
    print(f"  DAO (hizalama sonrasi): F1={len(stars1_raw)}, F2={len(stars2_raw)}")

    stars1_raw = filter_stars_from_border(stars1_raw, w, h, 10)
    stars2_raw = filter_stars_from_border(stars2_raw, w, h, 10)

    # Derinlik dengeleme: SADECE epoch2 >> epoch1 durumunda
    n1_raw, n2_raw = len(stars1_raw), len(stars2_raw)
    if n2_raw > n1_raw:
        stars1_raw, stars2_raw = match_detection_depth(
            stars1_raw, stars2_raw, max_ratio=2.0)
        print(f"  Derinlik denklemesi (F2>F1): F1={len(stars1_raw)}, F2={len(stars2_raw)}")
    else:
        print(f"  Derinlik denklemesi: gerek yok (F1={n1_raw} >= F2={n2_raw})")

    stars1 = stars1_raw
    stars2 = stars2_raw

    # ── Eslestirme ───────────────────────────────────────────────
    matched, unmatched1, unmatched2 = match_stars_mutual(
        stars1, stars2, max_distance=match_radius)
    print(f"\n  Eslestirme (r={match_radius}px):")
    print(f"    Matched     : {len(matched)}")
    print(f"    Unmatched F1: {len(unmatched1)}")
    print(f"    Unmatched F2: {len(unmatched2)}")

    moving_count = 0
    brightened_list = []
    dimmed_list = []

    # ── Hareket tespiti ──────────────────────────────────────────
    if len(matched) >= 4:
        moving_idx, med_dx, med_dy, thr_r, residuals = detect_moving_outliers(
            matched,
            iqr_multiplier=4.0,
            min_residual_px=5.0,
            max_moving_frac=0.02)
        moving_count = len(moving_idx)
        print(f"\n  Hareket tespiti:")
        print(f"    Moving objects  : {moving_count}")
        print(f"    Median dx={med_dx:.2f}  dy={med_dy:.2f}  threshold={thr_r:.1f}px")
        for mi in moving_idx[:8]:
            m = matched[mi]
            i, j, dx, dy, dist = m
            print(f"      S1[{i}]=({stars1[i][0]:.0f},{stars1[i][1]:.0f}) -> "
                  f"S2[{j}]=({stars2[j][0]:.0f},{stars2[j][1]:.0f})  "
                  f"dx={dx:.1f} dy={dy:.1f}  res={residuals[mi]:.1f}px")

    # ── Parlaklik tespiti ────────────────────────────────────────
    if len(matched) >= 4:
        result = detect_brightness_outliers(
            matched, stars1, stars2,
            flux1.astype(np.float32), flux2_aligned,
            r_inner=4, r_outer=8,
            flux_ratio_thr=2.5, min_flux=50.0)
        brightened_list, dimmed_list, stationary, changes_arr = result
        print(f"\n  Parlaklik tespiti:")
        print(f"    Brightened  : {len(brightened_list)}")
        print(f"    Dimmed      : {len(dimmed_list)}")
        for b in brightened_list[:5]:
            print(f"      ({b['x1']:.0f},{b['y1']:.0f})  b1={b['b1']:.0f} -> "
                  f"b2={b['b2']:.0f}  change={b['change_pct']:+.0f}%")
        for d in dimmed_list[:3]:
            print(f"      ({d['x1']:.0f},{d['y1']:.0f})  b1={d['b1']:.0f} -> "
                  f"b2={d['b2']:.0f}  change={d['change_pct']:+.0f}%")

    # ── Eslesmemis F2 (yeni kaynak adaylari) ─────────────────────
    if unmatched2:
        print(f"\n  Eslesmemis F2 (potansiyel yeni kaynaklar): {len(unmatched2)}")
        unm2_bright = sorted(unmatched2, key=lambda j: -stars2[j][3])[:5]
        for j in unm2_bright:
            s = stars2[j]
            print(f"      S2[{j}]: ({s[0]:.0f},{s[1]:.0f})  peak_abs={s[3]:.0f}")

    print()
    return {
        "matched": len(matched),
        "unmatched_f1": len(unmatched1),
        "unmatched_f2": len(unmatched2),
        "moving": moving_count,
        "brightened": len(brightened_list),
        "dimmed": len(dimmed_list),
    }


# ════════════════════════════════════════════════════════════════
#  ORNEK 1: SN 2026fov -- Brightness Change (Transient)
# ════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  ORNEK 1: SN 2026fov -- NGC 7292 (Type II Supernova)")
print(f"{'='*60}")
print("  [Beklenen]: 'Brightened' veya 'Unmatched F2'de SN gorunmeli")
print("              (SN 14 Mart 2026 kesfedildi, mag=13.5)")

sn_pre  = os.path.join(BASE, "sn2026fov", "pre_sn.fits")
sn_post = os.path.join(BASE, "sn2026fov", "post_sn.fits")

res1 = analyze_pair(sn_pre, sn_post,
                    match_radius=8.0,
                    dao_sigma=5.0,
                    coarse_radius=25,
                    use_wcs_preshift=False)
if res1:
    print(f"  >>> SONUC: matched={res1['matched']} "
          f"brightened={res1['brightened']} "
          f"dimmed={res1['dimmed']} "
          f"unmatched_f2={res1['unmatched_f2']}")

# ════════════════════════════════════════════════════════════════
#  ORNEK 2: Moving Object -- ZTF field 522 (Leo/Virgo, ekliptik)
# ════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  ORNEK 2: Moving Object -- ZTF field 522 (~3 saat ara)")
print(f"{'='*60}")
print("  [Beklenen]: 'Moving' listesinde asteroid gorunmeli")
print("              (WCS on-kaydirma + coarse_radius=25)")

ep1 = os.path.join(BASE, "moving_object", "epoch1.fits")
ep2 = os.path.join(BASE, "moving_object", "epoch2.fits")

res2 = analyze_pair(ep1, ep2,
                    match_radius=80.0,
                    dao_sigma=5.0,
                    coarse_radius=25,
                    use_wcs_preshift=True)   # WCS offset: ~35px RA
if res2:
    print(f"  >>> SONUC: matched={res2['matched']} "
          f"moving={res2['moving']} "
          f"brightened={res2['brightened']} "
          f"unmatched_f2={res2['unmatched_f2']}")

print()
print("=" * 60)
print("  Test tamamlandi.")
print("=" * 60)
