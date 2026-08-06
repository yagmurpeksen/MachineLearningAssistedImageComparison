"""
detection.py
============
Star detection, image alignment, matching, and brightness/motion analysis.

Pipeline order within the module:
  1. detect_stars          – threshold + blob analysis
  2. star_affine_align     – RANSAC affine alignment
  3. match_stars_mutual    – bidirectional nearest-neighbour matching
  4. detect_moving_outliers – motion detection via IQR
  5. bg_subtracted_flux    – aperture photometry (helper)
  6. calculate_star_snr    – CCD SNR formula
  7. filter_stars_by_snr   – star filter by SNR threshold
  8. detect_brightness_outliers – brightness detection via flux + area change
"""

import numpy as np
import cv2 as cv
from scipy.spatial import KDTree

try:
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    DAO_AVAILABLE = True
except ImportError:
    DAO_AVAILABLE = False


# ════════════════════════════════════════════════════════════════
#  0. PAIR PREPROCESSING  (shared scale + warp border mask)
# ════════════════════════════════════════════════════════════════

def normalize_pair_joint(img1, img2, p_lo=1.0, p_hi=99.5):
    """
    Brings two frames to the same brightness scale (instead of separate min-max).

    When normalized separately, a percentile threshold (e.g. 97%) maps to a
    different absolute brightness in each frame → artificially high detections in F2.
    """
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    stack = np.concatenate([a.ravel(), b.ravel()])
    lo = float(np.percentile(stack, p_lo))
    hi = float(np.percentile(stack, p_hi))
    span = max(hi - lo, 1.0)

    def _scale(im):
        out = (im.astype(np.float32) - lo) / span * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    return _scale(img1), _scale(img2)


def suppress_warp_border(img, margin=12):
    """Zeroes out edge interpolation artifacts after affine warp."""
    if margin <= 0:
        return img
    out = img.copy()
    out[:margin, :] = 0
    out[-margin:, :] = 0
    out[:, :margin] = 0
    out[:, -margin:] = 0
    return out


def filter_stars_from_border(stars, width, height, margin=10):
    """Removes spurious detections in the border region."""
    if margin <= 0 or not stars:
        return stars
    m = float(margin)
    return [
        s for s in stars
        if m <= s[0] < width - m and m <= s[1] < height - m
    ]


# ════════════════════════════════════════════════════════════════
#  0b. DEPTH MATCHING  (star count balancing after DAO)
# ════════════════════════════════════════════════════════════════

def match_detection_depth(stars1, stars2, max_ratio=2.0):
    """
    Balances the detection depth of two star lists.

    In pairs with different noise levels (sky background), even with the same
    sigma-clipped threshold the absolute detection depth may differ. This function:
      - Sorts by peak brightness
      - Trims the longer list until:
        max(N1,N2) / min(N1,N2) <= max_ratio

    Parameters
    ----------
    stars1, stars2 : list of (cx, cy, area, peak)
    max_ratio      : float — maximum allowed ratio (default 2.0)

    Returns
    -------
    (stars1_trimmed, stars2_trimmed)
    """
    if not stars1 or not stars2:
        return stars1, stars2

    n1, n2 = len(stars1), len(stars2)
    ratio  = max(n1, n2) / max(min(n1, n2), 1)

    if ratio <= max_ratio:
        return stars1, stars2   # already balanced

    # Trim the longer list by peak (brightest first)
    cap = int(min(n1, n2) * max_ratio)
    if n1 > n2:
        stars1 = sorted(stars1, key=lambda s: -s[3])[:cap]
    else:
        stars2 = sorted(stars2, key=lambda s: -s[3])[:cap]

    return stars1, stars2


# ════════════════════════════════════════════════════════════════
#  1. STAR DETECTION
# ════════════════════════════════════════════════════════════════

def detect_stars(img, pct=97, min_area=3, max_area=150, min_peak=20):
    """
    Detects star-like bright spots in an image.

    Method: Gaussian blur → percentile threshold → morphological opening →
            connected component analysis.

    Parameters
    ----------
    img      : uint8 numpy array   — normalized image
    pct      : float               — threshold percentile (default 97)
    min_area : int                 — minimum blob area (pixels²)
    max_area : int                 — maximum blob area (pixels²)
    min_peak : int                 — minimum peak brightness

    Returns
    -------
    list of (cx, cy, area, peak)
    """
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
        # Saturated stars (peak=255): area limit not applied,
        # diffraction spikes inflate area but the centroid is correct
        saturated = (peak >= 254)
        if not saturated and not (min_area <= area <= max_area):
            continue
        if area < min_area:
            continue
        if peak >= min_peak:
            stars.append((float(cx), float(cy), int(area), peak))
    return stars


# ════════════════════════════════════════════════════════════════
#  1b. DAO STAR DETECTION  (sigma-clipped, FITS-native)
# ════════════════════════════════════════════════════════════════

def detect_stars_dao(img_float, fwhm=3.0, sigma_thresh=5.0,
                     min_area=3, max_area=2000):
    """
    Astronomical star detection using DAOStarFinder.

    Computes a threshold based on each image's own noise level (sigma_clipped std)
    → star counts are balanced in pairs with different exposure times or sky backgrounds.

    Parameters
    ----------
    img_float    : float32 numpy array — raw FITS data (ADU)
    fwhm         : float               — expected star FWHM (pixels)
    sigma_thresh : float               — detection threshold (how many sigma above background)
    min_area     : int                 — minimum source area (pixels²)
    max_area     : int                 — maximum source area (pixels²)

    Returns
    -------
    list of (cx, cy, area, peak)   — same format as detect_stars
    """
    if not DAO_AVAILABLE:
        return []

    img = np.asarray(img_float, dtype=np.float64)

    # Sigma-clipped background statistics
    _, median, std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
    if std <= 0 or not np.isfinite(std):
        return []

    threshold = sigma_thresh * std
    daofind = DAOStarFinder(fwhm=float(fwhm), threshold=float(threshold),
                             exclude_border=True)
    sources = daofind(img - median)

    if sources is None or len(sources) == 0:
        return []

    h, w = img.shape[:2]
    stars = []
    for row in sources:
        cx   = float(row['xcentroid'])
        cy   = float(row['ycentroid'])
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        peak_above = float(row['peak'])      # peak above background
        peak_abs   = peak_above + median     # absolute ADU value
        # npix = number of pixels occupied by the source (area approximation)
        npix = int(row['npix']) if 'npix' in row.colnames else \
               max(min_area, int(np.pi * (fwhm / 2.0) ** 2))
        area = int(np.clip(npix, min_area, max_area))
        if peak_above > 0:
            stars.append((cx, cy, area, float(peak_abs)))

    return stars


# ════════════════════════════════════════════════════════════════
#  2. ALIGNMENT  (star-centroid affine + RANSAC)
# ════════════════════════════════════════════════════════════════

def star_affine_align(img1_n, img2_n, pct=97, coarse_radius=25,
                      ransac_thresh=5.0, border_margin=12,
                      stars1=None, stars2=None):
    """
    Aligns img2 to the coordinate system of img1.

    Method:
      - Find star centroids in both images
      - Coarse matching with KDTree (within coarse_radius)
      - Estimate robust affine transform with RANSAC
      - Apply the transform with cv.warpAffine

    Parameters
    ----------
    stars1, stars2 : list of (cx,cy,area,peak) or None
        Pre-computed star lists (e.g. from DAO detection).
        If None, they are computed internally using the percentile method.

    Returns
    -------
    (img2_aligned, H, method_str)
      H = None → alignment failed, original img2 is returned
    """
    h, w = img1_n.shape[:2]
    s1 = stars1 if stars1 is not None else detect_stars(img1_n, pct=pct)
    s2 = stars2 if stars2 is not None else detect_stars(img2_n, pct=pct)
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
                                  flags=cv.INTER_LINEAR,
                                  borderMode=cv.BORDER_CONSTANT)
    img2_aligned = suppress_warp_border(img2_aligned, border_margin)
    inliers = int(mask.sum()) if mask is not None else 0
    return img2_aligned, H, f"star-affine({inliers} inliers)"


# ════════════════════════════════════════════════════════════════
#  3. MATCHING  (mutual nearest neighbour)
# ════════════════════════════════════════════════════════════════

def match_stars_mutual(stars1, stars2, max_distance):
    """
    Bidirectional nearest-neighbour matching (mutual nearest neighbour).

    A pair is counted as matched only when it is mutually the nearest from
    both sides — greatly reduces false positive matches.

    Returns
    -------
    matched : list of (i, j, dx, dy, dist)
    unm1    : unmatched star indices in epoch1
    unm2    : unmatched star indices in epoch2
    """
    if not stars1 or not stars2:
        return [], list(range(len(stars1))), list(range(len(stars2)))
    pts1 = np.array([[s[0], s[1]] for s in stars1])
    pts2 = np.array([[s[0], s[1]] for s in stars2])
    tree1 = KDTree(pts1)
    tree2 = KDTree(pts2)
    d_fwd, i_fwd = tree2.query(pts1, k=1)
    _, i_bwd = tree1.query(pts2, k=1)
    matched, used1, used2 = [], set(), set()
    for i, (dist, j) in enumerate(zip(d_fwd, i_fwd)):
        j = int(j)
        if dist <= max_distance and i_bwd[j] == i:
            matched.append((i, j,
                             float(stars2[j][0] - stars1[i][0]),
                             float(stars2[j][1] - stars1[i][1]),
                             float(dist)))
            used1.add(i)
            used2.add(j)
    unm1 = [i for i in range(len(stars1)) if i not in used1]
    unm2 = [j for j in range(len(stars2)) if j not in used2]
    return matched, unm1, unm2


# ════════════════════════════════════════════════════════════════
#  4. MOVING DETECTION  (IQR outlier on position residuals)
# ════════════════════════════════════════════════════════════════

def detect_moving_outliers(matched, iqr_multiplier=3.0, min_residual_px=3.0,
                            max_moving_frac=0.05):
    """
    Detects abnormal position changes among matched star pairs
    (moving objects: asteroids, satellites, comets).

    Method:
      - Compute global median dx/dy (telescope drift reference)
      - Each star's deviation from the median = residual
      - IQR-based upper threshold → outlier = moving candidate

    Adaptive PSF-mismatch correction:
      If median residual > 1px (different telescope/epoch), the threshold
      automatically rises → false positives are suppressed.

    max_moving_frac : at most this fraction of stars are flagged as "moving"
                      (0.05 = 5% upper limit)
    """
    if len(matched) < 4:
        return [], 0.0, 0.0, min_residual_px, np.array([])
    dxs = np.array([m[2] for m in matched])
    dys = np.array([m[3] for m in matched])
    med_dx, med_dy = float(np.median(dxs)), float(np.median(dys))
    residuals = np.sqrt((dxs - med_dx)**2 + (dys - med_dy)**2)
    q1, q3 = np.percentile(residuals, 25), np.percentile(residuals, 75)
    iqr = q3 - q1

    thr = max(q3 + iqr_multiplier * iqr, min_residual_px)

    # Adaptive PSF-mismatch correction
    median_res = float(np.median(residuals))
    if median_res > 1.0:
        adaptive_floor = median_res * 4.0
        thr = max(thr, adaptive_floor)

    # Fraction limit: at most max_moving_frac of stars are flagged as "moving"
    moving_idx = [i for i, r in enumerate(residuals) if r > thr]
    cap = max(1, int(len(matched) * max_moving_frac))
    if len(moving_idx) > cap:
        sorted_idx = sorted(range(len(residuals)), key=lambda i: -residuals[i])
        moving_idx = sorted_idx[:cap]
        thr = float(residuals[sorted_idx[cap - 1]])

    return moving_idx, med_dx, med_dy, float(thr), residuals


# ════════════════════════════════════════════════════════════════
#  4b. UNMATCHED-PAIR MOVING DETECTION  (fallback when mutual-NN fails)
# ════════════════════════════════════════════════════════════════

def detect_moving_from_unmatched(stars1, stars2, unm1, unm2,
                                  med_dx, med_dy,
                                  min_motion=5.0, max_motion=200.0):
    """
    Detects moving object candidates from unmatched star pairs.

    Mutual nearest-neighbour matching fails when a background star falls
    near the asteroid's epoch2 position — the asteroid falls into unm1,
    and the background star ends up matched. This function:

      1. Predicts the expected epoch2 position of unm1 stars using the
         global shift (med_dx, med_dy).
      2. Searches for an unm2 star within [min_motion, max_motion] of the
         expected position (real motion = expected + delta).
      3. If a match is found → moving object candidate.

    Parameters
    ----------
    stars1, stars2 : list of (cx, cy, area, peak)
    unm1           : list of int — unmatched star indices in epoch1
    unm2           : list of int — unmatched star indices in epoch2
    med_dx, med_dy : float — global offset computed from matched pairs
    min_motion     : float — motions below this distance are ignored (px)
                             — recommended to use the same value as the IQR threshold
    max_motion     : float — motions above this distance are out of scope

    Returns
    -------
    list of dict — keys x1,y1,x2,y2,dx,dy,residual for each candidate
    """
    if not unm1 or not unm2:
        return []

    unm2_pts = np.array([[stars2[j][0], stars2[j][1]] for j in unm2],
                         dtype=np.float64)
    tree2 = KDTree(unm2_pts)

    results = []
    used_unm2 = set()

    for i in unm1:
        cx1, cy1 = float(stars1[i][0]), float(stars1[i][1])
        # Expected epoch2 position using global shift
        exp_x = cx1 + med_dx
        exp_y = cy1 + med_dy

        # Find all unm2 candidates within max_motion
        candidates = tree2.query_ball_point([exp_x, exp_y], max_motion)

        best_motion = float('inf')
        best_c = -1
        for c in candidates:
            if c in used_unm2:
                continue
            cx2, cy2 = unm2_pts[c]
            motion = float(np.sqrt((cx2 - exp_x) ** 2 + (cy2 - exp_y) ** 2))
            if motion >= min_motion and motion < best_motion:
                best_motion = motion
                best_c = c

        if best_c >= 0:
            j = unm2[best_c]
            cx2 = float(stars2[j][0])
            cy2 = float(stars2[j][1])
            dx = cx2 - cx1
            dy = cy2 - cy1
            residual = float(np.sqrt((dx - med_dx) ** 2 + (dy - med_dy) ** 2))
            results.append({
                "x1": cx1, "y1": cy1,
                "x2": cx2, "y2": cy2,
                "dx": dx, "dy": dy,
                "res_dx": dx - med_dx, "res_dy": dy - med_dy,
                "residual": residual,
                "b1": float(stars1[i][3]),
                "b2": float(stars2[j][3]),
                "ml_score": 0.0,
                "from_unmatched": True,
            })
            used_unm2.add(best_c)

    return results


# ════════════════════════════════════════════════════════════════
#  5. APERTURE PHOTOMETRY  (helper function)
# ════════════════════════════════════════════════════════════════

def bg_subtracted_flux(img, cx, cy, r_inner=4, r_outer=8):
    """
    Aperture photometry with annulus background subtraction.

    Star flux = sum of inner circle − background × number of pixels
    Background = median of the outer annulus region (annulus subtraction).
    """
    h, w = img.shape
    r = r_outer
    y0 = max(0, int(cy) - r); y1 = min(h, int(cy) + r + 1)
    x0 = max(0, int(cx) - r); x1 = min(w, int(cx) + r + 1)
    patch = img[y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2     = (xx - cx)**2 + (yy - cy)**2
    annulus   = (dist2 > r_inner**2) & (dist2 <= r_outer**2)
    star_mask = dist2 <= r_inner**2
    if not star_mask.any():
        return 0.0
    bg = float(np.median(patch[annulus])) if annulus.any() else 0.0
    return max(float(patch[star_mask].sum()) - bg * int(star_mask.sum()), 0.0)


# ════════════════════════════════════════════════════════════════
#  6. SNR  (Signal-to-Noise Ratio)
# ════════════════════════════════════════════════════════════════

def calculate_star_snr(img, cx, cy, r_inner=4, r_outer=8):
    """
    Computes SNR using the CCD noise equation.

      SNR = signal / sqrt(signal + n_pix * bg_sigma²)

    signal   = background-subtracted star flux (ADU)
    bg_sigma = pixel noise in the annulus region (std)
    """
    h, w = img.shape
    r = r_outer
    y0 = max(0, int(cy) - r); y1 = min(h, int(cy) + r + 1)
    x0 = max(0, int(cx) - r); x1 = min(w, int(cx) + r + 1)
    patch  = img[y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2     = (xx - cx)**2 + (yy - cy)**2
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


# ════════════════════════════════════════════════════════════════
#  7. SNR FILTER
# ════════════════════════════════════════════════════════════════

def filter_stars_by_snr(stars, img, min_snr=3.0, r_inner=4, r_outer=8):
    """
    Removes stars below the SNR threshold.

    Saturated stars (peak >= 254) are exempt from the filter:
    the annulus still falls within the star so SNR computes low,
    but the star is real.

    Returns
    -------
    (filtered_stars, all_snr_list, kept_snr_list)
    """
    snr_list = [calculate_star_snr(img, s[0], s[1], r_inner, r_outer)
                for s in stars]
    filtered = [s for s, snr in zip(stars, snr_list)
                if snr >= min_snr or s[3] >= 254]
    snr_kept = [snr for s, snr in zip(stars, snr_list)
                if snr >= min_snr or s[3] >= 254]
    return filtered, snr_list, snr_kept


# ════════════════════════════════════════════════════════════════
#  8. BRIGHTNESS OUTLIER DETECTION
# ════════════════════════════════════════════════════════════════

def detect_brightness_outliers(matched, stars1, stars2, img1_n, img2_n,
                                r_inner=4, r_outer=8,
                                flux_ratio_thr=2.5, min_flux=50,
                                area_ratio_thr=10.0):
    """
    Detects abnormal brightness changes among matched star pairs
    (nova, transient, variable star, fading).

    Method:
      - Compute global median flux ratio (corrects for exposure difference)
      - For each pair: corrected flux ratio > flux_ratio_thr → brightened
      - Or flux ratio < 1/flux_ratio_thr → dimmed
      - Area ratio (blob size) > area_ratio_thr → strong nova signal

    Returns
    -------
    (brightened, dimmed, stationary, changes_array)
    """
    fluxes = []
    for (i, j, dx, dy, dist) in matched:
        b1 = bg_subtracted_flux(img1_n, stars1[i][0], stars1[i][1],
                                r_inner, r_outer)
        b2 = bg_subtracted_flux(img2_n, stars2[j][0], stars2[j][1],
                                r_inner, r_outer)
        fluxes.append((b1, b2))

    # Correct global exposure/sky difference using the median ratio
    ratios = [b2 / b1 for b1, b2 in fluxes if b1 > min_flux and b2 > 0]
    global_ratio = float(np.median(ratios)) if ratios else 1.0

    brightened, dimmed, stationary = [], [], []
    changes_list = []
    for idx, (i, j, dx, dy, dist) in enumerate(matched):
        b1, b2 = fluxes[idx]
        b2_corr = b2 / global_ratio if global_ratio > 0 else b2

        area1 = max(stars1[i][2], 1)
        area2 = max(stars2[j][2], 1)
        area_ratio = area2 / area1

        if b1 > min_flux:
            ratio = b2_corr / b1
            ch = (ratio - 1.0) * 100.0
        elif b2_corr > min_flux * flux_ratio_thr or area_ratio >= area_ratio_thr:
            # Faint in Epoch1 but bright in Epoch2 → nova / new source candidate
            ratio = b2_corr / max(b1, 1.0)
            ch = (ratio - 1.0) * 100.0
        else:
            ratio = 1.0
            ch = 0.0

        if area_ratio >= area_ratio_thr and area_ratio > ratio:
            ratio = area_ratio
            ch = (ratio - 1.0) * 100.0

        changes_list.append(ch)
        entry = {
            "idx": idx, "i": i, "j": j,
            "x1": stars1[i][0], "y1": stars1[i][1],
            "x2": stars2[j][0], "y2": stars2[j][1],
            "b1": b1, "b2": b2, "change_pct": ch,
            "peak1": stars1[i][3], "peak2": stars2[j][3],
            "area1": area1, "area2": area2,
        }
        if ratio >= flux_ratio_thr:
            brightened.append(entry)
        elif b1 > min_flux and ratio <= 1.0 / flux_ratio_thr:
            dimmed.append(entry)
        else:
            stationary.append(entry)

    brightened.sort(key=lambda x: x["change_pct"], reverse=True)
    dimmed.sort(key=lambda x: x["change_pct"])
    return brightened, dimmed, stationary, np.array(changes_list)
