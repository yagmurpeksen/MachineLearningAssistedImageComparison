"""
tests/test_detection.py
=======================
Unit tests for all functions in detection.py.

Synthetic images are used — no external files or network connections required.
"""

import numpy as np
import pytest
from detection import (
    detect_stars,
    normalize_pair_joint,
    filter_stars_from_border,
    star_affine_align,
    match_stars_mutual,
    detect_moving_outliers,
    bg_subtracted_flux,
    calculate_star_snr,
    filter_stars_by_snr,
    detect_brightness_outliers,
)


# ── Helper functions ─────────────────────────────────────────────

def make_blank(h=128, w=128, bg=10):
    """Blank image with flat background."""
    return np.full((h, w), bg, dtype=np.uint8)


def add_star(img, cy, cx, peak=220, sigma=2.0):
    """Add a Gaussian star to the image."""
    img = img.astype(float)
    r = int(sigma * 5)
    y0, y1 = max(0, cy - r), min(img.shape[0], cy + r + 1)
    x0, x1 = max(0, cx - r), min(img.shape[1], cx + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += peak * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return np.clip(img, 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════
#  detect_stars
# ════════════════════════════════════════════════════════════════

def test_detect_stars_finds_single_star():
    img = make_blank()
    img = add_star(img, cy=64, cx=64, peak=230)
    stars = detect_stars(img, pct=90, min_area=1, max_area=200, min_peak=50)
    assert len(stars) >= 1
    # Center coordinates should be correct (±3px tolerance)
    cx, cy = stars[0][0], stars[0][1]
    assert abs(cx - 64) < 3
    assert abs(cy - 64) < 3


def test_detect_stars_finds_multiple_stars():
    img = make_blank()
    positions = [(30, 30), (60, 90), (100, 40)]
    for cy, cx in positions:
        img = add_star(img, cy=cy, cx=cx, peak=220)
    stars = detect_stars(img, pct=90, min_area=1, max_area=200, min_peak=50)
    assert len(stars) >= 3


def test_detect_stars_empty_image_returns_nothing():
    img = make_blank(bg=5)
    stars = detect_stars(img, pct=97, min_area=3, max_area=150, min_peak=50)
    assert len(stars) == 0


def test_detect_stars_respects_min_peak():
    img = make_blank()
    img = add_star(img, cy=64, cx=64, peak=30)   # too faint
    stars = detect_stars(img, pct=90, min_area=1, max_area=200, min_peak=100)
    assert len(stars) == 0


def test_detect_stars_returns_correct_format():
    img = make_blank()
    img = add_star(img, cy=64, cx=64, peak=220)
    stars = detect_stars(img, pct=90, min_area=1, max_area=200, min_peak=50)
    assert len(stars) >= 1
    cx, cy, area, peak = stars[0]
    assert isinstance(cx, float)
    assert isinstance(cy, float)
    assert isinstance(area, int)
    assert isinstance(peak, int)
    assert peak > 0 and area > 0


# ════════════════════════════════════════════════════════════════
#  star_affine_align
# ════════════════════════════════════════════════════════════════

def make_star_field(positions, h=256, w=256, peak=220):
    img = make_blank(h=h, w=w, bg=5)
    for cy, cx in positions:
        img = add_star(img, cy=cy, cx=cx, peak=peak)
    return img


def test_align_identical_images():
    positions = [(40, 40), (40, 120), (120, 40), (120, 120), (80, 80)]
    img = make_star_field(positions)
    aligned, H, method = star_affine_align(img, img.copy(), pct=90, coarse_radius=30)
    # Same image → H should be found
    assert H is not None
    assert "inliers" in method


def test_align_shifted_image():
    """Image shifted 5px to the right should be alignable."""
    import cv2 as cv
    positions = [(50, 50), (50, 150), (150, 50), (150, 150), (100, 100)]
    img1 = make_star_field(positions)
    # Shift 5px to the right
    M = np.float32([[1, 0, 5], [0, 1, 0]])
    img2 = cv.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    aligned, H, method = star_affine_align(img1, img2, pct=90, coarse_radius=30)
    assert H is not None


def test_align_too_few_stars():
    img1 = make_blank()
    img1 = add_star(img1, cy=64, cx=64, peak=220)   # only 1 star
    img2 = make_blank()
    img2 = add_star(img2, cy=64, cx=70, peak=220)
    aligned, H, method = star_affine_align(img1, img2, pct=90)
    assert H is None
    assert method in ("no_stars", "few_pairs", "ransac_failed")


# ════════════════════════════════════════════════════════════════
#  match_stars_mutual
# ════════════════════════════════════════════════════════════════

def test_match_perfect_correspondence():
    stars1 = [(10.0, 10.0, 5, 200), (50.0, 50.0, 5, 200), (90.0, 30.0, 5, 200)]
    stars2 = [(10.0, 10.0, 5, 200), (50.0, 50.0, 5, 200), (90.0, 30.0, 5, 200)]
    matched, unm1, unm2 = match_stars_mutual(stars1, stars2, max_distance=5)
    assert len(matched) == 3
    assert len(unm1) == 0
    assert len(unm2) == 0


def test_match_with_extra_star_in_epoch2():
    stars1 = [(10.0, 10.0, 5, 200), (50.0, 50.0, 5, 200)]
    stars2 = [(10.0, 10.0, 5, 200), (50.0, 50.0, 5, 200), (90.0, 90.0, 5, 200)]
    matched, unm1, unm2 = match_stars_mutual(stars1, stars2, max_distance=5)
    assert len(matched) == 2
    assert len(unm2) == 1   # new star like a nova


def test_match_respects_max_distance():
    stars1 = [(10.0, 10.0, 5, 200)]
    stars2 = [(10.0, 25.0, 5, 200)]   # 15px away
    matched, _, _ = match_stars_mutual(stars1, stars2, max_distance=5)
    assert len(matched) == 0   # too far → no match


def test_match_empty_lists():
    matched, unm1, unm2 = match_stars_mutual([], [], max_distance=8)
    assert matched == []
    assert unm1 == []
    assert unm2 == []


def test_match_returns_correct_format():
    stars1 = [(20.0, 20.0, 5, 200)]
    stars2 = [(22.0, 20.0, 5, 200)]   # 2px difference
    matched, _, _ = match_stars_mutual(stars1, stars2, max_distance=5)
    assert len(matched) == 1
    i, j, dx, dy, dist = matched[0]
    assert i == 0 and j == 0
    assert abs(dx - 2.0) < 0.1
    assert abs(dist - 2.0) < 0.1


# ════════════════════════════════════════════════════════════════
#  detect_moving_outliers
# ════════════════════════════════════════════════════════════════

def make_matched(n=20, dx=1.0, dy=0.5):
    """Generate matches with all stars shifted by the same dx/dy."""
    return [(i, i, dx, dy, 0.0) for i in range(n)]


def test_no_moving_when_all_same_offset():
    matched = make_matched(20, dx=2.0, dy=1.0)
    moving, *_ = detect_moving_outliers(matched, min_residual_px=3.0)
    assert len(moving) == 0


def test_detects_one_moving_star():
    matched = make_matched(20, dx=0.0, dy=0.0)
    # Shift the last star 20px to the right
    matched[-1] = (19, 19, 20.0, 0.0, 0.0)
    moving, *_ = detect_moving_outliers(matched, min_residual_px=3.0)
    assert 19 in moving


def test_too_few_matched_returns_empty():
    matched = make_matched(3)
    moving, med_dx, med_dy, thr, residuals = detect_moving_outliers(matched)
    assert moving == []
    assert len(residuals) == 0


def test_max_moving_frac_caps_results():
    """max_moving_frac=0.05 → at most 1 star (20 matches × 0.05 = 1)."""
    matched = make_matched(20, dx=0.0, dy=0.0)
    # Set 3 stars with very large offsets
    for k in [17, 18, 19]:
        matched[k] = (k, k, 50.0 + k, 0.0, 0.0)
    moving, *_ = detect_moving_outliers(matched, max_moving_frac=0.05,
                                         min_residual_px=3.0)
    assert len(moving) <= max(1, int(20 * 0.05))


# ════════════════════════════════════════════════════════════════
#  bg_subtracted_flux
# ════════════════════════════════════════════════════════════════

def test_flux_bright_star():
    img = make_blank(bg=10).astype(float)
    img = add_star(img.astype(np.uint8), cy=64, cx=64, peak=200).astype(float)
    flux = bg_subtracted_flux(img, cx=64, cy=64, r_inner=4, r_outer=8)
    assert flux > 0


def test_flux_empty_region():
    img = np.full((64, 64), 10.0)
    flux = bg_subtracted_flux(img, cx=32, cy=32, r_inner=4, r_outer=8)
    assert flux == pytest.approx(0.0, abs=1.0)


# ════════════════════════════════════════════════════════════════
#  calculate_star_snr
# ════════════════════════════════════════════════════════════════

def test_snr_bright_star_positive():
    img = make_blank(bg=5).astype(float)
    img = add_star(img.astype(np.uint8), cy=32, cx=32, peak=200).astype(float)
    snr = calculate_star_snr(img, cx=32, cy=32, r_inner=4, r_outer=8)
    assert snr > 0


def test_snr_flat_image_near_zero():
    img = np.full((64, 64), 20.0)
    snr = calculate_star_snr(img, cx=32, cy=32, r_inner=4, r_outer=8)
    assert snr == pytest.approx(0.0, abs=1.0)


def test_snr_brighter_star_higher_snr():
    base = make_blank(bg=5)
    img_dim    = add_star(base.copy(), cy=32, cx=32, peak=80).astype(float)
    img_bright = add_star(base.copy(), cy=32, cx=32, peak=220).astype(float)
    snr_dim    = calculate_star_snr(img_dim,    cx=32, cy=32)
    snr_bright = calculate_star_snr(img_bright, cx=32, cy=32)
    assert snr_bright > snr_dim


# ════════════════════════════════════════════════════════════════
#  filter_stars_by_snr
# ════════════════════════════════════════════════════════════════

def test_filter_removes_low_snr():
    img = make_blank(bg=5)
    img = add_star(img, cy=64, cx=64, peak=220)
    stars = detect_stars(img, pct=90, min_area=1, max_area=200, min_peak=20)
    # Very high threshold → all should be filtered
    filtered, _, _ = filter_stars_by_snr(stars, img.astype(float),
                                          min_snr=10000.0)
    assert len(filtered) == 0


def test_filter_keeps_good_snr():
    img = make_blank(bg=5)
    img = add_star(img, cy=64, cx=64, peak=220)
    stars = detect_stars(img, pct=90, min_area=1, max_area=200, min_peak=20)
    filtered, _, _ = filter_stars_by_snr(stars, img.astype(float), min_snr=1.0)
    assert len(filtered) == len(stars)


def test_filter_saturated_stars_always_kept():
    """Peak=255 (saturated) stars are always kept regardless of SNR."""
    stars = [(64.0, 64.0, 10, 255)]   # peak=255
    img = np.full((128, 128), 100.0)
    # Very high SNR threshold
    filtered, _, _ = filter_stars_by_snr(stars, img, min_snr=99999.0)
    assert len(filtered) == 1


# ════════════════════════════════════════════════════════════════
#  detect_brightness_outliers
# ════════════════════════════════════════════════════════════════

def make_brightness_test():
    """
    Returns two images with 3 stars:
      - Star 0: stationary
      - Star 1: 5x brighter in epoch2 (brightened)
      - Star 2: faded in epoch2 (dimmed)
    """
    img1 = make_blank(h=200, w=200, bg=5)
    img2 = make_blank(h=200, w=200, bg=5)

    # Star 0 — same in both epochs
    img1 = add_star(img1, cy=50, cx=50, peak=150)
    img2 = add_star(img2, cy=50, cx=50, peak=150)

    # Star 1 — much brighter in epoch2
    img1 = add_star(img1, cy=50, cx=100, peak=80)
    img2 = add_star(img2, cy=50, cx=100, peak=220)

    # Star 2 — faded in epoch2
    img1 = add_star(img1, cy=100, cx=50, peak=200)
    img2 = add_star(img2, cy=100, cx=50, peak=30)

    stars = [(50.0, 50.0, 8, 150),
             (100.0, 50.0, 8, 80),
             (50.0, 100.0, 8, 200)]
    matched = [(0, 0, 0.0, 0.0, 0.0),
               (1, 1, 0.0, 0.0, 0.0),
               (2, 2, 0.0, 0.0, 0.0)]
    return img1.astype(float), img2.astype(float), stars, matched


def test_brightened_detected():
    img1, img2, stars, matched = make_brightness_test()
    brightened, dimmed, stationary, _ = detect_brightness_outliers(
        matched, stars, stars, img1, img2,
        r_inner=4, r_outer=8, flux_ratio_thr=2.5, min_flux=10)
    assert len(brightened) >= 1


def test_dimmed_detected():
    img1, img2, stars, matched = make_brightness_test()
    brightened, dimmed, stationary, _ = detect_brightness_outliers(
        matched, stars, stars, img1, img2,
        r_inner=4, r_outer=8, flux_ratio_thr=2.5, min_flux=10)
    assert len(dimmed) >= 1


def test_stationary_not_flagged():
    img1, img2, stars, matched = make_brightness_test()
    brightened, dimmed, stationary, _ = detect_brightness_outliers(
        matched, stars, stars, img1, img2,
        r_inner=4, r_outer=8, flux_ratio_thr=2.5, min_flux=10)
    assert len(stationary) >= 1


def test_returns_sorted_by_change():
    """Brightened sorted descending, dimmed sorted ascending by change_pct."""
    img1, img2, stars, matched = make_brightness_test()
    brightened, dimmed, _, _ = detect_brightness_outliers(
        matched, stars, stars, img1, img2,
        r_inner=4, r_outer=8, flux_ratio_thr=2.5, min_flux=10)
    if len(brightened) > 1:
        pcts = [r["change_pct"] for r in brightened]
        assert pcts == sorted(pcts, reverse=True)
    if len(dimmed) > 1:
        pcts = [r["change_pct"] for r in dimmed]
        assert pcts == sorted(pcts)


def test_normalize_pair_joint_balances_identical_images():
    """F1/F2 detection counts should be close for identical images (joint normalization)."""
    import cv2 as cv
    positions = [(40, 40), (40, 120), (120, 40), (120, 120), (80, 80), (60, 180)]
    img = make_star_field(positions, peak=230)
    img_sep = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    n_sep1 = len(detect_stars(img_sep, pct=97, min_peak=15))
    n_sep2 = len(detect_stars(img_sep.copy(), pct=97, min_peak=15))
    img1_n, img2_n = normalize_pair_joint(img, img.copy())
    n_j1 = len(detect_stars(img1_n, pct=97, min_peak=15))
    n_j2 = len(detect_stars(img2_n, pct=97, min_peak=15))
    assert abs(n_j1 - n_j2) <= 3
    assert abs(n_j1 - n_j2) <= abs(n_sep1 - n_sep2) + 5


def test_filter_stars_from_border():
    stars = [(5.0, 50.0, 10, 200), (100.0, 100.0, 10, 200)]
    kept = filter_stars_from_border(stars, 256, 256, margin=10)
    assert len(kept) == 1
    assert kept[0][0] == 100.0
