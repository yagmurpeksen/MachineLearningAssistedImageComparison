"""
tests/test_change_detection.py
==============================
Unit tests for the astronomical image change-detection pipeline.

Coverage:
  1. Precision / recall on synthetic transient data  (most important for thesis)
  2. match_detection_depth  — star-count balancing
  3. suppress_warp_border   — edge artifact masking
  4. detect_moving_from_unmatched — unmatched-pair moving object detection
All tests use only synthetic numpy data — no real FITS files are required.
"""

import numpy as np
import pytest

from detection import (
    suppress_warp_border,
    match_detection_depth,
    detect_moving_from_unmatched,
    detect_stars,
    match_stars_mutual,
    detect_brightness_outliers,
)


# ════════════════════════════════════════════════════════════════
#  Shared helpers
# ════════════════════════════════════════════════════════════════

def _blank(h=256, w=256, bg=10):
    """Return a uint8 image filled with a flat background value."""
    return np.full((h, w), bg, dtype=np.uint8)


def _add_star(img, cy, cx, peak=220, sigma=2.0):
    """
    Add a 2-D Gaussian star centred at (cx, cy).
    Works on uint8 or float arrays.
    """
    arr = img.astype(float)
    r = int(sigma * 5)
    y0 = max(0, cy - r); y1 = min(arr.shape[0], cy + r + 1)
    x0 = max(0, cx - r); x1 = min(arr.shape[1], cx + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    arr[y0:y1, x0:x1] += peak * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
    )
    return np.clip(arr, 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════
#  1.  Synthetic pair factory
# ════════════════════════════════════════════════════════════════

def make_synthetic_pair(n_stars=50, n_changes=5, img_size=256, seed=0):
    """
    Build a pre/post image pair for recall/precision tests.

    Transient simulation strategy
    ------------------------------
    ``detect_brightness_outliers`` only examines *matched* star pairs, so a
    source that is completely absent in epoch1 would fall into ``unm2`` and
    never be evaluated.  We therefore simulate transients the same way real
    astrophysical events work in the pipeline: a faint pre-image source
    (below reliable detection, peak≈15) that dramatically brightens in
    epoch2 (peak≈230).  This means both epochs have a detectable star at
    that position, they get matched by mutual-NN, and the flux ratio
    triggers the brightening flag.

    Parameters
    ----------
    n_stars   : int  — stable background stars present in both epochs
    n_changes : int  — variable/transient positions (faint→bright)
    img_size  : int  — image dimension (square)
    seed      : int  — RNG seed for reproducibility

    Returns
    -------
    img1         : uint8 ndarray (pre-image)
    img2         : uint8 ndarray (post-image with brightened transients)
    stars1       : list of (cx, cy, area, peak) — stars detected in img1
    matched      : list of (i, j, dx, dy, dist) — mutual-NN matched pairs
    gt_positions : list of (cx, cy) — ground-truth centres of transients
    stars2       : list of (cx, cy, area, peak) — stars detected in img2
    unm1, unm2   : unmatched index lists
    """
    rng = np.random.default_rng(seed)
    margin = 30  # keep stars away from border

    # --- Place n_stars stable background stars ---
    positions = []
    while len(positions) < n_stars:
        cx = int(rng.integers(margin, img_size - margin))
        cy = int(rng.integers(margin, img_size - margin))
        # Avoid overlap (>12 px apart)
        if all(abs(cx - p[0]) > 12 or abs(cy - p[1]) > 12 for p in positions):
            positions.append((cx, cy))

    img1 = _blank(img_size, img_size, bg=10)
    img2 = _blank(img_size, img_size, bg=10)
    for cx, cy in positions:
        peak = int(rng.integers(80, 170))
        img1 = _add_star(img1, cy=cy, cx=cx, peak=peak)
        img2 = _add_star(img2, cy=cy, cx=cx, peak=peak)

    # --- Inject n_changes transients (faint in epoch1, bright in epoch2) ---
    # Both epochs get a star at the same position so mutual-NN can match them.
    # The large peak ratio (≈15 vs 230, factor ~15) exceeds flux_ratio_thr=2.0.
    gt_positions = []
    attempts = 0
    while len(gt_positions) < n_changes and attempts < 1000:
        attempts += 1
        cx = int(rng.integers(margin, img_size - margin))
        cy = int(rng.integers(margin, img_size - margin))
        if all(abs(cx - p[0]) > 15 or abs(cy - p[1]) > 15
               for p in positions + gt_positions):
            img1 = _add_star(img1, cy=cy, cx=cx, peak=15)   # faint pre-image
            img2 = _add_star(img2, cy=cy, cx=cx, peak=230)  # bright post-image
            gt_positions.append((cx, cy))

    # --- Detect stars in both epochs ---
    stars1 = detect_stars(img1, pct=88, min_area=1, max_area=500, min_peak=10)
    stars2 = detect_stars(img2, pct=88, min_area=1, max_area=500, min_peak=10)
    matched, unm1, unm2 = match_stars_mutual(stars1, stars2, max_distance=10)

    return img1, img2, stars1, matched, gt_positions, stars2, unm1, unm2


# ════════════════════════════════════════════════════════════════
#  1a.  Precision / Recall tests
# ════════════════════════════════════════════════════════════════

def test_recall_detects_majority_of_changes():
    """
    Plant 5 variable-star transients (faint→bright) and run the
    brightness-outlier detector.  At least 4 of them (≥80% recall) must
    appear in the 'brightened' list — this is the core thesis metric.

    Transients are simulated as matched pairs (peak≈15 in epoch1, ≈230 in
    epoch2) so that detect_brightness_outliers can evaluate them.  The
    flux ratio of ~15× far exceeds flux_ratio_thr=2.0.
    """
    img1, img2, stars1, matched, gt_positions, stars2, _, _ = make_synthetic_pair(
        n_stars=40, n_changes=5, img_size=256, seed=1
    )

    brightened, dimmed, stationary, _ = detect_brightness_outliers(
        matched, stars1, stars2,
        img1.astype(float), img2.astype(float),
        r_inner=4, r_outer=8,
        flux_ratio_thr=2.0,
        min_flux=5,
    )

    # Count how many ground-truth positions are close to a detected brightening
    tolerance_px = 12
    detected = 0
    for gx, gy in gt_positions:
        for entry in brightened:
            if abs(entry["x2"] - gx) < tolerance_px and abs(entry["y2"] - gy) < tolerance_px:
                detected += 1
                break

    recall = detected / len(gt_positions)
    assert recall >= 0.8, (
        f"Recall {recall:.2f} < 0.80: only {detected}/{len(gt_positions)} "
        "transients detected"
    )


def test_no_false_positives_on_identical_images():
    """
    When pre-image and post-image are identical, the brightness-change
    detector must report zero 'brightened' events.
    Identical images share the same global flux ratio (≈1), so no source
    should cross the flux_ratio_thr.
    """
    rng = np.random.default_rng(5)
    img = _blank(128, 128, bg=10)
    for _ in range(20):
        cx = int(rng.integers(20, 108))
        cy = int(rng.integers(20, 108))
        img = _add_star(img, cy=cy, cx=cx, peak=int(rng.integers(80, 200)))

    stars = detect_stars(img, pct=90, min_area=1, max_area=500, min_peak=30)
    matched, _, _ = match_stars_mutual(stars, stars, max_distance=5)

    brightened, _, _, _ = detect_brightness_outliers(
        matched, stars, stars,
        img.astype(float), img.astype(float),
        r_inner=4, r_outer=8,
        flux_ratio_thr=2.5,
        min_flux=10,
    )

    assert len(brightened) == 0, (
        f"Expected 0 false positives on identical images, got {len(brightened)}"
    )


def test_precision_no_spurious_detections_stable_field():
    """
    A perfectly stable star field (same stars, same brightness) should
    produce zero detections — this validates precision at the pipeline level.
    A high false-positive rate on stable fields would make the system
    unusable for survey data.
    """
    img1, img2, stars1, matched, gt_positions, stars2, _, _ = make_synthetic_pair(
        n_stars=30, n_changes=0, img_size=200, seed=42
    )

    brightened, dimmed, stationary, _ = detect_brightness_outliers(
        matched, stars1, stars2,
        img1.astype(float), img2.astype(float),
        r_inner=4, r_outer=8,
        flux_ratio_thr=2.5,
        min_flux=30,
    )

    assert len(brightened) == 0, (
        f"Expected 0 detections in stable field, got {len(brightened)}"
    )


def test_single_bright_transient_always_detected():
    """
    A single very bright transient (faint in epoch1, peak≈230 in epoch2)
    injected into a stable field should always be detected by the
    brightness-outlier detector.

    Both epochs contain a source at the transient position so that
    mutual-NN matching succeeds.  The flux ratio (~15×) greatly exceeds
    flux_ratio_thr=2.0, so the event must always appear in 'brightened'.
    Failing this test means the system misses the most obvious events.
    """
    img1 = _blank(128, 128, bg=10)
    img2 = _blank(128, 128, bg=10)

    # Stable stars in both epochs
    bg_stars = [(30, 30), (30, 90), (90, 30), (90, 90), (60, 60)]
    for cx, cy in bg_stars:
        img1 = _add_star(img1, cy=cy, cx=cx, peak=120)
        img2 = _add_star(img2, cy=cy, cx=cx, peak=120)

    # Transient: faint pre-image, bright post-image — same position so they match
    transient_cx, transient_cy = 60, 90
    img1 = _add_star(img1, cy=transient_cy, cx=transient_cx, peak=15)
    img2 = _add_star(img2, cy=transient_cy, cx=transient_cx, peak=230)

    stars1 = detect_stars(img1, pct=85, min_area=1, max_area=500, min_peak=10)
    stars2 = detect_stars(img2, pct=85, min_area=1, max_area=500, min_peak=10)
    matched, _, _ = match_stars_mutual(stars1, stars2, max_distance=10)

    brightened, _, _, _ = detect_brightness_outliers(
        matched, stars1, stars2,
        img1.astype(float), img2.astype(float),
        r_inner=4, r_outer=8,
        flux_ratio_thr=2.0,
        min_flux=5,
    )

    tolerance_px = 12
    found = any(
        abs(e["x2"] - transient_cx) < tolerance_px
        and abs(e["y2"] - transient_cy) < tolerance_px
        for e in brightened
    )
    assert found, (
        f"Single bright transient at ({transient_cx},{transient_cy}) not detected. "
        f"Brightened list: {[(e['x2'], e['y2']) for e in brightened]}"
    )


# ════════════════════════════════════════════════════════════════
#  2.  match_detection_depth
# ════════════════════════════════════════════════════════════════

def _make_stars(n, base_peak=100):
    """Return n fake (cx, cy, area, peak) tuples with decreasing brightness."""
    return [(float(i * 5), float(i * 5), 5, base_peak - i) for i in range(n)]


def test_match_detection_depth_caps_larger_list():
    """
    If stars1 has 100 entries and stars2 has 10, with max_ratio=2.0
    stars1 must be trimmed to at most 10 × 2.0 = 20 entries.
    This prevents the brightness-outlier step from being swamped by
    detections that have no counterpart in the shallower epoch.
    """
    stars1 = _make_stars(100, base_peak=200)
    stars2 = _make_stars(10, base_peak=200)

    out1, out2 = match_detection_depth(stars1, stars2, max_ratio=2.0)

    assert len(out1) <= 10 * 2.0 + 1, (  # +1 for int rounding
        f"Expected ≤20, got {len(out1)}"
    )
    assert len(out2) == 10, "Shorter list should be unchanged"


def test_match_detection_depth_equal_lists_unchanged():
    """
    When both lists have the same length the ratio is 1.0 ≤ max_ratio,
    so neither list should be modified.
    """
    stars1 = _make_stars(15)
    stars2 = _make_stars(15)

    out1, out2 = match_detection_depth(stars1, stars2, max_ratio=2.0)

    assert len(out1) == 15
    assert len(out2) == 15


def test_match_detection_depth_keeps_brightest():
    """
    When truncating the longer list, the function must keep the brightest
    stars (sorted by peak descending), not an arbitrary subset.
    The brightest stars are the most reliable for alignment and matching.
    """
    # stars1[0] has peak=199 (brightest), stars1[99] has peak=100 (faintest)
    stars1 = _make_stars(100, base_peak=200)
    stars2 = _make_stars(10, base_peak=200)

    out1, out2 = match_detection_depth(stars1, stars2, max_ratio=2.0)

    peaks = [s[3] for s in out1]
    assert peaks == sorted(peaks, reverse=True), (
        "Trimmed list should be sorted brightest-first"
    )
    # The brightest star in the original list must survive
    max_original_peak = max(s[3] for s in stars1)
    assert max_original_peak in peaks, (
        "The brightest star must be kept after trimming"
    )


def test_match_detection_depth_empty_input():
    """
    Empty inputs are a legitimate edge case (no stars detected in one epoch).
    The function must return the inputs unchanged without raising.
    """
    stars1 = _make_stars(5)
    out1, out2 = match_detection_depth(stars1, [], max_ratio=2.0)
    assert out1 == stars1
    assert out2 == []


# ════════════════════════════════════════════════════════════════
#  3.  suppress_warp_border
# ════════════════════════════════════════════════════════════════

def test_suppress_warp_border_zeros_edges():
    """
    After affine warping the border pixels are interpolation artifacts.
    suppress_warp_border(img, margin) must set all border pixels to zero
    so they cannot trigger false detections.
    """
    img = np.full((100, 100), 200, dtype=np.uint8)
    margin = 12
    out = suppress_warp_border(img, margin=margin)

    # All four border bands must be zero
    assert np.all(out[:margin, :] == 0), "Top border not zeroed"
    assert np.all(out[-margin:, :] == 0), "Bottom border not zeroed"
    assert np.all(out[:, :margin] == 0), "Left border not zeroed"
    assert np.all(out[:, -margin:] == 0), "Right border not zeroed"


def test_suppress_warp_border_preserves_center():
    """
    Pixels strictly inside the border zone must not be modified.
    Zeroing out the centre would corrupt real astronomical data.
    """
    img = np.full((100, 100), 200, dtype=np.uint8)
    margin = 12
    out = suppress_warp_border(img, margin=margin)

    center = out[margin:-margin, margin:-margin]
    assert np.all(center == 200), (
        f"Center pixels were modified; expected 200 everywhere, got min={center.min()}"
    )


def test_suppress_warp_border_all_zero_image():
    """
    An image that is already all-zero must remain all-zero after suppression.
    This verifies the function does not accidentally introduce non-zero values.
    """
    img = np.zeros((64, 64), dtype=np.uint8)
    out = suppress_warp_border(img, margin=10)
    assert np.all(out == 0)


def test_suppress_warp_border_zero_margin_unchanged():
    """
    margin=0 is a no-op — the image must be returned unchanged.
    The function documents 'if margin <= 0: return img'.
    """
    img = np.full((50, 50), 128, dtype=np.uint8)
    out = suppress_warp_border(img, margin=0)
    np.testing.assert_array_equal(out, img)


# ════════════════════════════════════════════════════════════════
#  4.  detect_moving_from_unmatched
# ════════════════════════════════════════════════════════════════

def test_moving_from_unmatched_finds_displaced_source():
    """
    A source at (50, 50) in epoch1 that reappears at (55, 55) in epoch2
    (displacement = ~7px, within max_motion=200 and outside min_motion=5)
    should be detected as a moving object candidate.

    This is the core asteroid/satellite detection scenario: mutual-NN
    matching fails when the object moves far enough, so this fallback
    searches the unmatched pool.
    """
    # No global shift (telescope was perfectly tracked)
    stars1 = [(50.0, 50.0, 5, 150)]
    stars2 = [(55.0, 55.0, 5, 150)]
    unm1 = [0]
    unm2 = [0]

    results = detect_moving_from_unmatched(
        stars1, stars2, unm1, unm2,
        med_dx=0.0, med_dy=0.0,
        min_motion=5.0, max_motion=200.0,
    )

    assert len(results) == 1, f"Expected 1 candidate, got {len(results)}"
    r = results[0]
    assert abs(r["x1"] - 50.0) < 0.1
    assert abs(r["x2"] - 55.0) < 0.1
    assert r["from_unmatched"] is True


def test_moving_from_unmatched_no_match_outside_radius():
    """
    A source at (50, 50) in epoch1 and (200, 200) in epoch2 are far apart
    (displacement ~212px > max_motion=100) — they must not be paired.
    Matching across large distances would produce spurious cosmic-ray hits.
    """
    stars1 = [(50.0, 50.0, 5, 150)]
    stars2 = [(200.0, 200.0, 5, 150)]
    unm1 = [0]
    unm2 = [0]

    results = detect_moving_from_unmatched(
        stars1, stars2, unm1, unm2,
        med_dx=0.0, med_dy=0.0,
        min_motion=5.0, max_motion=100.0,
    )

    assert len(results) == 0, (
        f"Expected 0 candidates (displacement > max_motion), got {len(results)}"
    )


def test_moving_from_unmatched_empty_inputs():
    """
    Empty unm1 or unm2 (all stars matched) must return an empty list
    without raising any exception.
    """
    stars1 = [(10.0, 10.0, 5, 100)]
    stars2 = [(10.0, 10.0, 5, 100)]

    results_empty_unm1 = detect_moving_from_unmatched(
        stars1, stars2, [], [0],
        med_dx=0.0, med_dy=0.0,
    )
    results_empty_unm2 = detect_moving_from_unmatched(
        stars1, stars2, [0], [],
        med_dx=0.0, med_dy=0.0,
    )
    results_both_empty = detect_moving_from_unmatched(
        stars1, stars2, [], [],
        med_dx=0.0, med_dy=0.0,
    )

    assert results_empty_unm1 == []
    assert results_empty_unm2 == []
    assert results_both_empty == []


def test_moving_from_unmatched_result_dict_keys():
    """
    Each result dict must contain the documented keys so that downstream
    code (GUI display, ML features) can access them without KeyError.
    """
    stars1 = [(50.0, 50.0, 5, 150)]
    stars2 = [(60.0, 60.0, 5, 150)]
    unm1 = [0]
    unm2 = [0]

    results = detect_moving_from_unmatched(
        stars1, stars2, unm1, unm2,
        med_dx=0.0, med_dy=0.0,
        min_motion=5.0, max_motion=200.0,
    )

    assert len(results) == 1
    required_keys = {"x1", "y1", "x2", "y2", "dx", "dy", "residual", "from_unmatched"}
    for key in required_keys:
        assert key in results[0], f"Missing key '{key}' in result dict"
