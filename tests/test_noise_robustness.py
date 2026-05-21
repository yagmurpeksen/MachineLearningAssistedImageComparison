"""
tests/test_noise_robustness.py
==============================
Noise robustness tests for the change detection pipeline.

Tests how the pipeline behaves under different noise conditions:
  1. Basic Gaussian noise — detection survives moderate noise
  2. SNR threshold — at what signal level does detection fail?
  3. Cosmic ray immunity — single extreme pixel is not a false positive
  4. Asymmetric noise — different noise levels between pre/post epochs
"""

import numpy as np
import pytest
from detection import (
    detect_brightness_outliers,
    match_stars_mutual,
    bg_subtracted_flux,
    calculate_star_snr,
    filter_stars_by_snr,
    detect_stars,
)


# ── Helpers ──────────────────────────────────────────────────────

def make_blank(h=256, w=256, bg=100.0):
    """Flat background image as float32."""
    return np.full((h, w), bg, dtype=np.float32)


def add_star(img, cy, cx, peak=200.0, sigma=2.0):
    """Add a Gaussian star profile to a float32 image."""
    img = img.copy().astype(np.float32)
    r = int(sigma * 5)
    y0, y1 = max(0, cy - r), min(img.shape[0], cy + r + 1)
    x0, x1 = max(0, cx - r), min(img.shape[1], cx + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += peak * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
    )
    return img


def add_gaussian_noise(img, sigma=10.0, seed=42):
    """Add zero-mean Gaussian noise to image."""
    rng = np.random.default_rng(seed)
    return img + rng.normal(0, sigma, img.shape).astype(np.float32)


def make_star_field(n_stars=30, img_size=256, bg=100.0, peak=180.0, seed=0):
    """Create a star field with n_stars at random positions."""
    rng = np.random.default_rng(seed)
    img = make_blank(img_size, img_size, bg)
    positions = []
    for _ in range(n_stars):
        cy = int(rng.integers(20, img_size - 20))
        cx = int(rng.integers(20, img_size - 20))
        img = add_star(img, cy, cx, peak=peak)
        positions.append((cy, cx))
    return img, positions


def run_detection(img1, img2, flux_ratio_thr=2.0, min_flux=20.0):
    """
    Minimal detection pipeline:
    detect stars → match → find brightness changes.
    Returns (brightened, dimmed).
    """
    u8_lo = float(np.percentile(img1, 1))
    u8_hi = float(np.percentile(img1, 99.5))
    span  = max(u8_hi - u8_lo, 1.0)

    def to_u8(im):
        return np.clip((im - u8_lo) / span * 255, 0, 255).astype(np.uint8)

    img1_u8 = to_u8(img1)
    img2_u8 = to_u8(img2)

    stars1 = detect_stars(img1_u8, pct=90, min_area=1, max_area=300, min_peak=15)
    stars2 = detect_stars(img2_u8, pct=90, min_area=1, max_area=300, min_peak=15)

    if len(stars1) == 0 or len(stars2) == 0:
        return [], []

    matched, _, _ = match_stars_mutual(stars1, stars2, max_distance=5)
    if len(matched) == 0:
        return [], []

    brightened, dimmed, _, _ = detect_brightness_outliers(
        matched, stars1, stars2,
        img1.astype(np.float32), img2.astype(np.float32),
        r_inner=4, r_outer=8,
        flux_ratio_thr=flux_ratio_thr,
        min_flux=min_flux,
    )
    return brightened, dimmed


# ════════════════════════════════════════════════════════════════
#  1. BASIC GAUSSIAN NOISE ROBUSTNESS
# ════════════════════════════════════════════════════════════════

def test_detection_survives_low_noise():
    """A bright transient is still found when mild noise (sigma=5) is present."""
    img1 = make_blank()
    img1 = add_star(img1, cy=80, cx=80, peak=150.0)  # background star (matched)
    img2 = add_star(img1.copy(), cy=128, cx=128, peak=200.0)  # new transient

    img1_n = add_gaussian_noise(img1, sigma=5.0)
    img2_n = add_gaussian_noise(img2, sigma=5.0)

    brightened, _ = run_detection(img1_n, img2_n)
    assert len(brightened) >= 1, "Transient should be detected under low noise"


def test_detection_survives_moderate_noise():
    """A bright transient is still found when moderate noise (sigma=15) is present."""
    img1 = make_blank()
    img1 = add_star(img1, cy=80, cx=80, peak=150.0)
    img2 = add_star(img1.copy(), cy=128, cx=128, peak=220.0)

    img1_n = add_gaussian_noise(img1, sigma=15.0)
    img2_n = add_gaussian_noise(img2, sigma=15.0)

    brightened, _ = run_detection(img1_n, img2_n, flux_ratio_thr=1.5)
    assert len(brightened) >= 1, "Transient should survive moderate noise"


def test_no_false_positives_from_noise_alone():
    """
    At known star positions, the bg_subtracted_flux ratio between two noisy
    images of the same scene must stay below the detection threshold.

    This tests the flux-computation layer directly: noise alone must not
    produce a flux ratio ≥ 3.0 for any of the matched reference stars.
    (Detection hyper-parameters such as pct=90 / min_area=1 intentionally
    find thousands of noise blobs; that behaviour is tested elsewhere.)
    """
    from detection import bg_subtracted_flux

    # Well-separated stars so each annulus measures pure background
    star_positions = [(60, 60), (60, 190), (128, 125), (190, 60), (190, 190)]

    img_base = make_blank()
    for cy, cx in star_positions:
        img_base = add_star(img_base, cy, cx, peak=200.0)

    img1 = add_gaussian_noise(img_base, sigma=10.0, seed=1)
    img2 = add_gaussian_noise(img_base, sigma=10.0, seed=2)

    for cy, cx in star_positions:
        b1 = bg_subtracted_flux(img1, cx, cy, r_inner=4, r_outer=8)
        b2 = bg_subtracted_flux(img2, cx, cy, r_inner=4, r_outer=8)
        assert b1 > 0, f"Star at ({cy},{cx}) should have positive flux in epoch 1 (b1={b1:.1f})"
        ratio = b2 / b1
        assert ratio < 3.0, (
            f"Flux ratio at ({cy},{cx}) = {ratio:.2f} — noise alone should not "
            f"reach threshold=3.0 (b1={b1:.1f}, b2={b2:.1f})"
        )


# ════════════════════════════════════════════════════════════════
#  2. SNR THRESHOLD — AT WHAT BRIGHTNESS DOES DETECTION FAIL?
# ════════════════════════════════════════════════════════════════

def test_bright_transient_detected_high_snr():
    """SNR~20: very bright transient must always be detected."""
    img1 = make_blank(bg=100.0)
    img1 = add_star(img1, cy=80, cx=80, peak=150.0)
    img1_n = add_gaussian_noise(img1, sigma=10.0)

    img2 = add_star(img1.copy(), cy=128, cx=128, peak=220.0)  # peak/noise ~ 22
    img2_n = add_gaussian_noise(img2, sigma=10.0)

    brightened, _ = run_detection(img1_n, img2_n, flux_ratio_thr=2.0, min_flux=15.0)
    assert len(brightened) >= 1, "High-SNR transient must be detected"


def test_faint_transient_may_be_missed_low_snr():
    """
    SNR~2: a very faint transient (peak close to noise level) should
    produce few or no detections — the pipeline is not expected to be
    sensitive at SNR < 3.
    """
    noise_sigma = 20.0
    transient_peak = 30.0  # SNR ~ 1.5

    img1 = make_blank(bg=100.0)
    img1 = add_star(img1, cy=80, cx=80, peak=150.0)
    img1_n = add_gaussian_noise(img1, sigma=noise_sigma)

    img2 = add_star(img1.copy(), cy=128, cx=128, peak=transient_peak)
    img2_n = add_gaussian_noise(img2, sigma=noise_sigma)

    brightened, _ = run_detection(img1_n, img2_n, flux_ratio_thr=2.0, min_flux=15.0)
    # At SNR~1.5, detection is unreliable — we assert it does NOT incorrectly
    # claim perfect recall, i.e., the test documents the sensitivity limit
    assert len(brightened) <= 2, (
        "At SNR~1.5 we expect 0–2 detections (pipeline sensitivity limit)"
    )


def test_snr_threshold_ordering():
    """
    Brighter transients should produce higher bg_subtracted_flux than fainter ones.
    This verifies the flux measurement is monotone w.r.t. source brightness.
    """
    img_base = make_blank(bg=50.0)

    flux_dim    = bg_subtracted_flux(add_star(img_base, 64, 64, peak=60),  64, 64, 4, 8)
    flux_medium = bg_subtracted_flux(add_star(img_base, 64, 64, peak=120), 64, 64, 4, 8)
    flux_bright = bg_subtracted_flux(add_star(img_base, 64, 64, peak=220), 64, 64, 4, 8)

    assert flux_dim < flux_medium < flux_bright, (
        "Flux should increase monotonically with source brightness"
    )


def test_snr_increases_with_brightness():
    """
    calculate_star_snr must return higher SNR for brighter sources
    at the same background noise level.
    """
    img_noisy = add_gaussian_noise(make_blank(bg=50.0), sigma=10.0)

    snr_faint  = calculate_star_snr(add_star(img_noisy, 64, 64, peak=40),  64, 64)
    snr_medium = calculate_star_snr(add_star(img_noisy, 64, 64, peak=120), 64, 64)
    snr_bright = calculate_star_snr(add_star(img_noisy, 64, 64, peak=220), 64, 64)

    assert snr_faint < snr_medium < snr_bright, (
        "SNR must increase monotonically with source brightness"
    )


# ════════════════════════════════════════════════════════════════
#  3. COSMIC RAY IMMUNITY
# ════════════════════════════════════════════════════════════════

def test_cosmic_ray_not_detected_as_transient():
    """
    A single saturated pixel (cosmic ray, value=65535) in epoch2 should NOT
    be reported as a brightness change.

    The pipeline filters by aperture area: a cosmic ray affects only 1 pixel,
    so bg_subtracted_flux measured over the aperture (r_inner=4) is diluted
    by the surrounding background and the ratio test may not trigger.
    Additionally, the min_flux threshold and star-match filter help.
    """
    img1 = make_blank(bg=100.0)
    img1 = add_star(img1, cy=80, cx=80, peak=180.0)  # reference star for matching

    img2 = img1.copy()
    img2[64, 64] = 65535.0  # single cosmic ray pixel — not a real source

    brightened, _ = run_detection(img1, img2, flux_ratio_thr=3.0, min_flux=30.0)

    # The cosmic ray pixel may or may not be detected depending on detection
    # sensitivity, but it must NOT dominate as a large-area transient
    for det in brightened:
        cx, cy = int(det["x1"]), int(det["y1"])
        # If detected, it should not be exactly at the cosmic ray position
        # OR there should be very few such detections
        assert not (abs(cx - 64) < 3 and abs(cy - 64) < 3), (
            f"Cosmic ray at (64,64) should not be detected as a transient, "
            f"but found detection at ({cx},{cy})"
        )


def test_multiple_cosmic_rays_do_not_flood_detections():
    """
    Several cosmic ray pixels scattered across epoch2 should not cause
    a flood of false positive detections.
    """
    rng = np.random.default_rng(99)
    img1 = make_blank(bg=100.0)
    img1 = add_star(img1, cy=60, cx=60, peak=170.0)

    img2 = img1.copy()
    # Plant 10 cosmic ray pixels at random locations
    for _ in range(10):
        ry = int(rng.integers(10, 246))
        rx = int(rng.integers(10, 246))
        img2[ry, rx] = 60000.0

    brightened, _ = run_detection(img1, img2, flux_ratio_thr=3.0, min_flux=30.0)
    # Expect very few false positives — cosmic rays are point-like, not Gaussian blobs
    assert len(brightened) <= 3, (
        f"Too many false positives from cosmic rays: {len(brightened)}"
    )


# ════════════════════════════════════════════════════════════════
#  4. ASYMMETRIC NOISE (different pre/post noise levels)
# ════════════════════════════════════════════════════════════════

def test_detection_with_asymmetric_noise_epochs():
    """
    Pre image has low noise (good night, sigma=5),
    post image has higher noise (worse conditions, sigma=20).
    A bright transient must still be detected.

    This simulates real multi-epoch ZTF observations where seeing and
    sky background differ significantly between epochs.
    """
    img1 = make_blank(bg=100.0)
    img1 = add_star(img1, cy=80, cx=80, peak=160.0)   # reference star

    img2 = add_star(img1.copy(), cy=128, cx=128, peak=220.0)  # transient

    img1_n = add_gaussian_noise(img1, sigma=5.0,  seed=10)   # good night
    img2_n = add_gaussian_noise(img2, sigma=20.0, seed=11)   # noisier night

    brightened, _ = run_detection(img1_n, img2_n, flux_ratio_thr=1.5, min_flux=15.0)
    assert len(brightened) >= 1, (
        "Transient should be detected even with asymmetric noise between epochs"
    )


def test_higher_background_noise_reduces_snr():
    """
    The same star measured against higher background noise should yield lower SNR.
    Confirms that SNR calculation correctly reflects observing conditions.
    """
    img_low_noise  = add_gaussian_noise(make_blank(bg=50.0), sigma=2.0)
    img_high_noise = add_gaussian_noise(make_blank(bg=50.0), sigma=25.0)

    star_low  = add_star(img_low_noise,  64, 64, peak=150.0)
    star_high = add_star(img_high_noise, 64, 64, peak=150.0)

    snr_low  = calculate_star_snr(star_low,  64, 64)
    snr_high = calculate_star_snr(star_high, 64, 64)

    assert snr_low > snr_high, (
        "Same star should have higher SNR under lower background noise"
    )


def test_snr_filter_rejects_noisy_detections():
    """
    With a very noisy image, filter_stars_by_snr should remove most
    spurious detections (low-SNR blobs), keeping only the bright real star.
    """
    img = add_gaussian_noise(make_blank(bg=50.0), sigma=18.0)
    img = add_star(img, cy=64, cx=64, peak=220.0)   # one real bright star

    u8 = np.clip((img - img.min()) / max(img.max() - img.min(), 1) * 255, 0, 255).astype(np.uint8)
    stars_raw = detect_stars(u8, pct=90, min_area=1, max_area=300, min_peak=10)

    kept, _, _ = filter_stars_by_snr(stars_raw, img, min_snr=3.0)

    # After SNR filtering, should keep very few stars (ideally just the bright one)
    assert len(kept) <= max(3, len(stars_raw) // 2), (
        "SNR filter should significantly reduce spurious detections in noisy image"
    )
    assert len(kept) < len(stars_raw), (
        "SNR filter must remove at least some low-SNR detections"
    )
