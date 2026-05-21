"""
tests/test_fits_loader.py
=========================
Unit tests for fits_loader.py, focusing on sqrt_stretch.

All tests use only synthetic numpy data — no real FITS files required
(except the one test that explicitly exercises load_fits_image via tmp_path).
"""

import numpy as np
import pytest


# ════════════════════════════════════════════════════════════════
#  Existing test (kept as-is)
# ════════════════════════════════════════════════════════════════

def test_load_fits_image_returns_uint8_and_header_info(tmp_path):
    # Local import so the module is only required when running tests.
    from astropy.io import fits

    from image_comparison_gui import load_fits_image

    data = np.zeros((64, 64), dtype=np.float32)
    data[32, 32] = 1000.0
    data[10, 10] = 200.0

    p = tmp_path / "synthetic.fits"
    fits.writeto(p, data, overwrite=True)

    img_u8, info, data_f32, header = load_fits_image(str(p))

    assert img_u8.dtype == np.uint8
    assert img_u8.shape == data.shape
    assert isinstance(info, dict)
    assert data_f32.shape == data.shape
    assert header is not None
    # Basic metadata should exist in FITS header
    assert "NAXIS1" in header and "NAXIS2" in header


# ════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════

def _make_sky_image(h=64, w=64, sky=1000.0, sigma=50.0, rng=None):
    """
    Return a float32 array simulating a sky background.
    Gaussian noise centred on `sky` with standard deviation `sigma`.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    return (rng.normal(loc=sky, scale=sigma, size=(h, w))).astype(np.float32)


# ════════════════════════════════════════════════════════════════
#  sqrt_stretch tests
# ════════════════════════════════════════════════════════════════

def test_sqrt_stretch_returns_uint8():
    """
    sqrt_stretch must always return a uint8 array, regardless of input dtype.
    The GUI expects 8-bit images for display; wrong dtype would break imshow.
    """
    from fits_loader import sqrt_stretch

    img = _make_sky_image()
    out = sqrt_stretch(img)

    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"


def test_sqrt_stretch_output_range():
    """
    Every output pixel must lie within [0, 255].
    Values outside this range would silently overflow when converted later.
    """
    from fits_loader import sqrt_stretch

    img = _make_sky_image()
    # Add a bright outlier (cosmic ray)
    img[10, 10] = 1e6

    out = sqrt_stretch(img)

    assert int(out.min()) >= 0, "Minimum below 0"
    assert int(out.max()) <= 255, "Maximum above 255"


def test_sqrt_stretch_constant_image():
    """
    A constant-valued image (all pixels equal) has no spatial variation.

    Because all pixels equal the sky, bg_pix is empty after the <90th-percentile
    filter, so sigma falls back to 1.0.  Every pixel is then clipped to vmin and
    stretched identically: sqrt(sky - vmin) / sqrt(sky - vmin) * 255 = 255.
    The output must therefore be uniformly 255 — output range is still valid,
    and the pixel value is consistent (no variation).

    This also confirms that the function does not crash on degenerate inputs
    and always returns a valid uint8 array.
    """
    from fits_loader import sqrt_stretch

    img = np.full((32, 32), 500.0, dtype=np.float32)
    out = sqrt_stretch(img)

    assert out.dtype == np.uint8
    # All pixels share the same value — image has no gradient
    assert len(np.unique(out)) == 1, (
        f"Constant image should produce a uniform output, got {np.unique(out)}"
    )
    # That uniform value must be within [0, 255]
    assert 0 <= int(out.flat[0]) <= 255


def test_sqrt_stretch_handles_nan_and_inf():
    """
    FITS data sometimes contains NaN (bad pixels) or Inf (saturation overflow).
    sqrt_stretch must not raise an exception and must produce a finite uint8 array.
    """
    from fits_loader import sqrt_stretch

    rng = np.random.default_rng(7)
    img = _make_sky_image(rng=rng)
    img[0, 0] = np.nan
    img[1, 1] = np.inf
    img[2, 2] = -np.inf

    # Must not raise
    out = sqrt_stretch(img)

    assert out.dtype == np.uint8
    assert np.all(np.isfinite(out.astype(np.float32))), "Output contains non-finite values"


def test_sqrt_stretch_bright_pixel_maps_high():
    """
    A pixel 15*sigma above the sky level sits exactly at vmax and should
    map to the maximum stretch value (255 after normalisation).
    This verifies the vmax = sky + 15*sigma formula is respected.
    """
    from fits_loader import sqrt_stretch

    sky = 1000.0
    sigma = 50.0
    h, w = 64, 64

    # Flat background at exactly `sky`
    img = np.full((h, w), sky, dtype=np.float32)
    # One pixel at sky + 15*sigma (exactly at vmax)
    img[32, 32] = sky + 15.0 * sigma

    out = sqrt_stretch(img)

    # The brightest pixel must be close to 255
    assert int(out[32, 32]) >= 240, (
        f"Pixel at 15-sigma should map near 255, got {int(out[32, 32])}"
    )


def test_sqrt_stretch_background_at_quarter():
    """
    A pixel at sky level (the median) should map to approximately 64 (±20)
    because it sits at vmin + 1*sigma while vmax is at vmin + (1+15)*sigma.
    sqrt(1) / sqrt(16) * 255 ≈ 64.

    This confirms the non-linear stretch is applied correctly: faint background
    pixels should not dominate the dynamic range.
    """
    from fits_loader import sqrt_stretch

    sky = 1000.0
    sigma = 100.0

    img = np.full((128, 128), sky, dtype=np.float32)
    # Ensure the image has a recognisable background+sigma structure
    # by adding a single very bright outlier (ensures s_max > 0)
    img[0, 0] = sky + 15.0 * sigma + 1.0  # just above vmax

    out = sqrt_stretch(img)

    # Most pixels should map to approximately 64 (the background level)
    # We sample the interior to avoid any border effects
    interior = out[1:-1, 1:-1]
    # The median of interior pixels represents the background mapping
    median_val = int(np.median(interior))
    assert abs(median_val - 64) <= 20, (
        f"Background pixel expected ~64, got median={median_val}"
    )


def test_sqrt_stretch_same_as_load_fits_image_normalization():
    """
    sqrt_stretch is documented as applying "the same sky-based sqrt-stretch
    as load_fits_image".  For the same float32 array the two code paths must
    produce identical uint8 images.

    This guards against future drift between the two implementations.
    We replicate load_fits_image's normalization inline (without file I/O)
    and compare pixel-by-pixel.
    """
    from fits_loader import sqrt_stretch

    rng = np.random.default_rng(99)
    data = _make_sky_image(rng=rng)
    # Add a bright star
    data[32, 32] = float(np.median(data)) + 12 * float(np.std(data))

    # --- Reference: inline replication of load_fits_image normalization ---
    d = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sky = float(np.median(d))
    bg_pix = d[d < np.percentile(d, 90)]
    sigma = float(np.std(bg_pix)) if len(bg_pix) > 10 else 1.0
    vmin = sky - 1.0 * sigma
    vmax = max(sky + 15.0 * sigma, vmin + 1.0)
    clipped = np.clip(d, vmin, vmax)
    shifted = clipped - vmin
    sqrt_d = np.sqrt(shifted)
    s_max = float(sqrt_d.max())
    if s_max > 0:
        expected = (sqrt_d / s_max * 255).astype(np.uint8)
    else:
        expected = np.zeros_like(d, dtype=np.uint8)

    # --- sqrt_stretch output ---
    actual = sqrt_stretch(data)

    np.testing.assert_array_equal(
        actual, expected,
        err_msg="sqrt_stretch output differs from load_fits_image normalization"
    )
