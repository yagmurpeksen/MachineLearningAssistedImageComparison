"""
fits_loader.py
==============
FITS file loading and normalization.

Single responsibility: raw FITS → (uint8 image, header info, float32 data)
"""

import numpy as np

try:
    from astropy.io import fits as astropy_fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False


def sqrt_stretch(data: np.ndarray) -> np.ndarray:
    """
    Apply the same sky-based sqrt-stretch as load_fits_image to a float32 array.
    Returns uint8 [0-255].  Safe against cosmic rays and hot pixels.
    """
    data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sky    = float(np.median(data))
    bg_pix = data[data < np.percentile(data, 90)]
    sigma  = float(np.std(bg_pix)) if len(bg_pix) > 10 else 1.0
    vmin   = sky - 1.0 * sigma
    vmax   = max(sky + 15.0 * sigma, vmin + 1.0)
    shifted = np.sqrt(np.clip(data, vmin, vmax) - vmin)
    s_max   = float(shifted.max())
    if s_max > 0:
        return (shifted / s_max * 255).astype(np.uint8)
    return np.zeros_like(data, dtype=np.uint8)


def load_fits_image(path):
    """
    Loads a FITS file and prepares it for display.

    Parameters
    ----------
    path : str
        Full path to the FITS file.

    Returns
    -------
    data_u8 : numpy.ndarray (uint8)
        Sqrt-stretched image normalized to 0-255.
    info : dict
        Key-value pairs extracted from the header
        (EXPTIME, DATE-OBS, NAXIS1, NAXIS2, BITPIX ...).
    data : numpy.ndarray (float32)
        Raw pixel values (for flux measurements).
    header : astropy.io.fits.Header
        Full FITS header (for WCS computations).
    """
    if not FITS_AVAILABLE:
        raise ImportError("astropy is not installed.  pip install astropy")

    with astropy_fits.open(path) as hdul:
        data, header = None, None
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) >= 2:
                data   = hdu.data.astype(np.float32)
                header = hdu.header
                break
        if data is None:
            raise ValueError("No image data found in FITS file.")

    # If 3-dimensional cube, take the first frame  (e.g. [1, H, W])
    if data.ndim == 3:
        data = data[0]

    # Clean NaN / Inf
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Find sky background (median), bring out stars
    sky    = float(np.median(data))
    bg_pix = data[data < np.percentile(data, 90)]
    sigma  = float(np.std(bg_pix)) if len(bg_pix) > 10 else 1.0
    vmin   = sky - 1.0 * sigma
    vmax   = sky + 15.0 * sigma
    vmax   = max(vmax, vmin + 1.0)

    clipped = np.clip(data, vmin, vmax)
    # Sqrt stretch: makes both bright and faint stars visible
    shifted = clipped - vmin
    sqrt_d  = np.sqrt(shifted)
    s_max   = float(sqrt_d.max())
    if s_max > 0:
        data_u8 = (sqrt_d / s_max * 255).astype(np.uint8)
    else:
        data_u8 = np.zeros_like(data, dtype=np.uint8)

    # Extract useful fields from the header
    info = {}
    for key in ("EXPTIME", "DATE-OBS", "TELESCOP", "INSTRUME", "FILTER",
                "OBJECT",  "NAXIS1",   "NAXIS2",  "BITPIX"):
        if key in header:
            info[key] = header[key]

    return data_u8, info, data, header
