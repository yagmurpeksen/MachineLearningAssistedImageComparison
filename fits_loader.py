import numpy as np

try:
    from astropy.io import fits as astropy_fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False


def sqrt_stretch(data: np.ndarray) -> np.ndarray:
    """
    Apply the same sky-based sqrt-stretch as load_fits_image to a float32 array.
    Returns uint8 [0-255].
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

    # Clean NaN / Inf If the telescop saturated it could return inf
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Sqrt stretch: makes both bright and faint stars visible
    data_u8 = sqrt_stretch(data)

    # Extract useful fields from the header
    info = {}
    for key in ("EXPTIME", "DATE-OBS", "TELESCOP", "INSTRUME", "FILTER",
                "OBJECT",  "NAXIS1",   "NAXIS2",  "BITPIX"):
        if key in header:
            info[key] = header[key]

    return data_u8, info, data, header
