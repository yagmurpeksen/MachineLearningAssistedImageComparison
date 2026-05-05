import numpy as np


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

