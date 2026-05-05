import numpy as np


def test_detect_stars_finds_bright_blobs():
    from image_comparison_gui import detect_stars

    img = np.zeros((128, 128), dtype=np.uint8)

    # Make a few bright points
    img[30, 40] = 255
    img[90, 70] = 240
    img[60, 100] = 250

    stars = detect_stars(img, pct=90, min_area=1, max_area=20, min_peak=10)
    assert len(stars) >= 2

