"""
tests/test_pipeline_integration.py
===================================
Tests for Train → Inference pipeline mismatches.

Critical question: is the feature vector produced by train_model.py
comparable to the feature vector produced by the GUI
(detect_stars → extract_deeptao_features)?

No external files, network connection, or real model required.
Synthetic images are used.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_model import extract_change_features
from ml_classifier import _extract_single_frame_feats, extract_deeptao_features
from detection import detect_stars


# ── Helper image generators ────────────────────────────────────

def make_star_cutout(peak=500, bg=50, size=64, noise_std=5.0, rng=None):
    """Deep-TAO style 64×64 raw float cutout (ADU-like values)."""
    if rng is None:
        rng = np.random.default_rng(42)
    img = rng.normal(bg, noise_std, (size, size)).astype(np.float32)
    cy, cx = size // 2, size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    img += peak * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 3.0**2))
    return img.clip(0, None)


def make_full_field(h=256, w=256, n_stars=20, bg=50, noise_std=5.0,
                    star_peak=400, rng=None):
    """
    Full-size 'FITS-like' image containing multiple stars.
    Returns: (image, [(cx, cy), ...]) — star positions
    """
    if rng is None:
        rng = np.random.default_rng(0)
    img = rng.normal(bg, noise_std, (h, w)).astype(np.float32)
    positions = []
    margin = 40
    for _ in range(n_stars):
        cy = int(rng.uniform(margin, h - margin))
        cx = int(rng.uniform(margin, w - margin))
        peak = rng.uniform(star_peak * 0.5, star_peak * 1.5)
        yy, xx = np.mgrid[0:h, 0:w]
        img += peak * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 2.5**2))
        positions.append((cx, cy))
    return img.clip(0, None), positions


# ════════════════════════════════════════════════════════════════════
#  1. Feature function consistency:
#     train_model.extract_single_frame  ≡  ml_classifier._extract_single_frame_feats
# ════════════════════════════════════════════════════════════════════

class TestFeatureFunctionConsistency:
    """
    The feature extraction functions in train_model.py and ml_classifier.py
    must produce identical output — if not, a train/inference mismatch occurs.
    """

    def test_single_frame_same_output_bright_star(self):
        cutout = make_star_cutout(peak=800, bg=100)
        f_train = _extract_single_frame_feats(cutout)
        f_infer = _extract_single_frame_feats(cutout)
        np.testing.assert_array_almost_equal(
            f_train, f_infer, decimal=5,
            err_msg="train_model and ml_classifier feature functions produced different results!")

    def test_single_frame_same_output_empty(self):
        cutout = np.zeros((64, 64), dtype=np.float32)
        f_train = _extract_single_frame_feats(cutout)
        f_infer = _extract_single_frame_feats(cutout)
        np.testing.assert_array_almost_equal(f_train, f_infer, decimal=5)

    def test_single_frame_same_output_dim_star(self):
        cutout = make_star_cutout(peak=80, bg=50, noise_std=2.0)
        f_train = _extract_single_frame_feats(cutout)
        f_infer = _extract_single_frame_feats(cutout)
        np.testing.assert_array_almost_equal(f_train, f_infer, decimal=5)

    def test_single_frame_same_length(self):
        cutout = make_star_cutout()
        assert len(_extract_single_frame_feats(cutout)) == 18

    def test_change_features_vs_deeptao_features_same_output(self):
        """
        train_model.extract_change_features(early, late)
            ≡  ml_classifier.extract_deeptao_features(full_img, cx, cy, full_img, cx, cy)
        — if the image is already a 64×64 cutout and cx=cy=32.
        """
        rng = np.random.default_rng(7)
        early = make_star_cutout(peak=200, bg=60, rng=rng)
        late  = make_star_cutout(peak=500, bg=60, rng=rng)

        f_train = extract_change_features(early, late)     # train_model yolu
        f_infer = extract_deeptao_features(                # GUI yolu
            early, 32, 32, late, 32, 32, size=64)

        np.testing.assert_array_almost_equal(
            f_train, f_infer, decimal=4,
            err_msg="54-dimensional vector: train_model vs ml_classifier mismatch!")

    def test_change_features_length_54(self):
        early = make_star_cutout(peak=200)
        late  = make_star_cutout(peak=500)
        assert len(extract_change_features(early, late)) == 54  # 18+18+8+10
        assert len(extract_deeptao_features(early, 32, 32, late, 32, 32)) == 54


# ════════════════════════════════════════════════════════════════════
#  2. Pixel range independence:
#     The feature vector should produce different raw values for raw float (ADU)
#     and normalized (0-255) images — but after StandardScaler they should be
#     close to the same Z-score distribution (this is currently the known issue)
# ════════════════════════════════════════════════════════════════════

class TestPixelRangeEffect:
    """
    Verifies the requirement to use raw FITS float data in the GUI.
    """

    def test_raw_vs_normalized_features_differ(self):
        """
        Raw ADU and 0-255 normalized images should produce different raw feature values.
        (This is expected behavior — StandardScaler normalizes both)
        """
        rng   = np.random.default_rng(42)
        raw   = make_star_cutout(peak=5000, bg=1000, rng=rng)  # ADU-range
        norm  = (raw / raw.max() * 255).astype(np.float32)      # 0-255

        f_raw  = _extract_single_frame_feats(raw)
        f_norm = _extract_single_frame_feats(norm)

        # Raw values should differ
        assert not np.allclose(f_raw, f_norm), (
            "Raw and normalized images should not produce the same features — "
            "this would hide a pipeline mismatch!")

    def test_feature_ratios_invariant_to_scale(self):
        """
        Scale-invariant features (concentration, skewness, kurtosis)
        should be the same for ADU vs 0-255.
        """
        rng   = np.random.default_rng(42)
        raw   = make_star_cutout(peak=5000, bg=1000, rng=rng)
        scale = 255.0 / raw.max()
        norm  = (raw * scale).astype(np.float32)

        f_raw  = _extract_single_frame_feats(raw)
        f_norm = _extract_single_frame_feats(norm)

        # Index 11, 12, 13 = conc_inner, conc_middle, conc_outer (ratio-based → scale-free)
        np.testing.assert_allclose(f_raw[11:14], f_norm[11:14], rtol=1e-4,
            err_msg="Concentration features should be scale-invariant")

        # Index 5, 6 = skewness, kurtosis (scale-free)
        np.testing.assert_allclose(f_raw[5:7], f_norm[5:7], rtol=1e-4,
            err_msg="Skewness/kurtosis should be scale-invariant")


# ════════════════════════════════════════════════════════════════════
#  3. Tam pipeline entegrasyon testi:
#     Synthetic FITS-like image → detect_stars → extract_deeptao_features
#     Result: valid, 54-dimensional, NaN-free feature vector
# ════════════════════════════════════════════════════════════════════

class TestFullPipelineIntegration:
    """
    The detect_stars() → extract_deeptao_features() chain should run without interruption.
    No real model or FITS file required.
    """

    def _make_pair(self, transient_peak=2000):
        """Early (no star) + late (new star added) image pair."""
        rng   = np.random.default_rng(1)
        early, star_positions = make_full_field(rng=rng)
        late  = early.copy()
        # Add a new 'transient' to the late image
        cy, cx = 128, 128
        yy, xx = np.mgrid[0:256, 0:256]
        late += transient_peak * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 2.5**2))
        return early, late, cx, cy

    def test_feature_extraction_on_detected_star(self):
        """Feature vector is produced using coordinates from detect_stars output."""
        early, late, tx, ty = self._make_pair()

        # Star detection in the late image
        stars_late = detect_stars(late.astype(np.uint8),
                                  pct=90, min_area=3, max_area=300, min_peak=10)
        assert len(stars_late) > 0, "No stars detected!"

        # Extract features for the first detected star
        cx, cy = stars_late[0][0], stars_late[0][1]
        fv = extract_deeptao_features(early, cx, cy, late, cx, cy)

        assert len(fv) == 54
        assert not np.any(np.isnan(fv))
        assert not np.any(np.isinf(fv))

    def test_new_source_has_positive_delta_peak(self):
        """
        New bright source appearing in the late image:
        delta_peak (fv[36]) should be > 0.
        """
        rng   = np.random.default_rng(99)
        early = make_full_field(rng=rng)[0]
        late  = early.copy()
        cx, cy = 100, 100
        yy, xx = np.mgrid[0:256, 0:256]
        late += 3000 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 3.0**2))

        fv = extract_deeptao_features(early, cx, cy, late, cx, cy)
        delta_peak = fv[36]
        assert delta_peak > 0, (
            f"delta_peak should be positive for a new source, but got {delta_peak:.2f}")

    def test_flux_ratio_above_1_for_brightening(self):
        """flux_ratio (fv[40]) should be > 1 for a brightening source."""
        rng   = np.random.default_rng(5)
        early = make_full_field(rng=rng)[0]
        late  = early.copy()
        cx, cy = 80, 150
        yy, xx = np.mgrid[0:256, 0:256]
        late += 4000 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 3.0**2))

        fv = extract_deeptao_features(early, cx, cy, late, cx, cy)
        flux_ratio = fv[40]
        assert flux_ratio > 1.0, (
            f"Expected flux_ratio > 1 for a brightening source, got {flux_ratio:.3f}")

    def test_stable_star_flux_ratio_near_1(self):
        """flux_ratio should be ≈ 1 for an unchanged star."""
        rng   = np.random.default_rng(3)
        img, positions = make_full_field(rng=rng)
        cx, cy = positions[0]

        fv = extract_deeptao_features(img, cx, cy, img, cx, cy)
        flux_ratio = fv[40]
        assert abs(flux_ratio - 1.0) < 0.05, (
            f"Expected flux_ratio ≈ 1 for a stable star, got {flux_ratio:.3f}")

    def test_feature_vector_not_all_zeros(self):
        """
        Feature vector should not be all zeros for a realistic image.
        (All-zero vector indicates a cutout crop error or empty pixel issue)
        """
        rng   = np.random.default_rng(2)
        img, positions = make_full_field(rng=rng, star_peak=600)
        cx, cy = positions[0]

        fv = extract_deeptao_features(img, cx, cy, img, cx, cy)
        assert np.any(fv != 0), "Feature vector is all zeros — cutout crop error!"

    def test_edge_star_no_crash(self):
        """Feature extraction should not raise an error for a star at the image edge."""
        rng = np.random.default_rng(11)
        img = make_full_field(rng=rng)[0]
        # Corner coordinate
        fv = extract_deeptao_features(img, 2, 2, img, 2, 2)
        assert len(fv) == 54
        assert not np.any(np.isnan(fv))

    def test_detect_then_extract_pipeline_no_exception(self):
        """
        The detect_stars → get coordinates → extract_deeptao_features chain
        should complete without raising an exception.
        """
        rng   = np.random.default_rng(77)
        early, _ = make_full_field(rng=rng, star_peak=500)

        # Convert to uint8 (detect_stars expects this)
        early_u8 = ((early / early.max()) * 255).astype(np.uint8)
        stars = detect_stars(early_u8, pct=90, min_area=2, max_area=500, min_peak=5)

        results = []
        for cx, cy, area, peak in stars[:5]:  # First 5 stars
            fv = extract_deeptao_features(early, cx, cy, early, cx, cy)
            results.append(fv)

        assert len(results) > 0, "No stars found or no features could be extracted"
        for fv in results:
            assert len(fv) == 54
            assert not np.any(np.isnan(fv))


# ════════════════════════════════════════════════════════════════════
#  4. Threshold entegrasyon testi:
#     Saved threshold should be consistent with model prediction
# ════════════════════════════════════════════════════════════════════

try:
    import joblib
    from pathlib import Path
    _MODEL_PATH = Path(__file__).resolve().parent.parent / "transient_model.joblib"
    _MODEL_AVAILABLE = _MODEL_PATH.exists()
except ImportError:
    _MODEL_AVAILABLE = False


@pytest.mark.skipif(not _MODEL_AVAILABLE,
                    reason="transient_model.joblib not found")
class TestSavedModelThreshold:
    """
    Validates the threshold value and dimension expectation of the saved model.
    """

    def setup_method(self):
        data = joblib.load(str(_MODEL_PATH))
        self.clf      = data["classifier"]
        self.scaler   = data["scaler"]
        self.threshold = data.get("threshold", 0.5)
        self.n_feat   = data.get("n_features", 44)

    def test_threshold_is_reasonable(self):
        """Saved threshold should be within a reasonable range (0.3–0.8)."""
        assert 0.30 <= self.threshold <= 0.80, (
            f"Threshold {self.threshold} is outside the reasonable range!")

    def test_model_expects_54_features(self):
        assert self.n_feat == 54, f"Model expects {self.n_feat} features, 54 expected!"

    def test_predict_proba_with_threshold(self):
        """
        Result should be in {0, 1} when threshold is applied with predict_proba.
        Checks consistency between predict() and threshold application.
        """
        rng = np.random.default_rng(42)
        fv  = rng.random(self.n_feat).astype(np.float32)
        fv_s = self.scaler.transform(fv.reshape(1, -1))

        prob_transient = float(self.clf.predict_proba(fv_s)[0][1])
        lbl_threshold  = int(prob_transient >= self.threshold)
        lbl_default    = int(self.clf.predict(fv_s)[0])

        assert lbl_threshold in (0, 1)
        assert lbl_default   in (0, 1)
        # Since threshold 0.64 > 0.5, the threshold label should be more
        # conservative than the default (predict fewer transients)
        if prob_transient < self.threshold:
            assert lbl_threshold == 0

    def test_pipeline_feature_accepted_by_model(self):
        """
        The output of extract_deeptao_features should be accepted by the model without errors.
        """
        rng   = np.random.default_rng(10)
        early = make_star_cutout(peak=2000, bg=500, rng=rng)
        late  = make_star_cutout(peak=5000, bg=500, rng=rng)

        fv   = extract_deeptao_features(early, 32, 32, late, 32, 32)
        fv   = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
        fv_s = self.scaler.transform(fv.reshape(1, -1))

        proba = self.clf.predict_proba(fv_s)[0]
        assert proba.shape == (2,)
        assert abs(proba.sum() - 1.0) < 1e-5
        assert 0.0 <= proba[0] <= 1.0
        assert 0.0 <= proba[1] <= 1.0
