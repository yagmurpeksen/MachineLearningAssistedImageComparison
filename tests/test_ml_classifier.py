"""
tests/test_ml_classifier.py
===========================
Unit tests for the classes and functions in the ml_classifier.py module.

TransientClassifier, feature extraction, and Isolation Forest are tested.
No external files or network connection required.
"""

import numpy as np
import pytest
from ml_classifier import (
    TransientClassifier,
    extract_deeptao_features,
    extract_cutout_features,
    extract_pair_features,
    run_isolation_forest,
    ML_AVAILABLE,
)


# ── Helper functions ────────────────────────────────────────

def make_cutout(h=64, w=64, bg=5, star_peak=0):
    """Square image with an optional center star."""
    img = np.full((h, w), bg, dtype=np.float32)
    if star_peak > 0:
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        img += star_peak * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 3.0**2))
    return img


def make_full_image(h=128, w=128, bg=5):
    return np.full((h, w), bg, dtype=np.float32)


def make_star_image(h=128, w=128, cy=64, cx=64, peak=200, bg=5):
    img = make_full_image(h, w, bg)
    yy, xx = np.mgrid[0:h, 0:w]
    img += peak * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 3.0**2))
    return np.clip(img, 0, 255).astype(np.float32)


# ════════════════════════════════════════════════════════════════
#  extract_cutout_features
# ════════════════════════════════════════════════════════════════

def test_cutout_features_returns_9_values():
    img = make_cutout(star_peak=150)
    feats = extract_cutout_features(img, cx=32, cy=32)
    assert len(feats) == 9


def test_cutout_features_no_nan():
    img = make_cutout(star_peak=150)
    feats = extract_cutout_features(img, cx=32, cy=32)
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))


def test_cutout_features_brighter_star_higher_peak():
    img_dim    = make_cutout(star_peak=50)
    img_bright = make_cutout(star_peak=200)
    f_dim    = extract_cutout_features(img_dim,    cx=32, cy=32)
    f_bright = extract_cutout_features(img_bright, cx=32, cy=32)
    # Index 2 = peak
    assert f_bright[2] > f_dim[2]


def test_cutout_features_empty_image():
    img = make_cutout(star_peak=0)
    feats = extract_cutout_features(img, cx=32, cy=32)
    assert len(feats) == 9
    assert not np.any(np.isnan(feats))


# ════════════════════════════════════════════════════════════════
#  extract_deeptao_features
# ════════════════════════════════════════════════════════════════

def test_deeptao_features_returns_54_dims():
    early = make_star_image(peak=100)
    late  = make_star_image(peak=200)
    fv = extract_deeptao_features(early, 64, 64, late, 64, 64)
    assert len(fv) == 54  # 18+18+8 classic + 10 subtraction


def test_deeptao_features_no_nan():
    early = make_star_image(peak=100)
    late  = make_star_image(peak=200)
    fv = extract_deeptao_features(early, 64, 64, late, 64, 64)
    assert not np.any(np.isnan(fv))
    assert not np.any(np.isinf(fv))


def test_deeptao_features_same_image_flux_ratio_near_1():
    img = make_star_image(peak=150)
    fv = extract_deeptao_features(img, 64, 64, img, 64, 64)
    # Index 36+4 = flux_ratio (delta[4]) → should be close to 1.0
    flux_ratio = fv[40]
    assert abs(flux_ratio - 1.0) < 0.1


def test_deeptao_features_brightened_positive_delta_peak():
    early = make_star_image(peak=80)
    late  = make_star_image(peak=220)
    fv = extract_deeptao_features(early, 64, 64, late, 64, 64)
    # delta_peak = fv[36] → should be positive (late is brighter)
    delta_peak = fv[36]
    assert delta_peak > 0


# ════════════════════════════════════════════════════════════════
#  extract_pair_features
# ════════════════════════════════════════════════════════════════

def test_pair_features_correct_length():
    img1 = make_star_image(cy=40, cx=40, peak=150)
    img2 = make_star_image(cy=40, cx=40, peak=200)
    stars1 = [(40.0, 40.0, 8, 150)]
    stars2 = [(40.0, 40.0, 8, 200)]
    matched = [(0, 0, 0.0, 0.0, 0.0)]
    fv = extract_pair_features(img1, img2, stars1, stars2, matched, idx=0)
    # 9 + 9 + 7 = 25 dimensions
    assert len(fv) == 25


def test_pair_features_no_nan():
    img1 = make_star_image(cy=40, cx=40, peak=150)
    img2 = make_star_image(cy=40, cx=40, peak=200)
    stars1 = [(40.0, 40.0, 8, 150)]
    stars2 = [(40.0, 40.0, 8, 200)]
    matched = [(0, 0, 0.0, 0.0, 0.0)]
    fv = extract_pair_features(img1, img2, stars1, stars2, matched, idx=0)
    assert not np.any(np.isnan(fv))


# ════════════════════════════════════════════════════════════════
#  TransientClassifier
# ════════════════════════════════════════════════════════════════

pytestmark_ml = pytest.mark.skipif(not ML_AVAILABLE,
                                    reason="scikit-learn not installed")


@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_classifier_starts_untrained():
    clf = TransientClassifier()
    assert not clf.is_trained
    assert clf.n_samples == 0


@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_classifier_train_requires_min_4_samples():
    clf = TransientClassifier()
    fv = np.zeros(25)
    clf.add_sample(fv, 0)
    clf.add_sample(fv, 1)
    clf.add_sample(fv, 0)
    with pytest.raises(ValueError):
        clf.train()


@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_classifier_train_succeeds():
    clf = TransientClassifier()
    rng = np.random.default_rng(0)
    for label in [0, 1, 0, 1, 0, 1]:
        fv = rng.random(25).astype(np.float32)
        clf.add_sample(fv, label)
    clf.train()
    assert clf.is_trained


@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_classifier_predict_returns_label_and_prob():
    clf = TransientClassifier()
    rng = np.random.default_rng(1)
    for label in [0, 1, 0, 1, 0, 1]:
        fv = rng.random(25).astype(np.float32)
        clf.add_sample(fv, label)
    clf.train()
    fv_test = rng.random(25).astype(np.float32)
    label, prob = clf.predict(fv_test)
    assert label in (0, 1)
    assert 0.0 <= prob <= 1.0


@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_classifier_predict_untrained_returns_zero():
    clf = TransientClassifier()
    fv = np.zeros(25)
    label, prob = clf.predict(fv)
    assert label == 0
    assert prob == 0.0


@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_classifier_save_load_roundtrip(tmp_path):
    clf = TransientClassifier()
    rng = np.random.default_rng(2)
    for label in [0, 1, 0, 1, 0, 1]:
        fv = rng.random(25).astype(np.float32)
        clf.add_sample(fv, label)
    clf.train()

    path = str(tmp_path / "model.joblib")
    clf.save(path)

    clf2 = TransientClassifier()
    clf2.load(path)
    assert clf2.is_trained
    assert clf2.n_samples == clf.n_samples

    # Should produce the same prediction for the same input
    fv_test = rng.random(25).astype(np.float32)
    assert clf.predict(fv_test)[0] == clf2.predict(fv_test)[0]


@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_classifier_cv_accuracy_returns_float():
    clf = TransientClassifier()
    rng = np.random.default_rng(3)
    for label in ([0] * 5 + [1] * 5):
        fv = rng.random(25).astype(np.float32)
        clf.add_sample(fv, label)
    clf.train()
    acc = clf.cross_val_accuracy()
    assert acc is not None
    assert 0.0 <= acc <= 1.0


# ════════════════════════════════════════════════════════════════
#  run_isolation_forest
# ════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_isolation_forest_returns_same_length():
    feats = [np.random.rand(9) for _ in range(20)]
    scores = run_isolation_forest(feats)
    assert len(scores) == 20


@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_isolation_forest_scores_in_0_1():
    feats = [np.random.rand(9) for _ in range(20)]
    scores = run_isolation_forest(feats)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_isolation_forest_outlier_gets_high_score():
    """19 points from a normal distribution + 1 extreme outlier."""
    rng = np.random.default_rng(42)
    normal = [rng.normal(0, 1, 9) for _ in range(19)]
    outlier = [np.array([100.0] * 9)]
    feats = normal + outlier
    scores = run_isolation_forest(feats, contamination=0.1)
    # Outlier should receive the highest score
    assert scores[-1] == pytest.approx(scores.max(), abs=0.01)


@pytest.mark.skipif(not ML_AVAILABLE, reason="scikit-learn not installed")
def test_isolation_forest_too_few_samples_returns_zeros():
    feats = [np.random.rand(9) for _ in range(3)]
    scores = run_isolation_forest(feats)
    assert np.all(scores == 0.0)
