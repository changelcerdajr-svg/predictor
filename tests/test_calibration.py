# tests/test_calibration.py
import pytest
import numpy as np
from scipy.special import expit, logit

from pipeline.calibration.temperature_scaling import TemperatureScaler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_probs():
    np.random.seed(42)
    return np.random.uniform(0.1, 0.9, 500)


@pytest.fixture
def overconfident_probs():
    """Simulate model that's too extreme — needs T > 1 to calibrate."""
    np.random.seed(42)
    p = np.random.uniform(0.1, 0.9, 500)
    return np.clip(expit(logit(p) * 2.5), 1e-6, 1 - 1e-6)


@pytest.fixture
def labels(perfect_probs):
    np.random.seed(42)
    return (np.random.uniform(size=len(perfect_probs)) < perfect_probs).astype(int)


@pytest.fixture
def overconfident_labels(overconfident_probs):
    np.random.seed(7)
    true_p = expit(logit(np.clip(overconfident_probs, 1e-6, 1-1e-6)) / 2.5)
    return (np.random.uniform(size=len(true_p)) < true_p).astype(int)


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestTemperatureScalerAPI:
    def test_fit_returns_self(self, perfect_probs, labels):
        scaler = TemperatureScaler()
        result = scaler.fit(perfect_probs, labels)
        assert result is scaler

    def test_default_temperature(self):
        assert TemperatureScaler().T == 1.0

    def test_fit_transform_shape(self, perfect_probs, labels):
        out = TemperatureScaler().fit_transform(perfect_probs, labels)
        assert out.shape == perfect_probs.shape

    def test_transform_before_fit_uses_T1(self, perfect_probs):
        scaler = TemperatureScaler()
        out = scaler.transform(perfect_probs)
        np.testing.assert_allclose(out, perfect_probs, atol=1e-5)

    def test_output_in_unit_interval(self, perfect_probs, labels):
        out = TemperatureScaler().fit_transform(perfect_probs, labels)
        assert (out >= 0).all() and (out <= 1).all()


# ---------------------------------------------------------------------------
# Temperature behavior
# ---------------------------------------------------------------------------

class TestTemperatureValue:
    def test_overconfident_model_yields_T_gt_1(self, overconfident_probs, overconfident_labels):
        scaler = TemperatureScaler().fit(overconfident_probs, overconfident_labels)
        assert scaler.T > 1.0, f"Expected T > 1, got {scaler.T:.4f}"

    def test_T_within_bounds(self, overconfident_probs, overconfident_labels):
        scaler = TemperatureScaler().fit(overconfident_probs, overconfident_labels)
        assert 0.1 <= scaler.T <= 10.0

    def test_T1_is_identity(self, perfect_probs):
        scaler = TemperatureScaler()
        scaler.T = 1.0
        out = scaler.transform(perfect_probs)
        np.testing.assert_allclose(out, perfect_probs, atol=1e-5)

    def test_high_T_shrinks_toward_half(self):
        scaler = TemperatureScaler()
        scaler.T = 10.0
        extreme = np.array([0.05, 0.95])
        out = scaler.transform(extreme)
        assert out[0] > extreme[0]
        assert out[1] < extreme[1]
        assert abs(out[0] - 0.5) < abs(extreme[0] - 0.5)

    def test_low_T_pushes_away_from_half(self):
        scaler = TemperatureScaler()
        scaler.T = 0.5
        mid = np.array([0.4, 0.6])
        out = scaler.transform(mid)
        assert out[0] < mid[0]
        assert out[1] > mid[1]


# ---------------------------------------------------------------------------
# Calibration quality
# ---------------------------------------------------------------------------

class TestCalibrationQuality:
    def test_brier_improves_on_overconfident(self, overconfident_probs, overconfident_labels):
        from sklearn.metrics import brier_score_loss
        scaler = TemperatureScaler().fit(overconfident_probs, overconfident_labels)
        cal    = scaler.transform(overconfident_probs)
        assert brier_score_loss(overconfident_labels, cal) <= \
               brier_score_loss(overconfident_labels, overconfident_probs) + 1e-6

    def test_monotonicity_preserved(self, perfect_probs, labels):
        scaler = TemperatureScaler().fit(perfect_probs, labels)
        out    = scaler.transform(perfect_probs)
        order_in  = np.argsort(perfect_probs)
        order_out = np.argsort(out)
        assert np.array_equal(order_in, order_out)

    def test_mean_probability_conservative(self, overconfident_probs, overconfident_labels):
        """Calibrated mean prob should be closer to actual win rate."""
        scaler  = TemperatureScaler().fit(overconfident_probs, overconfident_labels)
        cal     = scaler.transform(overconfident_probs)
        wr      = overconfident_labels.mean()
        assert abs(cal.mean() - wr) <= abs(overconfident_probs.mean() - wr) + 1e-4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_clips_zero_and_one(self):
        scaler = TemperatureScaler()
        out = scaler.transform(np.array([0.0, 1.0]))
        assert np.isfinite(out).all()

    def test_single_sample(self):
        scaler = TemperatureScaler()
        out = scaler.transform(np.array([0.6]))
        assert out.shape == (1,)

    def test_all_same_prob(self):
        p   = np.full(100, 0.55)
        lbl = np.ones(100, dtype=int)
        scaler = TemperatureScaler().fit(p, lbl)
        out = scaler.transform(p)
        assert np.isfinite(out).all()

    def test_reproducible_fit(self, overconfident_probs, overconfident_labels):
        T1 = TemperatureScaler().fit(overconfident_probs, overconfident_labels).T
        T2 = TemperatureScaler().fit(overconfident_probs, overconfident_labels).T
        assert T1 == T2