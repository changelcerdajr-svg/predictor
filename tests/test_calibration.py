# tests/test_calibration.py
"""Tests for TemperatureScaler and Kelly criterion."""
import pytest
import numpy as np
from pipeline.calibration.temperature_scaling import TemperatureScaler
from pipeline.risk.kelly import size_bet, kelly_fraction


# ---------------------------------------------------------------------------
# TemperatureScaler
# ---------------------------------------------------------------------------

def test_temperature_scaler_identity_when_calibrated():
    """If raw probs are already calibrated, T should be close to 1.0."""
    rng    = np.random.default_rng(42)
    probs  = rng.uniform(0.1, 0.9, 500)
    labels = rng.binomial(1, probs).astype(float)
    scaler = TemperatureScaler().fit(probs, labels)
    assert 0.5 < scaler.T < 2.0, f"T={scaler.T} is out of expected range"


def test_temperature_scaler_reduces_overconfidence():
    """Overconfident probs (near 0/1) should result in T > 1."""
    rng    = np.random.default_rng(0)
    probs  = np.clip(rng.uniform(0.1, 0.9, 500) ** 0.2, 0.01, 0.99)  # pushed to extremes
    labels = rng.binomial(1, rng.uniform(0.3, 0.7, 500)).astype(float)
    scaler = TemperatureScaler().fit(probs, labels)
    assert scaler.T >= 1.0, f"Expected T>=1 for overconfident model, got T={scaler.T}"


def test_transform_stays_in_01():
    probs  = np.array([0.05, 0.3, 0.5, 0.7, 0.95])
    labels = np.array([0, 0, 1, 1, 1], dtype=float)
    cal    = TemperatureScaler().fit(probs, labels).transform(probs)
    assert np.all(cal > 0) and np.all(cal < 1)


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------

def test_kelly_no_edge_returns_zero():
    """No edge → Kelly fraction must be 0."""
    f = kelly_fraction(prob=0.50, decimal_odds=1.909)  # -110 implied ~0.524
    assert f == 0.0


def test_kelly_positive_edge():
    f = kelly_fraction(prob=0.60, decimal_odds=2.0)  # +100 odds
    assert f > 0


def test_size_bet_returns_none_below_min_edge():
    rec = size_bet(
        game_pk=1, side="home", prob=0.52,
        american_odds=-110, bankroll=1000.0,
        min_edge=0.05,   # edge ~0.52 - 0.524 = negative → None
    )
    assert rec is None


def test_size_bet_caps_at_max_fraction():
    rec = size_bet(
        game_pk=1, side="home", prob=0.70,
        american_odds=+200, bankroll=1000.0,
        kelly_multiplier=1.0,  # full Kelly
        max_fraction=0.05,
    )
    assert rec is not None
    assert rec.bet_fraction <= 0.05


def test_size_bet_invalid_prob():
    assert size_bet(1, "home", 0.0,  -110, 1000.0) is None
    assert size_bet(1, "home", 1.0,  -110, 1000.0) is None
    assert size_bet(1, "home", float("nan"), -110, 1000.0) is None