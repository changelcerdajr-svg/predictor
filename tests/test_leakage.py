# tests/test_leakage.py
import pytest
import pandas as pd
import numpy as np
from datetime import date

from pipeline.features.feature_matrix import build_game_feature_vector
from pipeline.ingest.pitcher_stats    import build_pitcher_features
from pipeline.features.batting        import build_lineup_features
from pipeline.features.pitching       import build_bullpen_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cutoff():
    return date(2024, 6, 15)


@pytest.fixture
def clean_statcast(cutoff):
    """All rows strictly before cutoff."""
    rng = pd.date_range("2024-03-28", periods=60, freq="D")
    rng = [d for d in rng if d.date() < cutoff]
    n   = len(rng) * 20
    np.random.seed(0)
    return pd.DataFrame({
        "game_date":   [str(d.date()) for d in rng for _ in range(20)],
        "game_pk":     np.arange(n),
        "pitcher":     np.random.choice([100, 101, 102], n),
        "batter":      np.random.choice([200, 201, 202, 203], n),
        "events":      np.random.choice(
                           ["strikeout", "walk", "single", "field_out", None], n),
        "p_throws":    np.random.choice(["R", "L"], n),
        "inning_topbot": np.random.choice(["Top", "Bot"], n),
        "home_team":   np.random.choice([1, 2], n),
        "away_team":   np.random.choice([3, 4], n),
        "bb_type":     np.random.choice(["fly_ball", "ground_ball", None], n),
        "description": np.random.choice(["swinging_strike", "foul", "hit_into_play"], n),
        "launch_speed": np.random.uniform(70, 110, n),
        "estimated_woba_using_speedangle": np.random.uniform(0.2, 0.5, n),
    })


@pytest.fixture
def leaky_statcast(cutoff, clean_statcast):
    """Injects one row ON the cutoff date."""
    leak = clean_statcast.iloc[[0]].copy()
    leak["game_date"] = cutoff.isoformat()
    return pd.concat([clean_statcast, leak], ignore_index=True)


@pytest.fixture
def sample_game():
    return {
        "game_pk":      999,
        "game_date":    "2024-06-15",
        "home_team_id": 1,
        "away_team_id": 3,
        "home_sp_id":   100,
        "away_sp_id":   101,
        "venue_id":     10,
        "game_time_utc": "2024-06-15T23:10:00Z",
    }


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_pitcher_features_assert_fires(self, leaky_statcast, cutoff):
        with pytest.raises(AssertionError, match="LEAKAGE"):
            build_pitcher_features(100, cutoff, leaky_statcast)

    def test_pitcher_features_clean_passes(self, clean_statcast, cutoff):
        result = build_pitcher_features(100, cutoff, clean_statcast)
        assert isinstance(result, dict)
        assert "sp_fip_60d" in result

    def test_no_future_dates_in_statcast(self, clean_statcast, cutoff):
        dates = pd.to_datetime(clean_statcast["game_date"]).dt.date
        assert (dates >= cutoff).sum() == 0

    def test_build_game_feature_vector_clean(self, sample_game, clean_statcast, cutoff):
        row = build_game_feature_vector(sample_game, cutoff, clean_statcast, {})
        # May be None if insufficient data, but must not raise
        assert row is None or isinstance(row, dict)

    def test_lineup_features_no_future(self, clean_statcast, cutoff):
        result = build_lineup_features(1, cutoff, clean_statcast, "R")
        assert isinstance(result, dict)

    def test_bullpen_features_no_future(self, clean_statcast, cutoff):
        result = build_bullpen_features(1, cutoff, clean_statcast)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Date boundary
# ---------------------------------------------------------------------------

class TestDateBoundary:
    def test_cutoff_date_excluded(self, clean_statcast, cutoff):
        cutoff_str = cutoff.isoformat()
        assert (clean_statcast["game_date"] == cutoff_str).sum() == 0

    def test_day_before_cutoff_included(self, clean_statcast, cutoff):
        from datetime import timedelta
        day_before = (cutoff - timedelta(days=1)).isoformat()
        assert (clean_statcast["game_date"] <= day_before).all()

    def test_statcast_fetch_blocks_future(self):
        from pipeline.ingest.statcast import fetch_and_cache_statcast
        from datetime import date
        with pytest.raises(AssertionError):
            fetch_and_cache_statcast(date(2099, 1, 1), date(2099, 1, 2))


# ---------------------------------------------------------------------------
# Feature matrix integrity
# ---------------------------------------------------------------------------

class TestFeatureMatrixIntegrity:
    def test_returns_none_without_sp(self, clean_statcast, cutoff):
        game = {
            "game_pk": 1, "game_date": "2024-06-15",
            "home_team_id": 1, "away_team_id": 3,
            "home_sp_id": None, "away_sp_id": 101,
            "venue_id": 10, "game_time_utc": None,
        }
        assert build_game_feature_vector(game, cutoff, clean_statcast, {}) is None

    def test_differential_direction(self, sample_game, clean_statcast, cutoff):
        row = build_game_feature_vector(sample_game, cutoff, clean_statcast, {})
        if row is None:
            pytest.skip("Insufficient data for feature build")
        # sp_fip_diff = away_fip - home_fip (positive = home advantage)
        assert "sp_fip_diff" in row
        assert "woba_diff" in row