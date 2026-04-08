# tests/test_leakage.py
"""Tests that no future data leaks into feature computation."""
import pytest
import pandas as pd
import numpy as np
from datetime import date
from pipeline.ingest.pitcher_stats import build_pitcher_features
 
 
def _make_statcast(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
 
 
def test_leakage_assert_fires_on_future_data():
    """build_pitcher_features must raise if statcast_df contains rows on as_of_date."""
    df = _make_statcast([
        {"pitcher": 1, "game_date": "2024-04-10", "events": "strikeout",
         "bb_type": None, "description": "swinging_strike",
         "estimated_woba_using_speedangle": 0.3, "p_throws": "R", "game_pk": 1},
    ])
    with pytest.raises(AssertionError, match="LEAKAGE"):
        build_pitcher_features(1, date(2024, 4, 10), df)
 
 
def test_no_leakage_with_prior_data():
    """build_pitcher_features must NOT raise when all data is before as_of_date."""
    df = _make_statcast([
        {"pitcher": 1, "game_date": "2024-04-09", "events": "strikeout",
         "bb_type": None, "description": "swinging_strike",
         "estimated_woba_using_speedangle": 0.3, "p_throws": "R", "game_pk": 1},
    ])
    result = build_pitcher_features(1, date(2024, 4, 10), df)
    assert isinstance(result, dict)
    assert "sp_fip_60d" in result
 
 
def test_empty_statcast_no_crash():
    """Empty DataFrame must return NaN features, not crash."""
    df = pd.DataFrame(columns=[
        "pitcher", "game_date", "events", "bb_type",
        "description", "estimated_woba_using_speedangle", "p_throws", "game_pk"
    ])
    result = build_pitcher_features(99, date(2024, 4, 10), df)
    assert result["sp_fip_60d"] is np.nan or result["sp_fip_60d"] != result["sp_fip_60d"]