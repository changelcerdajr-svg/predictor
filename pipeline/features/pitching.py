# pipeline/features/pitching.py
import pandas as pd
import numpy as np
from datetime import date
from pipeline.ingest.pitcher_stats import (
    compute_fip, FIP_CONSTANT,
    _k_pct, _bb_pct, _hr_fb_rate, _whiff_pct, _recent_pitch_count,
)


def _get_pitcher_hand(pitcher_id: int, statcast_df: pd.DataFrame) -> str:
    """Returns 'L' or 'R'. Defaults to 'R' if unknown."""
    rows = statcast_df[statcast_df["pitcher"] == pitcher_id]["p_throws"].dropna()
    return rows.mode()[0] if not rows.empty else "R"


def _identify_relievers(
    team_id: int,
    statcast_df: pd.DataFrame,
    trailing_start: str,
) -> pd.DataFrame:
    """
    Returns pitch-level rows for relievers of team_id within the trailing window.

    Logic:
      - A pitcher belongs to `team_id` if they pitched while that team was in the field:
          * home team fields when inning_topbot == 'Top'   (away bats)
          * away team fields when inning_topbot == 'Bot'   (home bats)
      - Relievers = pitchers whose per-game IP average is < 3.0 in this window
        (excludes starters without requiring a separate roster lookup).
    """
    required_cols = {"home_team", "away_team", "inning_topbot", "pitcher", "game_date"}
    missing = required_cols - set(statcast_df.columns)
    if missing:
        return pd.DataFrame()

    # Pitches where team_id was in the field
    in_field = (
        ((statcast_df["home_team"] == team_id) & (statcast_df["inning_topbot"] == "Top")) |
        ((statcast_df["away_team"] == team_id) & (statcast_df["inning_topbot"] == "Bot"))
    )

    window_df = statcast_df[in_field & (statcast_df["game_date"] >= trailing_start)].copy()

    if window_df.empty:
        return pd.DataFrame()

    # Identify relievers: avg IP per game appearance < 3.0
    # IP proxy: outs recorded per game / 3
    out_events = {
        "field_out", "strikeout", "double_play", "triple_play",
        "grounded_into_double_play", "fielders_choice_out",
        "force_out", "sac_fly", "sac_bunt", "strikeout_double_play",
    }
    window_df["is_out"] = window_df["events"].isin(out_events)

    per_game = (
        window_df.groupby(["pitcher", "game_pk"])["is_out"]
        .sum()
        .reset_index()
    )
    per_game["ip"] = per_game["is_out"] / 3.0
    avg_ip = per_game.groupby("pitcher")["ip"].mean()

    reliever_ids = avg_ip[avg_ip < 3.0].index
    return window_df[window_df["pitcher"].isin(reliever_ids)].copy()


def build_bullpen_features(
    team_id: int,
    as_of_date: date,
    statcast_df: pd.DataFrame,   # pre-filtered: game_date < as_of_date
    trailing_days: int = 14,
) -> dict:
    """
    Bullpen quality and fatigue features for team_id.
    All metrics derived from relievers only (avg IP/game < 3.0).
    """
    empty = {
        "bullpen_fip_14d":        np.nan,
        "bullpen_whiff_14d":      np.nan,
        "bullpen_pitch_count_3d": 0,
        "bullpen_k_pct_14d":      np.nan,
        "bullpen_bb_pct_14d":     np.nan,
        "bullpen_hr_fb_14d":      np.nan,
    }

    if statcast_df.empty:
        return empty

    trailing_start = (
        pd.Timestamp(as_of_date) - pd.Timedelta(days=trailing_days)
    ).strftime("%Y-%m-%d")

    relievers_df = _identify_relievers(team_id, statcast_df, trailing_start)

    if relievers_df.empty:
        return empty

    # Separate 3-day window for fatigue (pitch count only)
    fatigue_start = (
        pd.Timestamp(as_of_date) - pd.Timedelta(days=3)
    ).strftime("%Y-%m-%d")
    fatigue_df = relievers_df[relievers_df["game_date"] >= fatigue_start]

    return {
        "bullpen_fip_14d":        compute_fip(relievers_df),
        "bullpen_whiff_14d":      _whiff_pct(relievers_df),
        "bullpen_k_pct_14d":      _k_pct(relievers_df),
        "bullpen_bb_pct_14d":     _bb_pct(relievers_df),
        "bullpen_hr_fb_14d":      _hr_fb_rate(relievers_df),
        # Raw pitch count last 3 days — fatigue proxy used in feature_matrix
        "bullpen_pitch_count_3d": len(fatigue_df),
    }