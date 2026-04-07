# pipeline/ingest/pitcher_stats.py
"""
Builds per-pitcher statistics using ONLY data prior to a given cutoff date.
All metrics are derived from raw Statcast to guarantee temporal integrity.
"""
import pandas as pd
import numpy as np
from datetime import date

# Tom Tango FIP constant (league-wide, recalculate annually)
FIP_CONSTANT = 3.10

def compute_fip(df: pd.DataFrame) -> float:
    """
    FIP = ((13*HR + 3*(BB+HBP) - 2*K) / IP) + FIP_constant
    Requires pitcher-level event summary.
    """
    hr  = df["events"].eq("home_run").sum()
    bb  = df["events"].eq("walk").sum()
    hbp = df["events"].eq("hit_by_pitch").sum()
    k   = df["events"].isin(["strikeout", "strikeout_double_play"]).sum()
    # Approximate IP from outs recorded
    outs = df["events"].isin([
        "field_out", "strikeout", "double_play", "triple_play",
        "grounded_into_double_play", "fielders_choice_out",
        "force_out", "sac_fly", "sac_bunt"
    ]).sum()
    ip = outs / 3
    if ip < 1:
        return np.nan
    return ((13*hr + 3*(bb+hbp) - 2*k) / ip) + FIP_CONSTANT

def compute_xera_proxy(df: pd.DataFrame) -> float:
    """
    xERA proxy using Statcast expected metrics.
    True xERA requires Baseball Savant aggregation endpoint.
    Here we use: xwOBA allowed as the primary proxy.
    """
    xwoba_col = "estimated_woba_using_speedangle"
    if xwoba_col not in df.columns:
        return np.nan
    valid = df[xwoba_col].dropna()
    return valid.mean() if len(valid) >= 50 else np.nan

def build_pitcher_features(
    pitcher_id: int,
    as_of_date: date,
    statcast_df: pd.DataFrame,  # pre-filtered: game_date < as_of_date
    trailing_days_sp: int = 60,  # ~10 starts for rotation context
    trailing_days_short: int = 14,
) -> dict:
    """
    Build all pitcher features as of a cutoff date.
    statcast_df MUST already be filtered to game_date < as_of_date.
    """
    assert (pd.to_datetime(statcast_df["game_date"]).dt.date >= as_of_date).sum() == 0, \
        "LEAKAGE DETECTED: statcast_df contains data on or after as_of_date"
    
    pitcher_df = statcast_df[statcast_df["pitcher"] == pitcher_id].copy()
    
    cutoff = pd.Timestamp(as_of_date)
    start_60 = cutoff - pd.Timedelta(days=trailing_days_sp)
    start_14 = cutoff - pd.Timedelta(days=trailing_days_short)
    
    recent_60 = pitcher_df[pitcher_df["game_date"] >= start_60.strftime("%Y-%m-%d")]
    recent_14 = pitcher_df[pitcher_df["game_date"] >= start_14.strftime("%Y-%m-%d")]
    
    # Days of rest
    game_dates = sorted(pitcher_df["game_date"].unique(), reverse=True)
    days_rest = (
        as_of_date - pd.to_datetime(game_dates[0]).date()
    ).days if game_dates else 999
    
    return {
        "sp_fip_60d":          compute_fip(recent_60),
        "sp_xera_proxy_60d":   compute_xera_proxy(recent_60),
        "sp_k_pct_60d":        _k_pct(recent_60),
        "sp_bb_pct_60d":       _bb_pct(recent_60),
        "sp_hr_fb_60d":        _hr_fb_rate(recent_60),
        "sp_whiff_pct_60d":    _whiff_pct(recent_60),
        "sp_xwoba_allowed_14d": compute_xera_proxy(recent_14),
        "sp_days_rest":        days_rest,
        "sp_pitch_count_7d":   _recent_pitch_count(pitcher_df, as_of_date, days=7),
    }

def _k_pct(df: pd.DataFrame) -> float:
    pa = df[df["events"].notna()].shape[0]
    k  = df["events"].isin(["strikeout", "strikeout_double_play"]).sum()
    return k / pa if pa > 0 else np.nan

def _bb_pct(df: pd.DataFrame) -> float:
    pa = df[df["events"].notna()].shape[0]
    bb = df["events"].isin(["walk"]).sum()
    return bb / pa if pa > 0 else np.nan

def _hr_fb_rate(df: pd.DataFrame) -> float:
    fb  = df["bb_type"].eq("fly_ball").sum()
    hr  = df["events"].eq("home_run").sum()
    return hr / fb if fb >= 10 else np.nan

def _whiff_pct(df: pd.DataFrame) -> float:
    swings  = df["description"].isin(["swinging_strike", "foul", "hit_into_play"]).sum()
    whiffs  = df["description"].eq("swinging_strike").sum()
    return whiffs / swings if swings > 0 else np.nan

def _recent_pitch_count(df: pd.DataFrame, as_of: date, days: int) -> int:
    cutoff_str = (pd.Timestamp(as_of) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    return df[df["game_date"] >= cutoff_str].shape[0]