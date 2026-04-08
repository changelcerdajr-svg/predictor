# pipeline/features/batting.py
import pandas as pd
import numpy as np
from datetime import date
from pipeline.ingest.team_map import team_abr
 
 
def _woba(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    bb   = df["events"].eq("walk").sum()
    hbp  = df["events"].eq("hit_by_pitch").sum()
    s    = df["events"].eq("single").sum()
    d    = df["events"].eq("double").sum()
    t    = df["events"].eq("triple").sum()
    hr   = df["events"].eq("home_run").sum()
    ab   = df["events"].isin([
        "single","double","triple","home_run",
        "field_out","strikeout","grounded_into_double_play",
        "double_play","fielders_choice_out","force_out",
        "strikeout_double_play","triple_play","sac_fly",
    ]).sum()
    pa = ab + bb + hbp
    if pa < 10:
        return np.nan
    return (0.69*bb + 0.72*hbp + 0.89*s + 1.27*d + 1.62*t + 2.10*hr) / pa
 
 
def _xwoba(df: pd.DataFrame) -> float:
    col = "estimated_woba_using_speedangle"
    if col not in df.columns:
        return np.nan
    valid = df[col].dropna()
    return float(valid.mean()) if len(valid) >= 20 else np.nan
 
 
def _iso(df: pd.DataFrame) -> float:
    ab = df["events"].isin([
        "single","double","triple","home_run",
        "field_out","strikeout","grounded_into_double_play",
        "double_play","fielders_choice_out","force_out",
        "strikeout_double_play","triple_play",
    ]).sum()
    if ab < 10:
        return np.nan
    d  = df["events"].eq("double").sum()
    t  = df["events"].eq("triple").sum()
    hr = df["events"].eq("home_run").sum()
    return (d + 2*t + 3*hr) / ab
 
 
def _k_pct(df: pd.DataFrame) -> float:
    pa = df["events"].notna().sum()
    k  = df["events"].isin(["strikeout","strikeout_double_play"]).sum()
    return k / pa if pa > 0 else np.nan
 
 
def _bb_pct(df: pd.DataFrame) -> float:
    pa = df["events"].notna().sum()
    bb = df["events"].eq("walk").sum()
    return bb / pa if pa > 0 else np.nan
 
 
def _hard_hit_pct(df: pd.DataFrame) -> float:
    col = "launch_speed"
    if col not in df.columns:
        return np.nan
    batted = df[df[col].notna()]
    if len(batted) < 10:
        return np.nan
    return (batted[col] >= 95).sum() / len(batted)
 
 
def _filter_by_hand(df: pd.DataFrame, pitcher_hand: str) -> pd.DataFrame:
    if "p_throws" not in df.columns:
        return df
    filtered = df[df["p_throws"] == pitcher_hand]
    return filtered if filtered["events"].notna().sum() >= 30 else df
 
 
def _get_team_batters(
    team_id: int,
    statcast_df: pd.DataFrame,
    start_str: str,
) -> pd.DataFrame:
    """
    FIX: Statcast stores team as string abbreviation (e.g. "NYY"), not int.
    Convert team_id → abbreviation before filtering.
    """
    required = {"home_team", "away_team", "inning_topbot", "batter", "game_date"}
    if required - set(statcast_df.columns):
        return pd.DataFrame()
 
    abr = team_abr(team_id)
    if abr is None:
        return pd.DataFrame()
 
    at_bat = (
        ((statcast_df["home_team"] == abr) & (statcast_df["inning_topbot"] == "Bot")) |
        ((statcast_df["away_team"] == abr) & (statcast_df["inning_topbot"] == "Top"))
    )
 
    df = statcast_df[at_bat & (statcast_df["game_date"] >= start_str)].copy()
    return df[df["events"].notna()]
 
 
def build_lineup_features(
    team_id: int,
    as_of_date: date,
    statcast_df: pd.DataFrame,
    pitcher_hand: str = "R",
    trailing_days: int = 30,
) -> dict:
    empty = {
        "team_woba_vs_hand":     np.nan,
        "team_xwoba_vs_hand":    np.nan,
        "team_iso_vs_hand":      np.nan,
        "team_k_pct_vs_hand":    np.nan,
        "team_bb_pct_vs_hand":   np.nan,
        "team_hard_hit_vs_hand": np.nan,
        "team_woba_30d":         np.nan,
        "team_xwoba_30d":        np.nan,
    }
 
    if statcast_df.empty:
        return empty
 
    start_str = (
        pd.Timestamp(as_of_date) - pd.Timedelta(days=trailing_days)
    ).strftime("%Y-%m-%d")
 
    pa_df = _get_team_batters(team_id, statcast_df, start_str)
 
    if pa_df.empty:
        return empty
 
    split_df = _filter_by_hand(pa_df, pitcher_hand)
 
    return {
        "team_woba_vs_hand":     _woba(split_df),
        "team_xwoba_vs_hand":    _xwoba(split_df),
        "team_iso_vs_hand":      _iso(split_df),
        "team_k_pct_vs_hand":    _k_pct(split_df),
        "team_bb_pct_vs_hand":   _bb_pct(split_df),
        "team_hard_hit_vs_hand": _hard_hit_pct(split_df),
        "team_woba_30d":         _woba(pa_df),
        "team_xwoba_30d":        _xwoba(pa_df),
    }