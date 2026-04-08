# pipeline/features/feature_matrix.py
from datetime import date
import pandas as pd
from pipeline.features.pitching import (   # FIX: was missing — caused NameError
    build_bullpen_features,
    _get_pitcher_hand,
)
from pipeline.features.batting import build_lineup_features
from pipeline.ingest.pitcher_stats import build_pitcher_features
from pipeline.features.context import build_context_features
 
 
def build_game_feature_vector(
    game: dict,
    as_of_date: date,
    statcast_df: pd.DataFrame,
    park_factors: dict,
) -> dict | None:
    """
    Builds a single game-level feature row.
    Returns None if critical features are unavailable (missing SP, etc.).
 
    CRITICAL: statcast_df must be pre-filtered to game_date < as_of_date
    before this function is called. The caller is responsible for this.
    """
    if game["home_sp_id"] is None or game["away_sp_id"] is None:
        return None
 
    home_sp = build_pitcher_features(
        game["home_sp_id"], as_of_date, statcast_df
    )
    away_sp = build_pitcher_features(
        game["away_sp_id"], as_of_date, statcast_df
    )
    home_bp = build_bullpen_features(
        game["home_team_id"], as_of_date, statcast_df
    )
    away_bp = build_bullpen_features(
        game["away_team_id"], as_of_date, statcast_df
    )
 
    home_sp_hand = _get_pitcher_hand(game["home_sp_id"], statcast_df)
    away_sp_hand = _get_pitcher_hand(game["away_sp_id"], statcast_df)
 
    home_bat = build_lineup_features(
        game["home_team_id"], as_of_date, statcast_df,
        pitcher_hand=away_sp_hand
    )
    away_bat = build_lineup_features(
        game["away_team_id"], as_of_date, statcast_df,
        pitcher_hand=home_sp_hand
    )
 
    ctx = build_context_features(game, park_factors)
 
    row = {
        "game_pk":    game["game_pk"],
        "game_date":  game["game_date"],
 
        "sp_fip_diff":       (away_sp["sp_fip_60d"] or 0) - (home_sp["sp_fip_60d"] or 0),
        "sp_xera_diff":      (away_sp["sp_xera_proxy_60d"] or 0) - (home_sp["sp_xera_proxy_60d"] or 0),
        "sp_k_pct_diff":     (home_sp["sp_k_pct_60d"] or 0) - (away_sp["sp_k_pct_60d"] or 0),
        "sp_bb_pct_diff":    (away_sp["sp_bb_pct_60d"] or 0) - (home_sp["sp_bb_pct_60d"] or 0),
 
        "home_sp_fip":       home_sp["sp_fip_60d"],
        "away_sp_fip":       away_sp["sp_fip_60d"],
        "home_sp_days_rest": home_sp["sp_days_rest"],
        "away_sp_days_rest": away_sp["sp_days_rest"],
 
        "bp_fip_diff":       (away_bp["bullpen_fip_14d"] or 0) - (home_bp["bullpen_fip_14d"] or 0),
        "home_bp_fatigue":   home_bp["bullpen_pitch_count_3d"],
        "away_bp_fatigue":   away_bp["bullpen_pitch_count_3d"],
 
        "woba_diff":         (home_bat["team_woba_vs_hand"] or 0) - (away_bat["team_woba_vs_hand"] or 0),
        "xwoba_diff":        (home_bat["team_xwoba_vs_hand"] or 0) - (away_bat["team_xwoba_vs_hand"] or 0),
 
        **ctx,
    }
 
    return row