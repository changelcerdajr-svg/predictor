# pipeline/features/feature_matrix.py
import pandas as pd
import numpy as np
from datetime import date

from pipeline.ingest.pitcher_stats  import build_pitcher_features
from pipeline.features.pitching     import build_bullpen_features, _get_pitcher_hand
from pipeline.features.batting      import build_lineup_features
from pipeline.features.context      import build_context_features


def build_game_feature_vector(
    game:         dict,
    as_of_date:   date,
    statcast_df:  pd.DataFrame,
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

    # ── Pitcher features ────────────────────────────────────────────────────
    home_sp = build_pitcher_features(
        game["home_sp_id"], as_of_date, statcast_df
    )
    away_sp = build_pitcher_features(
        game["away_sp_id"], as_of_date, statcast_df
    )

    # ── Bullpen features ─────────────────────────────────────────────────────
    home_bp = build_bullpen_features(
        game["home_team_id"], as_of_date, statcast_df
    )
    away_bp = build_bullpen_features(
        game["away_team_id"], as_of_date, statcast_df
    )

    # ── SP handedness → sets batting split direction ─────────────────────────
    home_sp_hand = _get_pitcher_hand(game["home_sp_id"], statcast_df)
    away_sp_hand = _get_pitcher_hand(game["away_sp_id"], statcast_df)

    # ── Batting features (split vs opposing SP hand) ─────────────────────────
    home_bat = build_lineup_features(
        game["home_team_id"], as_of_date, statcast_df,
        pitcher_hand=away_sp_hand   # home bats face AWAY SP
    )
    away_bat = build_lineup_features(
        game["away_team_id"], as_of_date, statcast_df,
        pitcher_hand=home_sp_hand   # away bats face HOME SP
    )

    # ── Context features ─────────────────────────────────────────────────────
    ctx = build_context_features(game, park_factors)

    # ── Null-safe helper ─────────────────────────────────────────────────────
    def v(x):
        return x if (x is not None and not (isinstance(x, float) and np.isnan(x))) else 0.0

    # ── Assemble row ─────────────────────────────────────────────────────────
    # Differential features: positive value = home team advantage
    row = {
        "game_pk":   game["game_pk"],
        "game_date": game["game_date"],

        # SP differentials
        "sp_fip_diff":    v(away_sp["sp_fip_60d"])          - v(home_sp["sp_fip_60d"]),
        "sp_xera_diff":   v(away_sp["sp_xera_proxy_60d"])   - v(home_sp["sp_xera_proxy_60d"]),
        "sp_k_pct_diff":  v(home_sp["sp_k_pct_60d"])        - v(away_sp["sp_k_pct_60d"]),
        "sp_bb_pct_diff": v(away_sp["sp_bb_pct_60d"])       - v(home_sp["sp_bb_pct_60d"]),

        # SP absolutes (nonlinear models benefit from raw values)
        "home_sp_fip":       home_sp["sp_fip_60d"],
        "away_sp_fip":       away_sp["sp_fip_60d"],
        "home_sp_days_rest": home_sp["sp_days_rest"],
        "away_sp_days_rest": away_sp["sp_days_rest"],

        # SP recent load
        "home_sp_pitch_count_7d": home_sp["sp_pitch_count_7d"],
        "away_sp_pitch_count_7d": away_sp["sp_pitch_count_7d"],

        # SP whiff / HR-FB
        "home_sp_whiff":  home_sp["sp_whiff_pct_60d"],
        "away_sp_whiff":  away_sp["sp_whiff_pct_60d"],
        "home_sp_hr_fb":  home_sp["sp_hr_fb_60d"],
        "away_sp_hr_fb":  away_sp["sp_hr_fb_60d"],

        # SP short-window xwOBA allowed
        "home_sp_xwoba_14d": home_sp["sp_xwoba_allowed_14d"],
        "away_sp_xwoba_14d": away_sp["sp_xwoba_allowed_14d"],

        # Bullpen differentials
        "bp_fip_diff":     v(away_bp["bullpen_fip_14d"])   - v(home_bp["bullpen_fip_14d"]),
        "bp_whiff_diff":   v(home_bp["bullpen_whiff_14d"]) - v(away_bp["bullpen_whiff_14d"]),
        "bp_k_pct_diff":   v(home_bp["bullpen_k_pct_14d"]) - v(away_bp["bullpen_k_pct_14d"]),

        # Bullpen fatigue (raw pitch count last 3 days)
        "home_bp_fatigue": home_bp["bullpen_pitch_count_3d"],
        "away_bp_fatigue": away_bp["bullpen_pitch_count_3d"],

        # Bullpen absolutes
        "home_bp_fip": home_bp["bullpen_fip_14d"],
        "away_bp_fip": away_bp["bullpen_fip_14d"],

        # Batting differentials (hand-split)
        "woba_diff":       v(home_bat["team_woba_vs_hand"])  - v(away_bat["team_woba_vs_hand"]),
        "xwoba_diff":      v(home_bat["team_xwoba_vs_hand"]) - v(away_bat["team_xwoba_vs_hand"]),
        "iso_diff":        v(home_bat["team_iso_vs_hand"])   - v(away_bat["team_iso_vs_hand"]),
        "k_pct_bat_diff":  v(away_bat["team_k_pct_vs_hand"]) - v(home_bat["team_k_pct_vs_hand"]),
        "bb_pct_bat_diff": v(home_bat["team_bb_pct_vs_hand"])- v(away_bat["team_bb_pct_vs_hand"]),
        "hard_hit_diff":   v(home_bat["team_hard_hit_vs_hand"]) - v(away_bat["team_hard_hit_vs_hand"]),

        # Batting 30d baseline (no split)
        "home_woba_30d": home_bat["team_woba_30d"],
        "away_woba_30d": away_bat["team_woba_30d"],
        "home_xwoba_30d": home_bat["team_xwoba_30d"],
        "away_xwoba_30d": away_bat["team_xwoba_30d"],

        # Context (park, schedule, time)
        **ctx,
    }

    return row