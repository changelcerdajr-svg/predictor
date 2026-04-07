# pipeline/features/pitching.py — bullpen section

def build_bullpen_features(
    team_id: int,
    as_of_date: date,
    statcast_df: pd.DataFrame,   # filtered: game_date < as_of_date
    trailing_days: int = 14,
) -> dict:
    """
    Bullpen quality and fatigue signal.
    Key insight: recent FIP matters more than season FIP because rosters change.
    Key risk: must use as_of_date guard — never full-season.
    """
    cutoff_str = (
        pd.Timestamp(as_of_date) - pd.Timedelta(days=trailing_days)
    ).strftime("%Y-%m-%d")
    
    # Filter to team's bullpen appearances (non-starters: IP <= 2 typically)
    # Heuristic: pitchers with < 2 IP per appearance in trailing window
    team_pitches = statcast_df[
        (statcast_df["home_team"] == team_id) | 
        (statcast_df["away_team"] == team_id)
    ]
    # Identify relievers for this team in this window
    # (this requires a roster/role lookup; simplified here)
    team_relievers = statcast_df[
        (statcast_df["pitcher_team"] == team_id) &
        (statcast_df["game_date"] >= cutoff_str)
    ]
    
    return {
        "bullpen_fip_14d":      compute_fip(team_relievers),
        "bullpen_whiff_14d":    _whiff_pct(team_relievers),
        "bullpen_pitch_count_3d": _recent_pitch_count(
            team_relievers, as_of_date, days=3
        ),  # Fatigue proxy
    }