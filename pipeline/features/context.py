# pipeline/features/context.py
import pandas as pd
import numpy as np
from datetime import date, datetime


# Park factors: run-scoring index relative to league average (1.0 = neutral)
# Source: update annually from Baseball Reference or FanGraphs
# Format: {venue_id: {"overall": float, "R": float, "L": float}}
_DEFAULT_PARK_FACTOR = {"overall": 1.0, "R": 1.0, "L": 1.0}


def _day_of_week_encoded(game_date: str) -> dict:
    """
    One-hot encode day of week (Mon–Sun).
    Weekend games (Fri/Sat/Sun) correlate with larger crowds / travel fatigue.
    """
    dt = pd.Timestamp(game_date)
    dow = dt.dayofweek  # 0=Mon ... 6=Sun
    return {
        "dow_friday":   int(dow == 4),
        "dow_saturday": int(dow == 5),
        "dow_sunday":   int(dow == 6),
        "is_weekend":   int(dow >= 4),
    }


def _game_time_features(game_time_utc: str | None, venue_id: int) -> dict:
    """
    Day/night flag from UTC tip-off time.
    Night games start >= 17:00 local (approximated via UTC; good enough for signal).
    """
    if not game_time_utc:
        return {"is_night_game": -1}  # unknown → model treats as missing
    try:
        dt = pd.Timestamp(game_time_utc)
        # Rough: games before 20:00 UTC are likely day games (ET/CT venues)
        return {"is_night_game": int(dt.hour >= 20)}
    except Exception:
        return {"is_night_game": -1}


def _season_phase(game_date: str) -> dict:
    """
    Encode approximate season phase.
    Early (Apr-May), mid (Jun-Jul), late (Aug-Sep), postseason (Oct+).
    Relevant for team fatigue and roster depth context.
    """
    month = pd.Timestamp(game_date).month
    return {
        "season_early":  int(month in (4, 5)),
        "season_mid":    int(month in (6, 7)),
        "season_late":   int(month in (8, 9)),
        "season_post":   int(month >= 10),
    }


def _series_position(game_pk: int, game_date: str, schedule_cache: dict | None) -> dict:
    """
    Placeholder: first/middle/last game of a series affects travel fatigue.
    Requires schedule context; returns neutral if unavailable.
    """
    # TODO: inject series metadata from schedule.py when available
    return {"series_game_num": -1}


def build_context_features(
    game: dict,
    park_factors: dict,
) -> dict:
    """
    Non-pitcher, non-batting contextual features for a game.

    Args:
        game: dict from schedule.py (must have game_pk, game_date,
              home_team_id, away_team_id, venue_id, game_time_utc)
        park_factors: {venue_id: {"overall", "R", "L"}} — caller supplies this

    Returns:
        Flat dict of context features, all numeric.
    """
    venue_id   = game.get("venue_id")
    game_date  = game.get("game_date", "")
    game_time  = game.get("game_time_utc")

    pf = park_factors.get(venue_id, _DEFAULT_PARK_FACTOR)

    features: dict = {}

    # Park factors
    features["park_factor_overall"] = float(pf.get("overall", 1.0))
    features["park_factor_R"]       = float(pf.get("R", 1.0))
    features["park_factor_L"]       = float(pf.get("L", 1.0))

    # Home field — always 1 for home team in this schema;
    # kept explicit so the model can learn its weight
    features["home_field_advantage"] = 1

    # Temporal
    features.update(_day_of_week_encoded(game_date))
    features.update(_game_time_features(game_time, venue_id))
    features.update(_season_phase(game_date))
    features.update(_series_position(game["game_pk"], game_date, None))

    return features


# ---------------------------------------------------------------------------
# Park factor loader (called once at pipeline startup)
# ---------------------------------------------------------------------------

def load_park_factors(path: str = "data/park_factors.csv") -> dict:
    """
    Load park factors from a CSV with columns:
        venue_id, overall, R, L

    Returns {venue_id: {"overall": float, "R": float, "L": float}}
    Falls back to neutral factors for any missing venue.
    """
    try:
        df = pd.read_csv(path)
        required = {"venue_id", "overall", "R", "L"}
        if not required.issubset(df.columns):
            raise ValueError(f"park_factors.csv missing columns: {required - set(df.columns)}")
        return {
            int(row["venue_id"]): {
                "overall": float(row["overall"]),
                "R":       float(row["R"]),
                "L":       float(row["L"]),
            }
            for _, row in df.iterrows()
        }
    except FileNotFoundError:
        # Non-fatal: model runs with neutral park factors
        return {}