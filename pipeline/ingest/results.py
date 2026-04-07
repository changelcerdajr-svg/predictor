# pipeline/ingest/results.py
import requests
import pandas as pd
from datetime import date
from pathlib import Path
import logging

log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/raw/results")
MLB_BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def _fetch_linescore(game_pk: int) -> dict | None:
    try:
        r = requests.get(MLB_BOXSCORE_URL.format(game_pk=game_pk), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Linescore fetch failed game_pk={game_pk}: {e}")
        return None


def _parse_linescore(data: dict) -> dict | None:
    try:
        home = data["teams"]["home"]["runs"]
        away = data["teams"]["away"]["runs"]
        return {"home_runs": int(home), "away_runs": int(away), "home_win": int(home > away)}
    except (KeyError, TypeError):
        return None


def fetch_outcomes_for_date(game_date: date) -> dict[int, int]:
    """
    Returns {game_pk: home_win (1|0)} for all final games on game_date.
    Uses cache at data/raw/results/YYYY-MM-DD.parquet.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = RESULTS_DIR / f"{game_date.isoformat()}.parquet"

    if cache.exists():
        df = pd.read_parquet(cache)
        return dict(zip(df["game_pk"], df["home_win"]))

    params = {
        "sportId": 1,
        "date":    game_date.strftime("%Y-%m-%d"),
        "hydrate": "linescore",
    }
    try:
        r = requests.get(MLB_SCHEDULE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error(f"Schedule fetch failed {game_date}: {e}")
        return {}

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue
            if game.get("gameType") not in ("R", "F", "D", "L", "W"):
                continue
            pk = game["gamePk"]
            ls = game.get("linescore") or {}
            try:
                home = ls["teams"]["home"]["runs"]
                away = ls["teams"]["away"]["runs"]
            except (KeyError, TypeError):
                # fallback: individual linescore endpoint
                raw = _fetch_linescore(pk)
                if raw is None:
                    continue
                parsed = _parse_linescore(raw)
                if parsed is None:
                    continue
                home, away = parsed["home_runs"], parsed["away_runs"]

            rows.append({
                "game_pk":   pk,
                "game_date": game_date.isoformat(),
                "home_runs": int(home),
                "away_runs": int(away),
                "home_win":  int(int(home) > int(away)),
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(cache, index=False)
        log.info(f"Cached {len(rows)} results for {game_date}")
        return dict(zip(df["game_pk"], df["home_win"]))

    return {}


def fetch_outcomes(game_pks: list[int], game_date: date) -> dict[int, int]:
    """
    Thin wrapper used by backtest engine.
    Fetches all outcomes for game_date, filters to requested game_pks.
    """
    all_outcomes = fetch_outcomes_for_date(game_date)
    return {pk: all_outcomes[pk] for pk in game_pks if pk in all_outcomes}