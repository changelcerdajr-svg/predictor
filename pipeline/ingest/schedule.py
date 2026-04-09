# pipeline/ingest/schedule.py
import requests
from datetime import date
import json
from pathlib import Path

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

def fetch_schedule(game_date: date) -> list[dict]:
    """
    Fetch MLB schedule from official Stats API.
    Returns list of game dicts with gamePk, home/away team IDs, venue, time.
    """
    params = {
        "sportId": 1,
        "date": game_date.strftime("%Y-%m-%d"),
        "hydrate": "probablePitcher,venue,weather,broadcasts"
    }
    resp = requests.get(MLB_SCHEDULE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            # Only regular season and playoffs
            if game.get("gameType") not in ("R", "F", "D", "L", "W"):
                continue
            games.append({
                "game_pk":        game["gamePk"],
                "game_date":      game_date.isoformat(),
                "home_team_id":   game["teams"]["home"]["team"]["id"],
                "away_team_id":   game["teams"]["away"]["team"]["id"],
                "home_sp_id":     _extract_pitcher(game, "home"),
                "away_sp_id":     _extract_pitcher(game, "away"),
                "venue_id":       game["venue"]["id"],
                "game_time_utc":  game.get("gameDate"),
            })
    return games

def _extract_pitcher(game: dict, side: str) -> int | None:
    try:
        return game["teams"][side]["probablePitcher"]["id"]
    except KeyError:
        return None