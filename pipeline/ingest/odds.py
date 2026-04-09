# pipeline/ingest/odds.py
"""
MLB moneyline scraper via SportsbookReview (sbrscrape).
Output: {game_pk: {"odds_home": int, "odds_away": int}}

Drop-in replacement for _fetch_odds() / _stub_odds() in:
  - run/daily_pipeline.py
  - run/backtest_runner.py

Install: pip install sbrscrape
"""
import logging
from datetime import date

log = logging.getLogger(__name__)

_FALLBACK = {"odds_home": -110, "odds_away": -110}

# Best lines first. SBR includes: pinnacle, draftkings, fanduel, betmgm,
# caesars, bovada, betonline, consensus, and others.
_BOOK_PRIORITY = [
    "pinnacle", "draftkings", "fanduel", "betmgm",
    "caesars", "bovada", "betonline", "consensus",
]

# SBR short name → MLB Stats API team_id
_ABBR_TO_ID: dict[str, int] = {
    "LAA": 108, "ARI": 109, "BAL": 110, "BOS": 111,
    "CHC": 112, "CIN": 113, "CLE": 114, "COL": 115,
    "DET": 116, "HOU": 117, "KC":  118, "LAD": 119,
    "WSH": 120, "NYM": 121, "OAK": 133, "PIT": 134,
    "SD":  135, "SEA": 136, "SF":  137, "STL": 138,
    "TB":  139, "TEX": 140, "TOR": 141, "MIN": 142,
    "PHI": 143, "ATL": 144, "CWS": 145, "MIA": 146,
    "NYY": 147, "MIL": 158,
    # SBR alternate abbreviations
    "WAS": 120, "CHW": 145, "KAN": 118, "SFG": 137,
    "SDP": 135, "TBR": 139, "KCR": 118,
}


def _best_ml(ml_dict: dict) -> int | None:
    """First non-None moneyline from priority book list, fallback to any."""
    for book in _BOOK_PRIORITY:
        val = ml_dict.get(book)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    for val in ml_dict.values():
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def _build_sbr_index(sbr_games: list[dict]) -> dict[tuple[int, int], dict]:
    """
    {(home_team_id, away_team_id): sbr_game}
    Skips games where either abbreviation is unresolvable.
    """
    index: dict[tuple[int, int], dict] = {}
    for g in sbr_games:
        h_id = _ABBR_TO_ID.get((g.get("home_team_abbr") or "").upper())
        a_id = _ABBR_TO_ID.get((g.get("away_team_abbr") or "").upper())
        if h_id and a_id:
            index[(h_id, a_id)] = g
    return index


def _scrape_sbr(game_date: date) -> list[dict]:
    """Scrape SportsbookReview via sbrscrape. Empty list on any error."""
    try:
        from sbrscrape import Scoreboard
        games = Scoreboard(sport="MLB", date=game_date.strftime("%Y-%m-%d")).games or []
        log.info(f"SBR: {len(games)} games scraped for {game_date}")
        return games
    except ImportError:
        log.error("sbrscrape not installed — run: pip install sbrscrape")
        return []
    except Exception as e:
        log.warning(f"SBR scrape failed ({game_date}): {e}")
        return []


def fetch_odds(games: list[dict], game_date: date | None = None) -> dict[int, dict]:
    """
    Drop-in replacement for _fetch_odds() and _stub_odds().

    Args:
        games:      list of game dicts from fetch_schedule()
                    requires keys: game_pk, home_team_id, away_team_id
        game_date:  defaults to today

    Returns:
        {game_pk: {"odds_home": int, "odds_away": int}}

    Guarantees:
        - Every game_pk in `games` has an entry in the output
        - Falls back to -110/-110 if scrape fails or match not found
    """
    if not games:
        return {}

    if game_date is None:
        from datetime import date as _d
        game_date = _d.today()

    sbr_games = _scrape_sbr(game_date)

    if not sbr_games:
        log.warning("SBR returned no data — all games get fallback odds")
        return {g["game_pk"]: _FALLBACK.copy() for g in games}

    sbr_index = _build_sbr_index(sbr_games)
    result: dict[int, dict] = {}
    matched = 0

    for g in games:
        pk  = g["game_pk"]
        key = (g["home_team_id"], g["away_team_id"])

        sbr_game = sbr_index.get(key)
        if sbr_game is None:
            log.debug(f"pk={pk} no SBR match (h={g['home_team_id']} a={g['away_team_id']})")
            result[pk] = _FALLBACK.copy()
            continue

        home_ml = _best_ml(sbr_game.get("home_ml") or {})
        away_ml = _best_ml(sbr_game.get("away_ml") or {})

        if home_ml is None or away_ml is None:
            log.debug(f"pk={pk} SBR matched but moneyline missing")
            result[pk] = _FALLBACK.copy()
            continue

        result[pk] = {"odds_home": home_ml, "odds_away": away_ml}
        matched += 1

    log.info(f"Odds: {matched}/{len(games)} matched from SBR")
    return result