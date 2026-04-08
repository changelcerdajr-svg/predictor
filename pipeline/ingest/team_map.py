# pipeline/ingest/team_map.py

_ALIASES: dict[str, str] = {
    "arizona diamondbacks":    "diamondbacks",
    "atlanta braves":          "braves",
    "baltimore orioles":       "orioles",
    "boston red sox":          "red sox",
    "chicago cubs":            "cubs",
    "chicago white sox":       "white sox",
    "cincinnati reds":         "reds",
    "cleveland guardians":     "guardians",
    "colorado rockies":        "rockies",
    "detroit tigers":          "tigers",
    "houston astros":          "astros",
    "kansas city royals":      "royals",
    "los angeles angels":      "angels",
    "los angeles dodgers":     "dodgers",
    "miami marlins":           "marlins",
    "milwaukee brewers":       "brewers",
    "minnesota twins":         "twins",
    "new york mets":           "mets",
    "new york yankees":        "yankees",
    "oakland athletics":       "athletics",
    "philadelphia phillies":   "phillies",
    "pittsburgh pirates":      "pirates",
    "san diego padres":        "padres",
    "san francisco giants":    "giants",
    "seattle mariners":        "mariners",
    "st. louis cardinals":     "cardinals",
    "st louis cardinals":      "cardinals",
    "tampa bay rays":          "rays",
    "texas rangers":           "rangers",
    "toronto blue jays":       "blue jays",
    "washington nationals":    "nationals",
}


def normalize_team_name(name: str) -> str:
    key = name.strip().lower()
    return _ALIASES.get(key, key)


TEAM_ID_TO_ABR: dict[int, str] = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}
 
def team_abr(team_id: int) -> str | None:
    """Returns Statcast abbreviation for a given MLB Stats API team_id."""
    return TEAM_ID_TO_ABR.get(team_id)