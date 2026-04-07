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


# Stats API team_id → canonical short name
TEAM_ID_TO_NAME: dict[int, str] = {
    109: "diamondbacks", 144: "braves",    110: "orioles",
    111: "red sox",      112: "cubs",      145: "white sox",
    113: "reds",         114: "guardians", 115: "rockies",
    116: "tigers",       117: "astros",    118: "royals",
    108: "angels",       119: "dodgers",   146: "marlins",
    158: "brewers",      142: "twins",     121: "mets",
    147: "yankees",      133: "athletics", 143: "phillies",
    134: "pirates",      135: "padres",    137: "giants",
    136: "mariners",     138: "cardinals", 139: "rays",
    140: "rangers",      141: "blue jays", 120: "nationals",
}


def team_id_to_name(team_id: int) -> str:
    return TEAM_ID_TO_NAME.get(team_id, str(team_id))