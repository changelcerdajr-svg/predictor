# pipeline/risk/kelly.py
import numpy as np
from dataclasses import dataclass


@dataclass
class BetRecommendation:
    game_pk:        int
    side:           str          # "home" | "away"
    prob:           float        # calibrated P(side wins)
    american_odds:  int
    decimal_odds:   float
    implied_prob:   float
    edge:           float        # prob - implied_prob
    kelly_fraction: float        # full Kelly
    bet_fraction:   float        # fractional Kelly applied
    bet_units:      float        # fraction * bankroll
    ev:             float        # expected value per unit wagered


# ---------------------------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------------------------

def american_to_decimal(american: int) -> float:
    if american > 0:
        return american / 100 + 1.0
    return 100 / abs(american) + 1.0


def decimal_to_implied(decimal: float) -> float:
    return 1.0 / decimal


def remove_vig(odds_home: int, odds_away: int) -> tuple[float, float]:
    """
    Remove bookmaker vig via additive normalization.
    Returns (fair_p_home, fair_p_away).
    """
    imp_h = decimal_to_implied(american_to_decimal(odds_home))
    imp_a = decimal_to_implied(american_to_decimal(odds_away))
    total = imp_h + imp_a
    return imp_h / total, imp_a / total


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------

def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """
    Full Kelly: f = (p * b - q) / b
    where b = decimal_odds - 1, q = 1 - p.
    Returns 0.0 if no edge.
    """
    b = decimal_odds - 1.0
    q = 1.0 - prob
    f = (prob * b - q) / b
    return max(f, 0.0)


# ---------------------------------------------------------------------------
# Bet sizing
# ---------------------------------------------------------------------------

def size_bet(
    game_pk:       int,
    side:          str,
    prob:          float,
    american_odds: int,
    bankroll:      float,
    kelly_multiplier: float = 0.25,   # quarter-Kelly default
    min_edge:      float = 0.03,      # minimum edge to bet
    max_fraction:  float = 0.05,      # hard cap: never risk > 5% bankroll
) -> BetRecommendation | None:
    """
    Returns a BetRecommendation or None if no edge / invalid inputs.

    kelly_multiplier: fraction of full Kelly to use (0.25 = quarter-Kelly)
    min_edge:         floor on (prob - implied_prob) to place bet
    max_fraction:     hard cap regardless of Kelly output
    """
    if not (0.0 < prob < 1.0):
        return None
    if np.isnan(prob):
        return None

    dec_odds    = american_to_decimal(american_odds)
    implied     = decimal_to_implied(dec_odds)
    edge        = prob - implied

    if edge < min_edge:
        return None

    full_kelly  = kelly_fraction(prob, dec_odds)
    frac_kelly  = min(full_kelly * kelly_multiplier, max_fraction)

    if frac_kelly <= 0.0:
        return None

    ev = prob * (dec_odds - 1.0) - (1.0 - prob)

    return BetRecommendation(
        game_pk        = game_pk,
        side           = side,
        prob           = prob,
        american_odds  = american_odds,
        decimal_odds   = dec_odds,
        implied_prob   = implied,
        edge           = edge,
        kelly_fraction = full_kelly,
        bet_fraction   = frac_kelly,
        bet_units      = frac_kelly * bankroll,
        ev             = ev,
    )


# ---------------------------------------------------------------------------
# Slate sizing (multiple games)
# ---------------------------------------------------------------------------

def size_slate(
    predictions: list[dict],        # [{game_pk, prob_home, odds_home, odds_away}, ...]
    bankroll:    float,
    kelly_multiplier: float = 0.25,
    min_edge:    float = 0.03,
    max_fraction: float = 0.05,
    max_exposure: float = 0.20,     # max total bankroll at risk across slate
) -> list[BetRecommendation]:
    """
    Size all bets in a slate with total exposure cap.
    Bets are sorted by edge descending; exposure cap halts further bets.
    """
    bets: list[BetRecommendation] = []
    total_fraction = 0.0

    candidates = []
    for p in predictions:
        for side, prob, odds in [
            ("home", p["prob_home"], p["odds_home"]),
            ("away", 1.0 - p["prob_home"], p["odds_away"]),
        ]:
            rec = size_bet(
                game_pk        = p["game_pk"],
                side           = side,
                prob           = prob,
                american_odds  = odds,
                bankroll       = bankroll,
                kelly_multiplier = kelly_multiplier,
                min_edge       = min_edge,
                max_fraction   = max_fraction,
            )
            if rec is not None:
                candidates.append(rec)

    # Sort by edge descending — take highest-edge bets first
    candidates.sort(key=lambda r: r.edge, reverse=True)

    for rec in candidates:
        if total_fraction + rec.bet_fraction > max_exposure:
            # Re-size to remaining exposure budget
            remaining = max_exposure - total_fraction
            if remaining < 0.005:   # < 0.5% not worth placing
                break
            rec = BetRecommendation(
                **{**rec.__dict__,
                   "bet_fraction": remaining,
                   "bet_units":    remaining * bankroll}
            )
        bets.append(rec)
        total_fraction += rec.bet_fraction

    return bets