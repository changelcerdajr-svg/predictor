# run/backtest_runner.py
"""
Walk-forward backtest over historical seasons.
Usage:
    python -m run.backtest_runner --start 2022-04-07 --end 2023-10-01
    python -m run.backtest_runner --start 2022-04-07 --end 2023-10-01 --min-edge 0.04
"""
import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from pipeline.ingest.schedule         import fetch_schedule
from pipeline.ingest.statcast         import fetch_and_cache_statcast
from pipeline.features.feature_matrix import build_game_feature_vector
from pipeline.features.context        import load_park_factors
from pipeline.models.gradient_boost   import train, predict, FEATURE_COLS
from pipeline.risk.kelly              import size_slate
from pipeline.backtest.evaluator      import evaluate
from pipeline.ingest.odds import fetch_odds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
 
OUTPUT_DIR           = Path("data/backtest")
STATCAST_LOOKBACK    = 90
TRAIN_MINIMUM_DAYS   = 365
RETRAIN_EVERY_DAYS   = 30
 
 
# ---------------------------------------------------------------------------
# Statcast loader — FIX: accepts pre-loaded df to avoid re-reading on every day
# ---------------------------------------------------------------------------
 
def _load_full_statcast(start: date, end: date) -> pd.DataFrame:
    """Load entire backtest window once. Caller slices by date."""
    df = fetch_and_cache_statcast(start, end)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    return df
 
 
def _slice_statcast(full_df: pd.DataFrame, as_of_date: date) -> pd.DataFrame:
    """Return rows strictly before as_of_date — no re-read from disk."""
    cutoff = as_of_date.isoformat()
    start  = (as_of_date - timedelta(days=STATCAST_LOOKBACK)).isoformat()
    return full_df[
        (full_df["game_date"] >= start) & (full_df["game_date"] < cutoff)
    ].copy()
 
 
# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
 
def _fetch_outcomes(game_pks: list[int]) -> dict[int, int]:
    """
    Fetch actual home_win (1/0) for completed games via MLB Stats API.
    Returns {game_pk: 1|0}. Returns {} for games not yet final.
    """
    import requests
    results = {}
    for pk in game_pks:
        try:
            url = f"https://statsapi.mlb.com/api/v1/game/{pk}/linescore"
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            home_runs = data["teams"]["home"]["runs"]
            away_runs = data["teams"]["away"]["runs"]
            # Only record if game is final (both sides have runs recorded)
            if home_runs is not None and away_runs is not None:
                results[pk] = int(home_runs > away_runs)
        except Exception:
            pass  # game not final or API error — skip
    return results
 
 
# ---------------------------------------------------------------------------
# Training data accumulation  (FIX: uses shared full_df slice, not per-day reload)
# ---------------------------------------------------------------------------
 
def _accumulate_training_data(
    start:        date,
    end:          date,
    full_df:      pd.DataFrame,
    park_factors: dict,
) -> pd.DataFrame:
    rows     = []
    cur_date = start
 
    while cur_date <= end:
        games = fetch_schedule(cur_date)
        if games:
            statcast_df = _slice_statcast(full_df, cur_date)
            outcomes    = _fetch_outcomes([g["game_pk"] for g in games])
 
            for game in games:
                pk  = game["game_pk"]
                row = build_game_feature_vector(game, cur_date, statcast_df, park_factors)
                if row is None or pk not in outcomes:
                    continue
                row["home_win"]  = outcomes[pk]
                row["game_date"] = cur_date.isoformat()
                rows.append(row)
 
        cur_date += timedelta(days=1)
 
    return pd.DataFrame(rows) if rows else pd.DataFrame()
 
 
# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
 
 
def run_backtest(
    start:            date,
    end:              date,
    kelly_multiplier: float = 0.25,
    min_edge:         float = 0.03,
    max_fraction:     float = 0.05,
    max_exposure:     float = 0.20,
    initial_bankroll: float = 1000.0,
) -> pd.DataFrame:
    park_factors = load_park_factors()
 
    train_end = start + timedelta(days=TRAIN_MINIMUM_DAYS)
    if train_end >= end:
        log.error("Backtest window too short for minimum training period.")
        return pd.DataFrame()
 
    # FIX: load full Statcast window once — reused by every day via _slice_statcast()
    log.info(f"Loading full Statcast window {start} → {end - timedelta(days=1)}")
    full_statcast = _load_full_statcast(
        start - timedelta(days=STATCAST_LOOKBACK),
        end   - timedelta(days=1),
    )
    log.info(f"Statcast rows in memory: {len(full_statcast):,}")
 
    log.info(f"Accumulating training data {start} → {train_end}")
    train_df = _accumulate_training_data(start, train_end, full_statcast, park_factors)
 
    if train_df.empty:
        log.error("No training data accumulated. Aborting.")
        return pd.DataFrame()
 
    model, scaler, metrics = train(train_df)
    log.info(f"Initial model | AUC={metrics['oof_auc']:.4f} | "
             f"Brier={metrics['oof_brier_cal']:.4f} | T={metrics['temperature_T']:.3f}")
 
    bankroll      = initial_bankroll
    peak_bankroll = initial_bankroll
    last_retrain  = train_end
    ledger_rows   = []
    cur_date      = train_end + timedelta(days=1)
 
    while cur_date <= end:
        if (cur_date - last_retrain).days >= RETRAIN_EVERY_DAYS:
            new_train_df = _accumulate_training_data(start, cur_date, full_statcast, park_factors)
            if not new_train_df.empty:
                model, scaler, metrics = train(new_train_df)
                last_retrain = cur_date
                log.info(f"Re-trained {cur_date} | AUC={metrics['oof_auc']:.4f}")
 
        games = fetch_schedule(cur_date)
        if not games:
            cur_date += timedelta(days=1)
            continue
 
        statcast_df  = _slice_statcast(full_statcast, cur_date)
        feature_rows = []
 
        for game in games:
            row = build_game_feature_vector(game, cur_date, statcast_df, park_factors)
            if row is None:
                continue
            feature_rows.append(row)
 
        if not feature_rows:
            cur_date += timedelta(days=1)
            continue
 
        feat_df = pd.DataFrame(feature_rows)
        probs   = predict(feat_df, model, scaler)
        feat_df["prob_home"] = probs
        feat_df = feat_df.dropna(subset=["prob_home"])
 
        odds = fetch_odds(games, cur_date)
        slate = [
            {
                "game_pk":   int(row["game_pk"]),
                "prob_home": float(row["prob_home"]),
                "odds_home": odds[int(row["game_pk"])]["odds_home"],
                "odds_away": odds[int(row["game_pk"])]["odds_away"],
            }
            for _, row in feat_df.iterrows()
            if int(row["game_pk"]) in odds
        ]
 
        bets = size_slate(
            predictions      = slate,
            bankroll         = bankroll,
            kelly_multiplier = kelly_multiplier,
            min_edge         = min_edge,
            max_fraction     = max_fraction,
            max_exposure     = max_exposure,
        )
 
        outcomes = _fetch_outcomes([g["game_pk"] for g in games])
        day_pnl  = 0.0
 
        for bet in bets:
            outcome = outcomes.get(bet.game_pk)
            if outcome is None:
                continue
            home_won = bool(outcome)
            won      = (bet.side == "home" and home_won) or \
                       (bet.side == "away" and not home_won)
            pnl      = bet.bet_units * (bet.decimal_odds - 1.0) if won \
                       else -bet.bet_units
            day_pnl += pnl
 
            ledger_rows.append({
                "date":         cur_date.isoformat(),
                "game_pk":      bet.game_pk,
                "side":         bet.side,
                "prob":         bet.prob,
                "edge":         bet.edge,
                "odds":         bet.american_odds,
                "bet_units":    bet.bet_units,
                "won":          int(won),
                "pnl":          pnl,
                "bankroll_pre": bankroll,
            })
 
        bankroll      += day_pnl
        peak_bankroll  = max(peak_bankroll, bankroll)
 
        log.info(
            f"{cur_date} | bets={len(bets)} | "
            f"day_pnl=${day_pnl:+.2f} | bankroll=${bankroll:.2f}"
        )
 
        cur_date += timedelta(days=1)
 
    ledger = pd.DataFrame(ledger_rows)
    if not ledger.empty:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"backtest_{start}_{end}.csv"
        ledger.to_csv(path, index=False)
        log.info(f"Ledger saved → {path}")
 
        summary = evaluate(ledger, initial_bankroll)
        log.info("\n=== BACKTEST SUMMARY ===")
        for k, v in summary.items():
            log.info(f"  {k:<28} {v}")
 
    return ledger
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",    type=lambda s: date.fromisoformat(s), required=True)
    parser.add_argument("--end",      type=lambda s: date.fromisoformat(s), required=True)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--kelly",    type=float, default=0.25)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    args = parser.parse_args()
 
    run_backtest(
        start            = args.start,
        end              = args.end,
        min_edge         = args.min_edge,
        kelly_multiplier = args.kelly,
        initial_bankroll = args.bankroll,
    )