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
from pipeline.calibration.temperature_scaling import TemperatureScaler
from pipeline.risk.kelly              import size_slate
from pipeline.backtest.evaluator      import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

OUTPUT_DIR           = Path("data/backtest")
STATCAST_LOOKBACK    = 90    # days of history fed to feature builders
TRAIN_MINIMUM_DAYS   = 365   # don't start predicting until 1 full season exists
RETRAIN_EVERY_DAYS   = 30    # re-train model monthly


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_statcast_window(
    as_of_date: date,
    lookback_days: int = STATCAST_LOOKBACK,
) -> pd.DataFrame:
    start = as_of_date - timedelta(days=lookback_days)
    end   = as_of_date - timedelta(days=1)
    df    = fetch_and_cache_statcast(start, end)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    return df[df["game_date"] < as_of_date.isoformat()].copy()


def _build_feature_row(
    game:        dict,
    as_of_date:  date,
    statcast_df: pd.DataFrame,
    park_factors: dict,
) -> dict | None:
    return build_game_feature_vector(game, as_of_date, statcast_df, park_factors)


# ---------------------------------------------------------------------------
# Training data accumulation
# ---------------------------------------------------------------------------

def _fetch_outcomes(game_pks: list[int]) -> dict[int, int]:
    """
    Placeholder: fetch actual home_win (1/0) for completed games.
    Replace with Stats API or local results DB.
    Format: {game_pk: 1|0}
    """
    # TODO: wire to pipeline/ingest/results.py
    return {}


def _accumulate_training_data(
    start:       date,
    end:         date,
    park_factors: dict,
) -> pd.DataFrame:
    """
    Build labelled feature rows for every game in [start, end].
    Iterates day-by-day; skips games with missing outcomes.
    """
    rows     = []
    cur_date = start

    while cur_date <= end:
        games = fetch_schedule(cur_date)
        if games:
            statcast_df = _load_statcast_window(cur_date)
            outcomes    = _fetch_outcomes([g["game_pk"] for g in games])

            for game in games:
                pk  = game["game_pk"]
                row = _build_feature_row(game, cur_date, statcast_df, park_factors)
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

def _stub_odds(games: list[dict]) -> dict[int, dict]:
    """Neutral -110 both sides until live odds feed is wired."""
    return {g["game_pk"]: {"odds_home": -110, "odds_away": -110} for g in games}


def run_backtest(
    start:           date,
    end:             date,
    kelly_multiplier: float = 0.25,
    min_edge:        float  = 0.03,
    max_fraction:    float  = 0.05,
    max_exposure:    float  = 0.20,
    initial_bankroll: float = 1000.0,
) -> pd.DataFrame:
    """
    Walk-forward backtest:
      1. Accumulate training data up to train_end
      2. Train model
      3. Predict and size bets one day at a time
      4. Re-train every RETRAIN_EVERY_DAYS days
      5. Return daily P&L ledger
    """
    park_factors = load_park_factors()

    # ── Phase 1: accumulate initial training set ─────────────────────────────
    train_end  = start + timedelta(days=TRAIN_MINIMUM_DAYS)
    if train_end >= end:
        log.error("Backtest window too short for minimum training period.")
        return pd.DataFrame()

    log.info(f"Accumulating training data {start} → {train_end}")
    train_df = _accumulate_training_data(start, train_end, park_factors)

    if train_df.empty:
        log.error("No training data accumulated. Aborting.")
        return pd.DataFrame()

    # ── Phase 2: walk-forward ────────────────────────────────────────────────
    model, scaler, metrics = train(train_df)
    log.info(f"Initial model trained | AUC={metrics['oof_auc']:.4f} | "
             f"Brier={metrics['oof_brier_cal']:.4f} | T={metrics['temperature_T']:.3f}")

    bankroll       = initial_bankroll
    peak_bankroll  = initial_bankroll
    last_retrain   = train_end
    ledger_rows    = []
    cur_date       = train_end + timedelta(days=1)

    while cur_date <= end:
        # ── Periodic re-train ────────────────────────────────────────────────
        if (cur_date - last_retrain).days >= RETRAIN_EVERY_DAYS:
            new_train_df = _accumulate_training_data(start, cur_date, park_factors)
            if not new_train_df.empty:
                model, scaler, metrics = train(new_train_df)
                last_retrain = cur_date
                log.info(f"Re-trained {cur_date} | AUC={metrics['oof_auc']:.4f}")

        # ── Daily prediction + sizing ─────────────────────────────────────────
        games = fetch_schedule(cur_date)
        if not games:
            cur_date += timedelta(days=1)
            continue

        statcast_df  = _load_statcast_window(cur_date)
        feature_rows = []
        game_map     = {}

        for game in games:
            row = _build_feature_row(game, cur_date, statcast_df, park_factors)
            if row is None:
                continue
            feature_rows.append(row)
            game_map[game["game_pk"]] = game

        if not feature_rows:
            cur_date += timedelta(days=1)
            continue

        feat_df = pd.DataFrame(feature_rows)
        probs   = predict(feat_df, model, scaler)
        feat_df["prob_home"] = probs
        feat_df = feat_df.dropna(subset=["prob_home"])

        odds  = _stub_odds(games)
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

        # ── Settle bets ───────────────────────────────────────────────────────
        outcomes = _fetch_outcomes([g["game_pk"] for g in games])
        day_pnl  = 0.0

        for bet in bets:
            outcome = outcomes.get(bet.game_pk)
            if outcome is None:
                continue   # game not settled yet
            home_won = bool(outcome)
            won      = (bet.side == "home" and home_won) or \
                       (bet.side == "away" and not home_won)
            pnl      = bet.bet_units * (bet.decimal_odds - 1.0) if won \
                       else -bet.bet_units
            day_pnl += pnl

            ledger_rows.append({
                "date":          cur_date.isoformat(),
                "game_pk":       bet.game_pk,
                "side":          bet.side,
                "prob":          bet.prob,
                "edge":          bet.edge,
                "odds":          bet.american_odds,
                "bet_units":     bet.bet_units,
                "won":           int(won),
                "pnl":           pnl,
                "bankroll_pre":  bankroll,
            })

        bankroll      += day_pnl
        peak_bankroll  = max(peak_bankroll, bankroll)

        log.info(
            f"{cur_date} | bets={len(bets)} | "
            f"day_pnl=${day_pnl:+.2f} | bankroll=${bankroll:.2f}"
        )

        cur_date += timedelta(days=1)

    # ── Save ledger ───────────────────────────────────────────────────────────
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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