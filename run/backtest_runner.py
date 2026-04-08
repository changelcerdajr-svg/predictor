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

import pandas as pd

from pipeline.backtest.engine         import WalkForwardEngine
from pipeline.backtest.evaluator      import evaluate, calibration_report, edge_decay_report
from pipeline.ingest.results          import fetch_outcomes_for_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/backtest")


def _patch_engine_outcomes(engine: WalkForwardEngine) -> None:
    """
    Monkey-patch the engine's internal _fetch_outcomes to use
    the real results ingest instead of the stub.
    """
    from pipeline.ingest.results import fetch_outcomes_for_date

    def _real_outcomes(game_pks: list[int]) -> dict[int, int]:
        # engine sets self._cur_date before calling _fetch_outcomes
        all_outcomes = fetch_outcomes_for_date(engine._cur_date)
        return {pk: all_outcomes[pk] for pk in game_pks if pk in all_outcomes}

    import pipeline.backtest.engine as eng_mod
    eng_mod._fetch_outcomes = _real_outcomes


def _save_reports(
    ledger:          pd.DataFrame,
    start:           date,
    end:             date,
    initial_bankroll: float,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start}_{end}"

    ledger_path = OUTPUT_DIR / f"backtest_{tag}.csv"
    ledger.to_csv(ledger_path, index=False)
    log.info(f"Ledger → {ledger_path}")

    cal = calibration_report(ledger)
    cal.to_csv(OUTPUT_DIR / f"calibration_{tag}.csv", index=False)

    decay = edge_decay_report(ledger)
    decay.to_csv(OUTPUT_DIR / f"edge_decay_{tag}.csv", index=False)

    summary = evaluate(ledger, initial_bankroll)
    log.info("\n=== BACKTEST SUMMARY ===")
    for k, v in summary.items():
        log.info(f"  {k:<30} {v}")

    import json
    with open(OUTPUT_DIR / f"summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary → {OUTPUT_DIR / f'summary_{tag}.json'}")


def run_backtest(
    start:            date,
    end:              date,
    kelly_multiplier: float = 0.25,
    min_edge:         float = 0.03,
    max_fraction:     float = 0.05,
    max_exposure:     float = 0.20,
    initial_bankroll: float = 1000.0,
    train_min_days:   int   = 365,
    retrain_every:    int   = 30,
    drawdown_halt:    float = 0.10,
) -> pd.DataFrame:

    engine = WalkForwardEngine(
        start            = start,
        end              = end,
        initial_bankroll = initial_bankroll,
        kelly_multiplier = kelly_multiplier,
        min_edge         = min_edge,
        max_fraction     = max_fraction,
        max_exposure     = max_exposure,
        train_min_days   = train_min_days,
        retrain_every    = retrain_every,
        drawdown_halt    = drawdown_halt,
    )

    _patch_engine_outcomes(engine)

    ledger = engine.run()

    if ledger.empty:
        log.warning("Empty ledger — no settled bets recorded.")
        return ledger

    _save_reports(ledger, start, end, initial_bankroll)
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",        type=lambda s: date.fromisoformat(s), required=True)
    parser.add_argument("--end",          type=lambda s: date.fromisoformat(s), required=True)
    parser.add_argument("--min-edge",     type=float, default=0.03)
    parser.add_argument("--kelly",        type=float, default=0.25)
    parser.add_argument("--bankroll",     type=float, default=1000.0)
    parser.add_argument("--train-days",   type=int,   default=365)
    parser.add_argument("--retrain-days", type=int,   default=30)
    parser.add_argument("--drawdown",     type=float, default=0.10)
    args = parser.parse_args()

    run_backtest(
        start            = args.start,
        end              = args.end,
        min_edge         = args.min_edge,
        kelly_multiplier = args.kelly,
        initial_bankroll = args.bankroll,
        train_min_days   = args.train_days,
        retrain_every    = args.retrain_days,
        drawdown_halt    = args.drawdown,
    )