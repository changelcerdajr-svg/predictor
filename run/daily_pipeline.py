# run/daily_pipeline.py
"""
Daily production pipeline.
Usage:
    python -m run.daily_pipeline --date 2025-04-06
    python -m run.daily_pipeline              # defaults to today
"""
import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
 
import pandas as pd
 
from pipeline.ingest.schedule         import fetch_schedule
from pipeline.ingest.statcast         import fetch_and_cache_statcast
from pipeline.features.feature_matrix import build_game_feature_vector
from pipeline.features.context        import load_park_factors
from pipeline.models.gradient_boost   import load_model, predict
from pipeline.risk.kelly              import size_slate
from pipeline.risk.drawdown           import check_drawdown_protection
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
 
STATCAST_LOOKBACK_DAYS = 90
BANKROLL_PATH          = Path("data/bankroll.json")
MODEL_TAG              = "latest"
OUTPUT_DIR             = Path("data/outputs")
 
 
def _load_bankroll() -> dict:
    import json
    if BANKROLL_PATH.exists():
        with open(BANKROLL_PATH) as f:
            return json.load(f)
    default = {"current": 1000.0, "peak": 1000.0, "history": []}
    _save_bankroll(default)
    return default
 
 
def _save_bankroll(state: dict) -> None:
    import json
    BANKROLL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BANKROLL_PATH, "w") as f:
        json.dump(state, f, indent=2)
 
 
def _load_statcast(as_of_date: date) -> pd.DataFrame:
    start = as_of_date - timedelta(days=STATCAST_LOOKBACK_DAYS)
    end   = as_of_date - timedelta(days=1)
    log.info(f"Loading Statcast {start} → {end}")
    df = fetch_and_cache_statcast(start, end)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df = df[df["game_date"] < as_of_date.isoformat()].copy()
    log.info(f"Statcast rows loaded: {len(df):,}")
    return df
 
 
def _build_features(
    games:        list[dict],
    as_of_date:   date,
    statcast_df:  pd.DataFrame,
    park_factors: dict,
) -> pd.DataFrame:
    rows = []
    skipped = 0
    for game in games:
        row = build_game_feature_vector(game, as_of_date, statcast_df, park_factors)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    log.info(f"Features built: {len(rows)} games | skipped (no SP): {skipped}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()
 
 
def _run_predictions(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()
    model, scaler = load_model(MODEL_TAG)
    probs = predict(feature_df, model, scaler)
    feature_df = feature_df.copy()
    feature_df["prob_home"] = probs
    feature_df = feature_df.dropna(subset=["prob_home"])
    log.info(f"Predictions generated: {len(feature_df)}")
    return feature_df
 
 
def _fetch_odds(games: list[dict]) -> dict[int, dict]:
    """
    Placeholder: replace with live odds feed (Pinnacle, DraftKings API, etc.).
    WARNING: currently returns neutral -110/-110 for all games.
    """
    log.warning("_fetch_odds() is using placeholder odds (-110/-110). Wire a real feed before betting.")
    return {g["game_pk"]: {"odds_home": -110, "odds_away": -110} for g in games}
 
 
def _build_slate(pred_df: pd.DataFrame, odds: dict[int, dict]) -> list[dict]:
    slate = []
    for _, row in pred_df.iterrows():
        pk = int(row["game_pk"])
        if pk not in odds:
            continue
        slate.append({
            "game_pk":   pk,
            "prob_home": float(row["prob_home"]),
            "odds_home": odds[pk]["odds_home"],
            "odds_away": odds[pk]["odds_away"],
        })
    return slate
 
 
def _save_output(pred_df: pd.DataFrame, bets: list, run_date: date) -> None:
    import json
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = run_date.isoformat()
 
    pred_path = OUTPUT_DIR / f"predictions_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info(f"Predictions saved → {pred_path}")
 
    bets_path = OUTPUT_DIR / f"bets_{tag}.json"
    with open(bets_path, "w") as f:
        json.dump([b.__dict__ for b in bets], f, indent=2, default=str)
    log.info(f"Bets saved → {bets_path}  ({len(bets)} bets)")
 
 
def run(target_date: date) -> None:
    log.info(f"=== Daily pipeline | {target_date} ===")
 
    bankroll_state = _load_bankroll()
 
    if check_drawdown_protection(
        bankroll_history = bankroll_state["history"],
        current_bankroll = bankroll_state["current"],
        peak_bankroll    = bankroll_state["peak"],
        halt_threshold   = 0.10,
    ):
        log.warning(
            f"DRAWDOWN HALT — current: {bankroll_state['current']:.2f} | "
            f"peak: {bankroll_state['peak']:.2f}. No bets placed."
        )
        return
 
    games = fetch_schedule(target_date)
    if not games:
        log.info("No games scheduled. Exiting.")
        return
    log.info(f"Games on slate: {len(games)}")
 
    statcast_df  = _load_statcast(target_date)
    park_factors = load_park_factors()
 
    feature_df = _build_features(games, target_date, statcast_df, park_factors)
    if feature_df.empty:
        log.info("No feature rows built. Exiting.")
        return
 
    pred_df = _run_predictions(feature_df)
    if pred_df.empty:
        log.info("No predictions. Exiting.")
        return
 
    odds  = _fetch_odds(games)
    slate = _build_slate(pred_df, odds)
    bets  = size_slate(
        predictions      = slate,
        bankroll         = bankroll_state["current"],
        kelly_multiplier = 0.25,
        min_edge         = 0.03,
        max_fraction     = 0.05,
        max_exposure     = 0.20,
    )
 
    _save_output(pred_df, bets, target_date)
 
    total_risk = sum(b.bet_units for b in bets)
 
    # FIX: update peak_bankroll after each run so drawdown protection is accurate
    projected_bankroll = bankroll_state["current"]   # bets settle later; track pre-bet
    new_peak = max(bankroll_state["peak"], projected_bankroll)
    if new_peak != bankroll_state["peak"]:
        bankroll_state["peak"] = new_peak
        _save_bankroll(bankroll_state)
 
    log.info(
        f"Bets: {len(bets)} | "
        f"Total risk: ${total_risk:.2f} | "
        f"Bankroll: ${bankroll_state['current']:.2f}"
    )
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
        help="Target date (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()
    run(args.date)