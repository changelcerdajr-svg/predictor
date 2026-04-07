# pipeline/backtest/engine.py
import numpy as np
import pandas as pd
from datetime import date, timedelta
import logging

from pipeline.ingest.schedule         import fetch_schedule
from pipeline.ingest.statcast         import fetch_and_cache_statcast
from pipeline.features.feature_matrix import build_game_feature_vector
from pipeline.features.context        import load_park_factors
from pipeline.models.gradient_boost   import train, predict, load_model
from pipeline.calibration.temperature_scaling import TemperatureScaler
from pipeline.risk.kelly              import size_slate
from pipeline.risk.drawdown           import check_drawdown_protection

log = logging.getLogger(__name__)

STATCAST_LOOKBACK = 90


def _load_statcast_window(as_of_date: date) -> pd.DataFrame:
    start = as_of_date - timedelta(days=STATCAST_LOOKBACK)
    end   = as_of_date - timedelta(days=1)
    df    = fetch_and_cache_statcast(start, end)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    return df[df["game_date"] < as_of_date.isoformat()].copy()


def _fetch_outcomes(game_pks: list[int]) -> dict[int, int]:
    """Stub — replace with pipeline/ingest/results.py"""
    return {}


def build_labelled_rows(
    start:        date,
    end:          date,
    park_factors: dict,
) -> pd.DataFrame:
    rows, cur = [], start
    while cur <= end:
        games = fetch_schedule(cur)
        if games:
            sc       = _load_statcast_window(cur)
            outcomes = _fetch_outcomes([g["game_pk"] for g in games])
            for game in games:
                pk  = game["game_pk"]
                row = build_game_feature_vector(game, cur, sc, park_factors)
                if row is None or pk not in outcomes:
                    continue
                row["home_win"]  = outcomes[pk]
                row["game_date"] = cur.isoformat()
                rows.append(row)
        cur += timedelta(days=1)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


class WalkForwardEngine:
    def __init__(
        self,
        start:             date,
        end:               date,
        initial_bankroll:  float = 1000.0,
        kelly_multiplier:  float = 0.25,
        min_edge:          float = 0.03,
        max_fraction:      float = 0.05,
        max_exposure:      float = 0.20,
        train_min_days:    int   = 365,
        retrain_every:     int   = 30,
        drawdown_halt:     float = 0.10,
    ):
        self.start            = start
        self.end              = end
        self.bankroll         = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.peak_bankroll    = initial_bankroll
        self.kelly_multiplier = kelly_multiplier
        self.min_edge         = min_edge
        self.max_fraction     = max_fraction
        self.max_exposure     = max_exposure
        self.train_min_days   = train_min_days
        self.retrain_every    = retrain_every
        self.drawdown_halt    = drawdown_halt
        self.park_factors     = load_park_factors()
        self.model            = None
        self.scaler           = None
        self.ledger_rows: list[dict] = []

    # ------------------------------------------------------------------
    def _stub_odds(self, games: list[dict]) -> dict[int, dict]:
        return {g["game_pk"]: {"odds_home": -110, "odds_away": -110} for g in games}

    def _maybe_retrain(self, cur_date: date, last_retrain: date) -> date:
        if (cur_date - last_retrain).days < self.retrain_every:
            return last_retrain
        df = build_labelled_rows(self.start, cur_date, self.park_factors)
        if df.empty:
            return last_retrain
        self.model, self.scaler, m = train(df)
        log.info(f"Retrained {cur_date} | AUC={m['oof_auc']:.4f} | T={m['temperature_T']:.3f}")
        return cur_date

    def _settle(self, bets, outcomes: dict[int, int]) -> float:
        day_pnl = 0.0
        for bet in bets:
            outcome = outcomes.get(bet.game_pk)
            if outcome is None:
                continue
            won  = (bet.side == "home" and bool(outcome)) or \
                   (bet.side == "away" and not bool(outcome))
            pnl  = bet.bet_units * (bet.decimal_odds - 1.0) if won else -bet.bet_units
            day_pnl += pnl
            self.ledger_rows.append({
                "date":         str(self._cur_date),
                "game_pk":      bet.game_pk,
                "side":         bet.side,
                "prob":         bet.prob,
                "edge":         bet.edge,
                "odds":         bet.american_odds,
                "bet_units":    bet.bet_units,
                "won":          int(won),
                "pnl":          pnl,
                "bankroll_pre": self.bankroll,
            })
        return day_pnl

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        train_end = self.start + timedelta(days=self.train_min_days)
        if train_end >= self.end:
            log.error("Window too short.")
            return pd.DataFrame()

        df = build_labelled_rows(self.start, train_end, self.park_factors)
        if df.empty:
            log.error("No training data.")
            return pd.DataFrame()

        self.model, self.scaler, m = train(df)
        log.info(f"Initial train | AUC={m['oof_auc']:.4f}")

        last_retrain   = train_end
        self._cur_date = train_end + timedelta(days=1)

        while self._cur_date <= self.end:
            # drawdown guard
            if check_drawdown_protection(
                [], self.bankroll, self.peak_bankroll, self.drawdown_halt
            ):
                log.warning(f"DRAWDOWN HALT {self._cur_date}")
                self._cur_date += timedelta(days=1)
                continue

            last_retrain = self._maybe_retrain(self._cur_date, last_retrain)

            games = fetch_schedule(self._cur_date)
            if not games:
                self._cur_date += timedelta(days=1)
                continue

            sc   = _load_statcast_window(self._cur_date)
            rows = []
            for game in games:
                row = build_game_feature_vector(game, self._cur_date, sc, self.park_factors)
                if row is not None:
                    rows.append(row)

            if rows:
                feat_df = pd.DataFrame(rows)
                probs   = predict(feat_df, self.model, self.scaler)
                feat_df["prob_home"] = probs
                feat_df = feat_df.dropna(subset=["prob_home"])

                odds  = self._stub_odds(games)
                slate = [
                    {
                        "game_pk":   int(r["game_pk"]),
                        "prob_home": float(r["prob_home"]),
                        "odds_home": odds[int(r["game_pk"])]["odds_home"],
                        "odds_away": odds[int(r["game_pk"])]["odds_away"],
                    }
                    for _, r in feat_df.iterrows()
                    if int(r["game_pk"]) in odds
                ]

                bets     = size_slate(slate, self.bankroll, self.kelly_multiplier,
                                      self.min_edge, self.max_fraction, self.max_exposure)
                outcomes = _fetch_outcomes([g["game_pk"] for g in games])
                day_pnl  = self._settle(bets, outcomes)

                self.bankroll     += day_pnl
                self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
                log.info(f"{self._cur_date} | bets={len(bets)} | "
                         f"pnl=${day_pnl:+.2f} | bankroll=${self.bankroll:.2f}")

            self._cur_date += timedelta(days=1)

        return pd.DataFrame(self.ledger_rows)