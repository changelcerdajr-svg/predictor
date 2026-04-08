# pipeline/backtest/evaluator.py
import numpy as np
import pandas as pd
from pipeline.risk.kelly import american_to_decimal, decimal_to_implied  # FIX: was __import__ inside apply()
 
 
def _roi(ledger: pd.DataFrame) -> float:
    total_wagered = ledger["bet_units"].sum()
    total_pnl     = ledger["pnl"].sum()
    return total_pnl / total_wagered if total_wagered > 0 else np.nan
 
 
def _win_rate(ledger: pd.DataFrame) -> float:
    return ledger["won"].mean() if len(ledger) > 0 else np.nan
 
 
def _sharpe(daily_pnl: pd.Series, periods: int = 252) -> float:
    if daily_pnl.std() == 0 or len(daily_pnl) < 2:
        return np.nan
    return (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(periods)
 
 
def _max_drawdown(bankroll_curve: pd.Series) -> float:
    roll_max  = bankroll_curve.cummax()
    drawdowns = (bankroll_curve - roll_max) / roll_max
    return float(drawdowns.min())
 
 
def _calmar(total_return: float, max_dd: float, years: float) -> float:
    if max_dd == 0 or years == 0:
        return np.nan
    return (total_return / years) / abs(max_dd)
 
 
def _brier_score(ledger: pd.DataFrame) -> float:
    if "prob" not in ledger.columns:
        return np.nan
    return float(((ledger["prob"] - ledger["won"]) ** 2).mean())
 
 
def _log_loss(ledger: pd.DataFrame) -> float:
    eps = 1e-7
    p   = ledger["prob"].clip(eps, 1 - eps)
    y   = ledger["won"]
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
 
 
def _edge_realized(ledger: pd.DataFrame) -> float:
    try:
        implied = ledger["odds"].apply(
            lambda o: decimal_to_implied(american_to_decimal(int(o)))
        )
        return float(ledger["won"].mean() - implied.mean())
    except Exception:
        return np.nan
 
 
def _kelly_efficiency(ledger: pd.DataFrame) -> float:
    """FIX: removed __import__ inside apply(); now uses module-level import."""
    try:
        ev = ledger["prob"] * (
            ledger["odds"].apply(lambda o: american_to_decimal(int(o)) - 1)
        ) - (1 - ledger["prob"])
 
        actual = ledger["pnl"] / ledger["bet_units"].replace(0, np.nan)
        mask   = ev.abs() > 0.001
        if mask.sum() == 0:
            return np.nan
        return float((actual[mask] / ev[mask]).mean())
    except Exception:
        return np.nan
 
 
def _longest_losing_streak(ledger: pd.DataFrame) -> int:
    streak = max_streak = 0
    for w in ledger["won"]:
        if w == 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak
 
 
def _profit_factor(ledger: pd.DataFrame) -> float:
    gross_win  = ledger.loc[ledger["pnl"] > 0, "pnl"].sum()
    gross_loss = ledger.loc[ledger["pnl"] < 0, "pnl"].abs().sum()
    return gross_win / gross_loss if gross_loss > 0 else np.nan
 
 
def _calibration_table(ledger: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    ledger = ledger.copy()
    ledger["bin"] = pd.cut(ledger["prob"], bins=n_bins, labels=False)
    table = (
        ledger.groupby("bin")
        .agg(
            mean_pred=("prob", "mean"),
            actual_wr=("won",  "mean"),
            n_bets   =("won",  "count"),
        )
        .reset_index(drop=True)
    )
    table["calibration_error"] = (table["mean_pred"] - table["actual_wr"]).abs()
    return table
 
 
def _ece(ledger: pd.DataFrame, n_bins: int = 10) -> float:
    table = _calibration_table(ledger, n_bins)
    total = table["n_bets"].sum()
    if total == 0:
        return np.nan
    return float((table["calibration_error"] * table["n_bets"] / total).sum())
 
 
def evaluate(
    ledger:           pd.DataFrame,
    initial_bankroll: float = 1000.0,
) -> dict:
    if ledger.empty:
        return {"error": "empty ledger"}
 
    ledger = ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"])
 
    daily = ledger.groupby("date")["pnl"].sum().sort_index()
    bankroll_curve = initial_bankroll + daily.cumsum()
 
    total_pnl    = ledger["pnl"].sum()
    total_return = total_pnl / initial_bankroll
    n_days       = (ledger["date"].max() - ledger["date"].min()).days
    years        = max(n_days / 365, 1 / 365)
    max_dd       = _max_drawdown(bankroll_curve)
 
    return {
        "n_bets":                len(ledger),
        "n_days_active":         int(daily.astype(bool).sum()),
        "bets_per_day":          round(len(ledger) / max(n_days, 1), 2),
        "total_pnl":             round(total_pnl, 2),
        "total_return_pct":      round(total_return * 100, 2),
        "roi_pct":               round(_roi(ledger) * 100, 2),
        "final_bankroll":        round(initial_bankroll + total_pnl, 2),
        "sharpe":                round(_sharpe(daily), 4),
        "max_drawdown_pct":      round(max_dd * 100, 2),
        "calmar":                round(_calmar(total_return, max_dd, years), 4),
        "profit_factor":         round(_profit_factor(ledger), 4),
        "win_rate_pct":          round(_win_rate(ledger) * 100, 2),
        "brier_score":           round(_brier_score(ledger), 4),
        "log_loss":              round(_log_loss(ledger), 4),
        "ece":                   round(_ece(ledger), 4),
        "mean_edge_pct":         round(ledger["edge"].mean() * 100, 2),
        "realized_edge_pct":     round(_edge_realized(ledger) * 100, 2),
        "longest_losing_streak": _longest_losing_streak(ledger),
    }
 
 
def calibration_report(ledger: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    return _calibration_table(ledger, n_bins)
 
 
def edge_decay_report(ledger: pd.DataFrame) -> pd.DataFrame:
    ledger = ledger.copy()
    ledger["date"]  = pd.to_datetime(ledger["date"])
    ledger["month"] = ledger["date"].dt.to_period("M")
 
    return (
        ledger.groupby("month")
        .apply(lambda g: pd.Series({
            "n_bets":    len(g),
            "roi_pct":   round(_roi(g) * 100, 2),
            "win_rate":  round(_win_rate(g) * 100, 2),
            "mean_edge": round(g["edge"].mean() * 100, 2),
            "pnl":       round(g["pnl"].sum(), 2),
        }))
        .reset_index()
    )