"""
backtesting/performance_metrics.py

All performance metrics used by the dashboard.
Every function takes a pd.Series of portfolio values
(or daily returns where noted) and returns a scalar or dict.

Matches these dashboard displays:
  - Total Return, Ann. Return, Volatility, Sharpe      → all pages top KPIs
  - Max Drawdown, Win Rate                             → all pages top KPIs
  - Calmar Ratio, Sortino Ratio                        → Trade Log stats card
  - Historical VaR, Parametric VaR, CVaR              → Risk VaR page
  - Profit Factor, Avg Win, Avg Loss                   → Trade Log stats card
  - Confusion matrix, Accuracy, Precision, Recall etc  → ML Prediction page
"""

import numpy as np
import pandas as pd


# ── Return series helper ──────────────────────────────────────────────────────

def compute_returns(portfolio: pd.Series) -> pd.Series:
    """Daily percentage returns from a portfolio value series."""
    return portfolio.pct_change().dropna()


# ── Individual metrics ────────────────────────────────────────────────────────

def total_return(portfolio: pd.Series) -> float:
    """Total return as a percentage. e.g. 96.84 means +96.84%"""
    return (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100


def annualised_return(portfolio: pd.Series) -> float:
    """CAGR as a percentage, assuming 252 trading days per year."""
    r   = compute_returns(portfolio)
    tot = total_return(portfolio)
    if len(r) == 0:
        return 0.0
    return ((1 + tot / 100) ** (252 / len(r)) - 1) * 100


def volatility(portfolio: pd.Series) -> float:
    """Annualised volatility as a percentage."""
    r = compute_returns(portfolio)
    return r.std() * np.sqrt(252) * 100


def sharpe_ratio(portfolio: pd.Series, rf: float = 0.05) -> float:
    """
    Annualised Sharpe ratio.
    rf : annual risk-free rate as a decimal (e.g. 0.05 = 5%)
    """
    ann = annualised_return(portfolio) / 100
    vol = volatility(portfolio) / 100
    if vol == 0:
        return 0.0
    return (ann - rf) / vol


def max_drawdown(portfolio: pd.Series) -> float:
    """Maximum drawdown as a percentage (negative number)."""
    rolling_max = portfolio.cummax()
    drawdown    = (portfolio - rolling_max) / rolling_max * 100
    return drawdown.min()


def win_rate(portfolio: pd.Series) -> float:
    """Percentage of trading days with positive returns."""
    r = compute_returns(portfolio)
    return (r > 0).mean() * 100


def calmar_ratio(portfolio: pd.Series) -> float:
    """
    Calmar ratio = annualised return / abs(max drawdown).
    Higher is better. Shown in Trade Log stats card.
    """
    ann = annualised_return(portfolio)
    mdd = abs(max_drawdown(portfolio))
    if mdd == 0:
        return 0.0
    return ann / mdd


def sortino_ratio(portfolio: pd.Series, rf: float = 0.05) -> float:
    """
    Sortino ratio — like Sharpe but only penalises downside volatility.
    Shown in Trade Log stats card.
    """
    r           = compute_returns(portfolio)
    ann         = annualised_return(portfolio) / 100
    daily_rf    = rf / 252
    downside    = r[r < daily_rf]
    if len(downside) == 0:
        return 0.0
    downside_std = downside.std() * np.sqrt(252)
    if downside_std == 0:
        return 0.0
    return (ann - rf) / downside_std


# ── Risk metrics ──────────────────────────────────────────────────────────────

def value_at_risk(portfolio: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR as a percentage at given confidence level.
    e.g. -2.679 means you expect to lose at most 2.679% on 95% of days.
    Used by Risk VaR page.
    """
    r = compute_returns(portfolio)
    return np.percentile(r, (1 - confidence) * 100) * 100


def parametric_var(portfolio: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (normal distribution) VaR as a percentage.
    Uses μ − z*σ formula. z=1.645 for 95% confidence.
    """
    r = compute_returns(portfolio)
    z = 1.645  # 95% one-tailed
    return (r.mean() - z * r.std()) * 100


def cvar(portfolio: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall) as a percentage.
    Average loss on the worst (1-confidence)% of days.
    """
    r        = compute_returns(portfolio)
    var_val  = np.percentile(r, (1 - confidence) * 100)
    tail     = r[r <= var_val]
    if len(tail) == 0:
        return var_val * 100
    return tail.mean() * 100


def rolling_var(portfolio: pd.Series,
                window: int = 60,
                confidence: float = 0.95) -> pd.Series:
    """
    Rolling historical VaR as a percentage Series.
    Used for the rolling VaR chart in Risk VaR page.
    """
    r = compute_returns(portfolio)
    return r.rolling(window).apply(
        lambda x: np.percentile(x, (1 - confidence) * 100)
    ) * 100


# ── Trade-level metrics ───────────────────────────────────────────────────────

def profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Profit factor = gross profit / gross loss.
    Uses the PnL column from trades_df (sell trades only).
    Shown in Trade Log stats card.
    """
    if trades_df.empty or "PnL" not in trades_df.columns:
        return 0.0
    pnl   = trades_df[trades_df["Type"] == "SELL"]["PnL"]
    wins  = pnl[pnl > 0].sum()
    losses= abs(pnl[pnl < 0].sum())
    if losses == 0:
        return float("inf")
    return round(wins / losses, 2)


def avg_win(trades_df: pd.DataFrame) -> float:
    """Average profit on winning trades ($)."""
    if trades_df.empty or "PnL" not in trades_df.columns:
        return 0.0
    pnl = trades_df[trades_df["Type"] == "SELL"]["PnL"]
    w   = pnl[pnl > 0]
    return round(w.mean(), 2) if len(w) else 0.0


def avg_loss(trades_df: pd.DataFrame) -> float:
    """Average loss on losing trades ($) — negative number."""
    if trades_df.empty or "PnL" not in trades_df.columns:
        return 0.0
    pnl = trades_df[trades_df["Type"] == "SELL"]["PnL"]
    l   = pnl[pnl < 0]
    return round(l.mean(), 2) if len(l) else 0.0


def trade_win_rate(trades_df: pd.DataFrame) -> float:
    """Win rate based on completed round-trip trades (%)."""
    if trades_df.empty or "PnL" not in trades_df.columns:
        return 0.0
    pnl  = trades_df[trades_df["Type"] == "SELL"]["PnL"]
    if len(pnl) == 0:
        return 0.0
    return round((pnl > 0).mean() * 100, 1)


def total_pnl(trades_df: pd.DataFrame) -> float:
    """Total net P&L across all completed trades ($)."""
    if trades_df.empty or "PnL" not in trades_df.columns:
        return 0.0
    return round(trades_df[trades_df["Type"] == "SELL"]["PnL"].sum(), 2)


# ── Summary dict — matches dashboard metric cards exactly ────────────────────

def performance_summary(portfolio: pd.Series, rf: float = 0.05) -> dict:
    """
    Returns a dict of formatted strings matching the dashboard KPI cards:
      Total Return, Ann. Return, Volatility, Sharpe, Max Drawdown, Win Rate

    Usage in dashboard:
        metrics = performance_summary(df["Portfolio"], rf=rf)
        for col, (k, v) in zip(columns, metrics.items()):
            col.metric(k, v)
    """
    tot = total_return(portfolio)
    ann = annualised_return(portfolio)
    vol = volatility(portfolio)
    sr  = sharpe_ratio(portfolio, rf)
    mdd = max_drawdown(portfolio)
    wr  = win_rate(portfolio)

    return {
        "Total Return": f"{tot:+.2f}%",
        "Ann. Return":  f"{ann:+.2f}%",
        "Volatility":   f"{vol:.2f}%",
        "Sharpe":       f"{sr:.3f}",
        "Max Drawdown": f"{mdd:.2f}%",
        "Win Rate":     f"{wr:.1f}%",
    }


def extended_summary(portfolio: pd.Series,
                     trades_df: pd.DataFrame = None,
                     rf: float = 0.05,
                     confidence: float = 0.95) -> dict:
    """
    Extended metrics used by Trade Log and Risk VaR pages.
    Includes everything in performance_summary plus:
    Calmar, Sortino, VaR, CVaR, trade stats.
    """
    base = performance_summary(portfolio, rf)

    extra = {
        "Calmar Ratio":    f"{calmar_ratio(portfolio):.3f}",
        "Sortino Ratio":   f"{sortino_ratio(portfolio, rf):.3f}",
        "Historical VaR":  f"{value_at_risk(portfolio, confidence):.3f}%",
        "Parametric VaR":  f"{parametric_var(portfolio, confidence):.3f}%",
        "CVaR / ES":       f"{cvar(portfolio, confidence):.3f}%",
        "Ann. Volatility": f"{volatility(portfolio):.2f}%",
    }

    if trades_df is not None and not trades_df.empty:
        sell_pnl = trades_df[trades_df["Type"] == "SELL"]["PnL"] \
                   if "PnL" in trades_df.columns else pd.Series(dtype=float)
        extra.update({
            "Total Trades":    str(len(trades_df)),
            "Buy Orders":      str(len(trades_df[trades_df["Type"] == "BUY"])),
            "Sell Orders":     str(len(trades_df[trades_df["Type"] == "SELL"])),
            "Trade Win Rate":  f"{trade_win_rate(trades_df):.1f}%",
            "Total P&L":       f"${total_pnl(trades_df):,.0f}",
            "Avg P&L/Trade":   f"${sell_pnl.mean():,.0f}" if len(sell_pnl) else "$0",
            "Avg Win":         f"${avg_win(trades_df):,.0f}",
            "Avg Loss":        f"${avg_loss(trades_df):,.0f}",
            "Profit Factor":   f"{profit_factor(trades_df):.2f}",
            "Best Trade":      f"${sell_pnl.max():,.0f}" if len(sell_pnl) else "$0",
            "Worst Trade":     f"${sell_pnl.min():,.0f}" if len(sell_pnl) else "$0",
        })

    return {**base, **extra}


# ── Monte Carlo VaR ───────────────────────────────────────────────────────────

def monte_carlo_var(portfolio: pd.Series,
                    n_simulations: int = 50_000,
                    confidence: float = 0.95,
                    seed: int = 7) -> float:
    """
    Monte Carlo VaR using simulated returns from historical mu/sigma.
    Returns percentage loss at confidence level.
    Used by Risk VaR page (Monte Carlo VaR card).
    """
    r = compute_returns(portfolio)
    np.random.seed(seed)
    sim = np.random.normal(r.mean(), r.std(), n_simulations)
    return np.percentile(sim, (1 - confidence) * 100) * 100


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick sanity check with synthetic data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    prices = pd.Series(100_000 * np.exp(np.cumsum(
        np.random.normal(0.0004, 0.018, 500))), index=dates)

    print("── Performance Summary ──────────────────────")
    for k, v in performance_summary(prices).items():
        print(f"  {k:<20} {v}")

    print("\n── Risk Metrics ─────────────────────────────")
    print(f"  Historical VaR (95%):  {value_at_risk(prices):.3f}%")
    print(f"  Parametric VaR (95%):  {parametric_var(prices):.3f}%")
    print(f"  CVaR / ES (95%):       {cvar(prices):.3f}%")
    print(f"  Monte Carlo VaR (95%): {monte_carlo_var(prices):.3f}%")
    print(f"  Calmar Ratio:          {calmar_ratio(prices):.3f}")
    print(f"  Sortino Ratio:         {sortino_ratio(prices):.3f}")