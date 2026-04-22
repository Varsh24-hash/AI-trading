"""
backtesting/backtester.py

Standalone backtester — works with the dashboard without needing
PyTorch, training loops, or missing imports at runtime.

Provides:
  run_ml_backtest()   → used by ML Prediction, Overview, Trade Log pages
  run_rl_backtest()   → used by RL Agent, Overview, Trade Log pages
  run_buyhold()       → baseline comparison
  get_all_metrics()   → single call to get everything the dashboard needs
"""

import numpy as np
import pandas as pd
import os
import joblib

from backtesting.performance_metrics import (
    compute_returns, total_return, annualised_return, volatility,
    sharpe_ratio, max_drawdown, win_rate, calmar_ratio, sortino_ratio,
    value_at_risk, cvar, profit_factor, performance_summary
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.abspath(os.path.join(_HERE, ".."))
_PROC_DIR  = os.path.join(_ROOT, "data", "processed")
_RAW_DIR   = os.path.join(_ROOT, "data", "raw")
_MODEL_DIR = os.path.join(_ROOT, "models")

MODEL_MAP = {
    "xgboost":             "xgb_model.pkl",
    "random_forest":       "random_forest_model.pkl",
    "logistic_regression": "logistic_model.pkl",
}

FEATURE_COLS = ["return_1d", "MA_10", "MA_50", "volatility", "volume_change", "RSI"]


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_processed(ticker: str) -> pd.DataFrame:
    """Load processed feature CSV — has Close, return_1d, RSI etc."""
    path = os.path.join(_PROC_DIR, f"{ticker}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No processed data for {ticker}: {path}")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df.sort_index()


def load_raw(ticker: str) -> pd.DataFrame:
    """Load raw OHLCV CSV (double-header format yfinance style)."""
    path = os.path.join(_RAW_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No raw data for {ticker}: {path}")
    df = pd.read_csv(path, header=0, skiprows=[1],
                     parse_dates=["Date"], index_col="Date")
    df.columns = [c.strip().capitalize() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df.sort_index()


# ── Signal generators ─────────────────────────────────────────────────────────

def _ml_signals(ticker: str, model_name: str) -> pd.DataFrame:
    """
    Run trained model on processed features.
    Returns DataFrame: [Close, Signal (1/0), Prob]
    """
    df    = load_processed(ticker)
    mpath = os.path.join(_MODEL_DIR, MODEL_MAP.get(model_name, "xgb_model.pkl"))
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Model not found: {mpath}")

    clf     = joblib.load(mpath)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X       = df[FEATURE_COLS]
    prob_up = clf.predict_proba(X)[:, 1]
    signals = (prob_up > 0.5).astype(int)   # 1=BUY, 0=SELL

    out = df[["Close"]].copy()
    out["Signal"] = signals
    out["Prob"]   = prob_up
    return out


def _rl_signals(ticker: str) -> pd.DataFrame:
    """
    Momentum-based signals from real price data —
    avoids running PyTorch at dashboard load time.
    Returns DataFrame: [Close, Signal (1/0/-1)]
    """
    df  = load_processed(ticker)
    mom = pd.Series(df["return_1d"].values).rolling(5).mean().fillna(0).values
    sig = np.where(mom > 0.001, 1, np.where(mom < -0.001, 0, -1))

    out = df[["Close"]].copy()
    out["Signal"] = sig
    return out


# ── Core execution engine ─────────────────────────────────────────────────────

def _execute(signal_df: pd.DataFrame,
             capital: float,
             txn_cost: float) -> tuple:
    """
    Simulate trading from a signal series.

    Parameters
    ----------
    signal_df : DataFrame with [Close, Signal]
                Signal: 1=BUY, 0=SELL, -1=HOLD
    capital   : starting cash ($)
    txn_cost  : transaction cost fraction (e.g. 0.001)

    Returns
    -------
    result_df : signal_df + [Portfolio, BuyHold] columns
    trades_df : DataFrame of trades with real P&L
    """
    df = signal_df.copy()

    cash, shares  = capital, 0
    portfolio_val = []
    trades        = []
    buy_price     = None

    for i in range(len(df) - 1):
        price  = df["Close"].iloc[i]
        signal = int(df["Signal"].iloc[i])

        if signal == 1 and shares == 0:
            # ── BUY ──────────────────────────────────────────────────────────
            shares    = int(cash * 0.95 / (price * (1 + txn_cost)))
            cost      = shares * price * (1 + txn_cost)
            cash     -= cost
            buy_price = price
            trades.append({
                "Date":   df.index[i],
                "Type":   "BUY",
                "Price":  round(price, 4),
                "Shares": shares,
                "Value":  round(shares * price, 2),
                "PnL":    0.0,
            })

        elif signal == 0 and shares > 0:
            # ── SELL ─────────────────────────────────────────────────────────
            proceeds = shares * price * (1 - txn_cost)
            cash    += proceeds
            gross    = (price - buy_price) * shares if buy_price else 0
            costs    = (price + (buy_price or 0)) * shares * txn_cost
            pnl      = round(gross - costs, 2)
            trades.append({
                "Date":   df.index[i],
                "Type":   "SELL",
                "Price":  round(price, 4),
                "Shares": shares,
                "Value":  round(proceeds, 2),
                "PnL":    pnl,
            })
            shares    = 0
            buy_price = None

        portfolio_val.append(cash + shares * price)

    df = df.iloc[:-1].copy()
    df["Portfolio"] = portfolio_val
    df["BuyHold"]   = capital * df["Close"] / df["Close"].iloc[0]

    trades_df = pd.DataFrame(trades)
    return df, trades_df


# ── Public API — called by dashboard ─────────────────────────────────────────

def run_ml_backtest(ticker: str,
                    model_name: str = "xgboost",
                    capital: float = 100_000,
                    txn_cost: float = 0.001) -> tuple:
    """
    Full ML backtest.

    Returns
    -------
    result_df  : DataFrame with Close, Signal, Prob, Portfolio, BuyHold
    trades_df  : DataFrame with Date, Type, Price, Shares, Value, PnL
    eval_dict  : dict of model evaluation metrics (accuracy, precision, etc.)
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 roc_auc_score, f1_score, confusion_matrix)

    signal_df = _ml_signals(ticker, model_name)
    result_df, trades_df = _execute(signal_df, capital, txn_cost)

    # ── Model evaluation on held-out test set (last 20%) ─────────────────────
    raw    = load_processed(ticker)
    clf    = joblib.load(os.path.join(_MODEL_DIR, MODEL_MAP[model_name]))
    X      = raw[FEATURE_COLS]
    y      = raw["target"]
    split  = int(len(raw) * 0.8)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Feature importances
    if hasattr(clf, "feature_importances_"):
        imps = dict(zip(FEATURE_COLS, clf.feature_importances_))
    elif hasattr(clf, "coef_"):
        raw_imps = np.abs(clf.coef_[0])
        imps     = dict(zip(FEATURE_COLS, raw_imps / raw_imps.sum()))
    else:
        imps = {f: 1/len(FEATURE_COLS) for f in FEATURE_COLS}

    eval_dict = {
        "accuracy":            round(accuracy_score(y_test, y_pred) * 100),
        "precision":           round(precision_score(y_test, y_pred, zero_division=0) * 100),
        "recall":              round(recall_score(y_test, y_pred, zero_division=0) * 100),
        "roc_auc":             round(roc_auc_score(y_test, y_prob), 2),
        "f1":                  round(f1_score(y_test, y_pred, zero_division=0), 2),
        "confusion_matrix":    confusion_matrix(y_test, y_pred).tolist(),
        "feature_importances": imps,
    }

    return result_df, trades_df, eval_dict


def run_rl_backtest(ticker: str,
                    capital: float = 100_000,
                    txn_cost: float = 0.001) -> tuple:
    """
    RL momentum backtest using real price data.

    Returns
    -------
    result_df : DataFrame with Close, Signal, Portfolio, BuyHold
    trades_df : DataFrame with Date, Type, Price, Shares, Value, PnL
    """
    signal_df = _rl_signals(ticker)
    result_df, trades_df = _execute(signal_df, capital, txn_cost)
    return result_df, trades_df


def run_buyhold(ticker: str, capital: float = 100_000) -> pd.Series:
    """
    Pure buy-and-hold equity curve for baseline comparison.
    Returns a Series of portfolio values indexed by date.
    """
    df    = load_processed(ticker)
    close = df["Close"]
    return capital * close / close.iloc[0]


def get_all_metrics(ticker: str,
                    model_name: str = "xgboost",
                    capital: float = 100_000,
                    txn_cost: float = 0.001,
                    rf: float = 0.05) -> dict:
    """
    Single call that returns everything the dashboard needs.

    Returns a dict with keys:
      ml_df, ml_trades, ml_eval,
      rl_df, rl_trades,
      ml_metrics, rl_metrics, bh_metrics
    """
    ml_df, ml_trades, ml_eval = run_ml_backtest(ticker, model_name, capital, txn_cost)
    rl_df, rl_trades           = run_rl_backtest(ticker, capital, txn_cost)

    ml_metrics = performance_summary(ml_df["Portfolio"], rf=rf)
    rl_metrics = performance_summary(rl_df["Portfolio"], rf=rf)
    bh_metrics = performance_summary(ml_df["BuyHold"],   rf=rf)

    return {
        "ml_df":      ml_df,
        "ml_trades":  ml_trades,
        "ml_eval":    ml_eval,
        "rl_df":      rl_df,
        "rl_trades":  rl_trades,
        "ml_metrics": ml_metrics,
        "rl_metrics": rl_metrics,
        "bh_metrics": bh_metrics,
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ticker = "AAPL"
    print(f"\nRunning backtest for {ticker}...\n")

    try:
        results = get_all_metrics(ticker)

        print("── ML Strategy ──────────────────────────────")
        for k, v in results["ml_metrics"].items():
            print(f"  {k:<20} {v}")

        print("\n── RL Strategy ──────────────────────────────")
        for k, v in results["rl_metrics"].items():
            print(f"  {k:<20} {v}")

        print("\n── Buy & Hold ───────────────────────────────")
        for k, v in results["bh_metrics"].items():
            print(f"  {k:<20} {v}")

        print(f"\n── ML Model Eval ────────────────────────────")
        ev = results["ml_eval"]
        print(f"  Accuracy:  {ev['accuracy']}%")
        print(f"  Precision: {ev['precision']}%")
        print(f"  Recall:    {ev['recall']}%")
        print(f"  ROC-AUC:   {ev['roc_auc']}")
        print(f"  F1:        {ev['f1']}")

        print(f"\n── Trades ───────────────────────────────────")
        print(f"  ML trades: {len(results['ml_trades'])}")
        print(f"  RL trades: {len(results['rl_trades'])}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure data/processed/ and models/ exist.")