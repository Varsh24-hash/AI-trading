"""
data/live_feed.py

Fetches the latest stock data from Yahoo Finance up to yesterday,
computes the same features used during model training, and returns
the most recent row ready for prediction.

Features match training pipeline exactly:
  return_1d, MA_10, MA_50, volatility, volume_change, RSI
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

FEATURE_COLS = ["return_1d", "MA_10", "MA_50", "volatility", "volume_change", "RSI"]


def get_latest_data(ticker: str, lookback_days: int = 120) -> pd.DataFrame:
    """
    Download the last `lookback_days` of OHLCV data from Yahoo Finance
    up to yesterday. Returns a DataFrame with all features computed.

    Parameters
    ----------
    ticker       : stock ticker e.g. "AAPL"
    lookback_days: how many days of history to fetch (need at least 60
                   for MA_50 + RSI to be stable)

    Returns
    -------
    DataFrame with columns:
      Open, High, Low, Close, Volume,
      return_1d, MA_10, MA_50, volatility, volume_change, RSI
    indexed by Date
    """
    end   = datetime.today().date()
    start = end - timedelta(days=lookback_days + 30)  # buffer for weekends

    try:
        raw = yf.download(ticker, start=str(start), end=str(end),
                          progress=False, auto_adjust=True)
    except Exception as e:
        raise ConnectionError(f"Failed to download {ticker} from Yahoo Finance: {e}")

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol.")

    # Flatten multi-level columns if present (yfinance sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw = raw.apply(pd.to_numeric, errors="coerce").dropna()
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    return _compute_features(raw)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the exact same features used during model training.
    Must match ml_price_prediction/train_model.py feature pipeline.
    """
    df = df.copy()

    # return_1d — daily percentage return
    df["return_1d"] = df["Close"].pct_change()

    # MA_10 and MA_50 — simple moving averages
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    # volatility — 10-day rolling std of returns (annualised)
    df["volatility"] = df["return_1d"].rolling(10).std() * np.sqrt(252)

    # volume_change — daily percentage change in volume
    df["volume_change"] = df["Volume"].pct_change()

    # RSI (14-period)
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Also compute extra indicator columns used for text explanation + chart
    df["MA_20"]    = df["Close"].rolling(20).mean()
    df["MA_200"]   = df["Close"].rolling(200).mean()
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    df["BB_Std"]   = df["Close"].rolling(20).std()
    df["BB_Up"]    = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lo"]    = df["BB_Mid"] - 2 * df["BB_Std"]
    e12 = df["Close"].ewm(span=12).mean()
    e26 = df["Close"].ewm(span=26).mean()
    df["MACD"]     = e12 - e26
    df["MACD_Sig"] = df["MACD"].ewm(span=9).mean()
    df["MACD_H"]   = df["MACD"] - df["MACD_Sig"]
    df["Vol_20"]   = df["return_1d"].rolling(20).std() * np.sqrt(252)

    return df.dropna()


def get_prediction_row(ticker: str) -> dict:
    """
    Returns everything the Daily Signal page needs:
      - df        : full feature DataFrame (for charts)
      - features  : dict of the 6 feature values for latest row
      - latest    : Series of the latest row (all columns)
      - date      : the date of the prediction
      - prev_close: previous day close (for % change display)
    """
    df = get_latest_data(ticker)

    if len(df) < 2:
        raise ValueError(f"Not enough data for {ticker} after feature computation.")

    latest     = df.iloc[-1]
    prev_close = df.iloc[-2]["Close"]

    features = {col: float(latest[col]) for col in FEATURE_COLS}

    return {
        "df":         df,
        "features":   features,
        "latest":     latest,
        "date":       df.index[-1].strftime("%A, %d %B %Y"),
        "prev_close": prev_close,
    }


def explain_signal_text(latest: pd.Series, signal: str, prob: float) -> list:
    """
    Generate human-readable text explanations for the signal.
    Returns a list of (reason, sentiment) tuples where sentiment
    is 'bullish', 'bearish', or 'neutral'.
    """
    reasons = []
    close   = latest["Close"]
    rsi     = latest["RSI"]
    macd    = latest["MACD"]
    macd_s  = latest["MACD_Sig"]
    ma10    = latest["MA_10"]
    ma50    = latest["MA_50"]
    bb_up   = latest.get("BB_Up", np.nan)
    bb_lo   = latest.get("BB_Lo", np.nan)
    vol     = latest.get("Vol_20", np.nan)
    ret1d   = latest["return_1d"] * 100

    # RSI
    if rsi > 70:
        reasons.append((f"RSI is {rsi:.1f} — overbought territory, suggesting the stock "
                         "may be due for a pullback.", "bearish"))
    elif rsi < 30:
        reasons.append((f"RSI is {rsi:.1f} — oversold territory, suggesting a potential "
                         "bounce or recovery.", "bullish"))
    else:
        reasons.append((f"RSI is {rsi:.1f} — in neutral zone, no extreme momentum signal.", "neutral"))

    # MACD
    if macd > macd_s:
        reasons.append((f"MACD ({macd:.3f}) is above its signal line ({macd_s:.3f}) — "
                         "bullish momentum crossover.", "bullish"))
    else:
        reasons.append((f"MACD ({macd:.3f}) is below its signal line ({macd_s:.3f}) — "
                         "bearish momentum crossover.", "bearish"))

    # Moving averages
    if close > ma50:
        reasons.append((f"Price ({close:.2f}) is above MA50 ({ma50:.2f}) — "
                         "medium-term uptrend intact.", "bullish"))
    else:
        reasons.append((f"Price ({close:.2f}) is below MA50 ({ma50:.2f}) — "
                         "medium-term downtrend.", "bearish"))

    if ma10 > ma50:
        reasons.append((f"MA10 ({ma10:.2f}) crossed above MA50 ({ma50:.2f}) — "
                         "short-term momentum is positive.", "bullish"))
    else:
        reasons.append((f"MA10 ({ma10:.2f}) is below MA50 ({ma50:.2f}) — "
                         "short-term momentum is negative.", "bearish"))

    # Bollinger Bands
    if not np.isnan(bb_up) and not np.isnan(bb_lo):
        bb_pct = (close - bb_lo) / (bb_up - bb_lo) * 100 if (bb_up - bb_lo) > 0 else 50
        if bb_pct > 80:
            reasons.append((f"Price is near the upper Bollinger Band ({bb_pct:.0f}%B) — "
                             "potential overbought signal.", "bearish"))
        elif bb_pct < 20:
            reasons.append((f"Price is near the lower Bollinger Band ({bb_pct:.0f}%B) — "
                             "potential oversold signal.", "bullish"))
        else:
            reasons.append((f"Price is within Bollinger Bands ({bb_pct:.0f}%B) — "
                             "no extreme band signal.", "neutral"))

    # Yesterday's return
    if ret1d > 1.5:
        reasons.append((f"Yesterday's return was +{ret1d:.2f}% — strong upward move.", "bullish"))
    elif ret1d < -1.5:
        reasons.append((f"Yesterday's return was {ret1d:.2f}% — strong downward move.", "bearish"))
    else:
        reasons.append((f"Yesterday's return was {ret1d:.2f}% — relatively flat.", "neutral"))

    # Model confidence
    conf = abs(prob - 0.5) * 200
    if conf > 60:
        reasons.append((f"Model confidence is high ({prob*100:.1f}% probability of upward move).", "bullish" if prob > 0.5 else "bearish"))
    else:
        reasons.append((f"Model confidence is moderate ({prob*100:.1f}% probability of upward move) — "
                         "signal is not strong.", "neutral"))

    return reasons