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

FEATURE_COLS = ["return_1d", "MA_10", "MA_50", "volatility", "volume_change", "RSI"]


def get_latest_data(ticker: str, lookback_days: int = 300) -> pd.DataFrame:
    """
    Download the last `lookback_days` of OHLCV data from Yahoo Finance.
    300 days gives enough history after dropna for MA_50 + RSI + volatility.
    """
    end   = datetime.today().date() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    try:
        raw = yf.download(
            ticker,
            start=str(start),
            end=str(end),
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        raise ConnectionError(
            f"Failed to download {ticker} from Yahoo Finance: {e}"
        )

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for {ticker}. "
            f"Check the ticker symbol and your internet connection."
        )

    # Flatten multi-level columns (yfinance v0.2+ returns MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Normalise column names
    rename = {}
    for c in raw.columns:
        cl = c.lower()
        if cl == "open":    rename[c] = "Open"
        elif cl == "high":  rename[c] = "High"
        elif cl == "low":   rename[c] = "Low"
        elif cl == "close": rename[c] = "Close"
        elif cl == "volume":rename[c] = "Volume"
    raw = raw.rename(columns=rename)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in raw.columns:
            raw[col] = np.nan

    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index().dropna(subset=["Close"])

    if len(raw) < 60:
        raise ValueError(
            f"Only {len(raw)} rows downloaded for {ticker}. "
            f"Need at least 60 trading days."
        )

    return _compute_features(raw)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features — matches training pipeline exactly."""
    df = df.copy()

    df["return_1d"]     = df["Close"].pct_change()
    df["MA_10"]         = df["Close"].rolling(10).mean()
    df["MA_50"]         = df["Close"].rolling(50).mean()
    df["volatility"]    = df["return_1d"].rolling(10).std() * np.sqrt(252)
    df["volume_change"] = df["Volume"].pct_change()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Extra indicators for charts and text explanation
    df["MA_20"]    = df["Close"].rolling(20).mean()
    df["MA_200"]   = df["Close"].rolling(200).mean()
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    df["BB_Std"]   = df["Close"].rolling(20).std()
    df["BB_Up"]    = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lo"]    = df["BB_Mid"] - 2 * df["BB_Std"]
    e12            = df["Close"].ewm(span=12).mean()
    e26            = df["Close"].ewm(span=26).mean()
    df["MACD"]     = e12 - e26
    df["MACD_Sig"] = df["MACD"].ewm(span=9).mean()
    df["MACD_H"]   = df["MACD"] - df["MACD_Sig"]
    df["Vol_20"]   = df["return_1d"].rolling(20).std() * np.sqrt(252)

    # Only drop rows where model features are NaN
    df = df.dropna(subset=FEATURE_COLS)

    return df


def get_prediction_row(ticker: str) -> dict:
    """
    Returns everything the Daily Signal page needs.
    """
    df = get_latest_data(ticker)

    if len(df) < 2:
        raise ValueError(
            f"Not enough clean data for {ticker} after feature computation. "
            f"Got {len(df)} rows — need at least 2."
        )

    latest     = df.iloc[-1]
    prev_close = df.iloc[-2]["Close"]

    for col in FEATURE_COLS:
        if col not in df.columns or pd.isna(latest[col]):
            raise ValueError(
                f"Feature '{col}' is missing or NaN for {ticker}. "
                f"Data may be incomplete."
            )

    features = {col: float(latest[col]) for col in FEATURE_COLS}

    return {
        "df":         df,
        "features":   features,
        "latest":     latest,
        "date":       df.index[-1].strftime("%A, %d %B %Y"),
        "prev_close": float(prev_close),
    }


def explain_signal_text(latest: pd.Series, signal: str, prob: float) -> list:
    """
    Returns a list of (reason_text, sentiment) tuples.
    sentiment: 'bullish', 'bearish', or 'neutral'
    """
    reasons = []
    close   = float(latest["Close"])
    rsi     = float(latest["RSI"])
    macd    = float(latest.get("MACD",     0.0))
    macd_s  = float(latest.get("MACD_Sig", 0.0))
    ma10    = float(latest["MA_10"])
    ma50    = float(latest["MA_50"])
    bb_up   = float(latest.get("BB_Up", np.nan))
    bb_lo   = float(latest.get("BB_Lo", np.nan))
    ret1d   = float(latest["return_1d"]) * 100

    if rsi > 70:
        reasons.append((f"RSI is {rsi:.1f} — overbought. Pullback may be due.", "bearish"))
    elif rsi < 30:
        reasons.append((f"RSI is {rsi:.1f} — oversold. A bounce may be near.", "bullish"))
    else:
        reasons.append((f"RSI is {rsi:.1f} — neutral zone, no extreme momentum.", "neutral"))

    if macd > macd_s:
        reasons.append((f"MACD ({macd:.3f}) above signal ({macd_s:.3f}) — bullish crossover.", "bullish"))
    else:
        reasons.append((f"MACD ({macd:.3f}) below signal ({macd_s:.3f}) — bearish crossover.", "bearish"))

    if close > ma50:
        reasons.append((f"Price ({close:.2f}) above MA50 ({ma50:.2f}) — uptrend intact.", "bullish"))
    else:
        reasons.append((f"Price ({close:.2f}) below MA50 ({ma50:.2f}) — downtrend.", "bearish"))

    if ma10 > ma50:
        reasons.append((f"MA10 ({ma10:.2f}) above MA50 ({ma50:.2f}) — Golden Cross zone.", "bullish"))
    else:
        reasons.append((f"MA10 ({ma10:.2f}) below MA50 ({ma50:.2f}) — Death Cross zone.", "bearish"))

    if not (np.isnan(bb_up) or np.isnan(bb_lo)) and (bb_up - bb_lo) > 0:
        bb_pct = (close - bb_lo) / (bb_up - bb_lo) * 100
        if bb_pct > 80:
            reasons.append((f"Near upper Bollinger Band ({bb_pct:.0f}%B) — overbought signal.", "bearish"))
        elif bb_pct < 20:
            reasons.append((f"Near lower Bollinger Band ({bb_pct:.0f}%B) — oversold signal.", "bullish"))
        else:
            reasons.append((f"Within Bollinger Bands ({bb_pct:.0f}%B) — no extreme signal.", "neutral"))

    if ret1d > 1.5:
        reasons.append((f"Yesterday's return: +{ret1d:.2f}% — strong upward move.", "bullish"))
    elif ret1d < -1.5:
        reasons.append((f"Yesterday's return: {ret1d:.2f}% — strong downward move.", "bearish"))
    else:
        reasons.append((f"Yesterday's return: {ret1d:.2f}% — relatively flat.", "neutral"))

    conf      = abs(prob - 0.5) * 200
    direction = "bullish" if prob > 0.5 else "bearish"
    if conf > 60:
        reasons.append((f"High model confidence: {prob*100:.1f}% probability of upward move.", direction))
    elif conf > 20:
        reasons.append((f"Moderate confidence: {prob*100:.1f}% probability of upward move.", direction))
    else:
        reasons.append((f"Low confidence: {prob*100:.1f}% — market direction uncertain.", "neutral"))

    return reasons