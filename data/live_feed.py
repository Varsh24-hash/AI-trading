"""
data/live_feed.py  —  AlgoTrade Daily Signal
Uses explicit start/end dates (NOT period strings like '18mo') — more reliable on Streamlit Cloud.
Debug screenshot showed: 124 rows with "6mo" period, timezone 00:00:00-04:00 not stripped.
This version fetches 730 days explicitly → ~500 trading days → plenty for MA_50 + RSI warmup.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

FEATURE_COLS = ["return_1d", "MA_10", "MA_50", "volatility", "volume_change", "RSI"]


def get_latest_data(ticker: str) -> pd.DataFrame:
    """
    Fetch 2 years of data via explicit start/end dates.
    730 calendar days ≈ 500+ trading days — MA_50 warmup needs 50, RSI needs 14.
    After dropna we still have 430+ clean rows. No risk of running out.
    """
    today = datetime.today().date()
    start = today - timedelta(days=730)
    end   = today + timedelta(days=1)   # +1 so today is included if market has closed

    t = yf.Ticker(ticker)

    try:
        raw = t.history(
            start=str(start),
            end=str(end),
            auto_adjust=True,
            actions=False,
        )
    except Exception as e:
        raise ConnectionError(f"Yahoo Finance fetch failed for '{ticker}': {e}")

    if raw is None or raw.empty:
        raise ValueError(
            f"Yahoo Finance returned 0 rows for '{ticker}'. "
            "Check ticker symbol and internet connection."
        )

    # ── Flatten MultiIndex columns (yfinance >= 0.2 sometimes wraps in ticker level) ──
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    # ── Strip timezone — THIS was the silent bug (index showed 00:00:00-04:00) ──
    if hasattr(raw.index, "tz") and raw.index.tz is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    raw.index = pd.to_datetime(raw.index)

    # ── Normalise column names to Title case ──
    col_map = {}
    for c in raw.columns:
        cl = c.strip().lower()
        if   cl == "open":   col_map[c] = "Open"
        elif cl == "high":   col_map[c] = "High"
        elif cl == "low":    col_map[c] = "Low"
        elif cl == "close":  col_map[c] = "Close"
        elif cl == "volume": col_map[c] = "Volume"
    raw = raw.rename(columns=col_map)

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    raw  = raw[keep].copy()

    if "Volume" not in raw.columns:
        raw["Volume"] = 0.0

    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw.sort_index().dropna(subset=["Close"])

    n_raw = len(raw)
    if n_raw < 60:
        raise ValueError(
            f"Only {n_raw} trading days for '{ticker}' after cleaning. "
            "Need at least 60."
        )

    df = _compute_features(raw)

    n_after = len(df)
    if n_after < 2:
        raise ValueError(
            f"Not enough data for {ticker} after feature computation. "
            f"Raw rows: {n_raw}, after dropna: {n_after}. "
            "This should not happen with 730-day fetch — check for data gaps."
        )

    return df


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Model features — identical to training pipeline ──
    df["return_1d"]     = df["Close"].pct_change()
    df["MA_10"]         = df["Close"].rolling(10).mean()
    df["MA_50"]         = df["Close"].rolling(50).mean()
    df["volatility"]    = df["return_1d"].rolling(10).std() * np.sqrt(252)
    df["volume_change"] = df["Volume"].pct_change()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Chart-only indicators (kept separate — NOT in dropna) ──
    df["MA_20"]    = df["Close"].rolling(20).mean()
    df["MA_200"]   = df["Close"].rolling(200).mean()
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    df["BB_Std"]   = df["Close"].rolling(20).std()
    df["BB_Up"]    = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lo"]    = df["BB_Mid"] - 2 * df["BB_Std"]
    e12            = df["Close"].ewm(span=12, adjust=False).mean()
    e26            = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]     = e12 - e26
    df["MACD_Sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_H"]   = df["MACD"] - df["MACD_Sig"]

    # ── Drop ONLY rows where model features are NaN (NOT chart indicators) ──
    df = df.dropna(subset=FEATURE_COLS)

    return df


def get_prediction_row(ticker: str) -> dict:
    df         = get_latest_data(ticker)
    latest     = df.iloc[-1]
    prev_close = df.iloc[-2]["Close"]

    missing = [c for c in FEATURE_COLS if c not in df.columns or pd.isna(latest[c])]
    if missing:
        raise ValueError(f"Features NaN for '{ticker}': {missing}")

    return {
        "df":         df,
        "features":   {col: float(latest[col]) for col in FEATURE_COLS},
        "latest":     latest,
        "date":       df.index[-1].strftime("%A, %d %B %Y"),
        "prev_close": float(prev_close),
    }


def explain_signal_text(latest: pd.Series, signal: str, prob: float) -> list:
    reasons = []
    close  = float(latest["Close"])
    rsi    = float(latest["RSI"])
    ma10   = float(latest["MA_10"])
    ma50   = float(latest["MA_50"])
    ret1d  = float(latest["return_1d"]) * 100
    macd   = float(latest.get("MACD",     0.0))
    macd_s = float(latest.get("MACD_Sig", 0.0))
    bb_up  = float(latest.get("BB_Up",    np.nan))
    bb_lo  = float(latest.get("BB_Lo",    np.nan))

    if rsi > 70:
        reasons.append((f"RSI {rsi:.1f} — overbought. Pullback may be due.", "bearish"))
    elif rsi < 30:
        reasons.append((f"RSI {rsi:.1f} — oversold. Bounce could be near.", "bullish"))
    else:
        reasons.append((f"RSI {rsi:.1f} — neutral zone, no extreme momentum.", "neutral"))

    if macd > macd_s:
        reasons.append((f"MACD ({macd:.3f}) above signal ({macd_s:.3f}) — bullish crossover.", "bullish"))
    else:
        reasons.append((f"MACD ({macd:.3f}) below signal ({macd_s:.3f}) — bearish crossover.", "bearish"))

    if close > ma50:
        reasons.append((f"Price ${close:.2f} above MA50 ${ma50:.2f} — uptrend intact.", "bullish"))
    else:
        reasons.append((f"Price ${close:.2f} below MA50 ${ma50:.2f} — downtrend.", "bearish"))

    if ma10 > ma50:
        reasons.append((f"MA10 ${ma10:.2f} > MA50 ${ma50:.2f} — Golden Cross zone.", "bullish"))
    else:
        reasons.append((f"MA10 ${ma10:.2f} < MA50 ${ma50:.2f} — Death Cross zone.", "bearish"))

    if not (np.isnan(bb_up) or np.isnan(bb_lo)) and (bb_up - bb_lo) > 0:
        bb_pct = (close - bb_lo) / (bb_up - bb_lo) * 100
        if bb_pct > 80:
            reasons.append((f"Near upper Bollinger Band ({bb_pct:.0f}%B) — overbought.", "bearish"))
        elif bb_pct < 20:
            reasons.append((f"Near lower Bollinger Band ({bb_pct:.0f}%B) — oversold.", "bullish"))
        else:
            reasons.append((f"Within Bollinger Bands ({bb_pct:.0f}%B) — no band extreme.", "neutral"))

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
        reasons.append((f"Low confidence ({prob*100:.1f}%) — direction uncertain.", "neutral"))

    return reasons