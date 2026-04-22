"""
data/live_feed.py  —  AlgoTrade Daily Signal
Fetches latest Yahoo Finance data and computes model features.
Works on Streamlit Cloud (uses Ticker.history() not yf.download()).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

FEATURE_COLS = ["return_1d", "MA_10", "MA_50", "volatility", "volume_change", "RSI"]


# ─────────────────────────────────────────────
# 1. DATA FETCHING  (Ticker.history is more
#    reliable than yf.download on cloud)
# ─────────────────────────────────────────────
def get_latest_data(ticker: str) -> pd.DataFrame:
    """
    Download ~18 months of daily OHLCV via Ticker.history().
    18 months = ~390 trading days → plenty of buffer for MA_50 + RSI warmup.
    Falls back to 'max' period if the period string fails.
    """
    t = yf.Ticker(ticker)

    # Try 18 months first; some tickers need 'max' as fallback
    for period in ("18mo", "2y", "max"):
        try:
            raw = t.history(period=period, auto_adjust=True, actions=False)
            if raw is not None and len(raw) >= 60:
                break
        except Exception:
            continue
    else:
        raise ConnectionError(
            f"Yahoo Finance returned no usable data for '{ticker}'.\n"
            f"Check the ticker symbol and your internet connection."
        )

    # ── Flatten MultiIndex columns (yfinance ≥ 0.2.x) ──
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    # ── Normalise column names to Title case ──
    col_map = {}
    for c in raw.columns:
        cl = c.strip().lower()
        for target in ("open", "high", "low", "close", "volume"):
            if cl == target:
                col_map[c] = target.capitalize()
    raw = raw.rename(columns=col_map)

    # Keep only what we need
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    raw = raw[keep].copy()

    # Force numeric and drop rows with no Close
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw.index = pd.to_datetime(raw.index).tz_localize(None)  # strip timezone
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw.sort_index().dropna(subset=["Close"])

    if "Volume" not in raw.columns:
        raw["Volume"] = np.nan

    if len(raw) < 60:
        raise ValueError(
            f"Only {len(raw)} trading days available for '{ticker}'. "
            f"Need at least 60."
        )

    return _compute_features(raw)


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
#    Must be identical to training pipeline
# ─────────────────────────────────────────────
def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Model features (must match training exactly) ──
    df["return_1d"]     = df["Close"].pct_change()
    df["MA_10"]         = df["Close"].rolling(10).mean()
    df["MA_50"]         = df["Close"].rolling(50).mean()
    df["volatility"]    = df["return_1d"].rolling(10).std() * np.sqrt(252)
    df["volume_change"] = df["Volume"].pct_change()

    # RSI (14-period)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Extra indicators for charts / text explanation ──
    df["MA_20"]    = df["Close"].rolling(20).mean()
    df["MA_200"]   = df["Close"].rolling(200).mean()
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    df["BB_Std"]   = df["Close"].rolling(20).std()
    df["BB_Up"]    = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lo"]    = df["BB_Mid"] - 2 * df["BB_Std"]

    e12             = df["Close"].ewm(span=12, adjust=False).mean()
    e26             = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = e12 - e26
    df["MACD_Sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_H"]    = df["MACD"] - df["MACD_Sig"]

    # ── Drop ONLY rows where model features are NaN ──
    # Do NOT dropna on all columns — that would lose rows where
    # MA_200 is NaN (needs 200 days) but model features are fine.
    df = df.dropna(subset=FEATURE_COLS)

    return df


# ─────────────────────────────────────────────
# 3. MAIN ENTRY POINT
# ─────────────────────────────────────────────
def get_prediction_row(ticker: str) -> dict:
    """
    Returns a dict with everything the Daily Signal page needs:
      df          — full DataFrame with features
      features    — dict of the 6 model input values
      latest      — last row as pd.Series
      date        — human-readable date string
      prev_close  — previous day's close for % change display
    """
    df = get_latest_data(ticker)

    if len(df) < 2:
        raise ValueError(
            f"Not enough clean rows for '{ticker}' after feature engineering. "
            f"Got {len(df)} — need at least 2. "
            f"This usually means MA_50 warm-up ate all available data."
        )

    latest     = df.iloc[-1]
    prev_close = df.iloc[-2]["Close"]

    # Validate every feature the model will consume
    missing = []
    for col in FEATURE_COLS:
        if col not in df.columns or pd.isna(latest[col]):
            missing.append(col)
    if missing:
        raise ValueError(
            f"Feature(s) missing or NaN for '{ticker}': {missing}. "
            f"Raw rows after dropna: {len(df)}."
        )

    features = {col: float(latest[col]) for col in FEATURE_COLS}

    return {
        "df":         df,
        "features":   features,
        "latest":     latest,
        "date":       df.index[-1].strftime("%A, %d %B %Y"),
        "prev_close": float(prev_close),
    }


# ─────────────────────────────────────────────
# 4. TEXT EXPLANATIONS FOR XAI
# ─────────────────────────────────────────────
def explain_signal_text(latest: pd.Series, signal: str, prob: float) -> list:
    """
    Returns list of (reason_text, sentiment) tuples.
    sentiment: 'bullish' | 'bearish' | 'neutral'
    """
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

    # RSI
    if rsi > 70:
        reasons.append((f"RSI {rsi:.1f} — overbought. A pullback may be due.", "bearish"))
    elif rsi < 30:
        reasons.append((f"RSI {rsi:.1f} — oversold. A bounce could be near.", "bullish"))
    else:
        reasons.append((f"RSI {rsi:.1f} — neutral zone, no extreme momentum.", "neutral"))

    # MACD
    if macd > macd_s:
        reasons.append((f"MACD ({macd:.3f}) above signal ({macd_s:.3f}) — bullish crossover.", "bullish"))
    else:
        reasons.append((f"MACD ({macd:.3f}) below signal ({macd_s:.3f}) — bearish crossover.", "bearish"))

    # Price vs MA50
    if close > ma50:
        reasons.append((f"Price ${close:.2f} above MA50 ${ma50:.2f} — uptrend intact.", "bullish"))
    else:
        reasons.append((f"Price ${close:.2f} below MA50 ${ma50:.2f} — downtrend.", "bearish"))

    # MA cross
    if ma10 > ma50:
        reasons.append((f"MA10 ${ma10:.2f} > MA50 ${ma50:.2f} — Golden Cross zone.", "bullish"))
    else:
        reasons.append((f"MA10 ${ma10:.2f} < MA50 ${ma50:.2f} — Death Cross zone.", "bearish"))

    # Bollinger Band %B
    if not (np.isnan(bb_up) or np.isnan(bb_lo)) and (bb_up - bb_lo) > 0:
        bb_pct = (close - bb_lo) / (bb_up - bb_lo) * 100
        if bb_pct > 80:
            reasons.append((f"Near upper Bollinger Band ({bb_pct:.0f}%B) — overbought.", "bearish"))
        elif bb_pct < 20:
            reasons.append((f"Near lower Bollinger Band ({bb_pct:.0f}%B) — oversold.", "bullish"))
        else:
            reasons.append((f"Within Bollinger Bands ({bb_pct:.0f}%B) — no band extreme.", "neutral"))

    # Yesterday's return
    if ret1d > 1.5:
        reasons.append((f"Yesterday's return: +{ret1d:.2f}% — strong upward move.", "bullish"))
    elif ret1d < -1.5:
        reasons.append((f"Yesterday's return: {ret1d:.2f}% — strong downward move.", "bearish"))
    else:
        reasons.append((f"Yesterday's return: {ret1d:.2f}% — relatively flat.", "neutral"))

    # Model confidence
    conf      = abs(prob - 0.5) * 200
    direction = "bullish" if prob > 0.5 else "bearish"
    if conf > 60:
        reasons.append((f"High model confidence: {prob*100:.1f}% probability of upward move.", direction))
    elif conf > 20:
        reasons.append((f"Moderate confidence: {prob*100:.1f}% probability of upward move.", direction))
    else:
        reasons.append((f"Low confidence ({prob*100:.1f}%) — market direction uncertain.", "neutral"))

    return reasons