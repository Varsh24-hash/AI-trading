"""
pages/Daily_Signal.py  —  AlgoTrade · Daily Signal
Live BUY / HOLD / SELL prediction with SHAP + text explanations.
Runs on Streamlit Cloud using Ticker.history() (no yf.download).
"""

import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# ── Path setup so live_feed imports correctly ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.live_feed import get_prediction_row, explain_signal_text, FEATURE_COLS

# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(page_title="Daily Signal · AlgoTrade", layout="wide")

TICKERS = {
    "AAPL · Apple Inc.":       "AAPL",
    "MSFT · Microsoft":        "MSFT",
    "GOOGL · Alphabet":        "GOOGL",
    "AMZN · Amazon":           "AMZN",
    "NVDA · NVIDIA":           "NVDA",
    "META · Meta Platforms":   "META",
    "TSLA · Tesla":            "TSLA",
    "JPM · JPMorgan":          "JPM",
    "V · Visa":                "V",
    "WMT · Walmart":           "WMT",
}

SIGNAL_COLOR = {"BUY": "#1D9E75", "HOLD": "#BA7517", "SELL": "#E24B4A"}
SENTIMENT_ICON = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}

MODEL_PATHS = {
    "XGBoost":  "models/xgb_model.pkl",
    "Random Forest": "models/rf_model.pkl",
    "Logistic": "models/lr_model.pkl",
}

# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
@st.cache_resource
def load_models():
    loaded = {}
    for name, path in MODEL_PATHS.items():
        try:
            loaded[name] = joblib.load(path)
        except Exception:
            pass
    return loaded


def predict_signal(model, features: dict):
    X = pd.DataFrame([features])[FEATURE_COLS]
    prob = float(model.predict_proba(X)[0, 1])
    if prob > 0.6:
        sig = "BUY"
    elif prob < 0.4:
        sig = "SELL"
    else:
        sig = "HOLD"
    return sig, prob


def shap_chart(model, features: dict, model_name: str):
    """Return a plotly figure of SHAP values, or None if SHAP unavailable."""
    try:
        import shap
        X = pd.DataFrame([features])[FEATURE_COLS]
        if "XGB" in model_name or "Forest" in model_name:
            exp = shap.TreeExplainer(model)
        else:
            exp = shap.LinearExplainer(model, X)
        vals = exp.shap_values(X)
        # Binary classifiers may return list [neg_class, pos_class]
        if isinstance(vals, list):
            vals = vals[1]
        sv = vals[0]

        colors = ["#1D9E75" if v > 0 else "#E24B4A" for v in sv]
        fig = go.Figure(go.Bar(
            x=sv,
            y=FEATURE_COLS,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in sv],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#C2C0B6",
            height=280,
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(
                zeroline=True, zerolinecolor="#444",
                gridcolor="#2a2a2a",
                title="SHAP value (push toward BUY →)",
            ),
            yaxis=dict(gridcolor="#2a2a2a"),
        )
        return fig
    except Exception:
        return None


def price_chart(df: pd.DataFrame, signal: str, date_str: str):
    """90-day candlestick with MA10, MA50, BB, MACD, RSI panels."""
    df90 = df.tail(90).copy()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df90.index, open=df90["Open"], high=df90["High"],
        low=df90["Low"],  close=df90["Close"],
        increasing_line_color="#1D9E75", decreasing_line_color="#E24B4A",
        name="Price",
    ), row=1, col=1)

    for col, color, name in [
        ("MA_10", "#E8A838", "MA10"),
        ("MA_50", "#5B9BD5", "MA50"),
    ]:
        if col in df90.columns:
            fig.add_trace(go.Scatter(
                x=df90.index, y=df90[col],
                line=dict(color=color, width=1.2),
                name=name,
            ), row=1, col=1)

    # Bollinger Bands
    if "BB_Up" in df90.columns:
        fig.add_trace(go.Scatter(
            x=df90.index, y=df90["BB_Up"],
            line=dict(color="#888", width=0.8, dash="dot"),
            name="BB Upper", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df90.index, y=df90["BB_Lo"],
            line=dict(color="#888", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,150,0.07)",
            name="BB Lower", showlegend=False,
        ), row=1, col=1)

    # Signal vertical line on last date
    sig_color = SIGNAL_COLOR.get(signal, "#888")
    last_date = df90.index[-1]
    fig.add_vline(
        x=last_date, line_color=sig_color,
        line_width=2, line_dash="dash",
        annotation_text=signal,
        annotation_font_color=sig_color,
        annotation_font_size=13,
        row=1, col=1,
    )

    # ── MACD ──
    if "MACD" in df90.columns:
        macd_colors = ["#1D9E75" if v >= 0 else "#E24B4A"
                       for v in df90["MACD_H"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df90.index, y=df90["MACD_H"],
            marker_color=macd_colors, name="MACD Hist",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df90.index, y=df90["MACD"],
            line=dict(color="#E8A838", width=1), name="MACD",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df90.index, y=df90["MACD_Sig"],
            line=dict(color="#5B9BD5", width=1, dash="dot"), name="Signal",
        ), row=2, col=1)

    # ── RSI ──
    if "RSI" in df90.columns:
        fig.add_trace(go.Scatter(
            x=df90.index, y=df90["RSI"],
            line=dict(color="#E8A838", width=1.2), name="RSI",
        ), row=3, col=1)
        for level, color in [(70, "#E24B4A"), (30, "#1D9E75")]:
            fig.add_hline(
                y=level, line_color=color,
                line_dash="dot", line_width=0.8,
                row=3, col=3,
            )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#C2C0B6",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            bgcolor="rgba(0,0,0,0)", font_size=11,
        ),
        xaxis_rangeslider_visible=False,
    )
    for row in (1, 2, 3):
        fig.update_xaxes(gridcolor="#2a2a2a", showgrid=True, row=row, col=1)
        fig.update_yaxes(gridcolor="#2a2a2a", showgrid=True, row=row, col=1)

    return fig


# ══════════════════════════════════════════════
#  PAGE LAYOUT
# ══════════════════════════════════════════════
models = load_models()

# ── Header ──
st.markdown("""
<h1 style='margin-bottom:0'><span style='color:#fff'>Daily</span>
<span style='color:#E8A838'> Signal</span></h1>
""", unsafe_allow_html=True)

# ── Sidebar controls ──
with st.sidebar:
    st.markdown("### DAILY SIGNAL")
    ticker_label = st.selectbox("Stock", list(TICKERS.keys()), index=0)
    ticker = TICKERS[ticker_label]
    engine_name = st.selectbox(
        "Primary model",
        [k for k in MODEL_PATHS if k in models] or ["XGBoost"],
    )
    st.markdown("---")
    run_btn = st.button("▶ FETCH & PREDICT", use_container_width=True)

if not models:
    st.error("No trained models found. Make sure .pkl files are in the models/ folder.")
    st.stop()

if engine_name not in models:
    st.warning(f"{engine_name} not loaded. Using first available model.")
    engine_name = list(models.keys())[0]

# ── Subtitle ──
company = ticker_label.split("·")[1].strip() if "·" in ticker_label else ticker_label
st.markdown(
    f"<p style='color:#888;margin-top:0'>{company} · {ticker} · "
    f"Live prediction from Yahoo Finance</p>",
    unsafe_allow_html=True,
)

# ── Main logic ──
with st.spinner(f"Fetching latest data for {ticker}…"):
    try:
        data = get_prediction_row(ticker)
    except Exception as e:
        st.error(f"Could not fetch live data: {e}")

        # ── Debug expander — shows exactly what went wrong ──
        with st.expander("🔍 Debug details (expand to diagnose)"):
            st.markdown("**Attempting raw yfinance fetch to check connectivity:**")
            try:
                import yfinance as yf
                t = yf.Ticker(ticker)
                raw = t.history(period="6mo", auto_adjust=True, actions=False)
                st.success(f"Raw fetch OK — {len(raw)} rows returned")
                st.write("Columns:", list(raw.columns))
                st.write("Date range:", raw.index[0] if len(raw) else "empty",
                         "→", raw.index[-1] if len(raw) else "empty")
                st.dataframe(raw.tail(5))
            except Exception as e2:
                st.error(f"Raw fetch also failed: {e2}")

        st.info("Check your internet connection or try a different ticker.")
        st.stop()

# ── Unpack data ──
df        = data["df"]
features  = data["features"]
latest    = data["latest"]
date_str  = data["date"]
prev_close = data["prev_close"]
close      = float(latest["Close"])
pct_chg    = (close - prev_close) / prev_close * 100

# ── Predict with primary model ──
primary_model = models[engine_name]
signal, prob  = predict_signal(primary_model, features)
sig_color     = SIGNAL_COLOR[signal]

# ── BIG SIGNAL CARD ──
st.markdown(f"""
<div style='
    background: linear-gradient(135deg, #1a1a1a 0%, #111 100%);
    border: 1px solid {sig_color}44;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin: 1rem 0 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
'>
  <div>
    <div style='font-size:12px;letter-spacing:.1em;color:#888;text-transform:uppercase'>
      {engine_name} · {date_str}
    </div>
    <div style='font-size:72px;font-weight:700;color:{sig_color};line-height:1'>
      {signal}
    </div>
    <div style='font-size:15px;color:#aaa;margin-top:6px'>
      {prob*100:.1f}% probability of upward move
    </div>
  </div>
  <div style='text-align:right'>
    <div style='font-size:42px;font-weight:600;color:#fff'>${close:.2f}</div>
    <div style='font-size:16px;color:{"#1D9E75" if pct_chg>=0 else "#E24B4A"}'>
      {"▲" if pct_chg>=0 else "▼"} {pct_chg:+.2f}% vs prev close
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metric row ──
cols = st.columns(6)
metric_vals = {
    "RSI (14)":    f"{features.get('RSI', 0):.1f}",
    "MA10":        f"${features.get('MA_10', 0):.2f}",
    "MA50":        f"${features.get('MA_50', 0):.2f}",
    "Return 1d":   f"{features.get('return_1d', 0)*100:+.2f}%",
    "Volatility":  f"{features.get('volatility', 0)*100:.1f}%",
    "Vol Change":  f"{features.get('volume_change', 0)*100:+.1f}%",
}
for col, (k, v) in zip(cols, metric_vals.items()):
    col.metric(k, v)

st.markdown("---")

# ── All model signals ──
if len(models) > 1:
    st.markdown("#### All model signals")
    mcols = st.columns(len(models))
    for mc, (mname, mobj) in zip(mcols, models.items()):
        msig, mprob = predict_signal(mobj, features)
        mc.markdown(f"""
        <div style='text-align:center;background:#1a1a1a;border-radius:12px;padding:1rem;
                    border:1px solid {SIGNAL_COLOR[msig]}44'>
          <div style='font-size:11px;color:#888;margin-bottom:4px'>{mname}</div>
          <div style='font-size:28px;font-weight:700;color:{SIGNAL_COLOR[msig]}'>{msig}</div>
          <div style='font-size:12px;color:#aaa'>{mprob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

# ── Charts + SHAP side by side ──
left, right = st.columns([3, 2])

with left:
    st.markdown("#### Price chart — last 90 days")
    st.plotly_chart(price_chart(df, signal, date_str), use_container_width=True)

with right:
    st.markdown("#### Explainable AI — feature contributions (SHAP)")
    fig_shap = shap_chart(primary_model, features, engine_name)
    if fig_shap:
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("Green bars push toward BUY · Red bars push toward SELL")
    else:
        # Fallback: manual feature importance bar
        st.info("SHAP library not installed — showing raw feature values.")
        feat_df = pd.DataFrame(
            {"Feature": list(features.keys()), "Value": list(features.values())}
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Text explanations ──
st.markdown("#### Why this signal? — AI reasoning")
reasons = explain_signal_text(latest, signal, prob)

reason_cols = st.columns(2)
for i, (text, sentiment) in enumerate(reasons):
    icon = SENTIMENT_ICON.get(sentiment, "⚪")
    color_map = {"bullish": "#0F6E56", "bearish": "#A32D2D", "neutral": "#5F5E5A"}
    bg = color_map.get(sentiment, "#333")
    reason_cols[i % 2].markdown(f"""
    <div style='background:{bg}22;border-left:3px solid {bg};
                border-radius:8px;padding:.6rem .9rem;margin-bottom:8px;
                font-size:13px;color:#ccc'>
      {icon} {text}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Raw feature table ──
with st.expander("📊 Raw feature values fed to model"):
    feat_display = pd.DataFrame([{
        "Feature":     k,
        "Value":       round(v, 6),
        "Description": {
            "return_1d":     "Previous day's return",
            "MA_10":         "10-day moving average of close",
            "MA_50":         "50-day moving average of close",
            "volatility":    "Annualised 10-day rolling volatility",
            "volume_change": "Day-over-day volume change",
            "RSI":           "14-period Relative Strength Index",
        }.get(k, ""),
    } for k, v in features.items()])
    st.dataframe(feat_display, use_container_width=True, hide_index=True)

# ── Data freshness note ──
last_date = df.index[-1].strftime("%d %b %Y")
st.caption(
    f"Data as of {last_date} · Source: Yahoo Finance · "
    f"Model: {engine_name} (pre-trained, not retrained daily) · "
    f"Not financial advice."
)