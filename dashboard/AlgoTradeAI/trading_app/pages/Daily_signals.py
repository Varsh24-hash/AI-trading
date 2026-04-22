"""
pages/Daily_Signal.py  —  AlgoTrade · Daily Signal
Live BUY / HOLD / SELL with SHAP + text XAI.
"""

import sys, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
from datetime import datetime, timedelta

# ── Path so 'data.live_feed' resolves from any working directory ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.live_feed import get_prediction_row, explain_signal_text, FEATURE_COLS

# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="Daily Signal · AlgoTrade", layout="wide")
# ══════════════════════════════════════════════════════════

TICKERS = {
    "AAPL · Apple Inc.":     "AAPL",
    "MSFT · Microsoft":      "MSFT",
    "GOOGL · Alphabet":      "GOOGL",
    "AMZN · Amazon":         "AMZN",
    "NVDA · NVIDIA":         "NVDA",
    "META · Meta Platforms": "META",
    "TSLA · Tesla":          "TSLA",
    "JPM · JPMorgan":        "JPM",
    "V · Visa":              "V",
    "WMT · Walmart":         "WMT",
}
MODEL_PATHS = {
    "XGBoost":        "models/xgb_model.pkl",
    "Random Forest":  "models/rf_model.pkl",
    "Logistic":       "models/lr_model.pkl",
}
SIG_COLOR  = {"BUY": "#1D9E75", "HOLD": "#BA7517", "SELL": "#E24B4A"}
SENT_ICON  = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
SENT_BG    = {"bullish": "#0F6E56", "bearish": "#A32D2D", "neutral": "#3a3a3a"}


@st.cache_resource
def load_models():
    out = {}
    for name, path in MODEL_PATHS.items():
        try:
            out[name] = joblib.load(path)
        except Exception:
            pass
    return out


def predict(model, features):
    X   = pd.DataFrame([features])[FEATURE_COLS]
    prob = float(model.predict_proba(X)[0, 1])
    sig  = "BUY" if prob > 0.6 else ("SELL" if prob < 0.4 else "HOLD")
    return sig, prob


def make_shap_fig(model, features, name):
    try:
        import shap
        X  = pd.DataFrame([features])[FEATURE_COLS]
        ex = shap.TreeExplainer(model) if "Log" not in name else shap.LinearExplainer(model, X)
        sv = ex.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv[0]
        colors = ["#1D9E75" if v > 0 else "#E24B4A" for v in sv]
        fig = go.Figure(go.Bar(
            x=sv, y=FEATURE_COLS, orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in sv], textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#C2C0B6", height=260,
            margin=dict(l=10, r=70, t=8, b=8),
            xaxis=dict(
                zeroline=True, zerolinecolor="#555", gridcolor="#2a2a2a",
                title="← SELL    SHAP value    BUY →",
            ),
            yaxis=dict(gridcolor="#2a2a2a"),
        )
        return fig
    except Exception:
        return None


def make_price_fig(df, signal):
    df90 = df.tail(90).copy()
    fig  = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.22, 0.20],
        vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=df90.index, open=df90["Open"], high=df90["High"],
        low=df90["Low"], close=df90["Close"],
        increasing_line_color="#1D9E75", decreasing_line_color="#E24B4A",
        name="Price",
    ), row=1, col=1)

    for col, color, lname in [("MA_10","#E8A838","MA10"), ("MA_50","#5B9BD5","MA50")]:
        if col in df90:
            fig.add_trace(go.Scatter(
                x=df90.index, y=df90[col],
                line=dict(color=color, width=1.2), name=lname,
            ), row=1, col=1)

    if "BB_Up" in df90:
        fig.add_trace(go.Scatter(
            x=df90.index, y=df90["BB_Up"],
            line=dict(color="#666", width=0.8, dash="dot"),
            name="BB", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df90.index, y=df90["BB_Lo"],
            line=dict(color="#666", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(130,130,130,0.07)",
            name="BB Lo", showlegend=False,
        ), row=1, col=1)

    fig.add_vline(
        x=df90.index[-1], line_color=SIG_COLOR[signal],
        line_width=2, line_dash="dash",
        annotation_text=f"  {signal} signal",
        annotation_font_color=SIG_COLOR[signal],
        annotation_font_size=12,
    )

    if "MACD" in df90:
        bar_colors = ["#1D9E75" if v >= 0 else "#E24B4A"
                      for v in df90["MACD_H"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df90.index, y=df90["MACD_H"],
            marker_color=bar_colors, name="MACD Hist",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df90.index, y=df90["MACD"],
            line=dict(color="#E8A838", width=1), name="MACD",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df90.index, y=df90["MACD_Sig"],
            line=dict(color="#5B9BD5", width=1, dash="dot"), name="Signal",
        ), row=2, col=1)

    if "RSI" in df90:
        fig.add_trace(go.Scatter(
            x=df90.index, y=df90["RSI"],
            line=dict(color="#E8A838", width=1.2), name="RSI",
        ), row=3, col=1)
        for lvl, clr in [(70, "#E24B4A44"), (30, "#1D9E7544")]:
            fig.add_hline(y=lvl, line_color=clr, line_dash="dot",
                          line_width=1, row=3, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#C2C0B6", height=500,
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            bgcolor="rgba(0,0,0,0)", font_size=11,
        ),
    )
    for r in (1, 2, 3):
        fig.update_xaxes(gridcolor="#2a2a2a", row=r, col=1)
        fig.update_yaxes(gridcolor="#2a2a2a", row=r, col=1)
    return fig


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
models = load_models()

with st.sidebar:
    st.markdown("### DAILY SIGNAL")
    ticker_label  = st.selectbox("Stock", list(TICKERS.keys()))
    ticker        = TICKERS[ticker_label]
    available     = [k for k in MODEL_PATHS if k in models]
    engine_name   = st.selectbox("Primary model", available or ["XGBoost"])
    st.markdown("---")

# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='margin-bottom:0'>"
    "<span style='color:#fff'>Daily</span>"
    "<span style='color:#E8A838'> Signal</span>"
    "</h1>",
    unsafe_allow_html=True,
)
company = ticker_label.split("·")[1].strip() if "·" in ticker_label else ticker_label
st.markdown(
    f"<p style='color:#888;margin-top:2px'>"
    f"{company} · {ticker} · Live prediction from Yahoo Finance</p>",
    unsafe_allow_html=True,
)

if not models:
    st.error("No trained models found in models/ folder. Add .pkl files and restart.")
    st.stop()

# ══════════════════════════════════════════════════════════
#  FETCH DATA
# ══════════════════════════════════════════════════════════
with st.spinner(f"Fetching 2 years of {ticker} data from Yahoo Finance…"):
    try:
        data = get_prediction_row(ticker)
    except Exception as err:
        st.error(f"Could not fetch live data: {err}")

        with st.expander("🔍 Debug details — click to diagnose"):
            st.markdown("**Running raw yfinance fetch with explicit dates:**")
            try:
                import yfinance as yf
                today = datetime.today().date()
                start = today - timedelta(days=730)
                end   = today + timedelta(days=1)

                t   = yf.Ticker(ticker)
                raw = t.history(
                    start=str(start), end=str(end),
                    auto_adjust=True, actions=False,
                )

                if raw is None or raw.empty:
                    st.error("Raw fetch returned 0 rows — yfinance or network issue.")
                else:
                    st.success(f"Raw fetch OK — {len(raw)} rows returned")
                    st.write("**Columns:**", list(raw.columns))
                    st.write("**Index type:**", type(raw.index).__name__,
                             "| tz:", getattr(raw.index, "tz", "none"))
                    st.write("**Date range:**",
                             raw.index[0], "→", raw.index[-1])
                    st.dataframe(raw.tail(5))

                    # Simulate feature computation
                    st.markdown("**Simulating feature engineering:**")
                    raw2 = raw.copy()
                    raw2["return_1d"]  = raw2["Close"].pct_change()
                    raw2["MA_10"]      = raw2["Close"].rolling(10).mean()
                    raw2["MA_50"]      = raw2["Close"].rolling(50).mean()
                    raw2["volatility"] = raw2["return_1d"].rolling(10).std() * (252**0.5)
                    raw2["volume_change"] = raw2["Volume"].pct_change()
                    delta = raw2["Close"].diff()
                    gain  = delta.clip(lower=0).rolling(14).mean()
                    loss  = (-delta.clip(upper=0)).rolling(14).mean()
                    raw2["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))
                    before = len(raw2)
                    raw2   = raw2.dropna(subset=FEATURE_COLS)
                    after  = len(raw2)
                    st.info(f"Rows before dropna: {before} | After dropna: {after}")
                    if after > 0:
                        st.success("Feature engineering works — refresh page or clear cache.")
                        st.dataframe(raw2[FEATURE_COLS].tail(3))
                    else:
                        st.error("dropna removed ALL rows — data may have NaN gaps.")

            except Exception as e2:
                st.error(f"Debug fetch also failed: {e2}")

        st.info("Try refreshing the page. If this persists, select a different ticker.")
        st.stop()

# ══════════════════════════════════════════════════════════
#  UNPACK & PREDICT
# ══════════════════════════════════════════════════════════
df         = data["df"]
features   = data["features"]
latest     = data["latest"]
date_str   = data["date"]
prev_close = data["prev_close"]
close      = float(latest["Close"])
pct_chg    = (close - prev_close) / prev_close * 100

if engine_name not in models:
    engine_name = list(models.keys())[0]

signal, prob = predict(models[engine_name], features)
sig_col      = SIG_COLOR[signal]

# ══════════════════════════════════════════════════════════
#  BIG SIGNAL CARD
# ══════════════════════════════════════════════════════════
arrow = "▲" if pct_chg >= 0 else "▼"
pchg_color = "#1D9E75" if pct_chg >= 0 else "#E24B4A"

st.markdown(f"""
<div style="background:#111;border:1px solid {sig_col}55;border-radius:16px;
            padding:2rem 2.5rem;margin:1rem 0 1.5rem;
            display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div style="font-size:11px;letter-spacing:.1em;color:#666;text-transform:uppercase;margin-bottom:4px">
      {engine_name} · {date_str}
    </div>
    <div style="font-size:80px;font-weight:700;color:{sig_col};line-height:1">
      {signal}
    </div>
    <div style="font-size:14px;color:#999;margin-top:6px">
      {prob*100:.1f}% probability of upward move
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-size:44px;font-weight:600;color:#fff">${close:.2f}</div>
    <div style="font-size:15px;color:{pchg_color};margin-top:4px">
      {arrow} {pct_chg:+.2f}% vs previous close
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metric strip ──
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("RSI (14)",    f"{features['RSI']:.1f}")
c2.metric("MA10",        f"${features['MA_10']:.2f}")
c3.metric("MA50",        f"${features['MA_50']:.2f}")
c4.metric("Return 1d",   f"{features['return_1d']*100:+.2f}%")
c5.metric("Volatility",  f"{features['volatility']*100:.1f}%")
c6.metric("Vol Change",  f"{features['volume_change']*100:+.1f}%")

st.markdown("---")

# ── All model signals ──
if len(models) > 1:
    st.markdown("#### All model signals")
    mcols = st.columns(len(models))
    for mc, (mname, mobj) in zip(mcols, models.items()):
        ms, mp = predict(mobj, features)
        mc.markdown(f"""
        <div style="text-align:center;background:#1a1a1a;border-radius:12px;
                    padding:1rem;border:1px solid {SIG_COLOR[ms]}44;margin-bottom:8px">
          <div style="font-size:11px;color:#777;margin-bottom:4px">{mname}</div>
          <div style="font-size:30px;font-weight:700;color:{SIG_COLOR[ms]}">{ms}</div>
          <div style="font-size:12px;color:#999">{mp*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")

# ── Chart + SHAP ──
left, right = st.columns([3, 2])

with left:
    st.markdown("#### Price chart — last 90 days")
    st.plotly_chart(make_price_fig(df, signal), use_container_width=True)

with right:
    st.markdown("#### SHAP — feature contributions")
    shap_fig = make_shap_fig(models[engine_name], features, engine_name)
    if shap_fig:
        st.plotly_chart(shap_fig, use_container_width=True)
        st.caption("Green = pushed toward BUY · Red = pushed toward SELL")
    else:
        st.info("Install `shap` to enable SHAP explanations.")
        feat_df = pd.DataFrame({"Feature": list(features.keys()),
                                "Value":   [round(v, 5) for v in features.values()]})
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ── Text reasoning ──
st.markdown("#### Why this signal? — AI reasoning")
reasons = explain_signal_text(latest, signal, prob)
rc = st.columns(2)
for i, (text, sentiment) in enumerate(reasons):
    bg = SENT_BG.get(sentiment, "#333")
    rc[i % 2].markdown(f"""
    <div style="background:{bg}33;border-left:3px solid {bg};
                border-radius:8px;padding:.55rem .85rem;margin-bottom:8px;
                font-size:13px;color:#ccc;line-height:1.5">
      {SENT_ICON.get(sentiment,'')} {text}
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Raw values expander ──
with st.expander("📊 Raw feature values fed to model"):
    desc = {
        "return_1d":     "Previous day's % return",
        "MA_10":         "10-day moving average of close price",
        "MA_50":         "50-day moving average of close price",
        "volatility":    "Annualised 10-day rolling return volatility",
        "volume_change": "Day-over-day % change in trading volume",
        "RSI":           "14-period Relative Strength Index",
    }
    st.dataframe(pd.DataFrame([{
        "Feature":     k,
        "Value":       round(v, 6),
        "Description": desc.get(k, ""),
    } for k, v in features.items()]),
    hide_index=True, use_container_width=True)

last_date = df.index[-1].strftime("%d %b %Y")
st.caption(
    f"Data as of {last_date} · Source: Yahoo Finance · "
    f"Model: {engine_name} (pre-trained, not retrained daily) · Not financial advice."
)