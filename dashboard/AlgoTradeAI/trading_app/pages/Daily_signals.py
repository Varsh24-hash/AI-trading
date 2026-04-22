"""
📡 Daily Signal
Live BUY / HOLD / SELL prediction using latest Yahoo Finance data.
Uses existing trained models — no retraining required.
Updates automatically each day when new market data is available.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, sidebar_controls,
                   OR, OR2, GOLD, CREAM, MUTE, GRN, RED)

# ── Import live feed from data/ folder ───────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _ROOT)
from data.live_feed import get_prediction_row, explain_signal_text, FEATURE_COLS

_MODEL_DIR = os.path.join(_ROOT, "models")

MODEL_MAP = {
    "XGBoost":             "xgb_model.pkl",
    "Random Forest":       "random_forest_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
}

st.set_page_config(page_title="Daily Signal · AlgoTrade AI",
                   page_icon="📡", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

ticker = cfg["ticker"]
name   = TICKERS.get(ticker, (ticker,))[0]

page_header("Daily", "Signal",
            f"{name}  ·  {ticker}  ·  Live prediction from Yahoo Finance")

# ── Fetch live data ───────────────────────────────────────────────────────────
with st.spinner(f"Fetching latest market data for {ticker}..."):
    try:
        data = get_prediction_row(ticker)
    except Exception as e:
        st.error(f"Could not fetch live data: {e}")
        st.info("Check your internet connection or try a different ticker.")
        st.stop()

df      = data["df"]
latest  = data["latest"]
feats   = data["features"]
date    = data["date"]
prev_cl = data["prev_close"]
chg_pct = (latest["Close"] - prev_cl) / prev_cl * 100
X_live  = pd.DataFrame([feats])

# ── Run all three models ──────────────────────────────────────────────────────
model_results = {}
for mname, mfile in MODEL_MAP.items():
    mpath = os.path.join(_MODEL_DIR, mfile)
    if not os.path.exists(mpath):
        continue
    try:
        clf  = joblib.load(mpath)
        prob = float(clf.predict_proba(X_live)[0, 1])
        if prob > 0.60:
            sig = "BUY"
        elif prob < 0.40:
            sig = "SELL"
        else:
            sig = "HOLD"
        model_results[mname] = {"prob": prob, "signal": sig}
    except Exception:
        continue

if not model_results:
    st.error("No trained models found. Make sure .pkl files exist in models/ folder.")
    st.stop()

# ── Primary signal = XGBoost (or first available) ────────────────────────────
primary_model = "XGBoost" if "XGBoost" in model_results else list(model_results.keys())[0]
primary       = model_results[primary_model]
sig           = primary["signal"]
prob          = primary["prob"]

SIG_COLOR = {"BUY": GRN, "SELL": RED, "HOLD": GOLD}
SIG_PILL  = {"BUY": "green", "SELL": "red", "HOLD": "gold"}
sig_color = SIG_COLOR[sig]
sig_pill  = SIG_PILL[sig]

# ── Header strip ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:rgba(28,16,8,0.85);border:1px solid rgba(180,80,20,0.25);
            border-radius:16px;padding:2rem 2.4rem;margin-bottom:1.5rem;
            display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:1rem">
  <div>
    <div style="font-family:'Outfit',sans-serif;font-size:0.7rem;color:{MUTE};
                letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem">
      Signal for {date}
    </div>
    <div style="font-family:'Playfair Display',serif;font-size:3.8rem;
                font-weight:700;color:{sig_color};line-height:1;letter-spacing:-0.02em">
      {sig}
    </div>
    <div style="font-family:'Outfit',sans-serif;font-size:0.85rem;
                color:{MUTE};margin-top:0.4rem">
      {primary_model}  ·  {prob*100:.1f}% probability of upward move
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-family:'Playfair Display',serif;font-size:2.6rem;
                font-weight:700;color:{'#52B788' if chg_pct>=0 else '#D95F4B'};line-height:1">
      {latest['Close']:,.2f}
    </div>
    <div style="font-size:0.85rem;color:{'#52B788' if chg_pct>=0 else '#D95F4B'};
                margin-top:0.3rem">
      {'▲' if chg_pct>=0 else '▼'} {abs(chg_pct):.2f}% yesterday
    </div>
    <div style="font-size:0.72rem;color:{MUTE};margin-top:0.2rem">{ticker}  ·  {name}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── All 3 model signals ───────────────────────────────────────────────────────
section_label("All Model Signals")
mcols = st.columns(len(model_results))
for col, (mname, res) in zip(mcols, model_results.items()):
    sc = SIG_COLOR[res["signal"]]
    col.markdown(f"""
    <div class="glass-card" style="text-align:center;padding:1.2rem">
      <div style="font-size:0.65rem;color:{MUTE};letter-spacing:0.12em;
                  text-transform:uppercase;margin-bottom:0.5rem">{mname}</div>
      <div style="font-family:'Playfair Display',serif;font-size:2rem;
                  font-weight:700;color:{sc}">{res['signal']}</div>
      <div style="font-size:0.78rem;color:{MUTE};margin-top:0.4rem">
        {res['prob']*100:.1f}% prob up
      </div>
      <div style="margin-top:0.6rem">
        {pill(res['signal'], SIG_PILL[res['signal']])}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([2.2, 1], gap="large")

with left:
    # ── Price chart with indicators ───────────────────────────────────────────
    section_label("Price Chart — Last 90 Days with Signals")
    chart_df = df.tail(90).copy()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.60, 0.22, 0.18])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"],   close=chart_df["Close"],
        name="OHLC",
        increasing_line_color=GRN, decreasing_line_color=RED,
        increasing_fillcolor=GRN,  decreasing_fillcolor=RED,
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA_10"],
        line=dict(color=GOLD, width=1.2), name="MA10", opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA_50"],
        line=dict(color=OR2, width=1.2), name="MA50", opacity=0.8), row=1, col=1)

    # Bollinger Bands
    if "BB_Up" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Up"],
            line=dict(color=OR, width=0.8, dash="dot"),
            name="BB Upper", opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Lo"],
            line=dict(color=OR, width=0.8, dash="dot"),
            name="BB Lower", opacity=0.5,
            fill="tonexty", fillcolor="rgba(212,98,26,0.04)"), row=1, col=1)

    # Mark today's prediction on chart
    fig.add_vline(x=chart_df.index[-1], line_dash="dash",
                  line_color=sig_color, line_width=1.5,
                  annotation_text=f"→ {sig}",
                  annotation_font=dict(color=sig_color, size=11,
                                       family="JetBrains Mono"))

    # MACD subplot
    if "MACD" in chart_df.columns:
        bar_c = [GRN if v >= 0 else RED for v in chart_df["MACD_H"]]
        fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["MACD_H"],
            marker_color=bar_c, opacity=0.65, name="MACD Hist"), row=2, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD"],
            line=dict(color=OR, width=1.1), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD_Sig"],
            line=dict(color=CREAM, width=1, dash="dot"),
            opacity=0.7, name="Signal"), row=2, col=1)

    # RSI subplot
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["RSI"],
        line=dict(color=GOLD, width=1.3), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=RED,  line_width=0.7, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=GRN,  line_width=0.7, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(217,95,75,0.04)",  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(82,183,136,0.04)", line_width=0, row=3, col=1)

    layout = base_layout("", h=580)
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    fig.update_xaxes(rangeslider_visible=False)
    for i in range(1, 4):
        fig.update_yaxes(gridcolor="rgba(180,80,20,0.07)",
                         linecolor="rgba(180,80,20,0.15)",
                         tickfont=dict(size=9), row=i, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP feature importance ───────────────────────────────────────────────
    section_label("Explainable AI — SHAP Feature Contributions")

    shap_ok = False
    try:
        import shap
        clf_shap = joblib.load(os.path.join(_MODEL_DIR, MODEL_MAP[primary_model]))

        # TreeExplainer for XGBoost/RF, LinearExplainer for Logistic
        if primary_model == "Logistic Regression":
            explainer  = shap.LinearExplainer(clf_shap, X_live)
        else:
            explainer  = shap.TreeExplainer(clf_shap)

        shap_vals  = explainer.shap_values(X_live)

        # For binary classifiers shap_values may return list [class0, class1]
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]   # class 1 (upward move)
        else:
            sv = shap_vals[0]

        feat_names  = FEATURE_COLS
        shap_series = pd.Series(sv, index=feat_names).sort_values()
        colors_shap = [GRN if v > 0 else RED for v in shap_series.values]

        fig_shap = go.Figure(go.Bar(
            x=shap_series.values,
            y=shap_series.index,
            orientation="h",
            marker=dict(color=colors_shap),
            text=[f"{v:+.4f}" for v in shap_series.values],
            textposition="outside",
            textfont=dict(color=MUTE, size=9, family="JetBrains Mono")))

        fig_shap.update_layout(**base_layout("", h=300))
        fig_shap.update_xaxes(title_text="SHAP value (impact on prediction)",
                               title_font=dict(size=10))
        fig_shap.add_vline(x=0, line_color=MUTE, line_width=0.8)
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown(f"""
        <div style="font-size:0.78rem;color:{MUTE};line-height:1.8;
                    padding:0.8rem 1rem;background:rgba(28,16,8,0.5);
                    border-radius:8px;border-left:3px solid {OR}">
          <span style="color:{GRN}">Green bars</span> push the prediction toward BUY &nbsp;·&nbsp;
          <span style="color:{RED}">Red bars</span> push toward SELL &nbsp;·&nbsp;
          Bar length = how much that feature influenced today's signal.
        </div>
        """, unsafe_allow_html=True)
        shap_ok = True

    except ImportError:
        st.info("Install `shap` to see SHAP feature contributions: add `shap` to requirements.txt")
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    if not shap_ok:
        # Fallback: show raw feature values as a bar chart
        section_label("Feature Values (SHAP unavailable)")
        feat_df = pd.DataFrame({
            "Feature": list(feats.keys()),
            "Value":   list(feats.values()),
        })
        fig_f = go.Figure(go.Bar(
            x=feat_df["Value"], y=feat_df["Feature"],
            orientation="h", marker_color=OR,
            text=[f"{v:.4f}" for v in feat_df["Value"]],
            textposition="outside",
            textfont=dict(color=MUTE, size=9, family="JetBrains Mono")))
        fig_f.update_layout(**base_layout("", h=260))
        st.plotly_chart(fig_f, use_container_width=True)

with right:
    # ── Current indicator readings ────────────────────────────────────────────
    section_label("Current Indicator Readings")

    rsi      = latest["RSI"]
    rsi_kind = "red" if rsi > 70 else ("green" if rsi < 30 else "orange")
    macd_bull= latest["MACD"] > latest["MACD_Sig"]
    ma_bull  = latest["MA_10"] > latest["MA_50"]
    bb_range = latest.get("BB_Up", np.nan) - latest.get("BB_Lo", np.nan)
    bb_pct   = ((latest["Close"] - latest.get("BB_Lo", 0)) /
                bb_range * 100) if bb_range > 0 else 50

    glass_card(f"""
      {kv('Close',       f"{latest['Close']:,.4f}",         CREAM)}
      {kv('RSI (14)',    f"{rsi:.1f} {pill('OB' if rsi>70 else ('OS' if rsi<30 else 'Neutral'), rsi_kind)}")}
      {kv('MACD',        f"{latest['MACD']:.4f} {pill('Bullish' if macd_bull else 'Bearish','green' if macd_bull else 'red')}")}
      {kv('MA10',        f"{latest['MA_10']:.4f}")}
      {kv('MA50',        f"{latest['MA_50']:.4f}")}
      {kv('MA Cross',    pill('Golden' if ma_bull else 'Death','green' if ma_bull else 'red'))}
      {kv('BB %B',       f"{bb_pct:.1f}%")}
      {kv('Volatility',  f"{latest['volatility']*100:.2f}%")}
      {kv('Return 1d',   f"{latest['return_1d']*100:+.3f}%",
           GRN if latest['return_1d']>0 else RED)}
      {kv('Vol Change',  f"{latest['volume_change']*100:+.1f}%",
           GRN if latest['volume_change']>0 else RED)}
    """)

    # ── Confidence gauge ──────────────────────────────────────────────────────
    section_label("Model Confidence")
    conf_pct = prob * 100
    bar_w    = abs(conf_pct - 50) * 2   # 0–100 scale from centre
    glass_card(f"""
      <div style="text-align:center">
        <div style="font-family:'Playfair Display',serif;font-size:2.2rem;
                    font-weight:700;color:{sig_color}">{conf_pct:.1f}%</div>
        <div style="font-size:0.72rem;color:{MUTE};margin:0.3rem 0 0.8rem">
          probability of upward move
        </div>
        <div style="background:rgba(58,36,24,0.5);border-radius:6px;
                    height:10px;overflow:hidden;position:relative">
          <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;
                      background:{MUTE}"></div>
          <div style="height:100%;width:{bar_w}%;
                      background:{sig_color};border-radius:6px;
                      {'margin-left:50%' if conf_pct>50 else f'margin-left:{conf_pct}%'}">
          </div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.6rem;color:{MUTE};margin-top:0.3rem">
          <span>SELL</span><span>NEUTRAL</span><span>BUY</span>
        </div>
      </div>
    """)

    # ── Explainable AI — text reasons ─────────────────────────────────────────
    section_label("AI Decision Reasoning")
    reasons = explain_signal_text(latest, sig, prob)

    for reason_text, sentiment in reasons:
        sent_color = GRN if sentiment == "bullish" else \
                     (RED if sentiment == "bearish" else MUTE)
        sent_icon  = "▲" if sentiment == "bullish" else \
                     ("▼" if sentiment == "bearish" else "●")
        st.markdown(f"""
        <div style="display:flex;gap:0.6rem;align-items:flex-start;
                    padding:0.5rem 0.8rem;margin-bottom:0.4rem;
                    background:rgba(28,16,8,0.6);border-radius:8px;
                    border-left:3px solid {sent_color}">
          <span style="color:{sent_color};font-size:0.75rem;
                       margin-top:0.1rem;flex-shrink:0">{sent_icon}</span>
          <span style="font-size:0.75rem;color:{CREAM};
                       font-family:'Outfit',sans-serif;line-height:1.6">
            {reason_text}
          </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature values used for prediction ────────────────────────────────────
    section_label("Features Fed to Model")
    feat_labels = {
        "return_1d":     "1-day return",
        "MA_10":         "MA (10-day)",
        "MA_50":         "MA (50-day)",
        "volatility":    "Volatility",
        "volume_change": "Volume change",
        "RSI":           "RSI (14)",
    }
    rows_html = ""
    for col, label in feat_labels.items():
        val = feats[col]
        if col in ("return_1d", "volume_change"):
            display = f"{val*100:+.3f}%"
            color   = GRN if val > 0 else RED
        elif col == "volatility":
            display = f"{val*100:.2f}%"
            color   = CREAM
        elif col == "RSI":
            display = f"{val:.2f}"
            color   = RED if val > 70 else (GRN if val < 30 else CREAM)
        else:
            display = f"{val:.4f}"
            color   = CREAM
        rows_html += kv(label, display, color)
    glass_card(rows_html, small=True)

    # ── Data freshness ────────────────────────────────────────────────────────
    section_label("Data Info")
    glass_card(f"""
      {kv('Last data date', df.index[-1].strftime('%Y-%m-%d'), OR2)}
      {kv('Data points',    f"{len(df):,}",                   CREAM)}
      {kv('Source',         'Yahoo Finance (live)',           GRN)}
      {kv('Model used',     primary_model,                   GOLD)}
      {kv('Retrain needed', 'No — using saved .pkl',         CREAM)}
    """, small=True)