"""
🏠 Overview
Portfolio snapshot — respects engine selection fully.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, get_features, run_engine,
                   perf_metrics, sidebar_controls,
                   OR, OR2, GOLD, CREAM, MUTE, GRN, RED, CARD, BORD)

st.set_page_config(page_title="Overview · AlgoTrade AI",
                   page_icon="📊", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

ticker = cfg["ticker"]
rf     = cfg["rf"]
engine = cfg["engine"]
name   = TICKERS[ticker][0]

result = run_engine(cfg)
active_df      = result["active_df"]
active_metrics = result["active_metrics"]
ml_df          = result["ml_df"]
rl_df          = result["rl_df"]
ml_metrics     = result["ml_metrics"]
rl_metrics     = result["rl_metrics"]
bh_metrics     = result["bh_metrics"]
model_label    = result["model_label"]

# ── Live quote from real data ─────────────────────────────────────────────────
feat = get_features(ticker)
last = feat.iloc[-1]
prev = feat.iloc[-2]
chg  = (last["Close"] - prev["Close"]) / prev["Close"] * 100

page_header("Portfolio", "Overview",
            f"{name}  ·  {ticker}  ·  Capital ${cfg['capital']:,.0f}  ·  Engine: {engine}")

# ── Top KPIs — from active engine ────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
for col, (lbl, val) in zip([k1,k2,k3,k4,k5,k6], active_metrics.items()):
    col.metric(lbl, val)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([2.4, 1], gap="large")

with left:
    section_label("Equity Curve Comparison")
    fig = go.Figure()

    if engine in ("ML Prediction", "Both") and ml_df is not None:
        fig.add_trace(go.Scatter(x=ml_df.index, y=ml_df["Portfolio"],
            line=dict(color=OR, width=2.2), name=f"ML ({model_label})"))

    if engine in ("RL Agent", "Both") and rl_df is not None:
        fig.add_trace(go.Scatter(x=rl_df.index, y=rl_df["Portfolio"],
            line=dict(color=GRN, width=2.2), name="RL Agent (PPO)"))

    # Buy & Hold from whichever df is available
    ref = ml_df if ml_df is not None else rl_df
    fig.add_trace(go.Scatter(x=ref.index, y=ref["BuyHold"],
        line=dict(color=CREAM, width=1.2, dash="dot"), opacity=0.5, name="Buy & Hold"))

    fig.update_layout(**base_layout("", h=360))
    st.plotly_chart(fig, use_container_width=True)

    section_label("Drawdown")
    fig2 = go.Figure()
    if engine in ("ML Prediction", "Both") and ml_df is not None:
        dd_ml = (ml_df["Portfolio"] - ml_df["Portfolio"].cummax()) / ml_df["Portfolio"].cummax() * 100
        fig2.add_trace(go.Scatter(x=ml_df.index, y=dd_ml, fill="tozeroy",
            fillcolor="rgba(212,98,26,0.08)",
            line=dict(color=OR, width=1.3), name="ML Drawdown"))
    if engine in ("RL Agent", "Both") and rl_df is not None:
        dd_rl = (rl_df["Portfolio"] - rl_df["Portfolio"].cummax()) / rl_df["Portfolio"].cummax() * 100
        fig2.add_trace(go.Scatter(x=rl_df.index, y=dd_rl, fill="tozeroy",
            fillcolor="rgba(82,183,136,0.07)",
            line=dict(color=GRN, width=1.3), name="RL Drawdown"))
    fig2.update_layout(**base_layout("", h=220))
    st.plotly_chart(fig2, use_container_width=True)

with right:
    section_label("Live Quote")
    chg_col = GRN if chg >= 0 else RED
    glass_card(f"""
      <div style="font-family:'Playfair Display',serif;font-size:2.4rem;
                  font-weight:700;color:{chg_col};line-height:1">
        {last['Close']:,.2f}
      </div>
      <div style="font-size:0.78rem;color:{chg_col};margin:0.3rem 0 1rem">
        {'▲' if chg>=0 else '▼'} {abs(chg):.2f}%  today
      </div>
      {kv('Open',   f"{last['Open']:,.2f}")}
      {kv('High',   f"{last['High']:,.2f}", GRN)}
      {kv('Low',    f"{last['Low']:,.2f}",  RED)}
      {kv('Volume', f"{int(last['Volume']):,}")}
    """)

    # ── Strategy comparison table — only show active engines ──────────────────
    section_label("Strategy Metrics")
    labels = ["Total Return","Ann. Return","Sharpe","Volatility","Max Drawdown","Win Rate"]

    if engine == "Both":
        cols_header = "1.2fr 1fr 1fr 1fr"
        header_html = "<span>Metric</span><span style='text-align:right'>ML</span><span style='text-align:right'>RL</span><span style='text-align:right'>B&H</span>"
        rows = ""
        for lbl, m_val, r_val, b_val in zip(
            labels, ml_metrics.values(), rl_metrics.values(), bh_metrics.values()
        ):
            rows += f"""<div style="display:grid;grid-template-columns:{cols_header};
                        gap:0;padding:0.42rem 0;border-bottom:1px solid rgba(180,80,20,0.09);
                        font-size:0.75rem;align-items:center">
              <span style="color:{MUTE};font-family:'Outfit',sans-serif">{lbl}</span>
              <span style="color:{OR2};font-family:'Playfair Display',serif;font-weight:600;text-align:right">{m_val}</span>
              <span style="color:{GRN};font-family:'Playfair Display',serif;font-weight:600;text-align:right">{r_val}</span>
              <span style="color:{CREAM};font-family:'Playfair Display',serif;font-weight:600;text-align:right;opacity:0.55">{b_val}</span>
            </div>"""
    elif engine == "ML Prediction":
        cols_header = "1.4fr 1fr 1fr"
        header_html = "<span>Metric</span><span style='text-align:right'>ML</span><span style='text-align:right'>B&H</span>"
        rows = ""
        for lbl, m_val, b_val in zip(labels, ml_metrics.values(), bh_metrics.values()):
            rows += f"""<div style="display:grid;grid-template-columns:{cols_header};
                        gap:0;padding:0.42rem 0;border-bottom:1px solid rgba(180,80,20,0.09);
                        font-size:0.75rem;align-items:center">
              <span style="color:{MUTE};font-family:'Outfit',sans-serif">{lbl}</span>
              <span style="color:{OR2};font-family:'Playfair Display',serif;font-weight:600;text-align:right">{m_val}</span>
              <span style="color:{CREAM};font-family:'Playfair Display',serif;font-weight:600;text-align:right;opacity:0.55">{b_val}</span>
            </div>"""
    else:  # RL Agent
        cols_header = "1.4fr 1fr 1fr"
        header_html = "<span>Metric</span><span style='text-align:right'>RL</span><span style='text-align:right'>B&H</span>"
        rows = ""
        for lbl, r_val, b_val in zip(labels, rl_metrics.values(), bh_metrics.values()):
            rows += f"""<div style="display:grid;grid-template-columns:{cols_header};
                        gap:0;padding:0.42rem 0;border-bottom:1px solid rgba(180,80,20,0.09);
                        font-size:0.75rem;align-items:center">
              <span style="color:{MUTE};font-family:'Outfit',sans-serif">{lbl}</span>
              <span style="color:{GRN};font-family:'Playfair Display',serif;font-weight:600;text-align:right">{r_val}</span>
              <span style="color:{CREAM};font-family:'Playfair Display',serif;font-weight:600;text-align:right;opacity:0.55">{b_val}</span>
            </div>"""

    glass_card(f"""
      <div style="display:grid;grid-template-columns:{cols_header};gap:0;
                  padding:0 0 0.5rem;border-bottom:1px solid rgba(180,80,20,0.18);
                  font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;
                  color:#5A4030;font-family:'Outfit',sans-serif;font-weight:600">
        {header_html}
      </div>
      {rows}
    """)

    section_label("Signal Indicators")
    rsi      = last["RSI"]
    rsi_kind = "red" if rsi > 70 else ("green" if rsi < 30 else "orange")
    rsi_lbl  = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
    macd_kind= "green" if last["MACD"] > last["MACD_Sig"] else "red"
    macd_lbl = "Bullish"   if last["MACD"] > last["MACD_Sig"] else "Bearish"
    glass_card(f"""
      {kv('RSI (14)', f"{rsi:.1f} &nbsp; {pill(rsi_lbl, rsi_kind)}")}
      {kv('MACD',     f"{last['MACD']:.3f} &nbsp; {pill(macd_lbl, macd_kind)}")}
      {kv('BB Width', f"{(last['BB_Up']-last['BB_Lo'])/last['BB_Mid']*100:.1f}%")}
      {kv('Ann. Vol', f"{last['Vol_20']*100:.1f}%")}
      {kv('MA Cross', pill('Golden' if last['MA_50']>last['MA_200'] else 'Death',
                           'green'  if last['MA_50']>last['MA_200'] else 'red'))}
    """, small=True)