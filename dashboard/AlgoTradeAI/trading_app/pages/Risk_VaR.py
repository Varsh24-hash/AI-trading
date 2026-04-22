"""
⚠️ Risk & VaR
All values driven by sidebar: ticker, var_c (confidence level)
No scipy dependency — z-score computed manually.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, get_features, sidebar_controls,
                   OR, OR2, GOLD, CREAM, MUTE, GRN, RED)

st.set_page_config(page_title="Risk & VaR · AlgoTrade AI",
                   page_icon="⚠️", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

ticker = cfg["ticker"]
var_c  = cfg["var_c"]

feat = get_features(ticker)
rets = feat["Return_1d"].dropna()

# ── VaR calculations — all use var_c from sidebar ─────────────────────────────
hist_var = np.percentile(rets, (1 - var_c) * 100)
cvar_val = rets[rets <= hist_var].mean()
ann_vol  = rets.std() * np.sqrt(252) * 100

# z-score lookup table — no scipy needed
_Z = {0.90: 1.282, 0.91: 1.341, 0.92: 1.405, 0.93: 1.476,
      0.94: 1.555, 0.95: 1.645, 0.96: 1.751, 0.97: 1.881,
      0.98: 2.054, 0.99: 2.326}
z = _Z.get(round(var_c, 2), 1.645)
param_var = rets.mean() - z * rets.std()

page_header("Risk", "Management",
            f"{TICKERS[ticker][0]}  ·  {var_c*100:.0f}% Confidence  ·  Volatility Forecasting")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Historical VaR",  f"{hist_var*100:.3f}%",
          f"Daily {var_c*100:.0f}% loss limit")
c2.metric("Parametric VaR",  f"{param_var*100:.3f}%",
          "Normal distribution")
c3.metric("CVaR / ES",       f"{cvar_val*100:.3f}%",
          "Expected shortfall")
c4.metric("Ann. Volatility", f"{ann_vol:.2f}%",
          "Annualised")

st.markdown("<br>", unsafe_allow_html=True)

top_l, top_r = st.columns(2, gap="large")

with top_l:
    section_label("Daily Return Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rets * 100, nbinsx=70,
        marker_color=OR, opacity=0.72, name="Daily returns"))
    fig.add_vline(x=hist_var * 100, line_dash="dash", line_color=RED,
        annotation_text=f"VaR ({var_c*100:.0f}%) {hist_var*100:.2f}%",
        annotation_font=dict(color=RED, size=10, family="JetBrains Mono"),
        annotation_position="top left")
    fig.add_vline(x=rets.mean() * 100, line_dash="dot", line_color=GRN,
        annotation_text=f"Mean {rets.mean()*100:.3f}%",
        annotation_font=dict(color=GRN, size=10, family="JetBrains Mono"))
    fig.update_layout(**base_layout("", h=340))
    st.plotly_chart(fig, use_container_width=True)

with top_r:
    section_label(f"Rolling 60-Day VaR ({var_c*100:.0f}% confidence)")
    roll_var = rets.rolling(60).apply(
        lambda x: np.percentile(x, (1 - var_c) * 100)
    ) * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=roll_var.index, y=roll_var,
        fill="tozeroy", fillcolor="rgba(217,95,75,0.09)",
        line=dict(color=RED, width=1.5),
        name=f"Rolling {var_c*100:.0f}% VaR"))
    fig2.add_hline(y=hist_var * 100, line_dash="dot",
                   line_color=OR, line_width=0.8,
                   annotation_text=f"Overall {hist_var*100:.2f}%",
                   annotation_font=dict(color=OR, size=9, family="JetBrains Mono"))
    fig2.update_layout(**base_layout("", h=340))
    st.plotly_chart(fig2, use_container_width=True)

section_label("Volatility — Realised vs Predicted")
vol_real = feat["Vol_20"] * 100
vol_pred = rets.ewm(span=20).std() * np.sqrt(252) * 100
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=feat.index, y=vol_real,
    line=dict(color=CREAM, width=1.3), opacity=0.8, name="Realised vol (20d)"))
fig3.add_trace(go.Scatter(x=vol_pred.index, y=vol_pred,
    line=dict(color=OR2, width=1.5, dash="dot"), name="EWMA predicted vol"))
fig3.update_layout(**base_layout("", h=260))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
section_label(f"VaR Method Comparison  ·  {var_c*100:.0f}% confidence")
v1, v2, v3 = st.columns(3, gap="large")

np.random.seed(7)
mc_r   = np.random.normal(rets.mean(), rets.std(), 50_000)
mc_var = np.percentile(mc_r, (1 - var_c) * 100)

with v1:
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.8rem">Historical VaR</div>
      <div class="stat-number" style="color:{RED}">{hist_var*100:.3f}%</div>
      <div style="margin-top:0.8rem;font-size:0.75rem;color:{MUTE};line-height:1.8">
        Sorts past daily returns<br>
        Takes the {(1-var_c)*100:.0f}th percentile<br>
        No distribution assumption<br>
        Best for fat-tailed returns
      </div>
    """)

with v2:
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.8rem">Parametric VaR</div>
      <div class="stat-number" style="color:{OR}">{param_var*100:.3f}%</div>
      <div style="margin-top:0.8rem;font-size:0.75rem;color:{MUTE};line-height:1.8">
        Assumes normal distribution<br>
        Uses μ − {z:.3f}σ (z={var_c*100:.0f}%)<br>
        Fast and analytical<br>
        Underestimates tail risk
      </div>
    """)

with v3:
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.8rem">Monte Carlo VaR</div>
      <div class="stat-number" style="color:{GOLD}">{mc_var*100:.3f}%</div>
      <div style="margin-top:0.8rem;font-size:0.75rem;color:{MUTE};line-height:1.8">
        50,000 simulated returns<br>
        {(1-var_c)*100:.0f}th percentile of sim<br>
        Captures non-linearity<br>
        Used by major banks
      </div>
    """)