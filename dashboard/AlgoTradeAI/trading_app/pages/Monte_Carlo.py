"""
🎲 Monte Carlo Simulation
All values driven by sidebar: ticker, mc_n (paths), var_c (confidence)
Changing any sidebar value immediately recalculates everything.
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

st.set_page_config(page_title="Monte Carlo · AlgoTrade AI",
                   page_icon="🎲", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

# ── All values from sidebar ───────────────────────────────────────────────────
ticker = cfg["ticker"]
mc_n   = cfg["mc_n"]    # number of simulation paths (500–5000)
var_c  = cfg["var_c"]   # VaR confidence level (0.90–0.99)

# ── Real mu/sigma from actual price data ──────────────────────────────────────
feat   = get_features(ticker)
rets   = feat["Return_1d"].dropna()
last_p = feat["Close"].iloc[-1]
mu     = rets.mean()
sigma  = rets.std()

# ── Simulation — cached per (ticker, mu, sigma, mc_n) ─────────────────────────
# Passing mu and sigma as args means cache invalidates when ticker changes.
# mc_n is also a cache key so changing paths slider recalculates immediately.
@st.cache_data(show_spinner=False)
def run_sim(last_price: float, mu: float, sigma: float,
            n_sims: int, days: int = 252, seed: int = 7):
    np.random.seed(seed)
    paths       = np.zeros((n_sims, days))
    paths[:, 0] = last_price
    for t in range(1, days):
        shocks      = np.random.normal(mu, sigma, n_sims)
        paths[:, t] = paths[:, t-1] * np.exp(shocks)
    return paths

paths = run_sim(last_p, mu, sigma, mc_n)

# ── All percentiles computed from var_c — updates when slider changes ─────────
p5     = np.percentile(paths[:, -1], 5)
p25    = np.percentile(paths[:, -1], 25)
p50    = np.percentile(paths[:, -1], 50)
p75    = np.percentile(paths[:, -1], 75)
p95    = np.percentile(paths[:, -1], 95)

# VaR uses var_c from sidebar — changing the slider updates this immediately
var_pct = (1 - var_c) * 100
var_mc  = np.percentile((paths[:, -1] / last_p - 1) * 100, var_pct)

page_header("Monte Carlo", "Simulation",
            f"{mc_n:,} paths  ·  252-day horizon  ·  {var_c*100:.0f}% confidence  ·  GBM")

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price",   f"{last_p:,.2f}")
c2.metric("5th Percentile",  f"{p5:,.2f}",  f"{(p5/last_p-1)*100:+.1f}%")
c3.metric("Median Path",     f"{p50:,.2f}", f"{(p50/last_p-1)*100:+.1f}%")
c4.metric("95th Percentile", f"{p95:,.2f}", f"{(p95/last_p-1)*100:+.1f}%")
c5.metric(f"MC VaR ({var_c*100:.0f}%)", f"{var_mc:.2f}%", "252-day horizon")

st.markdown("<br>", unsafe_allow_html=True)

# ── Fan chart ─────────────────────────────────────────────────────────────────
section_label("Simulated Price Paths — Fan Chart")
days_ax = np.arange(paths.shape[1])
fig     = go.Figure()

# Sample paths — capped at 300 for performance
np.random.seed(42)
idx = np.random.choice(mc_n, min(300, mc_n), replace=False)
for i in idx:
    fig.add_trace(go.Scatter(x=days_ax, y=paths[i],
        mode="lines", line=dict(color=OR, width=0.25), opacity=0.10,
        showlegend=False, hoverinfo="skip"))

for pct, col, name, w in [
    (5,  RED,   "5th pct",  2.0),
    (25, OR,    "25th pct", 1.4),
    (50, CREAM, "Median",   2.2),
    (75, OR,    "75th pct", 1.4),
    (95, GRN,   "95th pct", 2.0),
]:
    band = np.percentile(paths, pct, axis=0)
    fig.add_trace(go.Scatter(x=days_ax, y=band,
        line=dict(color=col, width=w), name=name))

fig.add_hline(y=last_p, line_dash="dash", line_color=GOLD, line_width=1.0,
              annotation_text="Entry price",
              annotation_font=dict(color=GOLD, size=10, family="JetBrains Mono"))
fig.update_layout(**base_layout("", h=420))
fig.update_xaxes(title_text="Trading days", title_font=dict(size=10))
fig.update_yaxes(title_text="Price",        title_font=dict(size=10))
st.plotly_chart(fig, use_container_width=True)

# ── Distribution + stats ──────────────────────────────────────────────────────
left, right = st.columns([1.6, 1], gap="large")

with left:
    section_label(f"Distribution of Final Values (Day 252)")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=paths[:, -1], nbinsx=100,
        marker_color=OR, opacity=0.70, name="Final price"))
    for val, col, lbl in [(p5, RED, "P5"), (p50, CREAM, "P50"), (p95, GRN, "P95")]:
        fig2.add_vline(x=val, line_dash="dash", line_color=col, line_width=1.2,
            annotation_text=lbl,
            annotation_font=dict(color=col, size=10, family="JetBrains Mono"))
    # Also mark the VaR threshold
    var_price = last_p * (1 + var_mc / 100)
    fig2.add_vline(x=var_price, line_dash="dot", line_color=GOLD, line_width=1.0,
        annotation_text=f"VaR ({var_c*100:.0f}%)",
        annotation_font=dict(color=GOLD, size=9, family="JetBrains Mono"))
    fig2.update_layout(**base_layout("", h=320))
    st.plotly_chart(fig2, use_container_width=True)

with right:
    section_label("Scenario Summary")
    scenarios = [
        ("Bear (P5)",  p5,  (p5/last_p-1)*100,  "red"),
        ("Low (P25)",  p25, (p25/last_p-1)*100, "orange"),
        ("Base (P50)", p50, (p50/last_p-1)*100, "gold"),
        ("High (P75)", p75, (p75/last_p-1)*100, "green"),
        ("Bull (P95)", p95, (p95/last_p-1)*100, "green"),
    ]
    rows_html = ""
    for name, price, ret, kind in scenarios:
        rows_html += f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 0.8fr;gap:0;
                    padding:0.5rem 0;border-bottom:1px solid rgba(180,80,20,0.1);
                    align-items:center">
          <span style="font-family:'Outfit',sans-serif;font-size:0.78rem;
                       color:{MUTE}">{name}</span>
          <span style="font-family:'Playfair Display',serif;font-size:0.9rem;
                       color:{CREAM};font-weight:600">{price:,.2f}</span>
          <span>{pill(f"{'+'if ret>=0 else ''}{ret:.1f}%", kind)}</span>
        </div>"""
    glass_card(rows_html)

    section_label("Distribution Stats")
    final = paths[:, -1]
    glass_card(f"""
      {kv('Mean final price', f"{final.mean():,.2f}")}
      {kv('Std dev',          f"{final.std():,.2f}")}
      {kv('Skewness',         f"{pd.Series(final).skew():.3f}")}
      {kv('Kurtosis',         f"{pd.Series(final).kurtosis():.3f}")}
      {kv('Prob > entry',     f"{(final > last_p).mean()*100:.1f}%",      GRN)}
      {kv('Prob > +20%',      f"{(final > last_p*1.2).mean()*100:.1f}%",  GRN)}
      {kv('Prob < -20%',      f"{(final < last_p*0.8).mean()*100:.1f}%",  RED)}
      {kv(f'MC VaR ({var_c*100:.0f}%)', f"{var_mc:.3f}%",                 RED)}
    """)

    section_label("Simulation Parameters")
    glass_card(f"""
      {kv('Ticker',          ticker,                          OR2)}
      {kv('Paths',           f"{mc_n:,}",                    OR2)}
      {kv('Horizon',         "252 trading days",             CREAM)}
      {kv('Confidence',      f"{var_c*100:.0f}%",            GOLD)}
      {kv('Daily μ',         f"{mu*100:.4f}%",               CREAM)}
      {kv('Daily σ',         f"{sigma*100:.4f}%",            CREAM)}
      {kv('Ann. return est.',f"{mu*252*100:.2f}%",           GOLD)}
      {kv('Ann. vol est.',   f"{sigma*np.sqrt(252)*100:.2f}%", GOLD)}
      {kv('Data points',     f"{len(rets):,}",               CREAM)}
    """)