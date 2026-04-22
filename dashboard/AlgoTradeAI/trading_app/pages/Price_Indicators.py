"""
📈 Price & Indicators
Candlestick chart, MACD, RSI, Bollinger Bands, Volume
All values driven by sidebar: ticker, show_bb, show_ma, show_vol
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, get_features, sidebar_controls,
                   OR, OR2, GOLD, CREAM, MUTE, GRN, RED)

st.set_page_config(page_title="Price & Indicators · AlgoTrade AI",
                   page_icon="📈", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

# ── All values from sidebar ───────────────────────────────────────────────────
ticker   = cfg["ticker"]
show_bb  = cfg["show_bb"]
show_ma  = cfg["show_ma"]
show_vol = cfg["show_vol"]

# get_features is cached per ticker — changes automatically when ticker changes
feat = get_features(ticker)
last = feat.iloc[-1]

page_header("Price &", "Indicators",
            f"{TICKERS[ticker][0]}  ·  OHLCV  ·  Technical Analysis")

# ── Dynamic subplot layout based on show_vol ──────────────────────────────────
rows    = 4 if show_vol else 3
heights = [0.52, 0.20, 0.16, 0.12][:rows]

fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    vertical_spacing=0.025, row_heights=heights)

# ── Row 1 — Candlestick ───────────────────────────────────────────────────────
fig.add_trace(go.Candlestick(
    x=feat.index, open=feat["Open"], high=feat["High"],
    low=feat["Low"],  close=feat["Close"], name="OHLC",
    increasing_line_color=GRN, decreasing_line_color=RED,
    increasing_fillcolor=GRN,  decreasing_fillcolor=RED,
), row=1, col=1)

# ── Bollinger Bands (toggled by sidebar checkbox) ─────────────────────────────
if show_bb:
    fig.add_trace(go.Scatter(x=feat.index, y=feat["BB_Up"], name="BB Upper",
        line=dict(color=OR, width=0.9, dash="dot"), opacity=0.55), row=1, col=1)
    fig.add_trace(go.Scatter(x=feat.index, y=feat["BB_Lo"], name="BB Lower",
        line=dict(color=OR, width=0.9, dash="dot"), opacity=0.55,
        fill="tonexty", fillcolor="rgba(212,98,26,0.04)"), row=1, col=1)

# ── Moving Averages (toggled by sidebar checkbox) ─────────────────────────────
if show_ma:
    for period, color in [(20, GOLD), (50, OR2), (200, CREAM)]:
        col_name = f"MA_{period}"
        if col_name in feat.columns:
            fig.add_trace(go.Scatter(x=feat.index, y=feat[col_name],
                name=f"MA {period}", line=dict(color=color, width=1.2),
                opacity=0.8), row=1, col=1)

# ── Row 2 — MACD ──────────────────────────────────────────────────────────────
bar_cols = [GRN if v >= 0 else RED for v in feat["MACD_H"]]
fig.add_trace(go.Bar(x=feat.index, y=feat["MACD_H"],
    marker_color=bar_cols, opacity=0.65, name="MACD Hist"), row=2, col=1)
fig.add_trace(go.Scatter(x=feat.index, y=feat["MACD"],
    line=dict(color=OR, width=1.2), name="MACD"), row=2, col=1)
fig.add_trace(go.Scatter(x=feat.index, y=feat["MACD_Sig"],
    line=dict(color=CREAM, width=1, dash="dot"), opacity=0.7, name="Signal"), row=2, col=1)

# ── Row 3 — RSI ───────────────────────────────────────────────────────────────
fig.add_trace(go.Scatter(x=feat.index, y=feat["RSI"],
    line=dict(color=GOLD, width=1.3), name="RSI"), row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_color=RED, line_width=0.7, row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color=GRN, line_width=0.7, row=3, col=1)
fig.add_hrect(y0=70, y1=100, fillcolor="rgba(217,95,75,0.04)",  line_width=0, row=3, col=1)
fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(82,183,136,0.04)", line_width=0, row=3, col=1)

# ── Row 4 — Volume (toggled by sidebar checkbox) ──────────────────────────────
if show_vol:
    vol_col = [GRN if feat["Close"].iloc[i] >= feat["Close"].iloc[i-1] else RED
               for i in range(len(feat))]
    fig.add_trace(go.Bar(x=feat.index, y=feat["Volume"],
        marker_color=vol_col, opacity=0.5, name="Volume"), row=4, col=1)

layout = base_layout("", h=680)
layout["xaxis_rangeslider_visible"] = False
fig.update_layout(**layout)
for i in range(1, rows + 1):
    fig.update_yaxes(gridcolor="rgba(180,80,20,0.07)",
                     linecolor="rgba(180,80,20,0.15)",
                     tickfont=dict(size=9), row=i, col=1)
fig.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ── Current Readings — all computed live from real data ───────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_label("Current Readings")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    rsi      = last["RSI"]
    rsi_kind = "red" if rsi > 70 else ("green" if rsi < 30 else "orange")
    rsi_col  = RED if rsi > 70 else (GRN if rsi < 30 else OR)
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.6rem">RSI (14)</div>
      <div class="stat-number" style="color:{rsi_col}">{rsi:.1f}</div>
      <div style="margin-top:0.5rem">
        {pill('Overbought' if rsi>70 else ('Oversold' if rsi<30 else 'Neutral'), rsi_kind)}
      </div>
    """, small=True)

with c2:
    macd_bull = last["MACD"] > last["MACD_Sig"]
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.6rem">MACD</div>
      <div class="stat-number" style="color:{OR}">{last['MACD']:.3f}</div>
      <div style="margin-top:0.5rem">
        {pill('Bullish' if macd_bull else 'Bearish', 'green' if macd_bull else 'red')}
      </div>
    """, small=True)

with c3:
    bb_range = last["BB_Up"] - last["BB_Lo"]
    bb_pct   = (last["Close"] - last["BB_Lo"]) / bb_range * 100 if bb_range > 0 else 50
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.6rem">Bollinger %B</div>
      <div class="stat-number" style="color:{GOLD}">{bb_pct:.1f}%</div>
      <div style="margin-top:0.5rem;font-size:0.72rem;color:{MUTE}">
        Range {last['BB_Lo']:,.1f} – {last['BB_Up']:,.1f}
      </div>
    """, small=True)

with c4:
    ann_vol = last["Vol_20"] * 100
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.6rem">Ann. Volatility</div>
      <div class="stat-number" style="color:{OR2}">{ann_vol:.1f}%</div>
      <div style="margin-top:0.5rem;font-size:0.72rem;color:{MUTE}">20-day rolling</div>
    """, small=True)

with c5:
    cross = last["MA_50"] > last["MA_200"]
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.6rem">MA Cross</div>
      <div class="stat-number" style="color:{GRN if cross else RED}">
        {'Golden' if cross else 'Death'}
      </div>
      <div style="margin-top:0.5rem">
        {pill('MA50 > MA200' if cross else 'MA50 < MA200', 'green' if cross else 'red')}
      </div>
    """, small=True)