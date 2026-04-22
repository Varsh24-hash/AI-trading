"""
💼 Portfolio Optimizer
All values driven by sidebar: portfolio basket (multi), risk-free rate (rf)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, get_price_data, efficient_frontier,
                   sidebar_controls, OR, OR2, GOLD, CREAM, MUTE, GRN, RED)

st.set_page_config(page_title="Portfolio Optimizer · AlgoTrade AI",
                   page_icon="💼", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

multi = cfg["multi"]
rf    = cfg["rf"]

page_header("Portfolio", "Optimiser",
            f"Modern Portfolio Theory  ·  {len(multi)} assets  ·  rf={rf*100:.1f}%  ·  1,000 simulations")

if len(multi) < 2:
    st.warning("Please select at least 2 stocks in the Portfolio basket (sidebar).")
    st.stop()

try:
    frames  = {t: get_price_data(t)["Close"].pct_change().dropna() for t in multi}
    aligned = pd.DataFrame(frames).dropna()
except Exception as e:
    st.error(f"Error loading price data: {e}")
    st.stop()

if aligned.empty or len(aligned) < 30:
    st.error("Not enough aligned data for selected stocks. Try a different combination.")
    st.stop()

opt_df = efficient_frontier(tuple(multi), rf=rf)

if opt_df.empty:
    st.error("Could not compute efficient frontier. Try selecting different stocks.")
    st.stop()

sharpe_idx = opt_df["Sharpe"].idxmax()
vol_idx    = opt_df["Volatility"].idxmin()
best       = opt_df.loc[sharpe_idx]
min_vol    = opt_df.loc[vol_idx]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Max Sharpe",         f"{best['Sharpe']:.3f}")
c2.metric("Optimal Return",     f"{best['Return']:.2f}%")
c3.metric("Optimal Volatility", f"{best['Volatility']:.2f}%")
c4.metric("Min Vol Portfolio",  f"{min_vol['Volatility']:.2f}%")
c5.metric("Assets",             str(len(multi)))

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([2, 1], gap="large")

with left:
    section_label("Efficient Frontier")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=opt_df["Volatility"], y=opt_df["Return"], mode="markers",
        marker=dict(
            color=opt_df["Sharpe"],
            colorscale=[[0, "#1A0A02"], [0.4, GOLD], [0.8, OR], [1.0, OR2]],
            size=5, opacity=0.75, showscale=True,
            colorbar=dict(
                title=dict(text="Sharpe", font=dict(color=MUTE, size=10)),
                tickfont=dict(color=MUTE, size=9, family="JetBrains Mono"),
                thickness=10, len=0.8)),
        name="Portfolios",
        hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[best["Volatility"]], y=[best["Return"]], mode="markers+text",
        marker=dict(size=18, color=OR, symbol="star",
                    line=dict(color=CREAM, width=1.5)),
        text=["Max Sharpe"], textposition="top right",
        textfont=dict(color=CREAM, size=10, family="Outfit"),
        name="Optimal"))
    fig.add_trace(go.Scatter(
        x=[min_vol["Volatility"]], y=[min_vol["Return"]], mode="markers+text",
        marker=dict(size=14, color=GRN, symbol="diamond",
                    line=dict(color=CREAM, width=1)),
        text=["Min Vol"], textposition="top right",
        textfont=dict(color=CREAM, size=10, family="Outfit"),
        name="Min Volatility"))
    fig.update_layout(**base_layout("", h=420))
    fig.update_xaxes(title_text="Portfolio Volatility (%)", title_font=dict(size=10))
    fig.update_yaxes(title_text="Expected Return (%)",      title_font=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)

    section_label("Correlation Matrix")
    corr = aligned.corr()
    fig2 = go.Figure(go.Heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.columns),
        colorscale=[[0, "#0A0704"], [0.5, "#5A2808"], [1, OR]],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, family="JetBrains Mono", color=CREAM),
        showscale=True,
        colorbar=dict(
            thickness=10,
            tickfont=dict(color=MUTE, size=9, family="JetBrains Mono"))))
    fig2.update_layout(**base_layout("", h=320))
    st.plotly_chart(fig2, use_container_width=True)

with right:
    section_label("Optimal Weights")
    opt_w      = dict(zip(multi, best["Weights"]))
    colors_pie = [OR, GOLD, GRN, OR2, RED, CREAM, "#7A6050", "#52B788",
                  "#B05535", "#3B8BD4"]
    fig3 = go.Figure(go.Pie(
        labels=list(opt_w.keys()),
        values=[v * 100 for v in opt_w.values()],
        marker=dict(colors=colors_pie[:len(multi)]),
        textfont=dict(size=11, family="Outfit", color=CREAM),
        hole=0.50,
        textinfo="label+percent",
        insidetextorientation="radial"))
    fig3.update_layout(**base_layout("", h=300))
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    section_label("Optimal Portfolio Detail")
    glass_card(f"""
      {kv('Sharpe Ratio',    f"{best['Sharpe']:.4f}",         OR2)}
      {kv('Expected Return', f"{best['Return']:.3f}%",        GRN)}
      {kv('Portfolio Vol',   f"{best['Volatility']:.3f}%")}
      {kv('Risk-free rate',  f"{rf*100:.2f}%",                GOLD)}
      {kv('Excess return',   f"{best['Return']-rf*100:.3f}%", OR)}
    """)

    section_label("Individual Stock Returns")
    ind_rets = aligned.mean() * 252 * 100
    ind_vols = aligned.std() * np.sqrt(252) * 100
    for t in multi:
        if t not in ind_rets.index:
            continue
        ret_val = float(ind_rets[t])
        vol_val = float(ind_vols[t])
        weight  = opt_w.get(t, 0) * 100
        glass_card(f"""
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-family:'Playfair Display',serif;font-size:0.9rem;
                         font-weight:600;color:{CREAM}">{t}</span>
            <span style="font-size:0.65rem;color:{MUTE}">
              {TICKERS.get(t, (t,))[0]}
            </span>
          </div>
          <div style="display:flex;gap:0.6rem;margin-top:0.4rem;flex-wrap:wrap">
            {pill(f"Ret {ret_val:+.1f}%", 'green' if ret_val > 0 else 'red')}
            {pill(f"Vol {vol_val:.1f}%",  'orange')}
            {pill(f"Wt {weight:.1f}%",    'gold')}
          </div>
        """, small=True)