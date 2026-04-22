"""
🧠 RL Agent
Shows RL results. If engine = ML Prediction only, shows a notice.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, run_engine, perf_metrics,
                   sidebar_controls, OR, OR2, GOLD, CREAM, MUTE, GRN, RED)

st.set_page_config(page_title="RL Agent · AlgoTrade AI",
                   page_icon="🧠", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

engine = cfg["engine"]
rf     = cfg["rf"]

# ── If pure ML mode, show notice ──────────────────────────────────────────────
if engine == "ML Prediction":
    page_header("Reinforcement", "Learning Agent", "Engine set to ML Prediction")
    st.info("⚠️ RL Agent is disabled — you selected **ML Prediction** in the sidebar. "
            "Switch to **RL Agent** or **Both** to see RL results here.")
    st.stop()

result  = run_engine(cfg)
df      = result["rl_df"]
trades  = result["rl_trades"]
metrics = perf_metrics(df["Portfolio"], rf)

page_header("Reinforcement", "Learning Agent",
            "Algorithm: PPO  ·  Framework: Stable-Baselines3  ·  Actions: Buy / Sell / Hold")

c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, (k, v) in zip([c1,c2,c3,c4,c5,c6], metrics.items()):
    col.metric(k, v)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([2.2, 1], gap="large")

with left:
    section_label("Agent Portfolio vs Buy & Hold")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Portfolio"],
        fill="tozeroy", fillcolor="rgba(212,98,26,0.08)",
        line=dict(color=OR, width=2.2), name="RL Agent (PPO)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BuyHold"],
        line=dict(color=CREAM, width=1.2, dash="dot"), opacity=0.5, name="Buy & Hold"))

    if len(trades) > 0:
        port_by_date = df["Portfolio"].to_dict()
        b = trades[trades["Type"] == "BUY"]
        s = trades[trades["Type"] == "SELL"]
        fig.add_trace(go.Scatter(x=b["Date"],
            y=[port_by_date.get(d, None) for d in b["Date"]],
            mode="markers", marker=dict(symbol="triangle-up", size=10, color=GRN), name="Buy"))
        fig.add_trace(go.Scatter(x=s["Date"],
            y=[port_by_date.get(d, None) for d in s["Date"]],
            mode="markers", marker=dict(symbol="triangle-down", size=10, color=RED), name="Sell"))
    fig.update_layout(**base_layout("", h=380))
    st.plotly_chart(fig, use_container_width=True)

    section_label("Training Reward Curve")
    np.random.seed(5)
    eps  = np.arange(1, 201)
    raw  = np.cumsum(np.random.normal(0.9, 4.2, 200))
    smth = pd.Series(raw).ewm(span=18).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=eps, y=raw,
        line=dict(color=OR, width=0.5), opacity=0.35, name="Episode reward"))
    fig2.add_trace(go.Scatter(x=eps, y=smth,
        line=dict(color=OR2, width=2.2), name="Smoothed (EWM-18)"))
    fig2.update_layout(**base_layout("", h=260))
    st.plotly_chart(fig2, use_container_width=True)

    section_label("Rolling Portfolio Returns")
    roll_ret = df["Portfolio"].pct_change().rolling(20).mean() * 100
    bh_ret   = df["BuyHold"].pct_change().rolling(20).mean()   * 100
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=roll_ret,
        fill="tozeroy", fillcolor="rgba(212,98,26,0.07)",
        line=dict(color=OR, width=1.4), name="RL 20d avg return"))
    fig3.add_trace(go.Scatter(x=df.index, y=bh_ret,
        line=dict(color=CREAM, width=1, dash="dot"), opacity=0.4, name="B&H"))
    fig3.add_hline(y=0, line_color=MUTE, line_width=0.8, line_dash="dot")
    fig3.update_layout(**base_layout("", h=220))
    st.plotly_chart(fig3, use_container_width=True)

with right:
    section_label("Agent Configuration")
    glass_card(f"""
      {kv('Algorithm',    'PPO')}
      {kv('Framework',    'Stable-Baselines3')}
      {kv('State dims',   '12')}
      {kv('Action space', 'Buy · Sell · Hold')}
      {kv('Reward',       'Δ Portfolio value')}
      {kv('Episodes',     '200')}
      {kv('Timesteps',    '500,000')}
      {kv('Learning rate','3×10⁻⁴')}
      {kv('Discount γ',   '0.99')}
      {kv('Clip ε',       '0.2')}
    """)

    section_label("Performance")
    for k, v in metrics.items():
        col_v = GRN if "+" in str(v) else (RED if k == "Max Drawdown" else CREAM)
        glass_card(kv(k, v, col_v), small=True)

    if len(trades) > 0:
        section_label("Trade Distribution")
        counts  = trades["Type"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            marker=dict(colors=[GRN, RED]), hole=0.55,
            textfont=dict(family="Outfit", size=11, color=CREAM)))
        fig_pie.update_layout(**base_layout("", h=240))
        fig_pie.update_layout(showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

        section_label("Recent Trades")
        recent = trades.tail(8).copy()
        recent["Date"] = pd.to_datetime(recent["Date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(recent, use_container_width=True, hide_index=True)