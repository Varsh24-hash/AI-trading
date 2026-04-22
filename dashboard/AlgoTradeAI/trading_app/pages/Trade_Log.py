"""
📋 Trade Log
Shows trades for the active engine. Switches between ML/RL/Both.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, run_engine, calc_pnl,
                   sidebar_controls, OR, OR2, GOLD, CREAM, MUTE, GRN, RED)

st.set_page_config(page_title="Trade Log · AlgoTrade AI",
                   page_icon="📋", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

ticker  = cfg["ticker"]
txn     = cfg["txn"]
engine  = cfg["engine"]

result   = run_engine(cfg)
ml_df    = result["ml_df"]
rl_df    = result["rl_df"]
ml_tr    = result["ml_trades"]
rl_tr    = result["rl_trades"]

# ── Compute real P&L ──────────────────────────────────────────────────────────
if ml_tr is not None and not ml_tr.empty:
    ml_tr = calc_pnl(ml_tr, txn)
if rl_tr is not None and not rl_tr.empty:
    rl_tr = calc_pnl(rl_tr, txn)

page_header("Trade", "Execution Log",
            f"{TICKERS[ticker][0]}  ·  Engine: {engine}  ·  Full history")

# ── Engine tab selector ───────────────────────────────────────────────────────
if engine == "Both":
    view = st.radio("View trades from", ["ML Strategy", "RL Agent"], horizontal=True)
elif engine == "ML Prediction":
    view = "ML Strategy"
    st.info("Showing ML Strategy trades — switch Engine to see RL trades.")
else:
    view = "RL Agent"
    st.info("Showing RL Agent trades — switch Engine to see ML trades.")

trades = ml_tr if view == "ML Strategy" else rl_tr
port_df= ml_df if view == "ML Strategy" else rl_df

st.markdown("<br>", unsafe_allow_html=True)

if trades is None or len(trades) == 0:
    st.info("No trades generated for this configuration.")
    st.stop()

trades = trades.copy()
trades["Date"] = pd.to_datetime(trades["Date"])

buy_t     = trades[trades["Type"] == "BUY"]
sell_t    = trades[trades["Type"] == "SELL"]
pnl_sells = sell_t["PnL"] if "PnL" in sell_t.columns else pd.Series(dtype=float)
wins      = (pnl_sells > 0).sum()
n_sells   = len(pnl_sells)
total_pnl = pnl_sells.sum()

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Total Trades",   len(trades))
c2.metric("Buy Orders",     len(buy_t))
c3.metric("Sell Orders",    len(sell_t))
c4.metric("Winning Trades", f"{wins} ({wins/max(n_sells,1)*100:.0f}%)")
c5.metric("Total P&L",      f"${total_pnl:,.0f}")
c6.metric("Avg P&L/Trade",  f"${pnl_sells.mean() if n_sells else 0:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

top_l, top_r = st.columns([1.8, 1], gap="large")

with top_l:
    section_label("Cumulative P&L")
    cum_pnl = trades["PnL"].cumsum()
    bar_col = [GRN if v >= 0 else RED for v in trades["PnL"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trades["Date"], y=trades["PnL"],
        marker_color=bar_col, opacity=0.65, name="Trade P&L"))
    fig.add_trace(go.Scatter(x=trades["Date"], y=cum_pnl,
        line=dict(color=OR, width=2.2), name="Cumulative P&L", yaxis="y2"))
    fig.update_layout(**base_layout("", h=340))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right",
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(size=9, family="JetBrains Mono", color=OR)))
    st.plotly_chart(fig, use_container_width=True)

    section_label("P&L Distribution")
    pnl_nonzero = trades[trades["PnL"] != 0]["PnL"]
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=pnl_nonzero, nbinsx=30,
        marker_color=OR, opacity=0.7, name="P&L per trade"))
    fig2.add_vline(x=0, line_color=MUTE, line_width=1, line_dash="dot")
    if len(pnl_nonzero):
        fig2.add_vline(x=pnl_nonzero.mean(), line_color=GRN, line_width=1.2,
            line_dash="dash",
            annotation_text=f"Avg ${pnl_nonzero.mean():,.0f}",
            annotation_font=dict(color=GRN, size=9, family="JetBrains Mono"))
    fig2.update_layout(**base_layout("", h=240))
    st.plotly_chart(fig2, use_container_width=True)

with top_r:
    section_label("Trade Stats")
    win_pnl  = pnl_sells[pnl_sells > 0].mean() if (pnl_sells > 0).any() else 0
    loss_pnl = pnl_sells[pnl_sells < 0].mean() if (pnl_sells < 0).any() else 0
    pf = abs(win_pnl / loss_pnl) if loss_pnl != 0 else 0
    glass_card(f"""
      {kv('Total trades',     str(len(trades)))}
      {kv('Win rate',         f"{wins/max(n_sells,1)*100:.1f}%",      GRN)}
      {kv('Avg win',          f"${win_pnl:,.0f}",                     GRN)}
      {kv('Avg loss',         f"${loss_pnl:,.0f}",                    RED)}
      {kv('Profit factor',    f"{pf:.2f}",                            OR2)}
      {kv('Total P&L',        f"${total_pnl:,.0f}",
           GRN if total_pnl > 0 else RED)}
      {kv('Best trade',       f"${pnl_sells.max() if n_sells else 0:,.0f}", GRN)}
      {kv('Worst trade',      f"${pnl_sells.min() if n_sells else 0:,.0f}", RED)}
      {kv('Largest position', f"{trades['Shares'].max():,} shares")}
    """)

    section_label("Monthly Breakdown")
    trades["Month"] = trades["Date"].dt.to_period("M")
    monthly = trades.groupby("Month")["PnL"].sum().reset_index()
    monthly["Month"] = monthly["Month"].astype(str)
    cols = [GRN if v >= 0 else RED for v in monthly["PnL"]]
    fig3 = go.Figure(go.Bar(x=monthly["Month"], y=monthly["PnL"],
        marker_color=cols, opacity=0.75))
    fig3.update_layout(**base_layout("", h=260))
    fig3.update_xaxes(tickangle=45, tickfont=dict(size=8))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
section_label("Full Trade History")
display = trades.copy()
display["Date"]  = display["Date"].dt.strftime("%Y-%m-%d")
display["Price"] = display["Price"].apply(lambda x: f"{x:,.4f}")
display["Value"] = (trades["Price"] * trades["Shares"]).apply(lambda x: f"${x:,.0f}")
display["PnL"]   = trades["PnL"].apply(lambda x: f"${x:+,.0f}" if x != 0 else "—")
display = display[["Date","Type","Price","Shares","Value","PnL"]]
st.dataframe(display, use_container_width=True, hide_index=True)