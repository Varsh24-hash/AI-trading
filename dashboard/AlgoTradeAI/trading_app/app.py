"""
AlgoTrade AI — Home
Entry point: project overview + navigation guide
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import inject_css, TICKERS, OR, OR2, GOLD, CREAM, MUTE, GRN, RED, CARD

st.set_page_config(
    page_title="AlgoTrade AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Algo<span class="accent">Trade</span></div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.75rem;color:{MUTE};line-height:1.9;margin-top:0.5rem">
      AI Algorithmic Trading Simulator<br>
      Built for quantitative finance<br>
      research and internship portfolios.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f'<div style="font-size:0.65rem;color:#3A2418;letter-spacing:0.1em;text-transform:uppercase;font-family:Outfit,sans-serif;font-weight:600">Navigate</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.78rem;color:{MUTE};line-height:2.2;margin-top:0.4rem">
      📊 &nbsp; Overview<br>
      📈 &nbsp; Price & Indicators<br>
      🤖 &nbsp; ML Prediction<br>
      🧠 &nbsp; RL Agent<br>
      ⚠️ &nbsp; Risk & VaR<br>
      🎲 &nbsp; Monte Carlo<br>
      💼 &nbsp; Portfolio Optimizer<br>
      📋 &nbsp; Trade Log
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
  background: linear-gradient(135deg, rgba(35,18,8,0.95) 0%, rgba(55,28,10,0.85) 50%, rgba(35,18,8,0.95) 100%);
  border: 1px solid rgba(212,98,26,0.22);
  border-radius: 18px;
  padding: 3.5rem 3rem;
  margin-bottom: 2.5rem;
  position: relative;
  overflow: hidden;
">
  <div style="
    position:absolute;top:-20%;right:-5%;
    width:500px;height:500px;
    background:radial-gradient(ellipse, rgba(212,98,26,0.10) 0%, transparent 65%);
    pointer-events:none
  "></div>
  <div style="
    font-family:'Playfair Display',serif;
    font-size:3.2rem;font-weight:700;
    color:{CREAM};line-height:1.1;
    letter-spacing:-0.02em;margin-bottom:1rem
  ">
    AI Algorithmic<br>
    <span style="color:{OR}">Trading Simulator</span>
  </div>
  <div style="
    font-family:'Outfit',sans-serif;
    font-size:0.9rem;color:{MUTE};
    max-width:600px;line-height:1.8;margin-bottom:2rem
  ">
    A full-stack quantitative finance research platform combining
    supervised ML price prediction, reinforcement learning trading agents,
    portfolio optimisation, and advanced risk modelling.
    Built to demonstrate skills relevant to quant teams at top-tier firms.
  </div>
  <div style="display:flex;gap:0.8rem;flex-wrap:wrap">
    <span class="pill pill-orange">Machine Learning</span>
    <span class="pill pill-green">Reinforcement Learning</span>
    <span class="pill pill-gold">Quantitative Finance</span>
    <span class="pill pill-orange">Monte Carlo</span>
    <span class="pill pill-red">Risk Management</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Navigation cards ──────────────────────────────────────────────────────────
st.markdown(f'<div class="section-label">Platform Modules</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

pages = [
    ("📊", "Overview",           "Strategy KPIs, equity curves, live quote, signal dashboard",
     "ML · RL · B&H comparison", "orange"),
    ("📈", "Price & Indicators", "Candlestick OHLCV, Bollinger Bands, MACD, RSI, Volume",
     "Technical Analysis", "gold"),
    ("🤖", "ML Prediction",      "XGBoost · Random Forest · Logistic Regression signal engine",
     "Supervised Learning", "orange"),
    ("🧠", "RL Agent",           "PPO trading agent trained with Stable-Baselines3, reward curves",
     "Reinforcement Learning", "green"),
    ("⚠️", "Risk & VaR",         "Historical, Parametric & MC Value at Risk, volatility forecast",
     "Risk Management", "red"),
    ("🎲", "Monte Carlo",        "10,000 price path simulations, confidence bands, scenarios",
     "Stochastic Simulation", "gold"),
    ("💼", "Portfolio Optimizer","Efficient frontier, Sharpe maximisation, correlation matrix",
     "Modern Portfolio Theory", "green"),
    ("📋", "Trade Log",          "Full execution history, P&L analysis, monthly breakdown",
     "Trade Analytics", "orange"),
]

cols = st.columns(4)
for i, (icon, name, desc, tag, kind) in enumerate(pages):
    with cols[i % 4]:
        st.markdown(f"""
        <div class="glass-card" style="cursor:default;min-height:160px">
          <div style="font-size:1.8rem;margin-bottom:0.6rem">{icon}</div>
          <div style="font-family:'Playfair Display',serif;font-size:1rem;
                      font-weight:600;color:{CREAM};margin-bottom:0.4rem">{name}</div>
          <div style="font-family:'Outfit',sans-serif;font-size:0.74rem;
                      color:{MUTE};line-height:1.6;margin-bottom:0.8rem">{desc}</div>
          <span class="pill pill-{kind}">{tag}</span>
        </div>
        """, unsafe_allow_html=True)
        if i % 4 == 3 and i < len(pages)-1:
            st.markdown("<br>", unsafe_allow_html=True)

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f'<div class="section-label">Technology Stack</div>', unsafe_allow_html=True)

tc = st.columns(6)
stack = [
    ("Python", "Core language"),
    ("Pandas / NumPy", "Data engineering"),
    ("scikit-learn / XGBoost", "ML models"),
    ("Stable-Baselines3", "RL training"),
    ("Plotly", "Visualisation"),
    ("Streamlit", "Dashboard"),
]
for col, (tech, role) in zip(tc, stack):
    col.markdown(f"""
    <div class="card-sm" style="text-align:center">
      <div style="font-family:'Playfair Display',serif;font-size:0.82rem;
                  font-weight:600;color:{OR2};margin-bottom:0.2rem">{tech}</div>
      <div style="font-size:0.65rem;color:{MUTE};letter-spacing:0.04em">{role}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center;margin-top:3rem;font-size:0.65rem;
            color:#2A1A0E;letter-spacing:0.1em;font-family:'Outfit',sans-serif">
  ALGOTRADE AI  ·  SIMULATED DATA ONLY  ·  NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)
