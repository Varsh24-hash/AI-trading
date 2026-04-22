"""
utils.py — Shared theme, data loading, and chart helpers
AI Algorithmic Trading Simulator
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import os

# ── Palette ───────────────────────────────────────────────────────────────────
OR   = "#D4621A"
OR2  = "#E8844A"
GOLD = "#C4922A"
CREAM= "#EDE0D0"
MUTE = "#7A6050"
DIM  = "#3A2418"
BG   = "#0A0704"
CARD = "rgba(28,16,8,0.75)"
GRN  = "#52B788"
RED  = "#D95F4B"
BORD = "rgba(180,80,20,0.18)"

# ── Project root ──────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_RAW_DIR   = os.path.join(_ROOT, "data", "raw")
_PROC_DIR  = os.path.join(_ROOT, "data", "processed")
_MODEL_DIR = os.path.join(_ROOT, "models")

# ── Tickers that have actual data files ───────────────────────────────────────
TICKERS = {
    "AAPL":  ("Apple Inc.",        182.0),
    "TSLA":  ("Tesla Inc.",        242.0),
    "MSFT":  ("Microsoft Corp.",   374.0),
    "GOOGL": ("Alphabet Inc.",     141.0),
    "AMZN":  ("Amazon.com",        178.0),
    "NVDA":  ("NVIDIA Corp.",      620.0),
    "META":  ("Meta Platforms",    350.0),
    "JPM":   ("JPMorgan Chase",    195.0),
    "V":     ("Visa Inc.",         275.0),
    "WMT":   ("Walmart Inc.",      165.0),
}

# ── CSS ───────────────────────────────────────────────────────────────────────
FONT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');
"""

GLOBAL_CSS = FONT_CSS + """
html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif !important;
  background: #0A0704 !important;
  color: #EDE0D0 !important;
}
.stApp {
  background:
    radial-gradient(ellipse 70% 50% at 5% 0%,  rgba(160,60,10,0.10) 0%, transparent 55%),
    radial-gradient(ellipse 50% 40% at 95% 95%, rgba(120,45,8,0.09)  0%, transparent 50%),
    #0A0704 !important;
}
section[data-testid="stSidebar"] {
  background: rgba(10,7,4,0.97) !important;
  border-right: 1px solid rgba(180,80,20,0.15) !important;
  padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
[data-testid="stSidebarNav"] a {
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
  color: #7A6050 !important;
  border-radius: 6px !important;
  padding: 0.45rem 0.8rem !important;
  transition: all 0.2s !important;
}
[data-testid="stSidebarNav"] a:hover { color: #D4621A !important; background: rgba(212,98,26,0.08) !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  color: #D4621A !important;
  background: rgba(212,98,26,0.12) !important;
  border-left: 2px solid #D4621A !important;
}
[data-testid="stSidebarNav"] span { font-size: 0.82rem !important; }
.stButton > button {
  background: linear-gradient(135deg, #D4621A 0%, #A84D12 100%) !important;
  color: #fff !important; border: none !important;
  border-radius: 7px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important; font-size: 0.8rem !important;
  letter-spacing: 0.07em !important; text-transform: uppercase !important;
  padding: 0.55rem 1.6rem !important;
  box-shadow: 0 3px 16px rgba(212,98,26,0.25) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #E8844A 0%, #D4621A 100%) !important;
  box-shadow: 0 5px 24px rgba(212,98,26,0.40) !important;
  transform: translateY(-1px) !important;
}
.run-btn > div > button {
  width: 100% !important; padding: 0.8rem 1.6rem !important;
  font-size: 0.88rem !important; letter-spacing: 0.12em !important;
  box-shadow: 0 4px 24px rgba(212,98,26,0.40) !important;
}
.run-btn > div > button:hover {
  box-shadow: 0 6px 32px rgba(212,98,26,0.60) !important;
  transform: translateY(-2px) !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
  background: rgba(28,16,8,0.8) !important;
  border: 1px solid rgba(180,80,20,0.22) !important;
  border-radius: 7px !important;
  font-family: 'Outfit', sans-serif !important; font-size: 0.84rem !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
  background: #D4621A !important; border: 2px solid #E8844A !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] { color: #D4621A !important; }
[data-testid="stMetric"] {
  background: rgba(28,16,8,0.75) !important;
  border: 1px solid rgba(180,80,20,0.18) !important;
  border-radius: 12px !important; padding: 1.2rem 1.4rem !important;
  backdrop-filter: blur(16px); transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: rgba(212,98,26,0.4) !important; }
[data-testid="stMetricLabel"] {
  font-family: 'Outfit', sans-serif !important; font-size: 0.68rem !important;
  font-weight: 500 !important; letter-spacing: 0.12em !important;
  text-transform: uppercase !important; color: #7A6050 !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Playfair Display', serif !important; font-size: 1.75rem !important;
  font-weight: 600 !important; color: #EDE0D0 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
.stTabs [data-baseweb="tab-list"] {
  background: rgba(16,10,5,0.6) !important;
  border-bottom: 1px solid rgba(180,80,20,0.18) !important;
  gap: 4px !important; padding: 0 4px !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Outfit', sans-serif !important; font-size: 0.78rem !important;
  font-weight: 500 !important; letter-spacing: 0.05em !important;
  color: #5A4030 !important; background: transparent !important;
  border: none !important; border-radius: 6px 6px 0 0 !important;
  padding: 0.65rem 1.2rem !important; transition: color 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #EDE0D0 !important; }
.stTabs [aria-selected="true"] {
  color: #D4621A !important; border-bottom: 2px solid #D4621A !important;
  background: rgba(212,98,26,0.06) !important;
}
.stDataFrame thead th {
  background: rgba(212,98,26,0.12) !important; color: #D4621A !important;
  font-family: 'Outfit', sans-serif !important; font-size: 0.72rem !important;
  letter-spacing: 0.08em !important; text-transform: uppercase !important;
  font-weight: 600 !important;
}
.stDataFrame tbody td {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.76rem !important; color: #EDE0D0 !important;
}
.stDataFrame tbody tr:hover td { background: rgba(212,98,26,0.06) !important; }
.stNumberInput input {
  background: rgba(28,16,8,0.8) !important;
  border: 1px solid rgba(180,80,20,0.22) !important;
  color: #EDE0D0 !important; font-family: 'JetBrains Mono', monospace !important;
  border-radius: 7px !important; font-size: 0.84rem !important;
}
.stRadio label {
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.82rem !important; color: #9A7A62 !important;
}
.stCheckbox label {
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.82rem !important; color: #9A7A62 !important;
}
.stProgress > div > div { background: #D4621A !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0A0704; }
::-webkit-scrollbar-thumb { background: #D4621A; border-radius: 4px; }
hr { border-color: rgba(180,80,20,0.18) !important; margin: 1.5rem 0 !important; }
.glass-card {
  background: rgba(28,16,8,0.75);
  border: 1px solid rgba(180,80,20,0.18);
  border-radius: 14px; padding: 1.6rem 1.8rem;
  backdrop-filter: blur(18px); margin-bottom: 1.2rem;
  transition: border-color 0.25s, box-shadow 0.25s;
}
.glass-card:hover {
  border-color: rgba(212,98,26,0.38);
  box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.card-sm {
  background: rgba(28,16,8,0.75);
  border: 1px solid rgba(180,80,20,0.18);
  border-radius: 10px; padding: 1rem 1.2rem;
  backdrop-filter: blur(18px); margin-bottom: 0.8rem;
}
.page-title {
  font-family: 'Playfair Display', serif; font-size: 2.4rem;
  font-weight: 700; color: #EDE0D0; line-height: 1.15;
  letter-spacing: -0.02em;
}
.page-title .accent { color: #D4621A; }
.page-subtitle {
  font-family: 'Outfit', sans-serif; font-size: 0.82rem;
  color: #7A6050; letter-spacing: 0.04em; margin-top: 0.4rem;
}
.section-label {
  font-family: 'Outfit', sans-serif; font-size: 0.65rem;
  font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase;
  color: #D4621A; padding-bottom: 0.5rem; margin-bottom: 1rem;
  border-bottom: 1px solid rgba(212,98,26,0.2);
}
.pill {
  display: inline-block; padding: 3px 10px; border-radius: 20px;
  font-family: 'Outfit', sans-serif; font-size: 0.65rem;
  font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
}
.pill-green  { background: rgba(82,183,136,0.15);  color: #52B788; border: 1px solid rgba(82,183,136,0.3); }
.pill-red    { background: rgba(217,95,75,0.15);   color: #D95F4B; border: 1px solid rgba(217,95,75,0.3); }
.pill-orange { background: rgba(212,98,26,0.15);   color: #E8844A; border: 1px solid rgba(212,98,26,0.3); }
.pill-gold   { background: rgba(196,146,42,0.15);  color: #C4922A; border: 1px solid rgba(196,146,42,0.3); }
.kv-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.4rem 0; border-bottom: 1px solid rgba(180,80,20,0.08);
  font-size: 0.8rem;
}
.kv-label { color: #7A6050; font-family: 'Outfit', sans-serif; }
.kv-val   { color: #EDE0D0; font-family: 'Playfair Display', serif; font-weight: 600; font-size: 0.88rem; }
.stat-number { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; line-height: 1; }
.sidebar-brand {
  font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700;
  color: #EDE0D0; padding: 1.2rem 0.4rem 0.8rem;
  border-bottom: 1px solid rgba(180,80,20,0.18); margin-bottom: 1rem;
  letter-spacing: -0.01em;
}
.sidebar-brand .accent { color: #D4621A; }
.sidebar-section {
  font-family: 'Outfit', sans-serif; font-size: 0.6rem; font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase;
  color: #5A4030; margin: 1.2rem 0 0.5rem;
}
"""


# ── UI helpers ────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)

def page_header(title, accent, subtitle=""):
    st.markdown(f"""
    <div style="padding: 2rem 0 1.5rem">
      <div class="page-title">{title} <span class="accent">{accent}</span></div>
      {'<div class="page-subtitle">' + subtitle + '</div>' if subtitle else ''}
    </div>""", unsafe_allow_html=True)

def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

def glass_card(content_html, small=False):
    cls = "card-sm" if small else "glass-card"
    st.markdown(f'<div class="{cls}">{content_html}</div>', unsafe_allow_html=True)

def kv(label, value, color=CREAM):
    return (f'<div class="kv-row"><span class="kv-label">{label}</span>'
            f'<span class="kv-val" style="color:{color}">{value}</span></div>')

def pill(text, kind="orange"):
    return f'<span class="pill pill-{kind}">{text}</span>'

def base_layout(title="", h=400):
    return dict(
        title=dict(text=title, font=dict(family="Playfair Display", size=14, color=OR),
                   x=0, xanchor="left", pad=dict(l=4)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,10,5,0.55)",
        font=dict(family="Outfit", color=MUTE, size=11),
        height=h, margin=dict(l=52, r=20, t=48, b=44),
        xaxis=dict(gridcolor="rgba(180,80,20,0.07)", linecolor="rgba(180,80,20,0.18)",
                   tickfont=dict(size=10, family="JetBrains Mono"), zeroline=False),
        yaxis=dict(gridcolor="rgba(180,80,20,0.07)", linecolor="rgba(180,80,20,0.18)",
                   tickfont=dict(size=10, family="JetBrains Mono"), zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(180,80,20,0.2)",
                    borderwidth=1, font=dict(size=10, family="Outfit")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(18,10,5,0.92)", bordercolor=OR,
                        font=dict(family="Outfit", size=11, color=CREAM)),
    )


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_price_data(ticker: str) -> pd.DataFrame:
    path = os.path.join(_RAW_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return _simulated_price(ticker)
    try:
        df = pd.read_csv(path, header=0, skiprows=[1],
                         parse_dates=["Date"], index_col="Date")
        df.columns = [c.strip() for c in df.columns]
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "close":    rename[c] = "Close"
            elif cl == "open":   rename[c] = "Open"
            elif cl == "high":   rename[c] = "High"
            elif cl == "low":    rename[c] = "Low"
            elif cl == "volume": rename[c] = "Volume"
        df = df.rename(columns=rename)
        for col in ["Open","High","Low","Close","Volume"]:
            if col not in df.columns:
                df[col] = np.nan
        df = df[["Open","High","Low","Close","Volume"]]
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        return df.sort_index()
    except Exception:
        return _simulated_price(ticker)


def _simulated_price(ticker: str, n_days: int = 756) -> pd.DataFrame:
    start = TICKERS.get(ticker, (ticker, 100.0))[1]
    np.random.seed(abs(hash(ticker)) % 2**31)
    mu, sig = 0.00035, 0.017
    r     = np.random.normal(mu, sig, n_days)
    price = start * np.exp(np.cumsum(r))
    end   = pd.Timestamp.today().normalize()
    if end.weekday() == 5: end -= pd.Timedelta(days=1)
    elif end.weekday() == 6: end -= pd.Timedelta(days=2)
    dates = pd.date_range(end=end, periods=n_days, freq="B")
    hi  = price * (1 + np.abs(np.random.normal(0, 0.007, n_days)))
    lo  = price * (1 - np.abs(np.random.normal(0, 0.007, n_days)))
    op  = price * (1 + np.random.normal(0, 0.004, n_days))
    vol = np.random.lognormal(np.log(start * 75_000), 0.45, n_days).astype(int)
    return pd.DataFrame({"Open":op,"High":hi,"Low":lo,"Close":price,"Volume":vol},
                        index=dates)


@st.cache_data(show_spinner=False)
def get_features(ticker: str) -> pd.DataFrame:
    df = get_price_data(ticker).copy()
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["MA_20"]     = df["Close"].rolling(20).mean()
    df["MA_50"]     = df["Close"].rolling(50).mean()
    df["MA_200"]    = df["Close"].rolling(200).mean()
    df["Vol_20"]    = df["Return_1d"].rolling(20).std() * np.sqrt(252)
    d = df["Close"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df["RSI"]      = 100 - 100 / (1 + g / l.replace(0, np.nan))
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    df["BB_Std"]   = df["Close"].rolling(20).std()
    df["BB_Up"]    = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lo"]    = df["BB_Mid"] - 2 * df["BB_Std"]
    e12 = df["Close"].ewm(span=12).mean()
    e26 = df["Close"].ewm(span=26).mean()
    df["MACD"]     = e12 - e26
    df["MACD_Sig"] = df["MACD"].ewm(span=9).mean()
    df["MACD_H"]   = df["MACD"] - df["MACD_Sig"]
    df["volume_change"] = df["Volume"].pct_change()
    return df.dropna()


# ── Engine-aware runners ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_ml(ticker: str, model: str, capital: float, txn: float):
    """
    Run ML backtest using real trained model + processed CSV.
    Falls back to simulation if files are missing.
    Stores model evaluation metrics in st.session_state.
    """
    import joblib
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 roc_auc_score, f1_score, confusion_matrix)

    MODEL_MAP = {
        "xgboost":             "xgb_model.pkl",
        "random_forest":       "random_forest_model.pkl",
        "logistic_regression": "logistic_model.pkl",
    }
    FEATURE_COLS = ["return_1d", "MA_10", "MA_50", "volatility", "volume_change", "RSI"]

    csv_path   = os.path.join(_PROC_DIR, f"{ticker}_features.csv")
    model_path = os.path.join(_MODEL_DIR, MODEL_MAP.get(model, "xgb_model.pkl"))

    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        return _run_ml_simulated(ticker, model, capital, txn)

    raw = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    raw = raw.apply(pd.to_numeric, errors="coerce").dropna()

    if any(c not in raw.columns for c in FEATURE_COLS + ["Close", "target"]):
        return _run_ml_simulated(ticker, model, capital, txn)

    clf     = joblib.load(model_path)
    X       = raw[FEATURE_COLS]
    y       = raw["target"]
    prob_up = clf.predict_proba(X)[:, 1]
    signals = (prob_up > 0.5).astype(int)

    df = raw.copy()
    df["Signal"] = signals
    df["Prob"]   = prob_up

    cash, shares = capital, 0
    port, trades = [], []
    buy_price    = None
    for i in range(len(df) - 1):
        p = df["Close"].iloc[i]
        s = df["Signal"].iloc[i]
        if s == 1 and shares == 0:
            shares    = int(cash * 0.95 / (p * (1 + txn)))
            cash     -= shares * p * (1 + txn)
            buy_price = p
            trades.append({"Date":df.index[i],"Type":"BUY",
                           "Price":round(p,4),"Shares":shares,"PnL":0.0})
        elif s == 0 and shares > 0:
            cash += shares * p * (1 - txn)
            pnl  = round((p - buy_price) * shares - (p + buy_price) * shares * txn, 2) \
                   if buy_price else 0.0
            trades.append({"Date":df.index[i],"Type":"SELL",
                           "Price":round(p,4),"Shares":shares,"PnL":pnl})
            shares = 0; buy_price = None
        port.append(cash + shares * p)

    df = df.iloc[:-1].copy()
    df["Portfolio"] = port
    df["BuyHold"]   = capital * df["Close"] / df["Close"].iloc[0]

    # ── Model evaluation ──────────────────────────────────────────────────────
    split  = int(len(raw) * 0.8)
    X_test = X.iloc[split:]; y_test = y.iloc[split:]
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc  = round(accuracy_score(y_test, y_pred) * 100)
    prec = round(precision_score(y_test, y_pred, zero_division=0) * 100)
    rec  = round(recall_score(y_test, y_pred, zero_division=0) * 100)
    auc  = round(roc_auc_score(y_test, y_prob), 2)
    f1   = round(f1_score(y_test, y_pred, zero_division=0), 2)

    st.session_state["ml_metrics"] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "roc_auc": auc, "f1": f1,
    }

    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imps = np.abs(clf.coef_[0]); imps /= imps.sum()
    else:
        imps = np.ones(len(FEATURE_COLS)) / len(FEATURE_COLS)
    st.session_state["ml_feature_importances"] = dict(zip(FEATURE_COLS, imps))
    st.session_state["ml_confusion_matrix"]    = confusion_matrix(y_test, y_pred).tolist()
    st.session_state["ml_model_name"]          = model

    return df, pd.DataFrame(trades)


def _run_ml_simulated(ticker, model, capital, txn):
    df  = get_features(ticker).copy()
    acc = {"xgboost": 0.63, "random_forest": 0.59, "logistic_regression": 0.54}[model]
    np.random.seed(42)
    true_dir = (df["Close"].shift(-1) > df["Close"]).astype(int)
    noise    = np.random.random(len(df))
    sig      = np.where(noise < acc, true_dir, 1 - true_dir)
    prob     = np.clip(0.5 + (sig-0.5)*np.random.uniform(0.3,0.8,len(df)), 0.34, 0.96)
    df["Signal"] = sig; df["Prob"] = prob
    cash, shares = capital, 0
    port, trades = [], []
    for i in range(len(df) - 1):
        p = df["Close"].iloc[i]; s = df["Signal"].iloc[i]
        if s == 1 and shares == 0:
            shares = int(cash*0.95/(p*(1+txn))); cash -= shares*p*(1+txn)
            trades.append({"Date":df.index[i],"Type":"BUY","Price":round(p,4),"Shares":shares,"PnL":0.0})
        elif s == 0 and shares > 0:
            cash += shares*p*(1-txn)
            trades.append({"Date":df.index[i],"Type":"SELL","Price":round(p,4),"Shares":shares,"PnL":0.0})
            shares = 0
        port.append(cash + shares*p)
    df = df.iloc[:-1].copy()
    df["Portfolio"] = port
    df["BuyHold"]   = capital * df["Close"] / df["Close"].iloc[0]
    return df, pd.DataFrame(trades)


@st.cache_data(show_spinner=False)
def run_rl(ticker: str, capital: float, txn: float):
    """Momentum-based RL proxy using real price data."""
    df = get_features(ticker).copy()
    np.random.seed(99)
    mom = pd.Series(df["Return_1d"].values).rolling(5).mean().fillna(0).values
    sig = np.where(mom > 0.001, 1, np.where(mom < -0.001, 0, -1))
    cash, shares = capital, 0
    port, trades = [], []
    buy_price    = None
    for i in range(len(df) - 1):
        p = df["Close"].iloc[i]; s = sig[i]
        if s == 1 and shares == 0:
            shares    = int(cash*0.95/(p*(1+txn))); cash -= shares*p*(1+txn)
            buy_price = p
            trades.append({"Date":df.index[i],"Type":"BUY","Price":round(p,4),"Shares":shares,"PnL":0.0})
        elif s == 0 and shares > 0:
            cash += shares*p*(1-txn)
            pnl  = round((p-buy_price)*shares-(p+buy_price)*shares*txn,2) if buy_price else 0.0
            trades.append({"Date":df.index[i],"Type":"SELL","Price":round(p,4),"Shares":shares,"PnL":pnl})
            shares = 0; buy_price = None
        port.append(cash + shares*p)
    df = df.iloc[:-1].copy()
    df["Portfolio"] = port
    df["BuyHold"]   = capital * df["Close"] / df["Close"].iloc[0]
    return df, pd.DataFrame(trades)


def run_engine(cfg: dict):
    """
    ── MAIN ENGINE ROUTER ──────────────────────────────────────────────────────
    Called by every page. Routes based on cfg["engine"]:
      "ML Prediction" → runs selected ML model only
      "RL Agent"      → runs RL momentum strategy only
      "Both"          → runs both, returns combined results

    Returns a dict with keys the pages actually need:
      active_df       : portfolio DataFrame for the active/primary strategy
      active_trades   : trades for the active strategy
      active_metrics  : perf_metrics dict for the active strategy
      ml_df, ml_trades, rl_df, rl_trades  (always populated)
      bh_metrics      : buy & hold metrics
      engine          : which engine is active ("ML Prediction"/"RL Agent"/"Both")
      model_label     : human-readable model name
    """
    ticker  = cfg["ticker"]
    model   = cfg["model"]
    capital = cfg["capital"]
    txn     = cfg["txn"]
    rf      = cfg["rf"]
    engine  = cfg["engine"]

    ml_df = ml_trades = rl_df = rl_trades = None

    # ── Always run what's needed ───────────────────────────────────────────────
    if engine in ("ML Prediction", "Both"):
        ml_df, ml_trades = run_ml(ticker, model, capital, txn)

    if engine in ("RL Agent", "Both"):
        rl_df, rl_trades = run_rl(ticker, capital, txn)

    # ── For "Both", ensure both are populated ─────────────────────────────────
    if engine == "Both":
        pass  # both already set above

    # ── Buy & Hold always computed from ML df (or RL if ML not run) ──────────
    ref_df = ml_df if ml_df is not None else rl_df

    # ── Select active strategy for pages that show one primary result ─────────
    if engine == "ML Prediction":
        active_df     = ml_df
        active_trades = ml_trades
    elif engine == "RL Agent":
        active_df     = rl_df
        active_trades = rl_trades
    else:  # Both — use ML as primary display
        active_df     = ml_df
        active_trades = ml_trades

    active_metrics = perf_metrics(active_df["Portfolio"], rf)
    bh_metrics     = perf_metrics(ref_df["BuyHold"],      rf)

    ml_metrics = perf_metrics(ml_df["Portfolio"], rf) if ml_df is not None else {}
    rl_metrics = perf_metrics(rl_df["Portfolio"], rf) if rl_df is not None else {}

    model_label = model.replace("_", " ").title()

    return {
        "active_df":      active_df,
        "active_trades":  active_trades,
        "active_metrics": active_metrics,
        "ml_df":          ml_df,
        "ml_trades":      ml_trades,
        "rl_df":          rl_df,
        "rl_trades":      rl_trades,
        "ml_metrics":     ml_metrics,
        "rl_metrics":     rl_metrics,
        "bh_metrics":     bh_metrics,
        "engine":         engine,
        "model_label":    model_label,
    }


def perf_metrics(s: pd.Series, rf: float = 0.05) -> dict:
    r   = s.pct_change().dropna()
    tot = (s.iloc[-1] / s.iloc[0] - 1) * 100
    ann = ((1 + tot/100) ** (252/len(r)) - 1) * 100
    vol = r.std() * np.sqrt(252) * 100
    sr  = (ann/100 - rf) / (vol/100) if vol else 0
    dd  = ((s - s.cummax()) / s.cummax() * 100).min()
    wr  = (r > 0).mean() * 100
    return {"Total Return": f"{tot:+.2f}%", "Ann. Return": f"{ann:+.2f}%",
            "Volatility": f"{vol:.2f}%",    "Sharpe": f"{sr:.3f}",
            "Max Drawdown": f"{dd:.2f}%",   "Win Rate": f"{wr:.1f}%"}


def calc_pnl(trades: pd.DataFrame, txn: float = 0.001) -> pd.DataFrame:
    if trades.empty:
        return trades
    trades    = trades.copy().reset_index(drop=True)
    pnl_list  = [0.0] * len(trades)
    buy_stack = []
    for i, row in trades.iterrows():
        if row["Type"] == "BUY":
            buy_stack.append((i, row["Price"], row["Shares"]))
        elif row["Type"] == "SELL" and buy_stack:
            _, buy_price, _ = buy_stack.pop(0)
            gross = (row["Price"] - buy_price) * row["Shares"]
            costs = (row["Price"] + buy_price) * row["Shares"] * txn
            pnl_list[i] = round(gross - costs, 2)
    trades["PnL"] = pnl_list
    return trades


@st.cache_data(show_spinner=False)
def efficient_frontier(tickers: tuple, n_port: int = 1000, rf: float = 0.05):
    frames  = {t: get_price_data(t)["Close"].pct_change().dropna() for t in tickers}
    aligned = pd.DataFrame(frames).dropna()
    if len(aligned) < 30 or len(tickers) < 2:
        return pd.DataFrame(columns=["Return","Volatility","Sharpe","Weights"])
    mu  = aligned.mean() * 252
    cov = aligned.cov()  * 252
    if cov.isnull().values.any():
        return pd.DataFrame(columns=["Return","Volatility","Sharpe","Weights"])
    res = []
    np.random.seed(3)
    for _ in range(n_port):
        w    = np.random.dirichlet(np.ones(len(tickers)))
        r    = float(mu @ w * 100)
        v_sq = float(w @ cov.values @ w)
        if v_sq <= 0 or not np.isfinite(v_sq) or not np.isfinite(r):
            continue
        v = float(np.sqrt(v_sq) * 100)
        s = (r/100 - rf) / (v/100)
        if not np.isfinite(s):
            continue
        res.append({"Return":r,"Volatility":v,"Sharpe":s,"Weights":w})
    if not res:
        return pd.DataFrame(columns=["Return","Volatility","Sharpe","Weights"])
    df = pd.DataFrame(res).dropna(subset=["Return","Volatility","Sharpe"])
    return df[df["Volatility"] > 0].reset_index(drop=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar_controls():
    st.markdown('<div class="sidebar-brand">Algo<span class="accent">Trade</span></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Universe</div>', unsafe_allow_html=True)
    ticker = st.selectbox("Stock", list(TICKERS.keys()),
                          format_func=lambda x: f"{x}  ·  {TICKERS[x][0]}")
    multi  = st.multiselect("Portfolio basket", list(TICKERS.keys()),
                             default=["AAPL","TSLA","MSFT","GOOGL","NVDA"])
    if len(multi) < 2:
        multi = ["AAPL","TSLA","MSFT","GOOGL"]

    st.markdown('<div class="sidebar-section">Strategy</div>', unsafe_allow_html=True)

    # ── Engine selector (affects ALL pages) ───────────────────────────────────
    engine = st.radio("Engine", ["ML Prediction", "RL Agent", "Both"])

    # ── ML model selector (only relevant when engine includes ML) ─────────────
    ml_disabled = (engine == "RL Agent")
    model = st.selectbox(
        "ML Model",
        ["XGBoost", "Random Forest", "Logistic Regression"],
        disabled=ml_disabled,
        help="Disabled when RL Agent only is selected",
    )

    st.markdown('<div class="sidebar-section">Parameters</div>', unsafe_allow_html=True)
    capital = st.number_input("Capital ($)", value=100_000, step=10_000, min_value=10_000)
    txn     = st.slider("Transaction cost (%)", 0.0, 1.0, 0.10, 0.05) / 100
    rf      = st.slider("Risk-free rate (%)",   0.0, 10.0, 5.0, 0.5)  / 100
    mc_n    = st.slider("Monte Carlo paths",    500, 5000, 2000, 500)
    var_c   = st.slider("VaR confidence",       0.90, 0.99, 0.95, 0.01)

    st.markdown('<div class="sidebar-section">Chart overlays</div>', unsafe_allow_html=True)
    show_bb  = st.checkbox("Bollinger Bands", True)
    show_ma  = st.checkbox("Moving Averages", True)
    show_vol = st.checkbox("Volume",          True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
    run = st.button("▶  Run Simulation", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        get_price_data.clear()
        get_features.clear()
        run_ml.clear()
        run_rl.clear()
        efficient_frontier.clear()
        st.rerun()

    return dict(
        ticker=ticker, multi=multi,
        model=model.lower().replace(" ", "_"),
        engine=engine, capital=capital,
        txn=txn, rf=rf, mc_n=mc_n, var_c=var_c,
        show_bb=show_bb, show_ma=show_ma, show_vol=show_vol,
        run=run,
    )