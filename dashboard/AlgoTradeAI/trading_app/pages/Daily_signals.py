"""
📡 Daily Signal
Live BUY / HOLD / SELL prediction using latest Yahoo Finance data.
Uses the shared sidebar (ticker, model, engine) from sidebar_controls().
No add_vline — uses add_shape instead to avoid Plotly annotation bug.
"""

import sys, os, importlib, importlib.util
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# ── Path setup ────────────────────────────────────────────────────────────────
_PAGE_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_root(start: str) -> str:
    current = start
    for _ in range(10):
        if os.path.exists(os.path.join(current, "data", "live_feed.py")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        f"Cannot find data/live_feed.py walking up from {start}"
    )

try:
    _ROOT = _find_root(_PAGE_DIR)
except FileNotFoundError as _e:
    st.set_page_config(page_title="Daily Signal · AlgoTrade", layout="wide")
    st.error(str(_e))
    st.stop()

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Load live_feed via importlib so it works regardless of package structure
_lf_path = os.path.join(_ROOT, "data", "live_feed.py")
_spec     = importlib.util.spec_from_file_location("live_feed", _lf_path)
_lf_mod   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lf_mod)

get_prediction_row  = _lf_mod.get_prediction_row
explain_signal_text = _lf_mod.explain_signal_text
FEATURE_COLS        = _lf_mod.FEATURE_COLS

# ── Utils import (shared sidebar + theme) ────────────────────────────────────
_UTILS_DIR = os.path.dirname(_PAGE_DIR)
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, sidebar_controls,
                   OR, OR2, GOLD, CREAM, MUTE, GRN, RED)

_MODEL_DIR = os.path.join(_ROOT, "models")
MODEL_FILE_MAP = {
    "xgboost":             "xgb_model.pkl",
    "random_forest":       "random_forest_model.pkl",
    "logistic_regression": "logistic_model.pkl",
}

st.set_page_config(page_title="Daily Signal · AlgoTrade AI",
                   page_icon="📡", layout="wide")
inject_css()

# ── Shared sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    cfg = sidebar_controls()

ticker = cfg["ticker"]
model  = cfg["model"]   # e.g. "xgboost"
name   = TICKERS.get(ticker, (ticker,))[0]

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    path = os.path.join(_MODEL_DIR, MODEL_FILE_MAP.get(model_key, "xgb_model.pkl"))
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_all_models():
    out = {}
    for key, fname in MODEL_FILE_MAP.items():
        path = os.path.join(_MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                out[key] = joblib.load(path)
            except Exception:
                pass
    return out

all_models = load_all_models()

if not all_models:
    st.error(f"No trained models found in {_MODEL_DIR}. "
             "Ensure xgb_model.pkl etc. are committed to your repo.")
    st.stop()

# ── Fetch live data ───────────────────────────────────────────────────────────
page_header("Daily", "Signal",
            f"{name}  ·  {ticker}  ·  Live Yahoo Finance  ·  Model: {model.replace('_',' ').title()}")

with st.spinner(f"Fetching latest market data for {ticker}…"):
    try:
        data = get_prediction_row(ticker)
    except Exception as e:
        st.error(f"Could not fetch live data: {e}")
        st.info("Check internet connection or try a different ticker.")
        st.stop()

df         = data["df"]
features   = data["features"]
latest     = data["latest"]
date_str   = data["date"]
prev_close = data["prev_close"]
close      = float(latest["Close"])
pct_chg    = (close - prev_close) / prev_close * 100
X_live     = pd.DataFrame([features])[FEATURE_COLS]

# ── Run selected model ────────────────────────────────────────────────────────
primary_clf = all_models.get(model) or list(all_models.values())[0]
primary_key = model if model in all_models else list(all_models.keys())[0]

prob_primary = float(primary_clf.predict_proba(X_live)[0, 1])
sig_primary  = "BUY" if prob_primary > 0.6 else ("SELL" if prob_primary < 0.4 else "HOLD")

SIG_COLOR = {"BUY": GRN, "SELL": RED, "HOLD": GOLD}
SIG_PILL  = {"BUY": "green", "SELL": "red", "HOLD": "gold"}
sig_color = SIG_COLOR[sig_primary]

# ── Hero signal card ──────────────────────────────────────────────────────────
chg_col = GRN if pct_chg >= 0 else RED
arrow   = "▲" if pct_chg >= 0 else "▼"

st.markdown(f"""
<div style="background:rgba(28,16,8,0.9);
            border:2px solid {sig_color}55;
            border-radius:18px;padding:2rem 2.8rem;
            margin:0.5rem 0 1.8rem;
            display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;gap:1.5rem">
  <div>
    <div style="font-family:'Outfit',sans-serif;font-size:0.68rem;
                color:{MUTE};letter-spacing:0.14em;
                text-transform:uppercase;margin-bottom:0.5rem">
      {model.replace('_',' ').title()}  ·  Signal for {date_str}
    </div>
    <div style="font-family:'Playfair Display',serif;
                font-size:5rem;font-weight:700;
                color:{sig_color};line-height:0.9;
                letter-spacing:-0.02em">
      {sig_primary}
    </div>
    <div style="font-family:'Outfit',sans-serif;font-size:0.88rem;
                color:{MUTE};margin-top:0.6rem">
      {prob_primary*100:.1f}% probability of upward move tomorrow
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-family:'Playfair Display',serif;
                font-size:3rem;font-weight:700;
                color:{chg_col};line-height:1">
      {close:,.2f}
    </div>
    <div style="font-size:0.9rem;color:{chg_col};margin-top:0.3rem">
      {arrow} {abs(pct_chg):.2f}%  vs previous close
    </div>
    <div style="font-size:0.7rem;color:{MUTE};margin-top:0.2rem">
      {ticker}  ·  {name}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── All 3 model signals ───────────────────────────────────────────────────────
section_label("All Model Signals")
mcols = st.columns(3)
model_labels = {
    "xgboost":             "XGBoost",
    "random_forest":       "Random Forest",
    "logistic_regression": "Logistic Regression",
}
for col, (key, label) in zip(mcols, model_labels.items()):
    if key in all_models:
        p   = float(all_models[key].predict_proba(X_live)[0, 1])
        s   = "BUY" if p > 0.6 else ("SELL" if p < 0.4 else "HOLD")
        sc  = SIG_COLOR[s]
        active_border = f"border:2px solid {sc};" if key == primary_key else f"border:1px solid {sc}44;"
        col.markdown(f"""
        <div style="background:rgba(28,16,8,0.8);{active_border}
                    border-radius:12px;padding:1.3rem;text-align:center">
          <div style="font-size:0.62rem;color:{MUTE};letter-spacing:0.1em;
                      text-transform:uppercase;margin-bottom:0.4rem">{label}</div>
          <div style="font-family:'Playfair Display',serif;font-size:2.2rem;
                      font-weight:700;color:{sc}">{s}</div>
          <div style="font-size:0.78rem;color:{MUTE};margin-top:0.3rem">
            {p*100:.1f}% prob up
          </div>
          <div style="margin-top:0.5rem">{pill(s, SIG_PILL[s])}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col.markdown(f"""
        <div style="background:rgba(28,16,8,0.5);border:1px solid rgba(180,80,20,0.15);
                    border-radius:12px;padding:1.3rem;text-align:center">
          <div style="font-size:0.62rem;color:{MUTE}">{label}</div>
          <div style="font-size:0.85rem;color:{MUTE};margin-top:0.4rem">
            Model not found
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Current readings strip ────────────────────────────────────────────────────
section_label("Current Indicator Readings")
r1, r2, r3, r4, r5, r6 = st.columns(6)
rsi_val  = features["RSI"]
rsi_col  = RED if rsi_val > 70 else (GRN if rsi_val < 30 else OR)
r1.metric("RSI (14)",   f"{rsi_val:.1f}",
          "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
r2.metric("MA 10",      f"{features['MA_10']:.2f}")
r3.metric("MA 50",      f"{features['MA_50']:.2f}")
r4.metric("Return 1d",  f"{features['return_1d']*100:+.2f}%")
r5.metric("Volatility", f"{features['volatility']*100:.1f}%")
r6.metric("Vol Change", f"{features['volume_change']*100:+.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Main layout: chart left, explanation right ────────────────────────────────
left, right = st.columns([2.2, 1], gap="large")

with left:
    section_label("Price Chart — Last 90 Days")

    chart_df = df.tail(90).copy()

    # Convert index to numeric positions to avoid Plotly timestamp issues
    # Use string dates for display but integer x-axis internally
    dates_str = chart_df.index.strftime("%Y-%m-%d").tolist()
    n         = len(dates_str)
    x_idx     = list(range(n))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.22, 0.20],
        vertical_spacing=0.03,
    )

    # ── Candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x_idx,
        open=chart_df["Open"].values,
        high=chart_df["High"].values,
        low=chart_df["Low"].values,
        close=chart_df["Close"].values,
        name="Price",
        increasing_line_color=GRN, decreasing_line_color=RED,
        increasing_fillcolor=GRN,  decreasing_fillcolor=RED,
    ), row=1, col=1)

    # MA lines
    fig.add_trace(go.Scatter(x=x_idx, y=chart_df["MA_10"].values,
        line=dict(color=GOLD, width=1.2), name="MA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_idx, y=chart_df["MA_50"].values,
        line=dict(color=OR2, width=1.2), name="MA50"), row=1, col=1)

    # Bollinger Bands
    if "BB_Up" in chart_df.columns and "BB_Lo" in chart_df.columns:
        fig.add_trace(go.Scatter(x=x_idx, y=chart_df["BB_Up"].values,
            line=dict(color=OR, width=0.8, dash="dot"),
            name="BB Upper", opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_idx, y=chart_df["BB_Lo"].values,
            line=dict(color=OR, width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(212,98,26,0.04)",
            name="BB Lower", opacity=0.5), row=1, col=1)

    # ── Signal marker: vertical line using add_shape (no annotation bug) ──────
    last_x = n - 1
    fig.add_shape(
        type="line",
        x0=last_x, x1=last_x,
        y0=0, y1=1,
        yref="paper",
        line=dict(color=sig_color, width=2, dash="dash"),
        row=1, col=1,
    )
    # Signal label as annotation separately — avoids the _mean() crash
    fig.add_annotation(
        x=last_x,
        y=1.0,
        yref="paper",
        text=f"→ {sig_primary}",
        showarrow=False,
        font=dict(color=sig_color, size=12, family="JetBrains Mono"),
        xanchor="left",
        yanchor="top",
        row=1, col=1,
    )

    # ── MACD ──────────────────────────────────────────────────────────────────
    if "MACD_H" in chart_df.columns:
        bar_c = [GRN if v >= 0 else RED
                 for v in chart_df["MACD_H"].fillna(0).values]
        fig.add_trace(go.Bar(x=x_idx, y=chart_df["MACD_H"].values,
            marker_color=bar_c, opacity=0.65, name="MACD Hist",
            showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_idx, y=chart_df["MACD"].values,
            line=dict(color=OR, width=1.1), name="MACD",
            showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_idx, y=chart_df["MACD_Sig"].values,
            line=dict(color=CREAM, width=1, dash="dot"),
            opacity=0.7, name="Signal",
            showlegend=False), row=2, col=1)

    # ── RSI — use scatter lines instead of add_hline ──────────────────────────
    fig.add_trace(go.Scatter(x=x_idx, y=chart_df["RSI"].values,
        line=dict(color=GOLD, width=1.3), name="RSI",
        showlegend=False), row=3, col=1)

    # Overbought/oversold as scatter lines (safer than add_hline on subplots)
    for lvl, clr in [(70, RED), (30, GRN)]:
        fig.add_trace(go.Scatter(
            x=[0, n-1], y=[lvl, lvl], mode="lines",
            line=dict(color=clr, width=0.8, dash="dot"),
            showlegend=False,
        ), row=3, col=1)

    fig.add_hrect(y0=70, y1=100,
                  fillcolor="rgba(217,95,75,0.04)", line_width=0,
                  row=3, col=1)
    fig.add_hrect(y0=0, y1=30,
                  fillcolor="rgba(82,183,136,0.04)", line_width=0,
                  row=3, col=1)

    # ── X-axis: show date labels at sensible intervals ─────────────────────────
    tick_step  = max(1, n // 8)
    tick_vals  = list(range(0, n, tick_step))
    tick_texts = [dates_str[i] for i in tick_vals]

    layout = base_layout("", h=560)
    layout["xaxis_rangeslider_visible"]  = False
    layout["xaxis3_rangeslider_visible"] = False
    fig.update_layout(**layout)

    for row_i in (1, 2, 3):
        fig.update_xaxes(
            tickvals=tick_vals, ticktext=tick_texts,
            tickangle=0, tickfont=dict(size=9),
            gridcolor="rgba(180,80,20,0.07)",
            linecolor="rgba(180,80,20,0.15)",
            row=row_i, col=1,
        )
        fig.update_yaxes(
            gridcolor="rgba(180,80,20,0.07)",
            linecolor="rgba(180,80,20,0.15)",
            tickfont=dict(size=9),
            row=row_i, col=1,
        )

    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP feature contributions ────────────────────────────────────────────
    section_label("Explainable AI — SHAP Feature Contributions")
    shap_done = False
    try:
        import shap
        clf_shap = primary_clf

        if primary_key == "logistic_regression":
            explainer = shap.LinearExplainer(clf_shap, X_live)
        else:
            explainer = shap.TreeExplainer(clf_shap)

        sv = explainer.shap_values(X_live)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.array(sv).flatten()

        shap_s = pd.Series(sv, index=FEATURE_COLS).sort_values()
        colors_shap = [GRN if v > 0 else RED for v in shap_s.values]

        fig_sh = go.Figure(go.Bar(
            x=shap_s.values, y=shap_s.index, orientation="h",
            marker=dict(color=colors_shap),
            text=[f"{v:+.4f}" for v in shap_s.values],
            textposition="outside",
            textfont=dict(color=MUTE, size=9, family="JetBrains Mono"),
        ))
        fig_sh.update_layout(**base_layout("", h=280))
        fig_sh.add_shape(type="line", x0=0, x1=0, y0=-0.5,
                         y1=len(FEATURE_COLS)-0.5,
                         line=dict(color=MUTE, width=1))
        fig_sh.update_xaxes(title_text="← SELL influence    |    BUY influence →",
                             title_font=dict(size=10))
        st.plotly_chart(fig_sh, use_container_width=True)

        st.markdown(f"""
        <div style="font-size:0.76rem;color:{MUTE};line-height:1.8;
                    padding:0.7rem 1rem;background:rgba(28,16,8,0.5);
                    border-radius:8px;border-left:3px solid {OR}">
          <span style="color:{GRN}">■ Green bars</span> pushed today's prediction toward
          <b style="color:{GRN}">BUY</b> &nbsp;·&nbsp;
          <span style="color:{RED}">■ Red bars</span> pushed toward
          <b style="color:{RED}">SELL</b> &nbsp;·&nbsp;
          Bar length = magnitude of that feature's influence.
        </div>
        """, unsafe_allow_html=True)
        shap_done = True

    except ImportError:
        st.info("Add `shap` to requirements.txt to enable SHAP explanations.")
    except Exception as e:
        st.warning(f"SHAP unavailable: {e}")

    if not shap_done:
        # Fallback: show normalised feature values
        section_label("Feature Contributions (SHAP unavailable)")
        norm = {k: v / (abs(v) + 1e-9) for k, v in features.items()}
        norm_s = pd.Series(norm).sort_values()
        fig_fb = go.Figure(go.Bar(
            x=norm_s.values, y=norm_s.index, orientation="h",
            marker_color=[GRN if v > 0 else RED for v in norm_s.values],
            text=[f"{v:+.3f}" for v in norm_s.values],
            textposition="outside",
            textfont=dict(color=MUTE, size=9, family="JetBrains Mono"),
        ))
        fig_fb.update_layout(**base_layout("", h=260))
        st.plotly_chart(fig_fb, use_container_width=True)

with right:
    # ── Confidence gauge ──────────────────────────────────────────────────────
    section_label("Signal Confidence")
    conf_pct = prob_primary * 100
    glass_card(f"""
      <div style="text-align:center;padding:0.5rem 0">
        <div style="font-family:'Playfair Display',serif;font-size:2.8rem;
                    font-weight:700;color:{sig_color};line-height:1">
          {conf_pct:.1f}%
        </div>
        <div style="font-size:0.72rem;color:{MUTE};margin:0.3rem 0 1rem">
          probability of upward move
        </div>
        <div style="background:rgba(58,36,24,0.5);border-radius:6px;
                    height:12px;overflow:hidden;position:relative;margin-bottom:0.4rem">
          <div style="position:absolute;left:50%;top:0;bottom:0;
                      width:2px;background:{MUTE};transform:translateX(-50%)"></div>
          <div style="height:100%;width:{abs(conf_pct-50)*2:.0f}%;
                      background:{sig_color};border-radius:6px;
                      margin-left:{'50%' if conf_pct>50 else str(conf_pct)+'%'}">
          </div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.6rem;color:{MUTE}">
          <span>SELL ◄</span><span>NEUTRAL</span><span>► BUY</span>
        </div>
      </div>
    """)

    # ── AI reasoning ─────────────────────────────────────────────────────────
    section_label("AI Decision Reasoning")
    reasons = explain_signal_text(latest, sig_primary, prob_primary)
    SENT_COLOR = {"bullish": GRN, "bearish": RED, "neutral": MUTE}
    SENT_ICON  = {"bullish": "▲", "bearish": "▼", "neutral": "●"}

    for reason_text, sentiment in reasons:
        sc2 = SENT_COLOR.get(sentiment, MUTE)
        si  = SENT_ICON.get(sentiment, "●")
        st.markdown(f"""
        <div style="display:flex;gap:0.5rem;align-items:flex-start;
                    padding:0.5rem 0.8rem;margin-bottom:0.4rem;
                    background:rgba(28,16,8,0.65);border-radius:8px;
                    border-left:3px solid {sc2}">
          <span style="color:{sc2};font-size:0.72rem;
                       margin-top:0.15rem;flex-shrink:0">{si}</span>
          <span style="font-size:0.74rem;color:{CREAM};
                       font-family:'Outfit',sans-serif;line-height:1.6">
            {reason_text}
          </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Features used ─────────────────────────────────────────────────────────
    section_label("Features Fed to Model")
    feat_meta = {
        "return_1d":     ("1-day return",       "pct"),
        "MA_10":         ("MA (10-day)",         "price"),
        "MA_50":         ("MA (50-day)",         "price"),
        "volatility":    ("Volatility (10d)",    "pct"),
        "volume_change": ("Volume change",       "pct"),
        "RSI":           ("RSI (14)",            "rsi"),
    }
    rows_html = ""
    for col, (label, fmt) in feat_meta.items():
        val = features[col]
        if fmt == "pct":
            display = f"{val*100:+.3f}%"
            color   = GRN if val > 0 else RED
        elif fmt == "rsi":
            display = f"{val:.2f}"
            color   = RED if val > 70 else (GRN if val < 30 else CREAM)
        else:
            display = f"{val:.4f}"
            color   = CREAM
        rows_html += kv(label, display, color)
    glass_card(rows_html, small=True)

    # ── Data info ─────────────────────────────────────────────────────────────
    section_label("Data Info")
    last_date = df.index[-1]
    date_disp = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") \
                else str(last_date)[:10]
    glass_card(f"""
      {kv('Data as of',     date_disp,                              OR2)}
      {kv('Data points',    f"{len(df):,}",                         CREAM)}
      {kv('Source',         'Yahoo Finance (live)',                 GRN)}
      {kv('Model',          model.replace('_',' ').title(),        GOLD)}
      {kv('Retraining',     'Not required — using saved .pkl',     CREAM)}
    """, small=True)