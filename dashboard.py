import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from strategy import (
    run_all, COUNTRY_ETFS, SENSITIVITY,
    TICKER_TO_COUNTRY, build_scores, build_regime,
    run_backtest, compute_metrics
)

st.set_page_config(
    page_title="Global Macro Strategy",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Global Macro Country Rotation Strategy")
st.markdown("Long/Short Country ETFs | Macro + Commodity + Momentum Signals | 2004-2024")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    run_btn = st.button("Run / Refresh Strategy", type="primary", width="stretch")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data and running backtest (~1 min first time)...")
def load():
    return run_all()

if run_btn or "data" not in st.session_state:
    with st.spinner("Running strategy..."):
        st.session_state["data"] = load()

d = st.session_state["data"]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Macro Signals",
    "Country Scores",
    "Backtest Results",
    "Cost Breakdown",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Macro Signals
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Live Macro Signal Dashboard")
    st.caption(f"Latest reading: {d['signals'].index[-1].strftime('%B %Y')} | Positive = Bullish | Negative = Bearish")

    SIGNAL_LABELS = {
        "cpi":         "CPI Inflation Trend",
        "ppi":         "PPI Inflation Trend",
        "cfnai":       "CFNAI Growth Index",
        "real_yield":  "Real Yield (TIPS)",
        "indust_prod": "Industrial Production",
        "fed_funds":   "Fed Funds Momentum",
        "breakeven":   "Breakeven Inflation",
        "hy_spreads":  "HY Credit Spreads",
        "vix":         "VIX Risk Appetite",
        "oil":         "Oil Momentum",
        "gold":        "Gold Momentum",
        "dxy":         "Dollar Index (DXY)",
    }

    CATEGORIES = {
        "Inflation":        ["cpi", "ppi", "breakeven"],
        "Growth":           ["cfnai", "indust_prod"],
        "Monetary Policy":  ["fed_funds", "real_yield"],
        "Credit & Risk":    ["hy_spreads", "vix"],
        "Commodities":      ["oil", "gold", "dxy"],
    }

    latest = d["signals_norm"].iloc[-1]

    for cat, sigs in CATEGORIES.items():
        st.markdown(f"#### {cat}")
        cols = st.columns(len(sigs))
        for i, sig in enumerate(sigs):
            val = latest.get(sig, np.nan)
            if not np.isnan(val):
                cols[i].metric(
                    SIGNAL_LABELS.get(sig, sig),
                    f"{val:.2f}",
                    delta="Bullish" if val > 0 else "Bearish",
                    delta_color="normal" if val > 0 else "inverse"
                )

    st.divider()
    st.subheader("Signal History")
    cat_select = st.selectbox("Select category", list(CATEGORIES.keys()))
    sigs = [s for s in CATEGORIES[cat_select] if s in d["signals_norm"].columns]

    fig = go.Figure()
    for sig in sigs:
        fig.add_trace(go.Scatter(x=d["signals_norm"].index, y=d["signals_norm"][sig],
                                  name=SIGNAL_LABELS.get(sig, sig), mode="lines"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"{cat_select} Signals Over Time", height=400,
                      xaxis_title="Date", yaxis_title="Z-Score", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw FRED Data (last 24 months)")
        fred_disp = d["macro"].copy()
        fred_disp.index = fred_disp.index.strftime("%Y-%m")
        st.dataframe(fred_disp.tail(24).round(3), width="stretch")
    with col2:
        st.subheader("Commodity Prices (last 24 months)")
        comm_disp = d["commodities"].copy()
        comm_disp.index = comm_disp.index.strftime("%Y-%m")
        st.dataframe(comm_disp.tail(24).round(2), width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Country Scores
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Country Composite Scores")
    st.caption("Higher = more bullish. Top 4 go long, regime determines allocation.")

    latest_scores = d["scores"].iloc[-1].sort_values(ascending=False)
    latest_scores.index = [TICKER_TO_COUNTRY.get(t, t) for t in latest_scores.index]

    colors = ["green" if v > 0 else "red" for v in latest_scores.values]
    fig = go.Figure(go.Bar(
        x=latest_scores.index, y=latest_scores.values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in latest_scores.values],
        textposition="outside",
    ))
    fig.update_layout(title=f"Scores as of {d['scores'].index[-1].strftime('%B %Y')}",
                      height=450, xaxis_title="Country", yaxis_title="Score")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Score History")
    score_hist = d["scores"].copy()
    score_hist.columns = [TICKER_TO_COUNTRY.get(t, t) for t in score_hist.columns]
    fig2 = px.line(score_hist, title="Composite Score Over Time", height=400)
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Current Regime")
    current_regime = d["regime"].iloc[-1]
    regime_color = {"US_Dominant": "🔴", "International": "🟢", "Neutral": "🟡"}
    st.metric("Market Regime", f"{regime_color.get(current_regime, '')} {current_regime}")

    st.subheader("Regime History")
    regime_counts = d["regime"].value_counts()
    fig3 = go.Figure(go.Bar(x=regime_counts.index, y=regime_counts.values,
                             marker_color=["red", "yellow", "green"]))
    fig3.update_layout(title="Regime Distribution (months)", height=300)
    st.plotly_chart(fig3, width="stretch")

    st.subheader("Portfolio Holdings Frequency")
    holding_freq = (d["weights_df"] > 0).mean().sort_values(ascending=False)
    holding_freq.index = [TICKER_TO_COUNTRY.get(t, t) for t in holding_freq.index]
    fig4 = go.Figure(go.Bar(x=holding_freq.index, y=holding_freq.values * 100,
                             marker_color="steelblue"))
    fig4.update_layout(title="% of Months Each Country is Held",
                       height=350, yaxis_title="%")
    st.plotly_chart(fig4, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Backtest Results
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    metrics = d["metrics"]
    net_ret = d["df_returns"]["net"]
    spy_ret = d["spy_ret"]
    cum_str = d["df_returns"]["cum_net"]
    cum_spy = d["cum_spy"]

    st.subheader("Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Return",  metrics["Annual Return"],
                delta=f"SPY: {metrics['SPY Total Return']}")
    col2.metric("Sharpe Ratio",   str(metrics["Sharpe Ratio"]),
                delta=f"SPY: {metrics['SPY Sharpe']}")
    col3.metric("Max Drawdown",   metrics["Max Drawdown"])
    col4.metric("Total Return",   metrics["Total Return"])

    # Cumulative returns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_str.index, y=cum_str.values,
                              name="Strategy", line=dict(color="steelblue", width=2)))
    fig.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy.values,
                              name="S&P 500", line=dict(color="orange", width=2, dash="dash")))
    fig.update_layout(title="Cumulative Returns: Strategy vs S&P 500",
                      height=450, xaxis_title="Date", yaxis_title="Growth of $1")
    st.plotly_chart(fig, width="stretch")

    col_l, col_r = st.columns(2)

    # Drawdown
    roll_max = cum_str.cummax()
    dd = (cum_str - roll_max) / roll_max
    with col_l:
        fig2 = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100,
                                     fill="tozeroy", line=dict(color="red")))
        fig2.update_layout(title="Strategy Drawdown (%)", height=350)
        st.plotly_chart(fig2, width="stretch")

    # Annual returns
    with col_r:
        ann_s = net_ret.resample("YE").apply(lambda x: (1+x).prod()-1) * 100
        ann_b = spy_ret.resample("YE").apply(lambda x: (1+x).prod()-1) * 100
        years = [str(d.year) for d in ann_s.index]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=years, y=ann_s.values, name="Strategy", marker_color="steelblue"))
        fig3.add_trace(go.Bar(x=years, y=ann_b.values, name="S&P 500", marker_color="orange"))
        fig3.update_layout(title="Annual Returns (%)", height=350, barmode="group")
        st.plotly_chart(fig3, width="stretch")

    # Rolling Sharpe
    roll_sharpe = (net_ret.rolling(12).mean() / net_ret.rolling(12).std()) * np.sqrt(12)
    fig4 = go.Figure(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
                                 line=dict(color="green")))
    fig4.add_hline(y=0, line_dash="dash", line_color="red")
    fig4.add_hline(y=1, line_dash="dash", line_color="gray")
    fig4.update_layout(title="Rolling 12-Month Sharpe Ratio", height=350)
    st.plotly_chart(fig4, width="stretch")

    # Returns by regime
    st.subheader("Returns by Regime")
    regime_ret = d["df_returns"].groupby("regime")["net"].mean() * 12
    fig5 = go.Figure(go.Bar(
        x=regime_ret.index, y=regime_ret.values * 100,
        marker_color=["green" if v > 0 else "red" for v in regime_ret.values],
        text=[f"{v:.1%}" for v in regime_ret.values],
        textposition="outside"
    ))
    fig5.update_layout(title="Annualised Return by Market Regime", height=350, yaxis_title="%")
    st.plotly_chart(fig5, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Cost Breakdown
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Cost Breakdown (Annualised)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Gross Return",      metrics["Gross Return"])
    col2.metric("Transaction Costs", metrics["Transaction Costs"])
    col3.metric("Net Return",        metrics["Annual Return"])

    gross = d["df_returns"]["gross"].mean() * 12 * 100
    tx    = d["df_returns"]["tx_cost"].mean() * 12 * 100
    net   = d["df_returns"]["net"].mean() * 12 * 100

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Gross Return", "Transaction Costs", "Net Return"],
        y=[gross, -tx, 0],
        decreasing={"marker": {"color": "red"}},
        increasing={"marker": {"color": "green"}},
        totals={"marker": {"color": "steelblue"}},
    ))
    fig.update_layout(title="Return Waterfall (%)", height=400, yaxis_title="%")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Monthly Transaction Costs Over Time")
    fig2 = go.Figure(go.Bar(
        x=d["df_returns"].index,
        y=d["df_returns"]["tx_cost"] * 100,
        marker_color="red", name="Transaction Costs"
    ))
    fig2.update_layout(title="Monthly Transaction Costs (%)", height=350, yaxis_title="%")
    st.plotly_chart(fig2, width="stretch")
