import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Global Macro Strategy",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Global Macro Country Rotation Strategy")
st.markdown("Long/Short Country ETFs | Macro + Commodity + Momentum Signals")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_n    = st.slider("Long top N countries",   1, 6, 4)
    bottom_n = st.slider("Short bottom N countries", 1, 6, 4)
    vol_target = st.slider("Vol target (%)", 5, 20, 10) / 100
    run_btn  = st.button("Run Backtest", type="primary", width="stretch")

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching data (this takes ~1 min first time)...")
def load_data():
    from data.fetch_data import fetch_all
    return fetch_all()

@st.cache_data(show_spinner="Building signals...")
def load_scores(_data):
    from signals.construct_signals import build_composite_score
    return build_composite_score(_data["fred_data"], _data["commodity_prices"], _data["etf_prices"])

def run_backtest_cached(scores, etf_prices, fred_data, top_n, bottom_n, vol_target):
    from config import COUNTRY_ETFS, REBALANCE_FREQ, MAX_POSITION, MAX_NET_EXP
    from config import COMMISSION_PER_SHARE, SLIPPAGE_PCT, SHORT_BORROW_LIQUID, SHORT_BORROW_EM, MARGIN_SPREAD, EM_ETFS
    from backtest.engine import run_backtest, compute_metrics
    import config
    config.TOP_N    = top_n
    config.BOTTOM_N = bottom_n
    config.VOL_TARGET = vol_target
    results = run_backtest(scores, etf_prices, fred_data)
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
data = load_data()
scores = load_scores(data)

from config import COUNTRY_ETFS

country_names = {v: k for k, v in COUNTRY_ETFS.items()}

# ── Tab layout ────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Macro Signals",
    "Signal Scores",
    "Portfolio Weights",
    "Backtest Results",
    "Cost Breakdown",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — Macro Signals
# ─────────────────────────────────────────────────────────────────────────────
with tab0:
    st.subheader("Live Macro Signal Dashboard")
    st.caption("All signals z-scored. Green = bullish, Red = bearish for global risk appetite.")

    from signals.construct_signals import compute_macro_regime, compute_commodity_signals
    macro_signals  = compute_macro_regime(data["fred_data"])
    comm_signals   = compute_commodity_signals(data["commodity_prices"])

    SIGNAL_LABELS = {
        "yield_curve":           "Yield Curve (10Y-2Y)",
        "fed_funds_rate":        "Fed Funds Rate Momentum",
        "real_yield":            "Real Yield (TIPS)",
        "industrial_production": "Industrial Production",
        "ism_pmi":               "CFNAI (Growth Index)",
        "jobless_claims":        "Jobless Claims",
        "jobless_claims_chg":    "Jobless Claims Change",
        "unemployment_rate":     "Unemployment Rate",
        "cpi_trend":             "CPI Inflation Trend",
        "ppi_trend":             "PPI Inflation Trend",
        "breakeven_inflation":   "Breakeven Inflation",
        "hy_spreads":            "HY Credit Spreads",
        "oil_momentum":          "Oil Price Momentum",
        "copper_momentum":       "Copper Momentum",
        "gold_momentum":         "Gold Momentum",
        "dxy_momentum":          "Dollar Index (DXY)",
        "vix_level":             "VIX (Risk Appetite)",
    }

    SIGNAL_CATEGORIES = {
        "Monetary Policy":  ["yield_curve", "fed_funds_rate", "real_yield"],
        "Growth":           ["industrial_production", "ism_pmi"],
        "Labor Market":     ["unemployment_rate", "jobless_claims", "jobless_claims_chg"],
        "Inflation":        ["cpi_trend", "ppi_trend", "breakeven_inflation"],
        "Credit/Risk":      ["hy_spreads"],
        "Commodities":      ["oil_momentum", "copper_momentum", "gold_momentum"],
        "Dollar & Risk":    ["dxy_momentum", "vix_level"],
    }

    # Current signal snapshot
    all_signals = pd.concat([macro_signals, comm_signals], axis=1).resample("ME").last()
    latest_signals = all_signals.iloc[-1]

    st.markdown(f"**Latest reading: {all_signals.index[-1].strftime('%B %Y')}**")

    for category, signal_list in SIGNAL_CATEGORIES.items():
        st.markdown(f"#### {category}")
        cols = st.columns(len(signal_list))
        for i, sig in enumerate(signal_list):
            if sig in latest_signals.index:
                val = latest_signals[sig]
                label = SIGNAL_LABELS.get(sig, sig)
                if not np.isnan(val):
                    delta_color = "normal" if val > 0 else "inverse"
                    cols[i].metric(label, f"{val:.2f}", delta="Bullish" if val > 0 else "Bearish", delta_color=delta_color)
                else:
                    cols[i].metric(label, "N/A")

    st.divider()

    # Signal history charts
    st.subheader("Signal History Charts")
    selected_category = st.selectbox("Select category", list(SIGNAL_CATEGORIES.keys()))
    selected_signals  = SIGNAL_CATEGORIES[selected_category]
    available = [s for s in selected_signals if s in all_signals.columns]

    if available:
        fig = go.Figure()
        for sig in available:
            fig.add_trace(go.Scatter(
                x=all_signals.index,
                y=all_signals[sig],
                name=SIGNAL_LABELS.get(sig, sig),
                mode="lines",
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title=f"{selected_category} Signals Over Time",
            height=400,
            xaxis_title="Date",
            yaxis_title="Z-Score",
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # Raw FRED data
    st.subheader("Raw FRED Data")
    fred_display = data["fred_data"].copy()
    fred_display.index = fred_display.index.strftime("%Y-%m")
    st.dataframe(fred_display.tail(24).round(3), width="stretch")

    # Raw commodity data
    st.subheader("Raw Commodity Prices")
    comm_display = data["commodity_prices"].resample("ME").last().copy()
    comm_display.index = comm_display.index.strftime("%Y-%m")
    st.dataframe(comm_display.tail(24).round(2), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Signal Scores
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Current Signal Scores by Country")
    st.caption("Higher = more bullish. Positive = long candidate. Negative = short candidate.")

    latest = scores.dropna(how="all").iloc[-1].sort_values(ascending=False)
    latest.index = [country_names.get(t, t) for t in latest.index]

    colors = ["green" if v > 0 else "red" for v in latest.values]
    fig = go.Figure(go.Bar(
        x=latest.index,
        y=latest.values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in latest.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Signal Scores as of {scores.index[-1].strftime('%B %Y')}",
        xaxis_title="Country",
        yaxis_title="Composite Score",
        height=450,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Score History")
    score_hist = scores.copy()
    score_hist.columns = [country_names.get(t, t) for t in score_hist.columns]
    fig2 = px.line(score_hist, title="Composite Score Over Time", height=400)
    fig2.update_layout(xaxis_title="Date", yaxis_title="Score")
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Raw Scores Table (Last 12 months)")
    display = score_hist.tail(12).round(3)
    st.dataframe(display, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Portfolio Weights
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Current Portfolio Positions")

    if run_btn or "results" in st.session_state:
        if run_btn:
            with st.spinner("Running backtest..."):
                st.session_state["results"] = run_backtest_cached(
                    scores, data["etf_prices"], data["fred_data"], top_n, bottom_n, vol_target
                )
        results = st.session_state["results"]
        weights = results["weights"]
        latest_w = weights.dropna(how="all").iloc[-1]
        latest_w.index = [country_names.get(t, t) for t in latest_w.index]
        latest_w = latest_w[latest_w != 0].sort_values()

        colors = ["red" if v < 0 else "steelblue" for v in latest_w.values]
        fig = go.Figure(go.Bar(
            x=latest_w.values * 100,
            y=latest_w.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in latest_w.values * 100],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Positions as of {weights.index[-1].strftime('%B %Y')}",
            xaxis_title="Weight (%)",
            height=450,
        )
        st.plotly_chart(fig, width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("Gross Exposure", f"{weights.iloc[-1].abs().sum():.1%}")
        col2.metric("Net Exposure",   f"{weights.iloc[-1].sum():.1%}")
        col3.metric("# Positions",    f"{(weights.iloc[-1] != 0).sum()}")

        st.subheader("Weight History")
        w_hist = results["weights"].copy()
        w_hist.columns = [country_names.get(t, t) for t in w_hist.columns]
        fig3 = px.area(w_hist * 100, title="Portfolio Weights Over Time (%)", height=400)
        st.plotly_chart(fig3, width="stretch")
    else:
        st.info("Click **Run Backtest** in the sidebar to see portfolio weights.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Backtest Results
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if "results" in st.session_state:
        results = st.session_state["results"]
        from backtest.engine import compute_metrics

        net_ret = results["net_returns"].dropna()
        spy_ret = results["spy_returns"].dropna()
        net_ret, spy_ret = net_ret.align(spy_ret, join="inner")

        cum_strat = (1 + net_ret).cumprod()
        cum_spy   = (1 + spy_ret).cumprod()

        # KPI cards
        metrics     = compute_metrics(net_ret, spy_ret)
        spy_metrics = compute_metrics(spy_ret)

        st.subheader("Performance Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Annual Return",  metrics["Annual Return"],
                    delta=f"{float(metrics['Annual Return'].strip('%')) - float(spy_metrics['Annual Return'].strip('%')):.1f}% vs SPY")
        col2.metric("Sharpe Ratio",   metrics["Sharpe Ratio"],
                    delta=f"{float(metrics['Sharpe Ratio']) - float(spy_metrics['Sharpe Ratio']):.2f} vs SPY")
        col3.metric("Max Drawdown",   metrics["Max Drawdown"])
        col4.metric("Alpha",          metrics["Alpha"])
        col5.metric("Beta",           metrics["Beta"])

        # Cumulative returns chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_strat.index, y=cum_strat.values, name="Strategy", line=dict(color="steelblue", width=2)))
        fig.add_trace(go.Scatter(x=cum_spy.index,   y=cum_spy.values,   name="S&P 500",  line=dict(color="orange", width=2, dash="dash")))
        fig.update_layout(title="Cumulative Returns: Strategy vs S&P 500", height=450,
                          xaxis_title="Date", yaxis_title="Growth of $1")
        st.plotly_chart(fig, width="stretch")

        col_l, col_r = st.columns(2)

        # Drawdown
        roll_max = cum_strat.cummax()
        dd = (cum_strat - roll_max) / roll_max
        with col_l:
            fig2 = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, fill="tozeroy",
                                         line=dict(color="red"), name="Drawdown"))
            fig2.update_layout(title="Strategy Drawdown (%)", height=350,
                                xaxis_title="Date", yaxis_title="%")
            st.plotly_chart(fig2, width="stretch")

        # Annual returns bar chart
        with col_r:
            ann_strat = net_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
            ann_spy   = spy_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
            years = [str(d.year) for d in ann_strat.index]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=years, y=ann_strat.values, name="Strategy", marker_color="steelblue"))
            fig3.add_trace(go.Bar(x=years, y=ann_spy.values,   name="S&P 500",  marker_color="orange"))
            fig3.update_layout(title="Annual Returns (%)", height=350,
                                barmode="group", xaxis_title="Year", yaxis_title="%")
            st.plotly_chart(fig3, width="stretch")

        # Rolling Sharpe
        roll_sharpe = (net_ret.rolling(12).mean() / net_ret.rolling(12).std()) * np.sqrt(12)
        fig4 = go.Figure(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
                                     line=dict(color="green"), name="Rolling Sharpe"))
        fig4.add_hline(y=0, line_dash="dash", line_color="red")
        fig4.add_hline(y=1, line_dash="dash", line_color="gray")
        fig4.update_layout(title="Rolling 12-Month Sharpe Ratio", height=350)
        st.plotly_chart(fig4, width="stretch")

    else:
        st.info("Click **Run Backtest** in the sidebar to see results.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Cost Breakdown
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    if "results" in st.session_state:
        results = st.session_state["results"]
        gross   = results["gross_returns"].dropna()
        tx      = results["transaction_costs"].dropna()
        hc      = results["holding_costs"].dropna()
        net     = results["net_returns"].dropna()

        st.subheader("Cost Breakdown (Annualised)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Gross Return",       f"{gross.mean()*12:.2%}")
        col2.metric("Transaction Costs",  f"-{tx.mean()*12:.2%}")
        col3.metric("Holding Costs",      f"-{hc.mean()*12:.2%}")
        col4.metric("Net Return",         f"{net.mean()*12:.2%}")

        # Waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Returns",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Gross Return", "Transaction Costs", "Holding Costs", "Net Return"],
            y=[gross.mean()*12*100, -tx.mean()*12*100, -hc.mean()*12*100, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "steelblue"}},
        ))
        fig.update_layout(title="Return Waterfall (%)", height=400, yaxis_title="%")
        st.plotly_chart(fig, width="stretch")

        # Monthly cost history
        cost_df = pd.DataFrame({
            "Transaction Costs": tx * 100,
            "Holding Costs":     hc * 100,
        })
        fig2 = px.bar(cost_df, title="Monthly Costs Over Time (%)", height=350, barmode="stack")
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("Click **Run Backtest** in the sidebar to see cost breakdown.")
