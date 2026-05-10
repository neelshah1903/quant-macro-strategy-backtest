import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
FRED_API_KEY = "87c04efcdf543072107e53f957356b52"

COUNTRY_ETFS = {
    "USA":        "SPY",
    "Brazil":     "EWZ",
    "Japan":      "EWJ",
    "Germany":    "EWG",
    "Canada":     "EWC",
    "Australia":  "EWA",
    "UK":         "EWU",
    "China":      "MCHI",
    "SouthKorea": "EWY",
    "Taiwan":     "EWT",
    "Italy":      "EWI",
    "Spain":      "EWP",
    "India":      "INDA",
}

COMMODITIES = {
    "oil":    "CL=F",
    "copper": "HG=F",
    "gold":   "GC=F",
    "dxy":    "DX-Y.NYB",
    "vix":    "^VIX",
}

FRED_SERIES = {
    "fed_funds":      "FEDFUNDS",
    "treasury_10y":   "GS10",
    "treasury_2y":    "GS2",
    "real_yield":     "DFII10",
    "cpi":            "CPIAUCSL",
    "ppi":            "PPIACO",
    "breakeven":      "T10YIE",
    "unemployment":   "UNRATE",
    "jobless_claims": "ICSA",
    "indust_prod":    "INDPRO",
    "cfnai":          "CFNAI",
    "hy_spreads":     "BAMLH0A0HYM2",
}

SENSITIVITY = {
    "SPY":  {"oil":  0.0, "copper":  0.5, "gold":  0.0, "dxy":  0.0},
    "EWZ":  {"oil":  1.5, "copper":  1.0, "gold":  0.5, "dxy": -1.5},
    "EWJ":  {"oil": -1.0, "copper":  0.5, "gold":  0.5, "dxy":  0.5},
    "EWG":  {"oil": -0.5, "copper":  1.0, "gold":  0.0, "dxy":  0.5},
    "EWC":  {"oil":  1.5, "copper":  0.5, "gold":  1.0, "dxy": -1.0},
    "EWA":  {"oil":  0.5, "copper":  1.5, "gold":  1.5, "dxy": -1.0},
    "EWU":  {"oil":  0.5, "copper":  0.5, "gold":  0.5, "dxy":  0.0},
    "MCHI": {"oil": -0.5, "copper":  1.5, "gold":  0.5, "dxy": -1.5},
    "EWY":  {"oil": -1.0, "copper":  1.0, "gold":  0.0, "dxy": -1.0},
    "EWT":  {"oil": -0.5, "copper":  0.5, "gold":  0.0, "dxy": -0.5},
    "EWI":  {"oil": -0.5, "copper":  0.5, "gold":  0.0, "dxy":  0.5},
    "EWP":  {"oil": -0.5, "copper":  0.5, "gold":  0.0, "dxy":  0.5},
    "INDA": {"oil": -1.0, "copper":  0.5, "gold":  1.0, "dxy": -1.5},
}

START_DATE = "2004-01-01"
END_DATE   = "2024-12-31"
TOP_N      = 4
COMMISSION = 0.001
SLIPPAGE   = 0.001
BORROW_LIQUID = 0.003
BORROW_EM     = 0.0075
EM_ETFS    = {"EWZ", "MCHI", "EWY", "EWT", "INDA"}

TICKER_TO_COUNTRY = {v: k for k, v in COUNTRY_ETFS.items()}


# ── Data Fetching ─────────────────────────────────────────────────────────────
def fetch_data():
    tickers = list(COUNTRY_ETFS.values())
    prices = yf.download(tickers, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)["Close"]
    monthly_prices  = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change()

    comm_raw = yf.download(list(COMMODITIES.values()), start=START_DATE,
                            end=END_DATE, auto_adjust=True, progress=False)["Close"]
    comm_raw.columns = list(COMMODITIES.keys())
    commodities = comm_raw.resample("ME").last().ffill()

    fred = Fred(api_key=FRED_API_KEY)
    macro = {}
    for name, code in FRED_SERIES.items():
        try:
            macro[name] = fred.get_series(code, observation_start=START_DATE,
                                           observation_end=END_DATE)
        except:
            pass
    macro = pd.DataFrame(macro)
    macro.index = pd.to_datetime(macro.index)
    macro = macro.resample("ME").last().ffill()

    return monthly_prices, monthly_returns, commodities, macro


# ── Signal Construction ───────────────────────────────────────────────────────
def zscore(series, window=36):
    roll = series.rolling(window, min_periods=12)
    return (series - roll.mean()) / roll.std()


def build_signals(macro, commodities):
    signals = pd.DataFrame(index=macro.index)

    signals["cpi"]         = zscore(-macro["cpi"].pct_change(12))
    signals["ppi"]         = zscore(-macro["ppi"].pct_change(12))
    signals["cfnai"]       = zscore(macro["cfnai"])
    signals["real_yield"]  = zscore(-macro["real_yield"])
    signals["indust_prod"] = zscore(macro["indust_prod"].pct_change(3))
    signals["fed_funds"]   = zscore(-macro["fed_funds"].diff(3))
    signals["breakeven"]   = zscore(macro["breakeven"].diff(3))
    signals["hy_spreads"]  = zscore(macro["hy_spreads"].diff(3))
    signals["vix"]         = zscore(commodities["vix"])
    signals["oil"]         = zscore(commodities["oil"].pct_change(3))
    signals["gold"]        = zscore(commodities["gold"].pct_change(3))
    signals["dxy"]         = zscore(commodities["dxy"].pct_change(3))

    signals_norm = signals.sub(signals.mean()).div(signals.std())
    return signals, signals_norm


# ── Country Scoring ───────────────────────────────────────────────────────────
def build_scores(signals_norm, monthly_returns):
    macro_cols = ["cpi", "ppi", "cfnai", "real_yield", "indust_prod",
                  "fed_funds", "breakeven", "hy_spreads", "vix"]

    scores = pd.DataFrame(index=signals_norm.index,
                          columns=list(SENSITIVITY.keys()), dtype=float)

    for date in signals_norm.index:
        s    = signals_norm.loc[date]
        base = s[macro_cols].mean()

        for ticker, sens in SENSITIVITY.items():
            comm = (s["oil"]  * sens["oil"]  +
                    s["gold"] * sens["gold"] +
                    s["dxy"]  * sens["dxy"])

            mom = 0
            if date in monthly_returns.index and ticker in monthly_returns.columns:
                idx = monthly_returns.index.get_loc(date)
                if idx >= 12:
                    mom = monthly_returns[ticker].iloc[idx-12:idx-1].add(1).prod() - 1

            scores.loc[date, ticker] = base + 0.3 * comm + 0.5 * mom

    return scores


# ── Regime Detection ──────────────────────────────────────────────────────────
def build_regime(monthly_prices):
    spy_12m  = monthly_prices["SPY"].pct_change(12)
    intl     = [t for t in list(SENSITIVITY.keys()) if t != "SPY"]
    intl_12m = monthly_prices[intl].mean(axis=1).pct_change(12)
    us_dom   = spy_12m - intl_12m

    regime = pd.Series("Neutral", index=monthly_prices.index)
    regime[us_dom >  0.10] = "US_Dominant"
    regime[us_dom < -0.10] = "International"
    return regime


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(scores, monthly_returns, monthly_prices, macro, regime):
    portfolio_returns = []
    weights_history  = []
    tickers = list(SENSITIVITY.keys())

    for i in range(1, len(scores)):
        date      = scores.index[i]
        prev_date = scores.index[i-1]

        row = scores.loc[prev_date].dropna()
        if len(row) < TOP_N:
            continue

        current_regime = regime.loc[prev_date] if prev_date in regime.index else "Neutral"
        target = pd.Series(0.0, index=tickers)

        if current_regime == "US_Dominant":
            target["SPY"] = 1.0

        elif current_regime == "International":
            row_ex = row.drop("SPY", errors="ignore")
            ranked = row_ex.rank(ascending=False)
            longs  = ranked[ranked <= TOP_N].index.tolist()
            target[longs] = 1.0 / TOP_N

        else:
            target["SPY"] = 0.50
            row_ex = row.drop("SPY", errors="ignore")
            ranked = row_ex.rank(ascending=False)
            longs  = ranked[ranked <= 3].index.tolist()
            target[longs] = 0.50 / 3

        if date not in monthly_returns.index:
            continue

        ret      = monthly_returns.loc[date]
        gross    = (target * ret).sum()
        prev_w   = weights_history[-1]["weights"] if weights_history else pd.Series(0.0, index=tickers)
        turnover = (target - prev_w).abs().sum()
        tx_cost  = turnover * (COMMISSION + SLIPPAGE)
        net      = gross - tx_cost

        portfolio_returns.append({"date": date, "gross": gross,
                                   "tx_cost": tx_cost, "net": net,
                                   "regime": current_regime})
        weights_history.append({"date": date, "weights": target.copy()})

    df = pd.DataFrame(portfolio_returns).set_index("date")
    df["cum_net"]   = (1 + df["net"]).cumprod()
    df["cum_gross"] = (1 + df["gross"]).cumprod()

    spy_ret = monthly_returns["SPY"].reindex(df.index)
    cum_spy = (1 + spy_ret).cumprod()

    weights_df = pd.DataFrame([x["weights"] for x in weights_history],
                               index=[x["date"] for x in weights_history])

    return df, spy_ret, cum_spy, weights_df


# ── Performance Metrics ───────────────────────────────────────────────────────
def compute_metrics(df_returns, spy_ret, macro):
    rf     = macro["fed_funds"].reindex(df_returns.index).ffill() / 100 / 12
    excess = df_returns["net"] - rf
    sharpe = excess.mean() / excess.std() * np.sqrt(12)

    cum    = df_returns["cum_net"]
    max_dd = ((cum / cum.cummax()) - 1).min()
    total  = cum.iloc[-1] - 1

    spy_excess = spy_ret - rf
    spy_sharpe = spy_excess.mean() / spy_excess.std() * np.sqrt(12)
    spy_total  = (1 + spy_ret).cumprod().iloc[-1] - 1

    return {
        "Annual Return":      f"{df_returns['net'].mean()*12:.2%}",
        "Gross Return":       f"{df_returns['gross'].mean()*12:.2%}",
        "Transaction Costs":  f"{df_returns['tx_cost'].mean()*12:.2%}",
        "Sharpe Ratio":       round(sharpe, 2),
        "Max Drawdown":       f"{max_dd:.2%}",
        "Total Return":       f"{total:.2%}",
        "SPY Sharpe":         round(spy_sharpe, 2),
        "SPY Total Return":   f"{spy_total:.2%}",
    }


# ── Run Everything ────────────────────────────────────────────────────────────
def run_all():
    print("Fetching data...")
    monthly_prices, monthly_returns, commodities, macro = fetch_data()

    print("Building signals...")
    signals, signals_norm = build_signals(macro, commodities)

    print("Scoring countries...")
    scores = build_scores(signals_norm, monthly_returns)

    print("Detecting regimes...")
    regime = build_regime(monthly_prices)

    print("Running backtest...")
    df_returns, spy_ret, cum_spy, weights_df = run_backtest(
        scores, monthly_returns, monthly_prices, macro, regime)

    metrics = compute_metrics(df_returns, spy_ret, macro)

    return {
        "monthly_prices":  monthly_prices,
        "monthly_returns": monthly_returns,
        "commodities":     commodities,
        "macro":           macro,
        "signals":         signals,
        "signals_norm":    signals_norm,
        "scores":          scores,
        "regime":          regime,
        "df_returns":      df_returns,
        "spy_ret":         spy_ret,
        "cum_spy":         cum_spy,
        "weights_df":      weights_df,
        "metrics":         metrics,
    }


if __name__ == "__main__":
    results = run_all()
    print("\n=== Performance ===")
    for k, v in results["metrics"].items():
        print(f"  {k:<20} {v}")
