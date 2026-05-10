import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COUNTRY_ETFS, SIGNAL_WEIGHTS


def zscore(series: pd.Series, window: int = 36) -> pd.Series:
    roll = series.rolling(window, min_periods=12)
    return (series - roll.mean()) / roll.std()


def crosssectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row * 0, axis=1)


def compute_macro_regime(fred_data: pd.DataFrame) -> pd.DataFrame:
    m = fred_data.copy().resample("ME").last().ffill()
    signals = pd.DataFrame(index=m.index)

    if "treasury_10y" in m.columns and "treasury_2y" in m.columns:
        signals["yield_curve"] = zscore(m["treasury_10y"] - m["treasury_2y"])

    if "fed_funds_rate" in m.columns:
        signals["fed_funds_rate"] = zscore(-m["fed_funds_rate"].diff(3))

    if "real_yield_10y" in m.columns:
        signals["real_yield"] = zscore(-m["real_yield_10y"])

    if "industrial_production" in m.columns:
        signals["industrial_production"] = zscore(m["industrial_production"].pct_change(3))

    if "cfnai" in m.columns:
        signals["ism_pmi"] = zscore(m["cfnai"])

    if "jobless_claims" in m.columns:
        signals["jobless_claims"] = zscore(-m["jobless_claims"].pct_change(3))
        signals["jobless_claims_chg"] = zscore(-m["jobless_claims"].diff(3))

    if "unemployment_rate" in m.columns:
        signals["unemployment_rate"] = zscore(-m["unemployment_rate"].diff(3))

    if "cpi" in m.columns:
        signals["cpi_trend"] = zscore(-m["cpi"].pct_change(12))

    if "ppi" in m.columns:
        signals["ppi_trend"] = zscore(-m["ppi"].pct_change(12))

    if "breakeven_inflation" in m.columns:
        signals["breakeven_inflation"] = zscore(m["breakeven_inflation"].diff(3))

    if "hy_spreads" in m.columns:
        signals["hy_spreads"] = zscore(-m["hy_spreads"].diff(3))

    return signals


def compute_commodity_signals(commodity_prices: pd.DataFrame) -> pd.DataFrame:
    m = commodity_prices.resample("ME").last().ffill()
    signals = pd.DataFrame(index=m.index)

    if "oil" in m.columns:
        signals["oil_momentum"] = zscore(m["oil"].pct_change(3))
    if "copper" in m.columns:
        signals["copper_momentum"] = zscore(m["copper"].pct_change(3))
    if "gold" in m.columns:
        signals["gold_momentum"] = zscore(m["gold"].pct_change(3))
    if "dxy" in m.columns:
        signals["dxy_momentum"] = zscore(-m["dxy"].pct_change(3))
    if "vix" in m.columns:
        signals["vix_level"] = zscore(-m["vix"])

    return signals


COMMODITY_SENSITIVITY = {
    "EWZ":  {"oil": 1.5,  "copper": 1.0, "gold": 0.5,  "dxy": -1.5, "vix": -1.0},
    "EWJ":  {"oil": -1.0, "copper": 0.5, "gold": 0.5,  "dxy":  0.5, "vix": -0.5},
    "EWG":  {"oil": -0.5, "copper": 1.0, "gold": 0.0,  "dxy":  0.5, "vix": -0.5},
    "EWC":  {"oil": 1.5,  "copper": 0.5, "gold": 1.0,  "dxy": -1.0, "vix": -0.5},
    "EWA":  {"oil": 0.5,  "copper": 1.5, "gold": 1.5,  "dxy": -1.0, "vix": -0.5},
    "EWU":  {"oil": 0.5,  "copper": 0.5, "gold": 0.5,  "dxy":  0.0, "vix": -0.5},
    "MCHI": {"oil": -0.5, "copper": 1.5, "gold": 0.5,  "dxy": -1.5, "vix": -1.0},
    "EWY":  {"oil": -1.0, "copper": 1.0, "gold": 0.0,  "dxy": -1.0, "vix": -1.0},
    "EWT":  {"oil": -0.5, "copper": 0.5, "gold": 0.0,  "dxy": -0.5, "vix": -1.0},
    "EWI":  {"oil": -0.5, "copper": 0.5, "gold": 0.0,  "dxy":  0.5, "vix": -0.5},
    "EWP":  {"oil": -0.5, "copper": 0.5, "gold": 0.0,  "dxy":  0.5, "vix": -0.5},
    "INDA": {"oil": -1.0, "copper": 0.5, "gold": 1.0,  "dxy": -1.5, "vix": -1.0},
    "SPY":  {"oil":  0.0, "copper": 0.5, "gold": 0.0,  "dxy":  0.0, "vix": -0.5},
}

COMM_SIGNAL_MAP = {
    "oil":    "oil_momentum",
    "copper": "copper_momentum",
    "gold":   "gold_momentum",
    "dxy":    "dxy_momentum",
    "vix":    "vix_level",
}


def build_composite_score(fred_data, commodity_prices, etf_prices) -> pd.DataFrame:
    macro   = compute_macro_regime(fred_data)
    comm    = compute_commodity_signals(commodity_prices)
    monthly = etf_prices.resample("ME").last()

    # Momentum signals (cross-sectional z-score)
    mom_12_1 = crosssectional_zscore(monthly.pct_change(12) - monthly.pct_change(1))
    mom_3m   = crosssectional_zscore(monthly.pct_change(3))

    # Common date index (monthly)
    dates = macro.index
    tickers = list(COUNTRY_ETFS.values())

    composite = pd.DataFrame(index=dates, columns=tickers, dtype=float)

    # Macro weight total for normalisation
    macro_cols = [c for c in macro.columns if c in SIGNAL_WEIGHTS]
    macro_w    = sum(SIGNAL_WEIGHTS[c] for c in macro_cols)

    for date in dates:
        macro_row = macro.loc[date]
        comm_row  = comm.reindex([date]).iloc[0] if date in comm.index else pd.Series(dtype=float)

        # Macro regime score (same for all countries)
        macro_score = sum(
            macro_row.get(c, np.nan) * SIGNAL_WEIGHTS[c]
            for c in macro_cols
            if not np.isnan(macro_row.get(c, np.nan))
        )
        macro_norm = macro_score / macro_w if macro_w > 0 else 0.0

        for ticker in tickers:
            sens = COMMODITY_SENSITIVITY.get(ticker, {})

            # Commodity component (country-specific sensitivity)
            comm_score = 0.0
            for asset, sig_col in COMM_SIGNAL_MAP.items():
                if sig_col in comm_row.index and not np.isnan(comm_row.get(sig_col, np.nan)):
                    comm_score += comm_row[sig_col] * sens.get(asset, 0.0) * SIGNAL_WEIGHTS.get(sig_col, 0.0)

            # Momentum
            mom_score = 0.0
            if date in mom_12_1.index and ticker in mom_12_1.columns:
                v = mom_12_1.loc[date, ticker]
                if not np.isnan(v):
                    mom_score += v * SIGNAL_WEIGHTS["momentum_12_1"]
            if date in mom_3m.index and ticker in mom_3m.columns:
                v = mom_3m.loc[date, ticker]
                if not np.isnan(v):
                    mom_score += v * SIGNAL_WEIGHTS["momentum_3m"]

            composite.loc[date, ticker] = macro_norm + comm_score + mom_score

    composite = composite.apply(pd.to_numeric, errors="coerce")
    return composite.dropna(how="all")


if __name__ == "__main__":
    from data.fetch_data import fetch_all
    data = fetch_all()
    scores = build_composite_score(data["fred_data"], data["commodity_prices"], data["etf_prices"])
    print("Score shape:", scores.shape)
    print(scores.tail(3).round(3))
