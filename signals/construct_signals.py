import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COUNTRY_ETFS, SIGNAL_WEIGHTS


def zscore(series: pd.Series, window: int = 36) -> pd.Series:
    roll = series.rolling(window)
    return (series - roll.mean()) / roll.std()


def compute_macro_signals(fred_data: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame(index=fred_data.index)

    # Yield curve — steeper = risk on
    signals["yield_curve"] = fred_data["treasury_10y"] - fred_data["treasury_2y"]

    # Fed funds momentum — rising = tightening = risk off
    signals["fed_funds_rate"] = -fred_data["fed_funds_rate"].diff(3)

    # Real yield — higher real yield = tighter = risk off
    signals["real_yield"] = -fred_data["real_yield_10y"]

    # Industrial production momentum
    signals["industrial_production"] = fred_data["industrial_production"].pct_change(3)

    # ISM PMI — above 50 = expansion
    signals["ism_pmi"] = fred_data["ism_pmi"] - 50

    # Jobless claims — falling = good
    signals["jobless_claims"] = -fred_data["jobless_claims"].pct_change(3)
    signals["jobless_claims_chg"] = -fred_data["jobless_claims"].diff(3)

    # Unemployment — falling = good
    signals["unemployment_rate"] = -fred_data["unemployment_rate"].diff(3)

    # CPI trend — rising inflation = risk off
    signals["cpi_trend"] = -fred_data["cpi"].pct_change(12)

    # PPI trend — leading inflation signal
    signals["ppi_trend"] = -fred_data["ppi"].pct_change(12)

    # Breakeven inflation — rising = reflationary
    signals["breakeven_inflation"] = fred_data["breakeven_inflation"].diff(3)

    # HY spreads — tightening = risk on
    signals["hy_spreads"] = -fred_data["hy_spreads"].diff(3)

    # Z-score all signals for comparability
    for col in signals.columns:
        signals[col] = zscore(signals[col])

    return signals


def compute_commodity_signals(commodity_prices: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame(index=commodity_prices.index)

    # Monthly prices
    monthly = commodity_prices.resample("ME").last()

    # Oil momentum (3m)
    signals["oil_momentum"] = monthly["oil"].pct_change(3)

    # Copper momentum (3m) — global growth proxy
    signals["copper_momentum"] = monthly["copper"].pct_change(3)

    # Gold momentum — safe haven signal
    signals["gold_momentum"] = monthly["gold"].pct_change(3)

    # DXY — strong dollar hurts EM, invert
    signals["dxy_momentum"] = -monthly["dxy"].pct_change(3)

    # VIX — high VIX = risk off, invert
    signals["vix_level"] = -monthly["vix"].pct_change(1)

    for col in signals.columns:
        signals[col] = zscore(signals[col])

    return signals


def compute_momentum_signals(etf_prices: pd.DataFrame) -> pd.DataFrame:
    monthly = etf_prices.resample("ME").last()
    signals_12_1 = {}
    signals_3m = {}

    for country, ticker in COUNTRY_ETFS.items():
        if ticker not in monthly.columns:
            continue
        prices = monthly[ticker]
        # 12-1 month momentum (skip last month)
        signals_12_1[ticker] = prices.pct_change(12) - prices.pct_change(1)
        # 3 month momentum
        signals_3m[ticker] = prices.pct_change(3)

    mom_12_1 = pd.DataFrame(signals_12_1)
    mom_3m = pd.DataFrame(signals_3m)

    # Cross-sectional z-score (rank within universe each month)
    mom_12_1 = mom_12_1.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
    mom_3m = mom_3m.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

    return mom_12_1, mom_3m


def build_composite_score(fred_data, commodity_prices, etf_prices) -> pd.DataFrame:
    macro = compute_macro_signals(fred_data)
    commodity = compute_commodity_signals(commodity_prices)
    mom_12_1, mom_3m = compute_momentum_signals(etf_prices)

    # Align all to monthly
    macro = macro.resample("ME").last()
    commodity = commodity.resample("ME").last()

    # Global macro signals apply equally to all countries
    # Commodity signals are tilted by country sensitivity
    commodity_sensitivity = {
        "EWZ": {"oil": 1.5,  "copper": 1.0, "gold": 0.5, "dxy": -1.5},
        "EWJ": {"oil": -1.0, "copper": 0.5, "gold": 0.5, "dxy":  0.5},
        "EWG": {"oil": -0.5, "copper": 1.0, "gold": 0.0, "dxy":  0.5},
        "EWC": {"oil": 1.5,  "copper": 0.5, "gold": 1.0, "dxy": -1.0},
        "EWA": {"oil": 0.5,  "copper": 1.5, "gold": 1.5, "dxy": -1.0},
        "EWU": {"oil": 0.5,  "copper": 0.5, "gold": 0.5, "dxy":  0.0},
        "MCHI":{"oil": -0.5, "copper": 1.5, "gold": 0.5, "dxy": -1.5},
        "EWY": {"oil": -1.0, "copper": 1.0, "gold": 0.0, "dxy": -1.0},
        "EWT": {"oil": -0.5, "copper": 0.5, "gold": 0.0, "dxy": -0.5},
        "EWI": {"oil": -0.5, "copper": 0.5, "gold": 0.0, "dxy":  0.5},
        "EWP": {"oil": -0.5, "copper": 0.5, "gold": 0.0, "dxy":  0.5},
        "INDA":{"oil": -1.0, "copper": 0.5, "gold": 1.0, "dxy": -1.5},
        "SPY": {"oil":  0.0, "copper": 0.5, "gold": 0.0, "dxy":  0.0},
    }

    tickers = list(COUNTRY_ETFS.values())
    all_dates = macro.index.union(commodity.index).union(mom_12_1.index)
    macro = macro.reindex(all_dates).ffill()
    commodity = commodity.reindex(all_dates).ffill()
    mom_12_1 = mom_12_1.reindex(all_dates).ffill()
    mom_3m = mom_3m.reindex(all_dates).ffill()

    composite = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)

    macro_signal_cols = [c for c in macro.columns if c in SIGNAL_WEIGHTS]
    macro_weight_total = sum(SIGNAL_WEIGHTS[c] for c in macro_signal_cols)

    for ticker in tickers:
        sens = commodity_sensitivity.get(ticker, {})

        # Macro component (same for all countries — global regime)
        macro_score = sum(
            macro[col] * SIGNAL_WEIGHTS[col]
            for col in macro_signal_cols
            if col in macro.columns
        ) / macro_weight_total

        # Commodity component (country-specific sensitivity)
        comm_score = (
            commodity.get("oil_momentum", 0) * sens.get("oil", 0) * SIGNAL_WEIGHTS["oil_momentum"] +
            commodity.get("copper_momentum", 0) * sens.get("copper", 0) * SIGNAL_WEIGHTS["copper_momentum"] +
            commodity.get("gold_momentum", 0) * sens.get("gold", 0) * SIGNAL_WEIGHTS["gold_momentum"] +
            commodity.get("dxy_momentum", 0) * sens.get("dxy", 0) * SIGNAL_WEIGHTS["dxy_momentum"] +
            commodity.get("vix_level", 0) * SIGNAL_WEIGHTS["vix_level"] +
            commodity.get("hy_spreads", 0) * SIGNAL_WEIGHTS.get("hy_spreads", 0.05)
        )

        # Momentum component
        mom_score = (
            mom_12_1.get(ticker, pd.Series(0, index=all_dates)) * SIGNAL_WEIGHTS["momentum_12_1"] +
            mom_3m.get(ticker, pd.Series(0, index=all_dates)) * SIGNAL_WEIGHTS["momentum_3m"]
        )

        composite[ticker] = (
            macro_score * macro_weight_total +
            comm_score +
            mom_score
        )

    return composite.dropna(how="all")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fetch_data import fetch_all

    data = fetch_all()
    scores = build_composite_score(
        data["fred_data"],
        data["commodity_prices"],
        data["etf_prices"],
    )
    print("Composite scores (last 3 months):")
    print(scores.tail(3).round(3))
