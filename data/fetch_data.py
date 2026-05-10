import yfinance as yf
import pandas as pd
from fredapi import Fred
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRED_API_KEY, COUNTRY_ETFS, BENCHMARK, START_DATE, END_DATE

fred = Fred(api_key=FRED_API_KEY)

FRED_SERIES = {
    "fed_funds_rate":        "FEDFUNDS",
    "treasury_10y":          "GS10",
    "treasury_2y":           "GS2",
    "real_yield_10y":        "DFII10",
    "cpi":                   "CPIAUCSL",
    "ppi":                   "PPIACO",
    "breakeven_inflation":   "T10YIE",
    "unemployment_rate":     "UNRATE",
    "jobless_claims":        "ICSA",
    "industrial_production": "INDPRO",
    "cfnai":                 "CFNAI",
    "hy_spreads":            "BAMLH0A0HYM2",
    "baa_spread":            "BAA",
}

COMMODITY_TICKERS = {
    "oil":    "CL=F",
    "brent":  "BZ=F",
    "copper": "HG=F",
    "gold":   "GC=F",
    "dxy":    "DX-Y.NYB",
    "vix":    "^VIX",
}


def fetch_etf_prices() -> pd.DataFrame:
    tickers = list(COUNTRY_ETFS.values())
    if BENCHMARK not in tickers:
        tickers.append(BENCHMARK)
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    prices = raw["Close"]
    prices.index = pd.to_datetime(prices.index)
    return prices


def fetch_commodity_prices() -> pd.DataFrame:
    tickers = list(COMMODITY_TICKERS.values())
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    prices = raw["Close"]
    prices.columns = list(COMMODITY_TICKERS.keys())
    prices.index = pd.to_datetime(prices.index)
    return prices


def fetch_fred_data() -> pd.DataFrame:
    series = {}
    for name, code in FRED_SERIES.items():
        try:
            s = fred.get_series(code, observation_start=START_DATE, observation_end=END_DATE)
            series[name] = s
            print(f"  fetched {name} ({code})")
        except Exception as e:
            print(f"  WARN: could not fetch {name} ({code}): {e}")
    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index)
    df = df.resample("ME").last().ffill()
    return df


def fetch_all() -> dict:
    print("Fetching ETF prices...")
    etf_prices = fetch_etf_prices()

    print("Fetching commodity prices...")
    commodity_prices = fetch_commodity_prices()

    print("Fetching FRED macro data...")
    fred_data = fetch_fred_data()

    return {
        "etf_prices":        etf_prices,
        "commodity_prices":  commodity_prices,
        "fred_data":         fred_data,
    }


if __name__ == "__main__":
    data = fetch_all()
    print("\nData shapes:")
    for k, v in data.items():
        print(f"  {k}: {v.shape}")
    print("\nSample ETF prices:")
    print(data["etf_prices"].tail())
