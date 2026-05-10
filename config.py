import os

# ── FRED API ──────────────────────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "87c04efcdf543072107e53f957356b52")

# ── Universe ──────────────────────────────────────────────────────────────────
COUNTRY_ETFS = {
    "Brazil":      "EWZ",
    "Japan":       "EWJ",
    "Germany":     "EWG",
    "Canada":      "EWC",
    "Australia":   "EWA",
    "UK":          "EWU",
    "China":       "MCHI",
    "SouthKorea":  "EWY",
    "Taiwan":      "EWT",
    "Italy":       "EWI",
    "Spain":       "EWP",
    "India":       "INDA",
    "USA":         "SPY",
}

BENCHMARK = "SPY"

# ── Backtest ──────────────────────────────────────────────────────────────────
START_DATE = "2004-01-01"
END_DATE   = "2024-12-31"
REBALANCE_FREQ = "ME"          # Month-End

# ── Portfolio ─────────────────────────────────────────────────────────────────
TOP_N          = 4             # Long top N countries
BOTTOM_N       = 4             # Short bottom N countries
VOL_TARGET     = 0.10          # 10% annual volatility target
MAX_POSITION   = 0.20          # Max 20% per country
MAX_GROSS_EXP  = 2.0           # 200% gross exposure
MAX_NET_EXP    = 0.20          # ±20% net exposure

# ── Costs ─────────────────────────────────────────────────────────────────────
COMMISSION_PER_SHARE   = 0.005
SLIPPAGE_PCT           = 0.001          # 0.10% per side
SHORT_BORROW_LIQUID    = 0.003          # 0.30% annually (SPY, EWJ etc.)
SHORT_BORROW_EM        = 0.0075         # 0.75% annually (INDA, MCHI etc.)
MARGIN_SPREAD          = 0.005          # Fed Funds + 0.50%
DIVIDEND_WITHHOLDING   = 0.15           # 15% on foreign dividends

EM_ETFS = {"EWZ", "MCHI", "EWY", "EWT", "INDA"}

# ── Signal Weights ────────────────────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    # Monetary Policy (20%)
    "fed_funds_rate":        0.07,
    "yield_curve":           0.08,
    "real_yield":            0.05,

    # Growth (18%)
    "industrial_production": 0.07,
    "ism_pmi":               0.07,
    "jobless_claims":        0.04,

    # Inflation (12%)
    "cpi_trend":             0.05,
    "ppi_trend":             0.04,
    "breakeven_inflation":   0.03,

    # Labor Market (10%)
    "unemployment_rate":     0.05,
    "jobless_claims_chg":    0.05,

    # Commodities (15%)
    "oil_momentum":          0.06,
    "copper_momentum":       0.05,
    "gold_momentum":         0.04,

    # Dollar / Risk (15%)
    "dxy_momentum":          0.05,
    "vix_level":             0.05,
    "hy_spreads":            0.05,

    # Momentum (10%)
    "momentum_12_1":         0.06,
    "momentum_3m":           0.04,
}
