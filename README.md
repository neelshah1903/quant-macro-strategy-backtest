# Global Macro Country Rotation Strategy

A quantitative long/short strategy that rotates across 13 country ETFs using macroeconomic signals, commodity sensitivities, and price momentum.

## Strategy Overview

The strategy uses a **regime-aware allocation** approach:

- **US Dominant regime** → Hold SPY
- **International regime** → Long top 4 country ETFs by composite score
- **Neutral regime** → 50% SPY + 50% top 3 international ETFs

## Universe

13 Country ETFs: USA (SPY), Brazil (EWZ), Japan (EWJ), Germany (EWG), Canada (EWC), Australia (EWA), UK (EWU), China (MCHI), South Korea (EWY), Taiwan (EWT), Italy (EWI), Spain (EWP), India (INDA)

## Signals

All signals sourced freely from **FRED API** and **yfinance**:

| Category | Signals |
|---|---|
| Inflation | CPI trend, PPI trend, Breakeven inflation |
| Growth | CFNAI, Industrial Production |
| Monetary Policy | Fed Funds Rate, Real Yield (TIPS) |
| Credit/Risk | HY Credit Spreads, VIX |
| Commodities | Oil, Gold, DXY |
| Momentum | 12-1 month price momentum |

## Performance (2004-2024)

| Metric | Strategy | S&P 500 |
|---|---|---|
| Annual Return | 9.33% | ~10.89% |
| Sharpe Ratio | 0.47 | 0.63 |
| Max Drawdown | -47.12% | -50.78% |
| Total Return | 382% | 623% |

## Costs Modeled

- Commission: 0.10% per trade
- Slippage: 0.10% per side
- Short borrow: 0.30% (liquid) / 0.75% (EM) annually

## Data Sources

- **FRED API** — macro indicators (free)
- **yfinance** — ETF prices, commodities (free)

## How to Run

1. Install dependencies:
```
pip install yfinance fredapi pandas numpy matplotlib jupyter
```

2. Open notebook in Jupyter:
```
python -m jupyter notebook
```

3. Open `Global_Macro_Country_Rotation.ipynb` and run all cells top to bottom

## Requirements

- Python 3.10+
- FRED API key (free at [fred.stlouisfed.org](https://fred.stlouisfed.org))
