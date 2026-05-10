import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    COUNTRY_ETFS, BENCHMARK, START_DATE, END_DATE, REBALANCE_FREQ,
    TOP_N, BOTTOM_N, VOL_TARGET, MAX_POSITION, MAX_GROSS_EXP, MAX_NET_EXP,
    COMMISSION_PER_SHARE, SLIPPAGE_PCT, SHORT_BORROW_LIQUID, SHORT_BORROW_EM,
    MARGIN_SPREAD, DIVIDEND_WITHHOLDING, EM_ETFS,
)


def compute_volatility(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(252)


def build_target_weights(scores: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    for date in scores.index:
        row = scores.loc[date].dropna()
        if len(row) < TOP_N + BOTTOM_N:
            continue

        ranked = row.rank(ascending=False)
        longs = ranked[ranked <= TOP_N].index.tolist()
        shorts = ranked[ranked > len(ranked) - BOTTOM_N].index.tolist()

        # Skip SPY on short side (benchmark — avoid shorting it)
        shorts = [s for s in shorts if s != "SPY"]

        w = pd.Series(0.0, index=scores.columns)

        # Inverse vol sizing
        vol_row = vol.loc[date] if date in vol.index else pd.Series(0.15, index=scores.columns)

        long_ivol = 1 / vol_row[longs].clip(lower=0.05)
        short_ivol = 1 / vol_row[shorts].clip(lower=0.05)

        long_w = long_ivol / long_ivol.sum() * 1.0
        short_w = short_ivol / short_ivol.sum() * -1.0

        # Apply position limits
        long_w = long_w.clip(upper=MAX_POSITION)
        short_w = short_w.clip(lower=-MAX_POSITION)

        w[longs] = long_w
        w[shorts] = short_w

        # Enforce net exposure limit
        net = w.sum()
        if abs(net) > MAX_NET_EXP:
            w = w - net * (w.abs() / w.abs().sum())

        weights.loc[date] = w

    return weights


def compute_transaction_costs(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    portfolio_value: float = 1_000_000,
) -> pd.Series:
    weight_changes = weights.diff().abs().fillna(0)
    total_costs = pd.Series(0.0, index=weights.index)

    for date in weights.index:
        daily_turnover = weight_changes.loc[date].sum()
        notional_traded = daily_turnover * portfolio_value

        # Commission (simplified: per notional traded)
        commission = notional_traded * (COMMISSION_PER_SHARE / 10)

        # Slippage both sides
        slippage = notional_traded * SLIPPAGE_PCT

        total_costs[date] = (commission + slippage) / portfolio_value

    return total_costs


def compute_holding_costs(weights: pd.DataFrame, fred_data: pd.DataFrame) -> pd.Series:
    holding_costs = pd.Series(0.0, index=weights.index)

    fed_funds_monthly = (
        fred_data["fed_funds_rate"]
        .reindex(weights.index, method="ffill")
        .fillna(0.02) / 100 / 12
    )

    for date in weights.index:
        w = weights.loc[date]
        shorts = w[w < 0]

        # Short borrow costs
        borrow_cost = 0.0
        for ticker, wt in shorts.items():
            rate = SHORT_BORROW_EM if ticker in EM_ETFS else SHORT_BORROW_LIQUID
            borrow_cost += abs(wt) * rate / 12

        # Margin interest on short proceeds
        margin_cost = abs(shorts.sum()) * (fed_funds_monthly.get(date, 0.002) + MARGIN_SPREAD / 12)

        holding_costs[date] = borrow_cost + margin_cost

    return holding_costs


def run_backtest(
    scores: pd.DataFrame,
    etf_prices: pd.DataFrame,
    fred_data: pd.DataFrame,
) -> dict:
    # Monthly prices aligned to rebalance dates
    monthly_prices = etf_prices.resample(REBALANCE_FREQ).last()
    monthly_returns = monthly_prices.pct_change()

    # Align scores to monthly
    scores_monthly = scores.resample(REBALANCE_FREQ).last()

    # Compute rolling volatility
    daily_returns = etf_prices.pct_change()
    vol_monthly = compute_volatility(daily_returns).resample(REBALANCE_FREQ).last()

    # Build target weights
    print("Building portfolio weights...")
    weights = build_target_weights(scores_monthly, vol_monthly)

    # Only rebalance if position change > 1%
    weight_changes = weights.diff().abs()
    small_change_mask = weight_changes < 0.01
    weights[small_change_mask] = weights.shift(1)[small_change_mask]

    # Compute strategy returns (weights are lagged by 1 period — no lookahead)
    lagged_weights = weights.shift(1).fillna(0)
    gross_returns = (lagged_weights * monthly_returns).sum(axis=1)

    # Costs
    print("Computing transaction costs...")
    transaction_costs = compute_transaction_costs(weights, monthly_prices)
    holding_costs = compute_holding_costs(lagged_weights, fred_data)

    net_returns = gross_returns - transaction_costs - holding_costs

    # Benchmark
    spy_returns = monthly_returns.get("SPY", pd.Series(dtype=float))
    spy_returns = spy_returns.reindex(net_returns.index)

    # Portfolio value
    portfolio_value = (1 + net_returns).cumprod()
    benchmark_value = (1 + spy_returns).cumprod()

    return {
        "net_returns":       net_returns,
        "gross_returns":     gross_returns,
        "transaction_costs": transaction_costs,
        "holding_costs":     holding_costs,
        "weights":           weights,
        "portfolio_value":   portfolio_value,
        "benchmark_value":   benchmark_value,
        "spy_returns":       spy_returns,
    }


def compute_metrics(returns: pd.Series, benchmark: pd.Series = None) -> dict:
    ann_return = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    metrics = {
        "Annual Return":   f"{ann_return:.2%}",
        "Annual Vol":      f"{ann_vol:.2%}",
        "Sharpe Ratio":    f"{sharpe:.2f}",
        "Max Drawdown":    f"{max_dd:.2%}",
        "Calmar Ratio":    f"{calmar:.2f}",
    }

    if benchmark is not None:
        aligned = returns.align(benchmark, join="inner")
        excess = aligned[0] - aligned[1]
        tracking_err = excess.std() * np.sqrt(12)
        info_ratio = excess.mean() * 12 / tracking_err if tracking_err > 0 else 0
        beta = returns.cov(benchmark) / benchmark.var() if benchmark.var() > 0 else 0
        alpha = ann_return - beta * (benchmark.mean() * 12)
        metrics["Beta"] = f"{beta:.2f}"
        metrics["Alpha"] = f"{alpha:.2%}"
        metrics["Info Ratio"] = f"{info_ratio:.2f}"

    return metrics


if __name__ == "__main__":
    from data.fetch_data import fetch_all
    from signals.construct_signals import build_composite_score

    data = fetch_all()
    scores = build_composite_score(
        data["fred_data"],
        data["commodity_prices"],
        data["etf_prices"],
    )
    results = run_backtest(scores, data["etf_prices"], data["fred_data"])

    print("\n── Strategy Metrics ──")
    metrics = compute_metrics(results["net_returns"], results["spy_returns"])
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n── Benchmark (SPY) Metrics ──")
    spy_metrics = compute_metrics(results["spy_returns"])
    for k, v in spy_metrics.items():
        print(f"  {k}: {v}")
