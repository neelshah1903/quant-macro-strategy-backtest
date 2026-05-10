import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.engine import compute_metrics


def plot_results(results: dict, save_path: str = None):
    net_ret = results["net_returns"].dropna()
    spy_ret = results["spy_returns"].dropna()
    weights  = results["weights"]

    aligned = net_ret.align(spy_ret, join="inner")
    net_ret  = aligned[0]
    spy_ret  = aligned[1]

    cum_strategy  = (1 + net_ret).cumprod()
    cum_benchmark = (1 + spy_ret).cumprod()

    # Drawdown
    roll_max = cum_strategy.cummax()
    drawdown = (cum_strategy - roll_max) / roll_max

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Global Macro Country Rotation Strategy vs S&P 500", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Cumulative returns
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(cum_strategy.index,  cum_strategy.values,  label="Strategy", color="steelblue", linewidth=2)
    ax1.plot(cum_benchmark.index, cum_benchmark.values, label="S&P 500",  color="orange",   linewidth=1.5, linestyle="--")
    ax1.set_title("Cumulative Returns")
    ax1.set_ylabel("Growth of $1")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
    ax2.set_title("Strategy Drawdown")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)

    # 3. Monthly returns distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(net_ret.values * 100, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    ax3.axvline(0, color="red", linewidth=1)
    ax3.set_title("Monthly Returns Distribution")
    ax3.set_xlabel("Return %")
    ax3.grid(True, alpha=0.3)

    # 4. Rolling 12m Sharpe
    ax4 = fig.add_subplot(gs[2, 0])
    rolling_sharpe = (net_ret.rolling(12).mean() / net_ret.rolling(12).std()) * np.sqrt(12)
    ax4.plot(rolling_sharpe.index, rolling_sharpe.values, color="green", linewidth=1.5)
    ax4.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax4.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax4.set_title("Rolling 12m Sharpe Ratio")
    ax4.grid(True, alpha=0.3)

    # 5. Average weights over time
    ax5 = fig.add_subplot(gs[2, 1])
    avg_weights = weights.mean().sort_values()
    colors = ["red" if w < 0 else "steelblue" for w in avg_weights.values]
    ax5.barh(avg_weights.index, avg_weights.values * 100, color=colors)
    ax5.axvline(0, color="black", linewidth=0.8)
    ax5.set_title("Average Position Weights (%)")
    ax5.set_xlabel("Weight %")
    ax5.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()


def print_tearsheet(results: dict):
    net_ret = results["net_returns"].dropna()
    spy_ret = results["spy_returns"].dropna()
    gross   = results["gross_returns"].dropna()
    tx_cost = results["transaction_costs"].dropna()
    h_cost  = results["holding_costs"].dropna()

    print("\n" + "=" * 55)
    print("  GLOBAL MACRO COUNTRY ROTATION — PERFORMANCE TEARSHEET")
    print("=" * 55)

    print("\n── Strategy (Net) ──────────────────────────────────────")
    metrics = compute_metrics(net_ret, spy_ret)
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")

    print("\n── Benchmark S&P 500 ───────────────────────────────────")
    spy_metrics = compute_metrics(spy_ret)
    for k, v in spy_metrics.items():
        print(f"  {k:<20} {v}")

    print("\n── Cost Breakdown (annualized) ─────────────────────────")
    print(f"  Gross Return:        {gross.mean() * 12:.2%}")
    print(f"  Transaction Costs:   {tx_cost.mean() * 12:.2%}")
    print(f"  Holding Costs:       {h_cost.mean() * 12:.2%}")
    print(f"  Net Return:          {net_ret.mean() * 12:.2%}")

    print("\n── Annual Returns ──────────────────────────────────────")
    annual = net_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    spy_annual = spy_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    for yr in annual.index:
        strat_r = annual.loc[yr]
        spy_r   = spy_annual.loc[yr] if yr in spy_annual.index else float("nan")
        diff    = strat_r - spy_r
        print(f"  {yr.year}   Strategy: {strat_r:+.1%}   SPY: {spy_r:+.1%}   Alpha: {diff:+.1%}")

    print("=" * 55)


if __name__ == "__main__":
    from data.fetch_data import fetch_all
    from signals.construct_signals import build_composite_score
    from backtest.engine import run_backtest

    print("Loading data...")
    data = fetch_all()

    print("Building signals...")
    scores = build_composite_score(
        data["fred_data"],
        data["commodity_prices"],
        data["etf_prices"],
    )

    print("Running backtest...")
    results = run_backtest(scores, data["etf_prices"], data["fred_data"])

    print_tearsheet(results)

    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance.png")
    plot_results(results, save_path=output)
