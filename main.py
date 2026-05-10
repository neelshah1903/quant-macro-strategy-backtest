import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetch_data import fetch_all
from signals.construct_signals import build_composite_score
from backtest.engine import run_backtest
from results.report import print_tearsheet, plot_results


def main():
    print("=" * 55)
    print("  GLOBAL MACRO COUNTRY ROTATION STRATEGY")
    print("=" * 55)
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("\n[1/4] Fetching data...")
    data = fetch_all()

    print("\n[2/4] Building signals...")
    scores = build_composite_score(
        data["fred_data"],
        data["commodity_prices"],
        data["etf_prices"],
    )

    print("\n[3/4] Running backtest...")
    results = run_backtest(scores, data["etf_prices"], data["fred_data"])

    print("\n[4/4] Generating report...")
    print_tearsheet(results)

    chart_path = os.path.join("results", "performance.png")
    plot_results(results, save_path=chart_path)
    print(f"\nDone. Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
