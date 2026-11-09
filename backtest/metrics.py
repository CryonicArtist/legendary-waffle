import numpy as np

def generate_performance_report(pnl_history: list, latency_history: list, initial_cash: float = 100_000.0):
    """
    Computes key quantitative trading and latency analytics.
    """
    pnl_arr = np.array(pnl_history)
    equity_curve = initial_cash + pnl_arr
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # 1. PnL Metrics
    total_pnl = pnl_arr[-1]
    total_return_pct = (total_pnl / initial_cash) * 100.0
    
    # 2. Risk Metrics (Max Drawdown & Sharpe)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    max_drawdown_pct = np.max(drawdowns) * 100.0
    
    # Annualized Sharpe (Assuming 1-second tick intervals: ~252 days * 6.5 hrs * 3600s)
    annualization_factor = np.sqrt(252 * 6.5 * 3600)
    mean_ret = np.mean(returns) if len(returns) > 0 else 0.0
    std_ret = np.std(returns) if len(returns) > 0 else 1.0
    sharpe_ratio = (mean_ret / (std_ret + 1e-9)) * annualization_factor
    
    # 3. Latency Metrics (Exclude warm-up spikes > 100ms)
    filtered_latencies = [l for l in latency_history if l < 100_000.0]  # µs
    avg_latency_ms = (np.mean(filtered_latencies) / 1000.0) if filtered_latencies else 0.0
    p99_latency_ms = (np.percentile(filtered_latencies, 99) / 1000.0) if filtered_latencies else 0.0

    print("\n==================================================")
    print("           PHASE 5 BACKTEST PERFORMANCE           ")
    print("==================================================")
    print(f"Initial Capital:        ${initial_cash:,.2f}")
    print(f"Final Total Equity:     ${equity_curve[-1]:,.2f}")
    print(f"Net Realized/Unrealized PnL: ${total_pnl:+,.2f} ({total_return_pct:+.2f}%)")
    print(f"Max Drawdown:           {max_drawdown_pct:.2f}%")
    print(f"Annualized Sharpe Ratio:{sharpe_ratio:.2f}")
    print("--------------------------------------------------")
    print(f"Filtered Mean DA Latency: {avg_latency_ms:.2f} ms")
    print(f"99th Percentile Latency: {p99_latency_ms:.2f} ms")
    print("==================================================\n")