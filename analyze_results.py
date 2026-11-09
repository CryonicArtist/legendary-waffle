import pandas as pd
import numpy as np

def analyze_performance(telemetry_file: str = "telemetry_metrics.csv", trades_file: str = "trade_history.csv"):
    try:
        df_telemetry = pd.read_csv(telemetry_file)
        df_trades = pd.read_csv(trades_file)
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return

    print("==================================================")
    print("        POST-TRADE PERFORMANCE DIAGNOSTICS        ")
    print("==================================================")

    # 1. Latency Profile Analysis
    latencies = df_telemetry['latency_ms']
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    max_lat = np.max(latencies)

    print("\n[1. EXECUTION LATENCY PROFILE]")
    print(f"  * Total Ticks Processed: {len(df_telemetry)}")
    print(f"  * Median (P50) Latency:  {p50_lat:.2f} ms")
    print(f"  * 95th Percentile (P95): {p95_lat:.2f} ms")
    print(f"  * 99th Percentile (P99): {p99_lat:.2f} ms")
    print(f"  * Worst-Case Spike:      {max_lat:.2f} ms")
    if p99_lat < 15.0:
        print("  ✓ Latency Verdict: PASS (Sub-15ms frame budget met)")
    else:
        print("  ⚠ Latency Verdict: WARNING (Tail latency spikes detected)")

    # 2. Financial Returns & Risk Diagnostics
    pnl = df_telemetry['pnl']
    initial_capital = 100_000.0
    equity = initial_capital + pnl
    
    # Calculate Max Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown_pct = np.abs(np.min(drawdown)) * 100.0

    # Tick-by-tick returns for Sharpe calculation
    returns = pnl.diff().fillna(0.0) / initial_capital
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    # Annualized for 20 Hz intraday tick frequency (assuming 252 days * 6.5 hours * 3600 * 20 ticks)
    ticks_per_year = 252 * 6.5 * 3600 * 20
    annualized_sharpe = (mean_ret / std_ret * np.sqrt(ticks_per_year)) if std_ret > 0 else 0.0

    print("\n[2. FINANCIAL PERFORMANCE & RISK]")
    print(f"  * Total Net PnL:         ${pnl.iloc[-1]:,.2f} ({pnl.iloc[-1]/initial_capital*100:+.2f}%)")
    print(f"  * Peak Equity:           ${np.max(equity):,.2f}")
    print(f"  * Lowest Equity:         ${np.min(equity):,.2f}")
    print(f"  * Max Drawdown:          {max_drawdown_pct:.2f}%")
    print(f"  * Tick Sharpe Ratio:     {annualized_sharpe:.2f}")

    # 3. Trade Execution & Inventory Audit
    total_trades = len(df_trades)
    buys = len(df_trades[df_trades['side'] == 'BUY'])
    sells = len(df_trades[df_trades['side'] == 'SELL'])
    total_volume_contracts = df_trades['quantity'].sum()
    total_friction_cash = np.sum(df_trades['quantity'] * 100.0 * 0.03)  # Approximate bid-ask half-spread friction

    print("\n[3. TRADE EXECUTION AUDIT]")
    print(f"  * Total Orders Executed: {total_trades} ({buys} BUY / {sells} SELL)")
    print(f"  * Total Option Contracts: {total_volume_contracts} contracts")
    print(f"  * Estimated Friction/Spread Paid: ${total_friction_cash:,.2f}")

    # 4. Sequential Monte Carlo Filter Health
    ess_series = df_telemetry['ess_health']
    resample_events = (ess_series < 5000).sum()
    min_ess = ess_series.min()

    print("\n[4. SMC PARTICLE FILTER HEALTH]")
    print(f"  * Resampling Frequency:  {resample_events} triggers ({resample_events/len(df_telemetry)*100:.1f}% of ticks)")
    print(f"  * Lowest ESS Registered: {min_ess} / 10000 particles")
    if min_ess < 1000:
        print("  ⚠ Filter Verdict: DEGENERATE (Particle collapse detected; increase regularized noise)")
    else:
        print("  ✓ Filter Verdict: HEALTHY (Sufficient particle diversity maintained)")

    print("==================================================")

if __name__ == "__main__":
    analyze_performance()