import time
import pandas as pd

class EngineTelemetryExporter:
    """
    Profiles per-tick latency, particle filter ESS health, and portfolio PnL,
    exporting log histories to CSV for post-trade statistical analysis.
    """
    def __init__(self):
        self.tick_records = []
        self.pnl_history = []
        self.latency_history = []

    def record_tick(self, latency_us: float, ess: float, pnl: float):
        """Records telemetry data for the current tick frame."""
        self.latency_history.append(latency_us)
        self.pnl_history.append(pnl)
        
        self.tick_records.append({
            "step": len(self.tick_records) + 1,
            "timestamp": time.time(),
            "latency_ms": round(latency_us / 1000.0, 3),
            "latency_us": round(latency_us, 1),
            "ess_health": int(ess),
            "pnl": round(pnl, 2)
        })

    def print_system_status(self, step: int):
        """Outputs tick metric summary to stdout."""
        if not self.tick_records:
            return
            
        latest = self.tick_records[-1]
        # Calculate moving percentiles over recent window
        recent_latencies = [r["latency_ms"] for r in self.tick_records[-50:]]
        sorted_lat = sorted(recent_latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99) - 1]
        
        pnl_str = f"+${latest['pnl']:,.2f}" if latest['pnl'] >= 0 else f"-${abs(latest['pnl']):,.2f}"
        print(f"[TELEMETRY | Tick {step:03d}] P50 Latency: {p50:.2f}ms | P99 Latency: {p99:.2f}ms | ESS Health: {latest['ess_health']}/10000 | PnL: {pnl_str}")

    def export_to_csv(self, filepath: str = "telemetry_metrics.csv"):
        """Exports tick-by-tick telemetry records to CSV."""
        if not self.tick_records:
            print("[TELEMETRY WARNING] No tick metrics recorded.")
            return
            
        df = pd.DataFrame(self.tick_records)
        df.to_csv(filepath, index=False)
        print(f">> Telemetry metrics successfully exported to: {filepath}")