import time
import numpy as np

class EngineTelemetryExporter:
    """
    Lightweight operational telemetry tracker for engine health and latency profiling.
    """
    def __init__(self):
        self.latency_buffer = []
        self.ess_buffer = []
        self.pnl_buffer = []
        self.start_time = time.time()

    def record_tick(self, latency_us: float, ess: float, pnl: float):
        """Records per-tick telemetry metrics."""
        self.latency_buffer.append(latency_us)
        self.ess_buffer.append(ess)
        self.pnl_buffer.append(pnl)

    def print_system_status(self, step: int):
        """Prints live operational health stats to standard output."""
        recent_latencies = self.latency_buffer[-10:]
        p50 = np.median(recent_latencies) / 1000.0 if recent_latencies else 0.0
        p99 = np.percentile(recent_latencies, 99) / 1000.0 if recent_latencies else 0.0
        latest_ess = self.ess_buffer[-1] if self.ess_buffer else 0.0
        latest_pnl = self.pnl_buffer[-1] if self.pnl_buffer else 0.0

        print(f"[TELEMETRY | Tick {step:03d}] P50 Latency: {p50:.2f}ms | P99 Latency: {p99:.2f}ms | ESS Health: {latest_ess:.0f}/10000 | PnL: ${latest_pnl:+,.2f}")