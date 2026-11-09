import asyncio
import random
import jax.numpy as jnp
from ingest.ring_buffer import MarketDataBuffer

class SimulatedLiveStreamer:
    """
    Zero-cost real-time streaming worker. Generates tick updates 
    and pushes them into MarketDataBuffer to simulate a live WebSocket stream.
    """
    def __init__(self, buffer: MarketDataBuffer, strikes: jnp.ndarray, tick_interval_sec: float = 0.05):
        self.buffer = buffer
        self.strikes = strikes
        self.num_strikes = len(strikes)
        self.tick_interval_sec = tick_interval_sec
        self.is_running = False
        
        # Base spot price and strike mids
        self.spot = 510.0
        self.mids = [max(15.0 - 0.45 * abs(float(k) - self.spot), 0.50) for k in strikes]

    async def start_stream(self):
        self.is_running = True
        print(f">> Zero-Cost Live Streamer Active ({self.num_strikes} strikes @ {1.0/self.tick_interval_sec:.0f} Hz)...")
        
        tick_count = 0
        while self.is_running:
            tick_count += 1
            await asyncio.sleep(self.tick_interval_sec)

            # 1. Random walk for underlying spot price
            self.spot += random.gauss(0, 0.05)

            # 2. Update option mid-prices relative to spot
            current_mids = []
            current_spreads = []

            for i, k in enumerate(self.strikes):
                dist = abs(float(k) - self.spot)
                base_mid = max(15.0 - 0.45 * dist, 0.20) + random.gauss(0, 0.02)
                
                # Inject a transient arbitrage mispricing every ~20 ticks on strike index 2
                if tick_count % 20 == 0 and i == 2:
                    base_mid += 1.80  # Temporary pricing anomaly

                spread = max(0.04 + 0.01 * random.random(), 0.02)
                current_mids.append(base_mid)
                current_spreads.append(spread)

            # 3. Push live tick frame into engine ring buffer
            await self.buffer.push_tick(
                jnp.array(current_mids),
                jnp.array(current_spreads)
            )

    def stop_stream(self):
        self.is_running = False