import asyncio
import jax.numpy as jnp
from jax import random
from ingest.ring_buffer import MarketDataBuffer

async def stream_market_data(buffer: MarketDataBuffer, strikes: jnp.ndarray, key: random.PRNGKey, tick_interval_ms: float = 50.0):
    """
    Asynchronous market feed emulator. Streams L2/L3 mid prices and bid-ask spreads into the buffer.
    """
    base_mids = jnp.array([11.85, 7.82, 4.61, 2.38, 1.05])
    spreads = jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])
    
    tick_count = 0
    print(f">> Launching Async Market Feed (Interval: {tick_interval_ms} ms)...")
    
    while True:
        key, subkey = random.split(key)
        
        # Simulate small random market drift (+ Brownian noise)
        drift = random.normal(subkey, (5,)) * 0.02
        current_mids = jnp.maximum(base_mids + drift, 0.05)
        
        # Inject periodic fat-finger mispricing on Step 8 at K=105
        tick_count += 1
        if tick_count % 8 == 0:
            current_mids = current_mids.at[3].set(3.15)  # Mispricing spike on K=105
            print(f"\n[FEED INJECT] Artificial Anomaly Streamed on K=105 ($3.15)!")

        await buffer.push_tick(current_mids, spreads)
        await asyncio.sleep(tick_interval_ms / 1000.0)