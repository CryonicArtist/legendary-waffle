import asyncio
import jax.numpy as jnp
from jax import random
from ingest.ring_buffer import MarketDataBuffer

class HistoricalTickReplayer:
    """
    Simulates historical tick replay feed using synthetic market quotes.
    """
    def __init__(self, buffer: MarketDataBuffer, num_ticks: int = 20):
        self.buffer = buffer
        self.num_ticks = num_ticks

    async def start_replay(self, key: random.PRNGKey):
        base_mids = jnp.array([11.85, 7.82, 4.61, 2.38, 1.05])
        spreads = jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])
        
        print(">> Starting Historical Tick Replay...")
        for tick in range(1, self.num_ticks + 1):
            key, subkey = random.split(key)
            noise = random.normal(subkey, (5,)) * 0.03
            current_mids = jnp.maximum(base_mids + noise, 0.05)
            
            # Inject transient arbitrage opportunity on tick 5
            if tick == 5:
                current_mids = current_mids.at[2].set(3.80)  # K=100 underpriced
            # Inject transient anomaly on tick 12
            if tick == 12:
                current_mids = current_mids.at[3].set(3.20)  # K=105 overpriced
                
            await self.buffer.push_tick(current_mids, spreads)
            await asyncio.sleep(0.01)  # Replay tick delay