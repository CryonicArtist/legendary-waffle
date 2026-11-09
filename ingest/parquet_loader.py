import asyncio
import pandas as pd
import jax.numpy as jnp
from ingest.ring_buffer import MarketDataBuffer

class ParquetTickReplayer:
    """
    Asynchronous historical tick replayer that streams L2/L3 option quotes 
    from Parquet/CSV datasets into the ring buffer.
    """
    def __init__(self, buffer: MarketDataBuffer, file_path: str, strikes: jnp.ndarray):
        self.buffer = buffer
        self.file_path = file_path
        self.strikes = strikes
        self.df = None

    def load_data(self):
        """Loads and sorts tick dataset by timestamp."""
        print(f">> Loading Parquet Tick Data from: {self.file_path} ...")
        # Supports parquet or fallback to CSV
        if self.file_path.endswith('.parquet'):
            self.df = pd.read_parquet(self.file_path)
        else:
            self.df = pd.read_csv(self.file_path)
            
        # Standardize expected columns: ['timestamp', 'strike', 'bid', 'ask', 'spot']
        self.df = self.df.sort_values(by='timestamp').reset_index(drop=True)
        print(f">> Loaded {len(self.df):,} total quote records.")

    async def stream_ticks(self, playback_speed_ms: float = 10.0):
        """
        Groups tick records by timestamp and streams multi-strike bid/ask vectors to the buffer.
        """
        if self.df is None:
            self.load_data()

        # Group data by timestamp frame
        grouped = self.df.groupby('timestamp')
        
        for timestamp, group in grouped:
            # Pivot strikes to match required strike vector order
            strike_mids = []
            strike_spreads = []
            
            for k in self.strikes:
                k_float = float(k)
                row = group[group['strike'] == k_float]
                
                if not row.empty:
                    bid = row['bid'].values[0]
                    ask = row['ask'].values[0]
                    mid = (bid + ask) / 2.0
                    spread = max(ask - bid, 0.01)
                else:
                    # Fallback for missing strike quotes in current tick frame
                    mid, spread = 0.0, 0.10
                    
                strike_mids.append(mid)
                strike_spreads.append(spread)

            mids_tensor = jnp.array(strike_mids)
            spreads_tensor = jnp.array(strike_spreads)

            # Push tick to non-blocking ring buffer
            await self.buffer.push_tick(mids_tensor, spreads_tensor)
            await asyncio.sleep(playback_speed_ms / 1000.0)