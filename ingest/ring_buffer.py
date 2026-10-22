import asyncio
import jax.numpy as jnp

class MarketDataBuffer:
    """
    Non-blocking thread-safe buffer storing the latest multi-strike option quote vector Y_t.
    """
    def __init__(self, num_strikes=5):
        self._lock = asyncio.Lock()
        self._latest_mids = None
        self._latest_spreads = None
        self._has_new_data = False

    async def push_tick(self, mids: jnp.ndarray, spreads: jnp.ndarray):
        """Pushes a new market tick vector into the buffer."""
        async with self._lock:
            self._latest_mids = mids
            self._latest_spreads = spreads
            self._has_new_data = True

    async def pop_tick(self):
        """
        Retrieves the newest available market tick vector.
        Returns (mids, spreads, has_new_data).
        """
        async with self._lock:
            if not self._has_new_data:
                return self._latest_mids, self._latest_spreads, False
            self._has_new_data = False
            return self._latest_mids, self._latest_spreads, True