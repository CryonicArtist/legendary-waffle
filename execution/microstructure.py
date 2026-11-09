import jax.numpy as jnp

class MicrostructureModel:
    """
    Models realistic order book slippage, bid-ask spread crossing, and market impact.
    """
    def __init__(self, impact_factor: float = 0.05, min_fill_pct: float = 0.80):
        self.impact_factor = impact_factor
        self.min_fill_pct = min_fill_pct

    def compute_execution_fill(self, qty: float, mid_price: float, spread: float) -> tuple[float, float]:
        """
        Calculates effective fill price incorporating spread penalty and order size impact.
        Returns: (fill_price, executed_qty)
        """
        half_spread = spread / 2.0
        # Square-root law market impact: Impact = factor * sqrt(qty)
        impact = self.impact_factor * jnp.sqrt(abs(qty))
        
        if qty > 0:  # BUY order (pays half-spread + impact penalty)
            fill_price = float(mid_price + half_spread + impact)
        else:        # SELL order (receives bid - impact penalty)
            fill_price = float(mid_price - half_spread - impact)
            
        executed_qty = qty  # Assume 100% fill for standard contract sizes <= 10
        return fill_price, executed_qty