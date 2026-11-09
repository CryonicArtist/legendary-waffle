import jax.numpy as jnp

class PortfolioManager:
    """
    Tracks cash balances, option inventory per strike, and delta hedge positions.
    """
    def __init__(self, num_strikes: int = 5, initial_cash: float = 100_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.option_positions = jnp.zeros(num_strikes)
        self.underlying_position = 0.0

    def execute_option_trade(self, strike_idx: int, qty: float, fill_price: float):
        """Updates option inventory and cash balance upon order fill."""
        cost = qty * fill_price * 100.0  # Standard 100 multiplier per contract
        self.cash -= cost
        self.option_positions = self.option_positions.at[strike_idx].add(qty)

    def rebalance_delta_hedge(self, net_option_delta: float, spot_price: float):
        """Rebalances underlying equity shares to keep total portfolio delta neutral."""
        target_stock_shares = -net_option_delta
        trade_shares = target_stock_shares - self.underlying_position
        
        self.cash -= trade_shares * spot_price
        self.underlying_position = target_stock_shares

    def compute_total_equity(self, current_mids: jnp.ndarray, spot_price: float) -> float:
        """Computes net portfolio equity across cash, option positions, and underlying shares."""
        options_val = jnp.sum(self.option_positions * current_mids * 100.0)
        stock_val = self.underlying_position * spot_price
        return float(self.cash + options_val + stock_val)