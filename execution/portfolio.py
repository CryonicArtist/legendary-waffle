import jax.numpy as jnp

class PortfolioManager:
    """
    Tracks options positions, equity cash hedge, and PnL metrics.
    """
    def __init__(self, initial_cash: float = 100_000.0, fee_per_contract: float = 0.50):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.fee_per_contract = fee_per_contract
        
        # Position tracks quantity held per strike index [K_90, K_95, K_100, K_105, K_110]
        self.option_positions = jnp.zeros(5)
        self.underlying_position = 0.0  # Shares of underlying equity held for delta hedging
        
    def execute_option_trade(self, strike_idx: int, qty: float, fill_price: float):
        """Executes option order: qty > 0 (BUY), qty < 0 (SELL)."""
        cost = qty * fill_price * 100.0  # 1 contract = 100 shares
        fees = abs(qty) * self.fee_per_contract
        
        self.cash -= (cost + fees)
        self.option_positions = self.option_positions.at[strike_idx].add(qty)
        
    def rebalance_delta_hedge(self, net_portfolio_delta: float, current_spot: float):
        """
        Executes underlying equity trade to reset net portfolio delta to 0.0.
        net_portfolio_delta: total delta exposure from option inventory.
        """
        target_shares = -net_portfolio_delta * 100.0
        trade_shares = target_shares - self.underlying_position
        
        if abs(trade_shares) >= 1.0:
            cost = trade_shares * current_spot
            self.cash -= cost
            self.underlying_position += trade_shares
            
    def compute_total_equity(self, current_mids: jnp.ndarray, current_spot: float) -> float:
        """Evaluates liquidation value of all option and equity holdings."""
        options_val = jnp.sum(self.option_positions * current_mids * 100.0)
        stock_val = self.underlying_position * current_spot
        return self.cash + options_val + stock_val