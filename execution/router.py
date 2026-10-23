import jax.numpy as jnp
from execution.portfolio import PortfolioManager

class ExecutionRouter:
    """
    Generates multi-leg trades and executes immediate delta-neutral hedging.
    """
    def __init__(self, portfolio: PortfolioManager, max_trade_size: float = 5.0):
        self.portfolio = portfolio
        self.max_trade_size = max_trade_size

    def process_signals(self, z_scores: jnp.ndarray, signals: jnp.ndarray, market_mids: jnp.ndarray, spreads: jnp.ndarray, greeks: jnp.ndarray, current_spot: float):
        """
        Processes mispricing signals and triggers option trades and equity hedges.
        """
        # 1. Option Trade Execution
        for i in range(len(signals)):
            sig = signals[i]
            if sig == 1.0:    # BUY Signal (Underpriced)
                # Buy at Ask = Mid + Half-Spread
                ask_price = market_mids[i] + (spreads[i] / 2.0)
                self.portfolio.execute_option_trade(strike_idx=i, qty=self.max_trade_size, fill_price=ask_price)
                print(f"[EXECUTION] BOUGHT {self.max_trade_size}x Strike {i} @ ${ask_price:.2f}")
                
            elif sig == -1.0:  # SELL Signal (Overpriced)
                # Sell at Bid = Mid - Half-Spread
                bid_price = market_mids[i] - (spreads[i] / 2.0)
                self.portfolio.execute_option_trade(strike_idx=i, qty=-self.max_trade_size, fill_price=bid_price)
                print(f"[EXECUTION] SOLD {self.max_trade_size}x Strike {i} @ ${bid_price:.2f}")

        # 2. Compute Net Option Portfolio Delta
        # Approx ATM option delta scaling across inventory
        net_option_delta = jnp.sum(self.portfolio.option_positions) * greeks[0]
        
        # 3. Trigger Delta Hedge
        self.portfolio.rebalance_delta_hedge(net_option_delta, current_spot)