import jax.numpy as jnp
from execution.portfolio import PortfolioManager

class ExecutionRouter:
    def __init__(self, portfolio: PortfolioManager, max_trade_size: float = 5.0, max_pos_per_strike: float = 10.0):
        self.portfolio = portfolio
        self.max_trade_size = max_trade_size
        self.max_pos_per_strike = max_pos_per_strike

    def process_signals(self, z_scores: jnp.ndarray, signals: jnp.ndarray, market_mids: jnp.ndarray, spreads: jnp.ndarray, greeks: jnp.ndarray, current_spot: float):
        for i in range(len(signals)):
            sig = signals[i]
            current_pos = float(self.portfolio.option_positions[i])
            
            # Execute BUY only if position limit is not exceeded
            if sig == 1.0 and current_pos + self.max_trade_size <= self.max_pos_per_strike:
                ask_price = market_mids[i] + (spreads[i] / 2.0)
                self.portfolio.execute_option_trade(strike_idx=i, qty=self.max_trade_size, fill_price=ask_price)
                print(f"[EXECUTION] BOUGHT {self.max_trade_size}x Strike {i} @ ${ask_price:.2f}")
                
            # Execute SELL only if position limit is not exceeded
            elif sig == -1.0 and current_pos - self.max_trade_size >= -self.max_pos_per_strike:
                bid_price = market_mids[i] - (spreads[i] / 2.0)
                self.portfolio.execute_option_trade(strike_idx=i, qty=-self.max_trade_size, fill_price=bid_price)
                print(f"[EXECUTION] SOLD {self.max_trade_size}x Strike {i} @ ${bid_price:.2f}")

        # Rebalance delta hedge
        net_option_delta = jnp.sum(self.portfolio.option_positions) * greeks[0]
        self.portfolio.rebalance_delta_hedge(net_option_delta, current_spot)