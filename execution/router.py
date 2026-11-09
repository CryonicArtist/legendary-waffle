import jax.numpy as jnp
from execution.portfolio import PortfolioManager
from execution.microstructure import MicrostructureModel
from risk.limits import RiskLimitManager

class ExecutionRouter:
    """
    Microstructure-aware execution router guarded by institutional risk limits.
    """
    def __init__(self, portfolio: PortfolioManager, risk_manager: RiskLimitManager, microstructure: MicrostructureModel, max_trade_size: float = 5.0):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.microstructure = microstructure
        self.max_trade_size = max_trade_size

    def process_signals(self, z_scores: jnp.ndarray, signals: jnp.ndarray, market_mids: jnp.ndarray, spreads: jnp.ndarray, greeks: jnp.ndarray, current_spot: float):
        for i in range(len(signals)):
            sig = signals[i]
            if sig == 0.0:
                continue

            current_pos = float(self.portfolio.option_positions[i])
            trade_qty = self.max_trade_size if sig == 1.0 else -self.max_trade_size

            # Risk check pass before execution
            if self.risk_manager.check_trade_allowed(strike_idx=i, trade_qty=trade_qty, current_pos=current_pos):
                fill_price, exec_qty = self.microstructure.compute_execution_fill(
                    qty=trade_qty, mid_price=float(market_mids[i]), spread=float(spreads[i])
                )
                self.portfolio.execute_option_trade(strike_idx=i, qty=exec_qty, fill_price=fill_price)
                action_str = "BOUGHT" if exec_qty > 0 else "SOLD"
                print(f"[EXECUTION] {action_str} {abs(exec_qty):.1f}x Strike {i} @ ${fill_price:.2f} (Slippage Adjusted)")

        # Rebalance underlying equity delta hedge
        net_option_delta = jnp.sum(self.portfolio.option_positions) * greeks[0]
        self.portfolio.rebalance_delta_hedge(net_option_delta, current_spot)
        
        # Check end-of-step portfolio risk metrics
        self.risk_manager.update_and_check_risk(
            total_equity=self.portfolio.compute_total_equity(market_mids, current_spot),
            current_net_delta=net_option_delta
        )