import time
import pandas as pd
import jax.numpy as jnp

class PortfolioManager:
    """
    Tracks cash balances, option inventory per strike, delta hedges,
    and maintains an audit log of executed trades.
    """
    def __init__(self, num_strikes: int = 5, initial_cash: float = 100_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.option_positions = jnp.zeros(num_strikes)
        self.underlying_position = 0.0
        self.trade_log = []

    def execute_option_trade(
        self, 
        step: int, 
        order_id: str, 
        strike_idx: int, 
        strike_val: float, 
        qty: float, 
        fill_price: float
    ):
        """Updates option inventory and appends structured record to trade log."""
        cost = qty * fill_price * 100.0  # 100 multiplier per options contract
        self.cash -= cost
        self.option_positions = self.option_positions.at[strike_idx].add(qty)

        self.trade_log.append({
            "step": step,
            "timestamp": time.time(),
            "order_id": order_id,
            "strike_idx": strike_idx,
            "strike": float(strike_val),
            "side": "BUY" if qty > 0 else "SELL",
            "quantity": int(abs(qty)),
            "fill_price": round(float(fill_price), 4),
            "notional_cash_flow": round(-cost, 2),
            "position_after": float(self.option_positions[strike_idx]),
            "cash_after": round(self.cash, 2)
        })

    def rebalance_delta_hedge(self, net_option_delta: float, spot_price: float):
        """Rebalances underlying equity shares to maintain delta neutrality."""
        target_stock_shares = -net_option_delta
        trade_shares = target_stock_shares - self.underlying_position
        
        self.cash -= trade_shares * spot_price
        self.underlying_position = target_stock_shares

    def compute_total_equity(self, current_mids: jnp.ndarray, spot_price: float) -> float:
        """Computes total portfolio equity (cash + option mark-to-market + stock equity)."""
        options_val = jnp.sum(self.option_positions * current_mids * 100.0)
        stock_val = self.underlying_position * spot_price
        return float(self.cash + options_val + stock_val)

    def export_trades_to_csv(self, filepath: str = "trade_history.csv"):
        """Exports trade fill history log to CSV."""
        if not self.trade_log:
            print("[PORTFOLIO] No trades executed during session.")
            return
            
        df = pd.DataFrame(self.trade_log)
        df.to_csv(filepath, index=False)
        print(f">> Trade fill history successfully exported to: {filepath}")    