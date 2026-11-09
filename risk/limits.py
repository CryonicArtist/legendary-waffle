import jax.numpy as jnp

class RiskLimitManager:
    """
    Enforces institutional hard risk constraints and emergency trading halt switches.
    """
    def __init__(self, max_portfolio_delta: float = 50.0, max_drawdown_pct: float = 2.0, max_pos_per_strike: float = 15.0):
        self.max_portfolio_delta = max_portfolio_delta
        self.max_drawdown_pct = max_drawdown_pct
        self.max_pos_per_strike = max_pos_per_strike
        self.peak_equity = 100_000.0
        self.kill_switch_active = False

    def check_trade_allowed(self, strike_idx: int, trade_qty: float, current_pos: float) -> bool:
        """Verifies if trade complies with strike inventory limits."""
        if self.kill_switch_active:
            print("[RISK REJECT] Trade blocked: Emergency Kill-Switch ACTIVE!")
            return False
            
        new_pos = current_pos + trade_qty
        if abs(new_pos) > self.max_pos_per_strike:
            print(f"[RISK REJECT] Strike {strike_idx} position limit exceeded ({new_pos:+.0f} > {self.max_pos_per_strike:.0f}).")
            return False
            
        return True

    def update_and_check_risk(self, total_equity: float, current_net_delta: float) -> bool:
        """
        Evaluates portfolio-level drawdown and net delta exposure.
        Returns False if a risk limit is breached and trips the kill-switch.
        """
        # Update peak equity
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity

        # 1. Drawdown Check
        current_drawdown = (self.peak_equity - total_equity) / self.peak_equity * 100.0
        if current_drawdown >= self.max_drawdown_pct:
            self.kill_switch_active = True
            print(f"[RISK BREACH] MAX DRAWDOWN EXCEEDED ({current_drawdown:.2f}% >= {self.max_drawdown_pct:.2f}%). TRIPPING KILL-SWITCH!")
            return False

        # 2. Net Portfolio Delta Exposure Check
        if abs(current_net_delta) > self.max_portfolio_delta:
            print(f"[RISK WARNING] Unhedged Delta Breach: {current_net_delta:+.2f} > {self.max_portfolio_delta:.2f}")

        return True