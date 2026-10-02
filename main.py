import jax
import jax.numpy as jnp
from jax import random

from core.forecast import run_forecast
from assimilation.update import update_weights
from assimilation.resample import resample_particles
from pricer.batch_greeks import compute_ensemble_greeks
from signals.mispricing import evaluate_arbitrage_signal

def main():
    # Configuration
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0)
    steps = 5
    
    # Contract specs
    K, T, r, option_type = 100.0, 0.25, 0.05, 1.0
    params = jnp.array([0.05, 2.0, 0.04, 0.1, -0.7])
    
    key = random.PRNGKey(42)
    particles = jnp.ones((num_particles, 2)) * jnp.array([100.0, 0.04])
    weights = jnp.ones(num_particles) / num_particles
    
    # Simulated market stream containing an artificial mispricing at Step 4 ($5.20 vs expected ~$4.79)
    simulated_ticks = [4.616, 4.670, 4.720, 5.200, 4.850]
    
    print(f"Booting DA Engine with Real-Time AAD Greeks & Arbitrage Signals.\n")
    
    for t in range(steps):
        print(f"=== Time Step {t+1} ===")
        market_tick = simulated_ticks[t]
        
        # 1. Forecast Step
        particles, key = run_forecast(key, particles, params, dt, num_particles)
        
        # 2. Evaluate Arbitrage Signal BEFORE Assimilation
        exp_price, std_dev, z_score, signal = evaluate_arbitrage_signal(
            particles, weights, market_tick, K, T, r, option_type, z_threshold=2.0
        )
        
        sig_str = "BUY (Underpriced)" if signal == 1.0 else ("SELL (Overpriced)" if signal == -1.0 else "HOLD (Fair)")
        print(f"Market Tick: ${market_tick:.3f} | Model Exp: ${exp_price:.3f} ± ${std_dev:.3f}")
        print(f"Mispricing Z-Score: {z_score:+.2f} | Action Signal: {sig_str}")
        
        # 3. Data Assimilation Step
        observation_variance = 0.002  # Loosened slightly to reduce sample impoverishment
        weights = update_weights(
            particles, weights, market_tick, observation_variance, K, T, r, option_type
        )
        
        # 4. Check ESS and Resample
        ess = 1.0 / jnp.sum(weights**2)
        if ess < (num_particles / 2.0):
            particles, weights, key = resample_particles(key, particles, weights)
            
        # 5. Calculate Real-Time Ensemble Greeks via JAX AAD
        greeks = compute_ensemble_greeks(particles, weights, K, T, r, option_type)
        delta, gamma, vega = greeks[0], greeks[1], greeks[2]
        
        print(f"Ensemble Greeks -> Delta: {delta:.4f} | Gamma: {gamma:.4f} | Vega: {vega:.4f}\n")

if __name__ == "__main__":
    main()