import jax
import jax.numpy as jnp
from jax import random

from core.forecast import run_forecast
from assimilation.update import update_weights
from assimilation.resample import resample_particles
from pricer.batch_greeks import compute_ensemble_greeks
from signals.mispricing import evaluate_surface_arbitrage_signals

def main():
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0)
    steps = 5
    
    # Option Chain Configuration (5 Strike Prices)
    strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
    T, r, option_type = 0.25, 0.05, 1.0
    params = jnp.array([0.05, 2.0, 0.04, 0.1, -0.7])
    
    key = random.PRNGKey(42)
    particles = jnp.ones((num_particles, 2)) * jnp.array([100.0, 0.04])
    weights = jnp.ones(num_particles) / num_particles
    
    # Dynamic Order Book Stream: Mids and Spreads across 5 strikes
    # Format per step: (Mids_vector, Spreads_vector)
    chain_stream = [
        # Step 1: Normal fair quotes
        (jnp.array([11.85, 7.82, 4.61, 2.38, 1.05]), jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])),
        # Step 2: Slight spot tick upward
        (jnp.array([11.90, 7.87, 4.67, 2.42, 1.08]), jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])),
        # Step 3: Continued tick upward
        (jnp.array([11.95, 7.92, 4.72, 2.46, 1.10]), jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])),
        # Step 4: ANOMALY on K=105 ($3.10 vs expected ~$2.45)
        (jnp.array([11.98, 7.95, 4.75, 3.10, 1.12]), jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])),
        # Step 5: K=105 returns to normal ($2.50)
        (jnp.array([12.05, 8.02, 4.81, 2.50, 1.15]), jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])),
    ]
    
    print("Booting Phase 3 DA Engine: Multi-Strike Chain Assimilation with Dynamic Noise R_t.\n")
    
    for t in range(steps):
        print(f"=== Time Step {t+1} ===")
        market_mids, spreads = chain_stream[t]
        
        # 1. Forecast Step
        particles, key = run_forecast(key, particles, params, dt, num_particles)
        
        # 2. Evaluate Arbitrage Signals across Surface
        exp_prices, std_devs, z_scores, signals = evaluate_surface_arbitrage_signals(
            particles, weights, market_mids, strikes, T, r, option_type, z_threshold=2.0
        )
        
        # Print chain status
        print("Strike | Mkt Mid | Model Exp | Noise R_t |  Z-Score  | Signal")
        print("-" * 65)
        for i, K_val in enumerate(strikes):
            sig_txt = "BUY" if signals[i] == 1.0 else ("SELL" if signals[i] == -1.0 else "HOLD")
            R_val = (spreads[i] / 2.0)**2
            print(f" {K_val:5.1f} | ${market_mids[i]:6.2f} | ${exp_prices[i]:7.3f} | {R_val:.6f} | {z_scores[i]:+9.2f} | {sig_txt}")
        
        # 3. Data Assimilation Step with Selective Contract Masking
        weights = update_weights(
            particles, weights, market_mids, spreads, strikes, z_scores, T, r, option_type
        )
        
        # 4. Resample if ESS drops below threshold
        ess = 1.0 / jnp.sum(weights**2)
        if ess < (num_particles / 2.0):
            print(f">> ESS = {ess:.0f}/{num_particles}. Resampling ensemble...")
            particles, weights, key = resample_particles(key, particles, weights)
            
        # 5. ATM Contract (K=100) Greeks
        greeks = compute_ensemble_greeks(particles, weights, 100.0, T, r, option_type)
        print(f"ATM Greeks (K=100) -> Delta: {greeks[0]:.4f} | Gamma: {greeks[1]:.4f} | Vega: {greeks[2]:.4f}\n")

if __name__ == "__main__":
    main()