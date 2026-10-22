import time
import jax
import jax.numpy as jnp
from jax import random

from core.forecast import run_forecast
from assimilation.update import update_weights
from assimilation.resample import resample_and_regularize
from pricer.batch_greeks import compute_ensemble_greeks
from signals.mispricing import evaluate_surface_arbitrage_signals

def main():
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0)
    steps = 5
    
    strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
    T, r, option_type = 0.25, 0.05, 1.0
    params = jnp.array([0.05, 2.0, 0.04, 0.1, -0.7])
    
    key = random.PRNGKey(42)
    # Dispersed particle initialization (S ~ N(100, 0.5^2), v ~ LogNormal(0.04, 0.15^2))
    key, subkey_s, subkey_v = random.split(key, 3)
    init_spot = 100.0 + random.normal(subkey_s, (num_particles,)) * 0.5
    init_vol = 0.04 * jnp.exp(random.normal(subkey_v, (num_particles,)) * 0.15)
    particles = jnp.stack([init_spot, init_vol], axis=1)
    weights = jnp.ones(num_particles) / num_particles
    
    chain_stream = [
        (jnp.array([11.85, 7.82, 4.61, 2.38, 1.05]), jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])),
        (jnp.array([11.90, 7.87, 4.67, 2.42, 1.08]), jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])),
        (jnp.array([11.95, 7.92, 4.72, 2.46, 1.10]), jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])),
        (jnp.array([11.98, 7.95, 4.75, 3.10, 1.12]), jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])),
        (jnp.array([12.05, 8.02, 4.81, 2.50, 1.15]), jnp.array([0.10, 0.08, 0.06, 0.05, 0.04])),
    ]
    
    print("Booting Phase 4 High-Frequency Engine (@jax.jit + Liu-West Smoothing)...")
    
    # -------------------------------------------------------------------------
    # XLA Warm-up Pass (Forces JAX to compile CUDA kernels before live trading)
    # -------------------------------------------------------------------------
    t_start_warmup = time.perf_counter()
    dummy_mids, dummy_spreads = chain_stream[0]
    
    p_dummy, k_dummy = run_forecast(key, particles, params, dt, num_particles)
    exp_p, sd_p, z_p, sig_p = evaluate_surface_arbitrage_signals(p_dummy, weights, dummy_mids, strikes, T, r, option_type)
    w_dummy = update_weights(p_dummy, weights, dummy_mids, dummy_spreads, strikes, z_p, T, r, option_type)
    p_dummy, w_dummy, k_dummy = resample_and_regularize(k_dummy, p_dummy, w_dummy)
    g_dummy = compute_ensemble_greeks(p_dummy, w_dummy, 100.0, T, r, option_type)
    
    # Block until GPU finishes compilation
    g_dummy.block_until_ready()
    t_warmup = (time.perf_counter() - t_start_warmup) * 1000.0
    print(f"XLA Compilation Warm-Up Finished in {t_warmup:.2f} ms.\n")
    
    # -------------------------------------------------------------------------
    # Live Benchmark Loop
    # -------------------------------------------------------------------------
    latencies = []
    
    for t in range(steps):
        t_step_start = time.perf_counter()
        
        market_mids, spreads = chain_stream[t]
        
        # 1. Forecast Step
        particles, key = run_forecast(key, particles, params, dt, num_particles)
        
        # 2. Evaluate Arbitrage Signals
        exp_prices, std_devs, z_scores, signals = evaluate_surface_arbitrage_signals(
            particles, weights, market_mids, strikes, T, r, option_type, z_threshold=2.0
        )
        
        # 3. Data Assimilation Update
        weights = update_weights(
            particles, weights, market_mids, spreads, strikes, z_scores, T, r, option_type
        )
        
        # 4. Check ESS & Regularized Resampling
        ess = 1.0 / jnp.sum(weights**2)
        resampled_flag = False
        if ess < (num_particles / 2.0):
            particles, weights, key = resample_and_regularize(key, particles, weights, a=0.98)
            resampled_flag = True
            
        # 5. Calculate ATM AAD Greeks
        greeks = compute_ensemble_greeks(particles, weights, 100.0, T, r, option_type)
        
        # Force GPU synchronization for exact timer measurement
        greeks.block_until_ready()
        
        t_step_elapsed = (time.perf_counter() - t_step_start) * 1_000_000.0  # Microseconds
        latencies.append(t_step_elapsed)
        
        print(f"=== Time Step {t+1} [Latency: {t_step_elapsed:.1f} µs] ===")
        print(f"ESS: {ess:.0f}/{num_particles} | Resampled & Regularized: {resampled_flag}")
        print(f"ATM Delta: {greeks[0]:.4f} | ATM Gamma: {greeks[1]:.4f} | ATM Vega: {greeks[2]:.4f}\n")
        
    avg_lat = sum(latencies) / len(latencies)
    print(f"Benchmark Complete. Average Execution Latency per Tick: {avg_lat:.1f} µs ({avg_lat/1000.0:.3f} ms)")

if __name__ == "__main__":
    main()