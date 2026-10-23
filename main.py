import asyncio
import time
import jax
import jax.numpy as jnp
from jax import random

from core.forecast import run_forecast
from assimilation.update import update_weights
from assimilation.resample import resample_and_regularize
from pricer.batch_greeks import compute_ensemble_greeks
from signals.mispricing import evaluate_surface_arbitrage_signals
from ingest.ring_buffer import MarketDataBuffer
from ingest.feed import stream_market_data

async def run_trading_engine(buffer: MarketDataBuffer, strikes: jnp.ndarray, key: random.PRNGKey, num_steps: int = 10):
    """
    Asynchronous GPU Engine Loop consuming market data from the ring buffer.
    """
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0)
    T, r, option_type = 0.25, 0.05, 1.0
    params = jnp.array([0.05, 2.0, 0.04, 0.1, -0.7])
    
    # Dispersed particle initialization
    # Tighter initial particle dispersion (S ~ N(100.0, 0.05^2), v ~ LogNormal(0.04, 0.05^2))
    key, subkey_s, subkey_v = random.split(key, 3)
    # In main.py (run_trading_engine):
    # Set initial variance v_0 = 0.04 (which corresponds to 20% volatility: sqrt(0.04) = 0.20)
    init_spot = 100.0 + random.normal(subkey_s, (num_particles,)) * 0.05
    init_vol = 0.04 * jnp.exp(random.normal(subkey_v, (num_particles,)) * 0.05)
    particles = jnp.stack([init_spot, init_vol], axis=1)
    weights = jnp.ones(num_particles) / num_particles

    print("Booting Asynchronous JIT Engine & Waiting for Initial Data...")
    
    # Wait for first data tick to arrive
    while True:
        mids, spreads, has_data = await buffer.pop_tick()
        if has_data:
            break
        await asyncio.sleep(0.005)

    # XLA Warm-up Pass
    print("Executing XLA Compilation Warm-Up...")
    t_warmup_start = time.perf_counter()
    p_dummy, k_dummy = run_forecast(key, particles, params, dt, num_particles)
    exp_p, sd_p, z_p, sig_p = evaluate_surface_arbitrage_signals(p_dummy, weights, mids, strikes, T, r, option_type)
    w_dummy = update_weights(p_dummy, weights, mids, spreads, strikes, z_p, T, r, option_type)
    p_dummy, w_dummy, k_dummy = resample_and_regularize(k_dummy, p_dummy, w_dummy)
    g_dummy = compute_ensemble_greeks(p_dummy, w_dummy, 100.0, T, r, option_type)
    g_dummy.block_until_ready()
    print(f"XLA Warm-Up Complete in {(time.perf_counter() - t_warmup_start)*1000:.2f} ms.\n")

    # Main Asynchronous Execution Loop
    step = 0
    while step < num_steps:
        mids, spreads, has_data = await buffer.pop_tick()
        
        # If no new tick arrived yet, yield control briefy to avoid busy-spin
        if not has_data:
            await asyncio.sleep(0.001)
            continue
            
        step += 1
        t_start = time.perf_counter()

        # 1. Forecast Step
        particles, key = run_forecast(key, particles, params, dt, num_particles)

        # 2. Evaluate Arbitrage Signals
        exp_prices, std_devs, z_scores, signals = evaluate_surface_arbitrage_signals(
            particles, weights, mids, strikes, T, r, option_type, z_threshold=2.0
        )

        # 3. Data Assimilation
        weights = update_weights(
            particles, weights, mids, spreads, strikes, z_scores, T, r, option_type
        )

        # 4. ESS Check & Regularized Resampling
        ess = 1.0 / jnp.sum(weights**2)
        if ess < (num_particles / 2.0):
            particles, weights, key = resample_and_regularize(key, particles, weights, a=0.98)

        # 5. Calculate ATM Greeks
        greeks = compute_ensemble_greeks(particles, weights, 100.0, T, r, option_type)
        greeks.block_until_ready()
        
        latency_us = (time.perf_counter() - t_start) * 1_000_000.0

        print(f"=== Async Step {step} [DA Latency: {latency_us:.1f} µs] ===")
        print(f"Market Mid (K=100): ${mids[2]:.2f} | Model Exp: ${exp_prices[2]:.3f} | Z-Score: {z_scores[2]:+.2f}")
        
        # Check for anomalies across any contract
        anomalies = jnp.where(jnp.abs(z_scores) >= 2.0)[0]
        if len(anomalies) > 0:
            for idx in anomalies:
                sig_str = "BUY" if signals[idx] == 1.0 else "SELL"
                print(f"!! SIGNAL TRIGGERED on Strike {strikes[idx]:.1f}: {sig_str} (Z = {z_scores[idx]:+.2f})")
                
        print(f"ESS: {ess:.0f}/{num_particles} | ATM Delta: {greeks[0]:.4f} | ATM Vega: {greeks[2]:.4f}\n")

async def main():
    strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
    key = random.PRNGKey(42)
    buffer = MarketDataBuffer(num_strikes=5)

    # Launch Data Ingestion Feed and GPU Engine Concurrently
    feed_task = asyncio.create_task(stream_market_data(buffer, strikes, key, tick_interval_ms=50.0))
    engine_task = asyncio.create_task(run_trading_engine(buffer, strikes, key, num_steps=10))

    await engine_task
    feed_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())