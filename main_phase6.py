import asyncio
import time
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random

from core.forecast import run_forecast
from core.calibration import HestonSurfaceCalibrator
from assimilation.update import update_weights
from assimilation.resample import resample_and_regularize
from pricer.batch_greeks import compute_ensemble_greeks
from signals.mispricing import evaluate_surface_arbitrage_signals
from ingest.ring_buffer import MarketDataBuffer
from ingest.parquet_loader import ParquetTickReplayer
from execution.portfolio import PortfolioManager
from execution.router import ExecutionRouter
from backtest.metrics import generate_performance_report

def create_sample_parquet_dataset(file_path: str = "ticks_sample.parquet"):
    """Generates a sample multi-strike option tick Parquet file if non-existent."""
    timestamps = []
    strikes = []
    bids = []
    asks = []
    spots = []
    
    base_mids = [11.85, 7.82, 4.61, 2.38, 1.05]
    strike_list = [90.0, 95.0, 100.0, 105.0, 110.0]
    
    for t in range(20):
        for idx, k in enumerate(strike_list):
            timestamps.append(t)
            strikes.append(k)
            spots.append(100.0)
            
            mid = base_mids[idx]
            # Inject transient mispricing at t=6 for K=105
            if t == 6 and k == 105.0:
                mid = 3.25
                
            bids.append(mid - 0.03)
            asks.append(mid + 0.03)
            
    df = pd.DataFrame({
        'timestamp': timestamps,
        'strike': strikes,
        'bid': bids,
        'ask': asks,
        'spot': spots
    })
    df.to_parquet(file_path)
    print(f">> Created Sample Parquet File: {file_path}")

async def run_phase6_engine():
    dataset_path = "ticks_sample.parquet"
    create_sample_parquet_dataset(dataset_path)
    
    strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
    key = random.PRNGKey(2026)
    
    buffer = MarketDataBuffer(num_strikes=5)
    replayer = ParquetTickReplayer(buffer, dataset_path, strikes)
    
    # Launch Parquet streaming task
    replay_task = asyncio.create_task(replayer.stream_ticks(playback_speed_ms=10.0))
    
    # Fetch initial tick frame for calibration
    while True:
        mids, spreads, has_data = await buffer.pop_tick()
        if has_data:
            break
        await asyncio.sleep(0.002)

    # 1. Calibrate Heston Prior to Empirical Market Chain
    print("\nExecuting Empirical Surface Calibration...")
    calibrator = HestonSurfaceCalibrator(num_steps=200)
    calibrated_params, calibrated_v0 = calibrator.calibrate(
        spot=100.0, strikes=strikes, market_mids=mids, T=0.25, r=0.05, option_type=1.0
    )

    # 2. Initialize Ensemble around Calibrated Parameters
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0)
    T, r, option_type = 0.25, 0.05, 1.0
    
    key, subkey_s, subkey_v = random.split(key, 3)
    init_spot = 100.0 + random.normal(subkey_s, (num_particles,)) * 0.05
    init_vol = calibrated_v0 * jnp.exp(random.normal(subkey_v, (num_particles,)) * 0.05)
    particles = jnp.stack([init_spot, init_vol], axis=1)
    weights = jnp.ones(num_particles) / num_particles

    portfolio = PortfolioManager(initial_cash=100_000.0)
    router = ExecutionRouter(portfolio, max_trade_size=5.0)

    print("\n--- Starting Phase 6 Assimilation Engine ---")
    step = 0
    pnl_history = []
    latency_history = []

    while step < 15:
        mids, spreads, has_data = await buffer.pop_tick()
        if not has_data:
            await asyncio.sleep(0.001)
            continue
            
        step += 1
        t_start = time.perf_counter()

        # Forecast & Assimilation Loop
        particles, key = run_forecast(key, particles, calibrated_params, dt, num_particles)
        exp_prices, std_devs, z_scores, signals = evaluate_surface_arbitrage_signals(
            particles, weights, mids, strikes, T, r, option_type, z_threshold=2.0
        )
        weights = update_weights(particles, weights, mids, spreads, strikes, z_scores, T, r, option_type)
        
        ess = 1.0 / jnp.sum(weights**2)
        if ess < (num_particles / 2.0):
            particles, weights, key = resample_and_regularize(key, particles, weights, a=0.98)

        greeks = compute_ensemble_greeks(particles, weights, 100.0, T, r, option_type)
        greeks.block_until_ready()

        # Execution & Delta Hedging
        current_spot = float(jnp.sum(particles[:, 0] * weights))
        router.process_signals(z_scores, signals, mids, spreads, greeks, current_spot)
        
        total_equity = portfolio.compute_total_equity(mids, current_spot)
        pnl = total_equity - portfolio.initial_cash
        pnl_history.append(pnl)

        latency_us = (time.perf_counter() - t_start) * 1_000_000.0
        latency_history.append(latency_us)

        print(f"Parquet Tick {step:02d} | DA Latency: {latency_us:.1f} µs | Total Equity: ${total_equity:,.2f} | PnL: ${pnl:+,.2f}")

    replay_task.cancel()
    generate_performance_report(pnl_history, latency_history, portfolio.initial_cash)

if __name__ == "__main__":
    asyncio.run(run_phase6_engine())