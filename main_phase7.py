import asyncio
import time
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
from execution.microstructure import MicrostructureModel
from risk.limits import RiskLimitManager
from execution.router import ExecutionRouter
from backtest.metrics import generate_performance_report

async def run_phase7_engine():
    dataset_path = "ticks_sample.parquet"
    strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
    key = random.PRNGKey(2026)
    
    buffer = MarketDataBuffer(num_strikes=5)
    replayer = ParquetTickReplayer(buffer, dataset_path, strikes)
    replay_task = asyncio.create_task(replayer.stream_ticks(playback_speed_ms=10.0))
    
    # Fetch initial tick frame
    while True:
        mids, spreads, has_data = await buffer.pop_tick()
        if has_data:
            break
        await asyncio.sleep(0.002)

    # 1. Surface Calibration
    print("\nExecuting Phase 7 Calibrated Pipeline...")
    calibrator = HestonSurfaceCalibrator(num_steps=200)
    calibrated_params, calibrated_v0 = calibrator.calibrate(
        spot=100.0, strikes=strikes, market_mids=mids, T=0.25, r=0.05, option_type=1.0
    )

    # 2. Ensemble Initialization
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0)
    T, r, option_type = 0.25, 0.05, 1.0
    
    key, subkey_s, subkey_v = random.split(key, 3)
    init_spot = 100.0 + random.normal(subkey_s, (num_particles,)) * 0.05
    init_vol = calibrated_v0 * jnp.exp(random.normal(subkey_v, (num_particles,)) * 0.05)
    particles = jnp.stack([init_spot, init_vol], axis=1)
    weights = jnp.ones(num_particles) / num_particles

    # Phase 7 Microstructure & Risk Architecture
    portfolio = PortfolioManager(initial_cash=100_000.0)
    risk_manager = RiskLimitManager(max_portfolio_delta=50.0, max_drawdown_pct=2.0)
    microstructure = MicrostructureModel(impact_factor=0.02)
    router = ExecutionRouter(portfolio, risk_manager, microstructure, max_trade_size=5.0)

    print("\n--- Starting Phase 7 Production Simulation Loop ---")
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

        current_spot = float(jnp.sum(particles[:, 0] * weights))
        router.process_signals(z_scores, signals, mids, spreads, greeks, current_spot)
        
        total_equity = portfolio.compute_total_equity(mids, current_spot)
        pnl = total_equity - portfolio.initial_cash
        pnl_history.append(pnl)

        latency_us = (time.perf_counter() - t_start) * 1_000_000.0
        latency_history.append(latency_us)

        print(f"Phase 7 Tick {step:02d} | Latency: {latency_us:.1f} µs | Total Equity: ${total_equity:,.2f} | PnL: ${pnl:+,.2f}")

    replay_task.cancel()
    generate_performance_report(pnl_history, latency_history, portfolio.initial_cash)

if __name__ == "__main__":
    asyncio.run(run_phase7_engine())