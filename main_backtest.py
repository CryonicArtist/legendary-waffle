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
from backtest.replayer import HistoricalTickReplayer
from execution.portfolio import PortfolioManager
from execution.router import ExecutionRouter

async def run_backtest():
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0)
    T, r, option_type = 0.25, 0.05, 1.0
    params = jnp.array([0.05, 2.0, 0.04, 0.1, -0.7])
    strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
    
    key = random.PRNGKey(101)
    buffer = MarketDataBuffer(num_strikes=5)
    portfolio = PortfolioManager(initial_cash=100_000.0)
    router = ExecutionRouter(portfolio, max_trade_size=5.0)
    replayer = HistoricalTickReplayer(buffer, num_ticks=15)
    
    # Initialize particles
    key, subkey_s, subkey_v = random.split(key, 3)
    init_spot = 100.0 + random.normal(subkey_s, (num_particles,)) * 0.05
    init_vol = 0.04 * jnp.exp(random.normal(subkey_v, (num_particles,)) * 0.05)
    particles = jnp.stack([init_spot, init_vol], axis=1)
    weights = jnp.ones(num_particles) / num_particles

    # Launch historical replay task
    replay_task = asyncio.create_task(replayer.start_replay(key))
    
    # Wait for first tick
    while True:
        mids, spreads, has_data = await buffer.pop_tick()
        if has_data:
            break
        await asyncio.sleep(0.002)

    print("\n--- Starting Phase 5 Backtest Execution Loop ---")
    step = 0
    pnl_history = []

    while step < 15:
        mids, spreads, has_data = await buffer.pop_tick()
        if not has_data:
            await asyncio.sleep(0.001)
            continue
            
        step += 1
        t_start = time.perf_counter()

        # 1. DA Forecast & Signal Evaluation
        particles, key = run_forecast(key, particles, params, dt, num_particles)
        exp_prices, std_devs, z_scores, signals = evaluate_surface_arbitrage_signals(
            particles, weights, mids, strikes, T, r, option_type, z_threshold=2.0
        )
        
        # 2. DA Weight Update & Resampling
        weights = update_weights(particles, weights, mids, spreads, strikes, z_scores, T, r, option_type)
        ess = 1.0 / jnp.sum(weights**2)
        if ess < (num_particles / 2.0):
            particles, weights, key = resample_and_regularize(key, particles, weights, a=0.98)

        # 3. Calculate Risk Sensitivities
        greeks = compute_ensemble_greeks(particles, weights, 100.0, T, r, option_type)
        greeks.block_until_ready()
        
        # 4. Execute Multi-Leg Option Orders & Delta Hedge
        current_spot = float(jnp.sum(particles[:, 0] * weights))
        router.process_signals(z_scores, signals, mids, spreads, greeks, current_spot)
        
        # 5. Compute Portfolio Equity & PnL
        total_equity = portfolio.compute_total_equity(mids, current_spot)
        pnl = total_equity - portfolio.initial_cash
        pnl_history.append(pnl)
        
        latency_us = (time.perf_counter() - t_start) * 1_000_000.0
        print(f"Tick {step:02d} | DA Latency: {latency_us:.1f} µs | Total Equity: ${total_equity:,.2f} | PnL: ${pnl:+,.2f} | Stock Pos: {portfolio.underlying_position:+.0f} sh\n")

    await replay_task
    print("Backtest Complete.")

if __name__ == "__main__":
    asyncio.run(run_backtest())