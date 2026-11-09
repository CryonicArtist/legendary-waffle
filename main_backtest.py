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
from backtest.metrics import generate_performance_report


async def run_backtest():
    # Simulation Parameters
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0)
    T, r, option_type = 0.25, 0.05, 1.0
    params = jnp.array([0.05, 2.0, 0.04, 0.1, -0.7])  # [mu, kappa, theta, xi, rho]
    strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])

    key = random.PRNGKey(101)
    buffer = MarketDataBuffer(num_strikes=5)
    portfolio = PortfolioManager(initial_cash=100_000.0)
    router = ExecutionRouter(portfolio, max_trade_size=5.0)

    # Supply 25 ticks to give the 15-step loop ample headroom
    replayer = HistoricalTickReplayer(buffer, num_ticks=25)

    # Initial Particle Ensemble Dispersion
    key, subkey_s, subkey_v = random.split(key, 3)
    init_spot = 100.0 + random.normal(subkey_s, (num_particles,)) * 0.05
    init_vol = 0.04 * jnp.exp(random.normal(subkey_v, (num_particles,)) * 0.05)
    particles = jnp.stack([init_spot, init_vol], axis=1)
    weights = jnp.ones(num_particles) / num_particles

    # Launch historical feed task asynchronously
    replay_task = asyncio.create_task(replayer.start_replay(key))

    # Wait for the first quote tick to populate the buffer
    while True:
        mids, spreads, has_data = await buffer.pop_tick()
        if has_data:
            break
        await asyncio.sleep(0.002)

    # XLA Compilation Warm-up Pass
    print("Executing XLA Compilation Warm-Up...")
    t_warmup_start = time.perf_counter()
    p_dummy, k_dummy = run_forecast(key, particles, params, dt, num_particles)
    exp_p, sd_p, z_p, sig_p = evaluate_surface_arbitrage_signals(
        p_dummy, weights, mids, strikes, T, r, option_type
    )
    w_dummy = update_weights(
        p_dummy, weights, mids, spreads, strikes, z_p, T, r, option_type
    )
    p_dummy, w_dummy, k_dummy = resample_and_regularize(
        k_dummy, p_dummy, w_dummy
    )
    g_dummy = compute_ensemble_greeks(p_dummy, w_dummy, 100.0, T, r, option_type)
    g_dummy.block_until_ready()
    print(
        f"XLA Warm-Up Complete in {(time.perf_counter() - t_warmup_start)*1000:.2f} ms.\n"
    )

    print("--- Starting Phase 5 Backtest Execution Loop ---")
    step = 0
    pnl_history = []
    latency_history = []
    empty_polls = 0

    while step < 15:
        mids, spreads, has_data = await buffer.pop_tick()

        # Handle stream completion / buffer empty conditions safely
        if not has_data:
            empty_polls += 1
            if empty_polls > 50:  # Timeout after 50ms of no ticks
                print("\nReplayer stream completed.")
                break
            await asyncio.sleep(0.001)
            continue

        empty_polls = 0
        step += 1
        t_start = time.perf_counter()

        # 1. Sequential Monte Carlo Forecast Step
        particles, key = run_forecast(key, particles, params, dt, num_particles)

        # 2. Evaluate Arbitrage Signals across the Option Surface
        exp_prices, std_devs, z_scores, signals = evaluate_surface_arbitrage_signals(
            particles, weights, mids, strikes, T, r, option_type, z_threshold=2.0
        )

        # 3. Data Assimilation (Likelihood Weighting)
        weights = update_weights(
            particles, weights, mids, spreads, strikes, z_scores, T, r, option_type
        )

        # 4. Check ESS & Execute Liu-West Regularized Resampling if degraded
        ess = 1.0 / jnp.sum(weights**2)
        if ess < (num_particles / 2.0):
            particles, weights, key = resample_and_regularize(
                key, particles, weights, a=0.98
            )

        # 5. Compute Pathwise AAD Risk Sensitivities (Greeks)
        greeks = compute_ensemble_greeks(particles, weights, 100.0, T, r, option_type)
        greeks.block_until_ready()

        # 6. Execute Multi-Leg Orders and Rebalance Delta Hedge
        current_spot = float(jnp.sum(particles[:, 0] * weights))
        router.process_signals(
            z_scores, signals, mids, spreads, greeks, current_spot
        )

        # 7. Calculate Portfolio Equity & Track Performance
        total_equity = portfolio.compute_total_equity(mids, current_spot)
        pnl = total_equity - portfolio.initial_cash
        pnl_history.append(pnl)

        # Record Execution Latency in microseconds
        latency_us = (time.perf_counter() - t_start) * 1_000_000.0
        latency_history.append(latency_us)

        print(
            f"Tick {step:02d} | DA Latency: {latency_us:.1f} µs | Total Equity: ${total_equity:,.2f} | PnL: ${pnl:+,.2f} | Stock Pos: {portfolio.underlying_position:+.0f} sh\n"
        )

    # Clean up feed task
    replay_task.cancel()
    print("Backtest Complete.")

    # Generate Institutional Tear-Sheet Performance Report
    generate_performance_report(
        pnl_history, latency_history, portfolio.initial_cash
    )


if __name__ == "__main__":
    asyncio.run(run_backtest())