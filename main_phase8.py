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
from connectors.paper_broker import LivePaperBroker
from execution.portfolio import PortfolioManager
from execution.microstructure import MicrostructureModel
from risk.limits import RiskLimitManager
from backtest.metrics import generate_performance_report

async def run_phase8_paper_trading():
    dataset_path = "ticks_sample.parquet"
    strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
    key = random.PRNGKey(2026)
    
    # 1. Initialize Broker API Gateway
    broker = LivePaperBroker(simulated_rtt_ms=10.0)
    await broker.connect()
    await broker.subscribe_quotes(strikes)

    buffer = MarketDataBuffer(num_strikes=5)
    replayer = ParquetTickReplayer(buffer, dataset_path, strikes)
    replay_task = asyncio.create_task(replayer.stream_ticks(playback_speed_ms=20.0))
    
    # Wait for first live tick frame
    while True:
        mids, spreads, has_data = await buffer.pop_tick()
        if has_data:
            break
        await asyncio.sleep(0.002)

    # 2. Calibrate Surface Prior
    print("\n[PHASE 8] Calibrating Heston Surface Prior...")
    calibrator = HestonSurfaceCalibrator(num_steps=200)
    calibrated_params, calibrated_v0 = calibrator.calibrate(
        spot=100.0, strikes=strikes, market_mids=mids, T=0.25, r=0.05, option_type=1.0
    )

    # 3. Particle Ensemble Initialization
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0)
    T, r, option_type = 0.25, 0.05, 1.0
    
    key, subkey_s, subkey_v = random.split(key, 3)
    init_spot = 100.0 + random.normal(subkey_s, (num_particles,)) * 0.05
    init_vol = calibrated_v0 * jnp.exp(random.normal(subkey_v, (num_particles,)) * 0.05)
    particles = jnp.stack([init_spot, init_vol], axis=1)
    weights = jnp.ones(num_particles) / num_particles

    portfolio = PortfolioManager(initial_cash=100_000.0)
    risk_manager = RiskLimitManager(max_portfolio_delta=50.0, max_drawdown_pct=2.0)
    microstructure = MicrostructureModel(impact_factor=0.02)

    print("\n--- Starting Phase 8 Live Paper-Trading Engine ---")
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

        # Engine DA Processing
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

        # Process Signals via Broker Interface
        current_spot = float(jnp.sum(particles[:, 0] * weights))
        
        for i in range(len(signals)):
            sig = signals[i]
            if sig == 0.0:
                continue
                
            current_pos = float(portfolio.option_positions[i])
            trade_qty = 5.0 if sig == 1.0 else -5.0
            
            if risk_manager.check_trade_allowed(strike_idx=i, trade_qty=trade_qty, current_pos=current_pos):
                fill_price, exec_qty = microstructure.compute_execution_fill(
                    qty=trade_qty, mid_price=float(mids[i]), spread=float(spreads[i])
                )
                side_str = "BUY" if exec_qty > 0 else "SELL"
                
                # Asynchronously route order over network to broker API
                await broker.send_order("SPY", float(strikes[i]), side_str, abs(exec_qty), price=fill_price)
                portfolio.execute_option_trade(strike_idx=i, qty=exec_qty, fill_price=fill_price)

        # Rebalance Underlying Hedge
        net_option_delta = jnp.sum(portfolio.option_positions) * greeks[0]
        portfolio.rebalance_delta_hedge(net_option_delta, current_spot)
        
        total_equity = portfolio.compute_total_equity(mids, current_spot)
        pnl = total_equity - portfolio.initial_cash
        pnl_history.append(pnl)

        latency_us = (time.perf_counter() - t_start) * 1_000_000.0
        latency_history.append(latency_us)

        print(f"Paper Tick {step:02d} | DA+Routing Latency: {latency_us:.1f} µs | Total Equity: ${total_equity:,.2f} | PnL: ${pnl:+,.2f}\n")

    replay_task.cancel()
    generate_performance_report(pnl_history, latency_history, portfolio.initial_cash)

if __name__ == "__main__":
    asyncio.run(run_phase8_paper_trading())