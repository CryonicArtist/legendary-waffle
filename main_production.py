import asyncio
import time
import os
import jax
import jax.numpy as jnp
from jax import random

from core.forecast import run_forecast
from core.calibration import HestonSurfaceCalibrator
from core.device_stream import CUDAStreamManager
from assimilation.update import update_weights
from assimilation.resample import resample_and_regularize
from pricer.batch_greeks import compute_ensemble_greeks
from signals.mispricing import evaluate_surface_arbitrage_signals
from ingest.ring_buffer import MarketDataBuffer
from ingest.simulated_stream import SimulatedLiveStreamer
from connectors.paper_broker import LivePaperBroker
from execution.portfolio import PortfolioManager
from execution.microstructure import MicrostructureModel
from risk.limits import RiskLimitManager
from telemetry.metrics_exporter import EngineTelemetryExporter
from backtest.metrics import generate_performance_report

async def run_production_engine():
    print("==================================================")
    print("  LAUNCHING QUANTITATIVE OPTIONS ENGINE (LIVE WS) ")
    print("==================================================")

    stream_mgr = CUDAStreamManager()
    telemetry = EngineTelemetryExporter()
    
    # CPU Production Tuning (Sub-15ms Real-Time Response)
    num_particles = 10000
    strikes = stream_mgr.transfer_to_device(jnp.linspace(500.0, 520.0, 5))
    num_strikes = len(strikes)
    
    key = random.PRNGKey(2026)
    
    # 1. Connect Broker Order Gateway
    broker = LivePaperBroker(simulated_rtt_ms=10.0)
    await broker.connect()
    await broker.subscribe_quotes(strikes)

    # 2. Initialize zero-cost live streamer
    buffer = MarketDataBuffer(num_strikes=num_strikes)
    streamer = SimulatedLiveStreamer(buffer=buffer, strikes=strikes, tick_interval_sec=0.05)
    ws_task = asyncio.create_task(streamer.start_stream())

    # 3. Await first live valid quote frame
    print("\n>> Awaiting initial live market quotes from Streamer...")
    while True:
        mids, spreads, has_data = await buffer.pop_tick()
        if has_data and float(jnp.sum(mids)) > 0:
            print(">> Valid market quote frame received.")
            break
        await asyncio.sleep(0.01)

    # 4. Surface Prior Calibration on Live Market Quotes
    print("\n[PRODUCTION] Calibrating Heston Surface Prior to Live Market Chain...")
    calibrator = HestonSurfaceCalibrator(num_steps=200)
    calibrated_params, calibrated_v0 = calibrator.calibrate(
        spot=510.0, strikes=strikes, market_mids=mids, T=0.25, r=0.05, option_type=1.0
    )

    # 5. Particle Ensemble Initialization
    dt = 1.0 / (252.0 * 390.0)
    T, r, option_type = 0.25, 0.05, 1.0
    
    key, subkey_s, subkey_v = random.split(key, 3)
    init_spot = 510.0 + random.normal(subkey_s, (num_particles,)) * 0.05
    init_vol = calibrated_v0 * jnp.exp(random.normal(subkey_v, (num_particles,)) * 0.05)
    particles = stream_mgr.transfer_to_device(jnp.stack([init_spot, init_vol], axis=1))
    weights = stream_mgr.transfer_to_device(jnp.ones(num_particles) / num_particles)

    portfolio = PortfolioManager(num_strikes=num_strikes, initial_cash=100_000.0)
    risk_manager = RiskLimitManager(max_portfolio_delta=50.0, max_drawdown_pct=2.0)
    microstructure = MicrostructureModel(impact_factor=0.02)

    # 6. Comprehensive XLA JIT Warm-Up
    print("\nExecuting Comprehensive XLA Warm-Up...")
    t_warmup = time.perf_counter()
    mids_dev = stream_mgr.transfer_to_device(mids)
    spreads_dev = stream_mgr.transfer_to_device(spreads)
    
    p_dummy, k_dummy = run_forecast(key, particles, calibrated_params, dt, num_particles)
    exp_p, std_d, z_sc, sigs = evaluate_surface_arbitrage_signals(
        p_dummy, weights, mids_dev, strikes, T, r, option_type, z_threshold=2.0
    )
    w_dummy = update_weights(p_dummy, weights, mids_dev, spreads_dev, strikes, z_sc, T, r, option_type)
    p_dummy, w_dummy, k_dummy = resample_and_regularize(k_dummy, p_dummy, w_dummy, a=0.98)
    g_dummy = compute_ensemble_greeks(p_dummy, w_dummy, 510.0, T, r, option_type)
    g_dummy.block_until_ready()
    print(f"XLA Warm-Up Complete in {(time.perf_counter() - t_warmup)*1000:.2f} ms.\n")

    print("--- Starting Phase 9 Live Production Execution Loop ---")
    step = 0
    pnl_history = []
    latency_history = []

    while step < 100:
        mids, spreads, has_data = await buffer.pop_tick()
        if not has_data:
            await asyncio.sleep(0.001)
            continue
            
        step += 1
        t_start = time.perf_counter()

        mids_dev = stream_mgr.transfer_to_device(mids)
        spreads_dev = stream_mgr.transfer_to_device(spreads)

        # Step A: SDE Forecast
        particles, key = run_forecast(key, particles, calibrated_params, dt, num_particles)

        # Step B: Surface Pricing & Arbitrage Z-score Signals
        exp_prices, std_devs, z_scores, signals = evaluate_surface_arbitrage_signals(
            particles, weights, mids_dev, strikes, T, r, option_type, z_threshold=2.0
        )

        # Step C: Likelihood Weight Update
        weights = update_weights(particles, weights, mids_dev, spreads_dev, strikes, z_scores, T, r, option_type)
        
        # Step D: ESS Check & Resampling
        ess = float(1.0 / jnp.sum(weights**2))
        if ess < (num_particles / 2.0):
            particles, weights, key = resample_and_regularize(key, particles, weights, a=0.98)

        # Step E: Pathwise AAD Greeks
        greeks = compute_ensemble_greeks(particles, weights, 510.0, T, r, option_type)
        greeks.block_until_ready()

        # Step F: Concurrent Signal Order Routing
        current_spot = float(jnp.sum(particles[:, 0] * weights))
        order_tasks = []
        
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
                
                order_tasks.append(
                    broker.send_order("SPY", float(strikes[i]), side_str, abs(exec_qty), price=fill_price)
                )
                portfolio.execute_option_trade(strike_idx=i, qty=exec_qty, fill_price=fill_price)

        if order_tasks:
            await asyncio.gather(*order_tasks)

        # Step G: Delta Neutral Rebalance
        net_option_delta = float(jnp.sum(portfolio.option_positions) * greeks[0])
        portfolio.rebalance_delta_hedge(net_option_delta, current_spot)
        
        total_equity = portfolio.compute_total_equity(mids, current_spot)
        pnl = total_equity - portfolio.initial_cash
        pnl_history.append(pnl)

        latency_us = (time.perf_counter() - t_start) * 1_000_000.0
        latency_history.append(latency_us)

        telemetry.record_tick(latency_us, ess, pnl)
        telemetry.print_system_status(step)

    # Graceful Shutdown
    streamer.stop_stream()
    ws_task.cancel()
    
    print("\nLive Engine Session Concluded.")
    generate_performance_report(pnl_history, latency_history, portfolio.initial_cash)

if __name__ == "__main__":
    asyncio.run(run_production_engine())