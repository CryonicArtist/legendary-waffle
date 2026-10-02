import jax
import jax.numpy as jnp
from jax import random

from core.forecast import run_forecast
from assimilation.update import update_weights
from assimilation.resample import resample_particles
from pricer.black_scholes import black_scholes_price

def main():
    # 1. Configuration
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0)  # 1-minute time steps
    steps = 5
    
    # Option Contract Specs: ATM Call option, 3 months (0.25 yr) to expiry
    K = 100.0       # Strike Price
    T = 0.25        # Time to Expiration (Years)
    r = 0.05        # 5% Risk-free rate
    option_type = 1.0 # 1.0 = Call Option
    
    # Heston Parameters: [mu, kappa, theta, xi, rho]
    params = jnp.array([0.05, 2.0, 0.04, 0.1, -0.7])
    
    # Initialize state: [Spot = $100.0, Volatility = 0.04 (20% annualized vol)]
    key = random.PRNGKey(42)
    particles = jnp.ones((num_particles, 2)) * jnp.array([100.0, 0.04])
    weights = jnp.ones(num_particles) / num_particles
    
    # Calculate baseline option price at starting state
    base_price = black_scholes_price(jnp.array([100.0, 0.04]), K, T, r, option_type)
    print(f"Booting Option Data Assimilation Filter. N={num_particles} Particles.")
    print(f"Tracking Call Option (K={K}, T={T}yr). Theoretical Base Price: ${base_price:.3f}\n")
    
    # 2. Live Trading Loop (Simulating live option market quotes)
    # Market option ticks rise from $4.61 up to $4.85 due to underlying movement/vol shift
    simulated_option_ticks = [4.616, 4.670, 4.720, 4.790, 4.850]
    
    for t in range(steps):
        print(f"--- Time Step {t+1} ---")
        
        # A. Forecast Step: Push ensemble forward via Heston SDE
        particles, key = run_forecast(key, particles, params, dt, num_particles)
        
        # B. Ingest Live Option Quote
        market_option_tick = simulated_option_ticks[t]
        
        # C. Assimilation Step: Update particle weights using option price likelihood
        # Observation noise variance (accounts for bid-ask bounce/spread noise)
        observation_variance = 0.0005 
        weights = update_weights(
            particles, weights, market_option_tick, observation_variance, K, T, r, option_type
        )
        
        # D. Check Degeneracy & Resample
        ess = 1.0 / jnp.sum(weights**2)
        print(f"Option Market Tick: ${market_option_tick:.3f} | ESS: {ess:.0f}/{num_particles}")
        
        if ess < (num_particles / 2.0):
            print("Degeneracy detected. Resampling particles...")
            particles, weights, key = resample_particles(key, particles, weights)
            
        # E. Extract Latent State (Posterior expected Spot & Vol)
        expected_spot = jnp.sum(particles[:, 0] * weights)
        expected_vol = jnp.sum(particles[:, 1] * weights)
        implied_annual_vol = jnp.sqrt(expected_vol) * 100
        
        print(f"Latent State -> Extracted Spot: ${expected_spot:.2f} | Extracted Vol: {implied_annual_vol:.2f}%\n")

if __name__ == "__main__":
    main()