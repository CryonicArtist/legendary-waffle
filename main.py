import jax
import jax.numpy as jnp
from jax import random

# Import modularized functions
from core.forecast import run_forecast
from assimilation.update import update_weights
from assimilation.resample import resample_particles

def main():
    # 1. Configuration
    num_particles = 10000
    dt = 1.0 / (252.0 * 390.0) # 1-minute steps
    steps = 5 # Simulating 5 minutes of live order book ticks
    
    # Parameters: [mu, kappa, theta, xi, rho]
    params = jnp.array([0.05, 2.0, 0.04, 0.1, -0.7])
    
    # Initialize PRNG key and state
    key = random.PRNGKey(42)
    particles = jnp.ones((num_particles, 2)) * jnp.array([100.0, 0.04]) # [Spot, Vol]
    weights = jnp.ones(num_particles) / num_particles
    
    print(f"Booting Particle Filter. N={num_particles} Particles.\n")
    
    # 2. The Trading / Assimilation Loop
    for t in range(steps):
        print(f"--- Time Step {t+1} ---")
        
        # A. Forecast Step (Drive SDE forward)
        particles, key = run_forecast(key, particles, params, dt, num_particles)
        
        # B. Receive Live Market Tick (Simulated here as 100.05, 100.12, etc.)
        # In production, this pulls from Databento/Polygon websocket
        simulated_market_tick = 100.0 + (t * 0.05) 
        
        # C. Update Step (Data Assimilation)
        observation_variance = 0.01
        weights = update_weights(particles, weights, simulated_market_tick, observation_variance)
        
        # Calculate Effective Sample Size (ESS) to check for degeneracy
        ess = 1.0 / jnp.sum(weights**2)
        print(f"Market Tick: {simulated_market_tick:.2f} | ESS: {ess:.0f}/{num_particles}")
        
        # D. Resample if degenerate (e.g., ESS drops below 50%)
        if ess < (num_particles / 2.0):
            print("Degeneracy detected. Resampling particles...")
            particles, weights, key = resample_particles(key, particles, weights)
            
        # E. Calculate Expected State (Mean of the posterior distribution)
        expected_spot = jnp.sum(particles[:, 0] * weights)
        expected_vol = jnp.sum(particles[:, 1] * weights)
        print(f"Latent State Update -> Spot: {expected_spot:.2f}, Vol: {expected_vol:.4f}\n")

if __name__ == "__main__":
    main()