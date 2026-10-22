import jax.numpy as jnp
from jax import jit
from pricer.batch_pricer import evaluate_particles_option_surface

@jit
def evaluate_surface_arbitrage_signals(particles, weights, market_mids, strikes, T, r, option_type=1.0, z_threshold=2.0):
    particle_surfaces = evaluate_particles_option_surface(particles, strikes, T, r, option_type)
    weights_col = weights[:, None]
    expected_prices = jnp.sum(particle_surfaces * weights_col, axis=0)
    
    variance = jnp.sum(weights_col * (particle_surfaces - expected_prices)**2, axis=0)
    std_devs = jnp.sqrt(jnp.maximum(variance, 1e-8))
    
    # Safe Z-score calculation
    safe_std_devs = jnp.maximum(std_devs, 1e-6)
    z_scores = (market_mids - expected_prices) / safe_std_devs
    
    signals = jnp.where(z_scores < -z_threshold, 1.0, 
                jnp.where(z_scores > z_threshold, -1.0, 0.0))
    
    return expected_prices, std_devs, z_scores, signals