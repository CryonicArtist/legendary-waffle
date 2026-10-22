import jax.numpy as jnp
from pricer.batch_pricer import evaluate_particles_option_surface
from jax import jit

@jit
def evaluate_surface_arbitrage_signals(particles, weights, market_mids, strikes, T, r, option_type=1.0, z_threshold=2.0):
    """
    Evaluates statistical mispricings across an entire option chain vector Y_t.
    """
    # Evaluate model predictions: Shape (N, M)
    particle_surfaces = evaluate_particles_option_surface(particles, strikes, T, r, option_type)
    
    # Weighted mean price per strike: Shape (M,)
    weights_col = weights[:, None]  # Shape (N, 1) for broadcasting
    expected_prices = jnp.sum(particle_surfaces * weights_col, axis=0)
    
    # Weighted variance per strike: Shape (M,)
    variance = jnp.sum(weights_col * (particle_surfaces - expected_prices)**2, axis=0)
    std_devs = jnp.sqrt(jnp.maximum(variance, 1e-12))
    
    # Z-scores per strike: Shape (M,)
    z_scores = (market_mids - expected_prices) / std_devs
    
    # Signals per strike: +1.0 (BUY), -1.0 (SELL), 0.0 (HOLD)
    signals = jnp.where(z_scores < -z_threshold, 1.0, 
                jnp.where(z_scores > z_threshold, -1.0, 0.0))
    
    return expected_prices, std_devs, z_scores, signals