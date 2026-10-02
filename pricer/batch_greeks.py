import jax.numpy as jnp
from jax import vmap
from pricer.greeks import single_particle_greeks

# Vectorize over particles (axis 0 of state)
batch_greeks_fn = vmap(single_particle_greeks, in_axes=(0, None, None, None, None))

def compute_ensemble_greeks(particles, weights, K, T, r, option_type=1.0):
    """
    Computes weighted ensemble averages for Delta, Gamma, and Vega across all particles.
    Returns: jnp.array([ensemble_delta, ensemble_gamma, ensemble_vega])
    """
    # Evaluate Greeks for all particles in parallel: shape (N, 3)
    particle_greeks = batch_greeks_fn(particles, K, T, r, option_type)

    # Compute posterior weighted expected Greeks
    ensemble_delta = jnp.sum(particle_greeks[:, 0] * weights)
    ensemble_gamma = jnp.sum(particle_greeks[:, 1] * weights)
    ensemble_vega = jnp.sum(particle_greeks[:, 2] * weights)

    return jnp.array([ensemble_delta, ensemble_gamma, ensemble_vega])