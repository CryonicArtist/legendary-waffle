import jax
import jax.numpy as jnp
from jax import random, jit

@jit
def resample_and_regularize(key, particles, weights, a=0.98):
    """
    JIT-compiled systematic resampling with Liu-West Kernel Smoothing 
    and an Absolute Physical Variance Floor to prevent sample impoverishment.
    """
    num_particles = particles.shape[0]
    key, subkey1, subkey2 = random.split(key, 3)

    # 1. Systematic Resampling
    cumsum_weights = jnp.cumsum(weights)
    u = (random.uniform(subkey1) + jnp.arange(num_particles)) / num_particles
    indices = jnp.searchsorted(cumsum_weights, u)
    resampled_particles = particles[indices]

    # 2. Liu-West Kernel Smoothing with Variance Floor
    mean = jnp.mean(resampled_particles, axis=0)
    sample_var = jnp.var(resampled_particles, axis=0)

    # VARIANCE FLOOR: Force minimum jitter (Spot std >= $0.25, Vol std >= 0.005)
    # This prevents zero-jitter locks when sample_var collapses
    min_var = jnp.array([0.25**2, 0.005**2])
    effective_var = jnp.maximum(sample_var, min_var)

    # Shrink centers toward mean: m_i = a * x_i + (1 - a) * mean
    shrunk_centers = a * resampled_particles + (1.0 - a) * mean

    # Add kernel jitter: ~ N(0, (1 - a^2) * effective_var)
    noise_std = jnp.sqrt((1.0 - a**2) * effective_var)
    noise = random.normal(subkey2, shape=resampled_particles.shape) * noise_std
    regularized_particles = shrunk_centers + noise

    # Clean boundaries (S >= $1.0, v >= 1e-6)
    s_clean = jnp.maximum(regularized_particles[:, 0], 1.0)
    v_clean = jnp.maximum(regularized_particles[:, 1], 1e-6)
    clean_particles = jnp.stack([s_clean, v_clean], axis=1)

    # Reset weights
    new_weights = jnp.ones(num_particles) / num_particles

    return clean_particles, new_weights, key