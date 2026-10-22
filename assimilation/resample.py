import jax
import jax.numpy as jnp
from jax import random, jit

@jit
def resample_and_regularize(key, particles, weights, a=0.98):
    """
    JIT-compiled systematic resampling followed by Liu-West Kernel Smoothing.
    
    Parameters:
    - a: Shrinkage parameter (default 0.98). Preserves true posterior mean and covariance.
    """
    num_particles = particles.shape[0]
    key, subkey1, subkey2 = random.split(key, 3)

    # 1. JIT-compatible Systematic Resampling
    cumsum_weights = jnp.cumsum(weights)
    u = (random.uniform(subkey1) + jnp.arange(num_particles)) / num_particles
    indices = jnp.searchsorted(cumsum_weights, u)
    resampled_particles = particles[indices]

    # 2. Liu-West Kernel Smoothing Regularization
    mean = jnp.mean(resampled_particles, axis=0)
    var = jnp.var(resampled_particles, axis=0)

    # Shrink centers: m_i = a * x_i + (1 - a) * mean
    shrunk_centers = a * resampled_particles + (1.0 - a) * mean

    # Add kernel jitter: ~ N(0, (1 - a^2) * var)
    noise_std = jnp.sqrt(jnp.maximum((1.0 - a**2) * var, 1e-8))
    noise = random.normal(subkey2, shape=resampled_particles.shape) * noise_std
    regularized_particles = shrunk_centers + noise

    # Ensure physical boundaries (S > $1.0, v > 1e-8)
    s_clean = jnp.maximum(regularized_particles[:, 0], 1.0)
    v_clean = jnp.maximum(regularized_particles[:, 1], 1e-8)
    clean_particles = jnp.stack([s_clean, v_clean], axis=1)

    # Reset weights
    new_weights = jnp.ones(num_particles) / num_particles

    return clean_particles, new_weights, key