import jax.numpy as jnp
from jax import random

def resample_particles(key, particles, weights):
    """
    Multinomial resampling. Clones high-weight particles and drops low-weight ones.
    """
    num_particles = particles.shape[0]
    
    # Split key for randomness
    key, subkey = random.split(key)
    
    # Sample new indices based on current weights
    indices = random.choice(
        subkey, 
        jnp.arange(num_particles), 
        shape=(num_particles,), 
        p=weights, 
        replace=True
    )
    
    # Create new particle ensemble and reset weights to uniform
    resampled_particles = particles[indices]
    uniform_weights = jnp.ones(num_particles) / num_particles
    
    return resampled_particles, uniform_weights, key