import jax.numpy as jnp
from jax import vmap
from pricer.black_scholes import black_scholes_price

# Inner vmap: Single particle evaluated across M strikes K -> outputs shape (M,)
single_particle_chain_price = vmap(black_scholes_price, in_axes=(None, 0, None, None, None))

# Outer vmap: Vectorized across N particles -> outputs shape (N, M)
batch_chain_price = vmap(single_particle_chain_price, in_axes=(0, None, None, None, None))

def evaluate_particles_option_surface(particles, strikes, T, r, option_type=1.0):
    """
    Evaluates option prices across N particles and M strikes simultaneously.
    Returns: jnp.array of shape (N, M)
    """
    return batch_chain_price(particles, strikes, T, r, option_type)