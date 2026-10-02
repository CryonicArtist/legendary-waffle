from jax import vmap
from pricer.black_scholes import black_scholes_price

# Vectorize over particles (axis 0 of state). Keep K, T, r, option_type scalar (None).
batch_bs_price = vmap(black_scholes_price, in_axes=(0, None, None, None, None))

def evaluate_particles_option_prices(particles, K, T, r, option_type=1.0):
    """
    Evaluates option observation operator H(x_t) across all particles simultaneously.
    Returns an array of shape (N,) containing option prices.
    """
    return batch_bs_price(particles, K, T, r, option_type)