import jax.numpy as jnp
from jax.scipy.stats import norm

def black_scholes_price(state, K, T, r, option_type=1.0):
    S = jnp.maximum(state[0], 1.0)
    v = jnp.maximum(state[1], 1e-6)
    sigma = jnp.sqrt(v)
    
    std_dev = jnp.maximum(sigma * jnp.sqrt(T), 1e-6)
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / std_dev
    d2 = d1 - std_dev
    
    call_price = S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)
    put_price = K * jnp.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return jnp.where(option_type >= 0.0, call_price, put_price)