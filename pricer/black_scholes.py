import jax.numpy as jnp
from jax.scipy.stats import norm

def black_scholes_price(state, K, T, r, option_type=1.0):
    """
    Computes the Black-Scholes price for a single particle's state [S, v].
    
    Parameters:
    - state: jnp.array([S, v]) where S is spot, v is variance
    - K: Strike price
    - T: Time to maturity (years)
    - r: Risk-free rate
    - option_type: 1.0 for Call, -1.0 for Put
    """
    S, v = state
    # Ensure variance is non-negative and non-zero
    v_pos = jnp.maximum(v, 1e-8)
    sigma = jnp.sqrt(v_pos)
    
    std_dev = sigma * jnp.sqrt(T)
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / std_dev
    d2 = d1 - std_dev
    
    call_price = S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)
    put_price = K * jnp.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return jnp.where(option_type >= 0.0, call_price, put_price)