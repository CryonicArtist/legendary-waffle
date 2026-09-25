import jax.numpy as jnp

def heston_step(state, params, dW_S, dW_v, dt):
    """
    Advances a single particle forward by dt under Heston dynamics.
    """
    S, v = state
    mu, kappa, theta, xi, rho = params
    
    # Full Truncation to prevent negative variance
    v_pos = jnp.maximum(v, 0.0)
    
    # Correlated Brownian motions
    dZ_v = dW_v
    dZ_S = rho * dW_v + jnp.sqrt(1 - rho**2) * dW_S
    
    # Euler-Maruyama discretization
    S_new = S + mu * S * dt + jnp.sqrt(v_pos) * S * jnp.sqrt(dt) * dZ_S
    v_new = v + kappa * (theta - v_pos) * dt + xi * jnp.sqrt(v_pos) * jnp.sqrt(dt) * dZ_v
    
    return jnp.array([S_new, v_new])