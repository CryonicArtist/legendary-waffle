import jax.numpy as jnp
from jax import jit

@jit
def heston_step(state, params, dt, dW_S, dW_v):
    S, v = state[0], state[1]
    mu, kappa, theta, xi, rho = params[0], params[1], params[2], params[3], params[4]
    
    # Floor variance before calculation to avoid sqrt of negative values
    v_pos = jnp.maximum(v, 1e-6)
    
    # Scale Brownian motion increments by sqrt(dt)
    sqrt_dt = jnp.sqrt(dt)
    
    # Correct Euler-Maruyama Discretization: dW = sqrt(dt) * Z
    dS = mu * S * dt + jnp.sqrt(v_pos) * S * (dW_S * sqrt_dt)
    dv = kappa * (theta - v_pos) * dt + xi * jnp.sqrt(v_pos) * (dW_v * sqrt_dt)
    
    S_new = jnp.maximum(S + dS, 1.0)
    v_new = jnp.maximum(v + dv, 1e-6)
    
    return jnp.array([S_new, v_new])