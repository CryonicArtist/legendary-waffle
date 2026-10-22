import jax.numpy as jnp
from jax import jit

@jit
def heston_step(state, params, dt, dW_S, dW_v):
    S, v = state[0], state[1]
    mu, kappa, theta, xi, rho = params[0], params[1], params[2], params[3], params[4]
    
    # Floor variance before calculation to avoid sqrt of negative values
    v_pos = jnp.maximum(v, 1e-6)
    
    # SDE Discretization
    dS = mu * S * dt + jnp.sqrt(v_pos) * S * dW_S
    dv = kappa * (theta - v_pos) * dt + xi * jnp.sqrt(v_pos) * dW_v
    
    S_new = jnp.maximum(S + dS, 1.0)
    v_new = jnp.maximum(v + dv, 1e-6)
    
    return jnp.array([S_new, v_new])