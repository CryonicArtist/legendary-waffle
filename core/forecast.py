import jax.numpy as jnp
from jax import jit, random, vmap
from core.heston import heston_step

@jit(static_argnums=(4,))
def run_forecast(key, particles, params, dt, num_particles=10000):
    N = particles.shape[0]
    key, subkey1, subkey2 = random.split(key, 3)
    
    dW_S = random.normal(subkey1, (N,))
    dW_v_raw = random.normal(subkey2, (N,))
    
    rho = params[4]
    dW_v = rho * dW_S + jnp.sqrt(1.0 - rho**2) * dW_v_raw
    
    batch_heston = vmap(heston_step, in_axes=(0, None, None, 0, 0))
    updated_particles = batch_heston(particles, params, dt, dW_S, dW_v)
    
    return updated_particles, key