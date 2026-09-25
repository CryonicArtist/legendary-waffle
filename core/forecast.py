from jax import vmap, random
from core.heston import heston_step

# Vectorize over particles (axis 0). Keep params and dt un-batched (None)
batch_heston_step = vmap(heston_step, in_axes=(0, None, 0, 0, None))

def run_forecast(key, particles, params, dt, num_particles):
    """
    Drives the entire ensemble of particles forward simultaneously.
    """
    key, subkey1, subkey2 = random.split(key, 3)
    
    # Generate random shocks for the whole batch
    dW_S = random.normal(subkey1, (num_particles,))
    dW_v = random.normal(subkey2, (num_particles,))
    
    # Push batch forward
    new_particles = batch_heston_step(particles, params, dW_S, dW_v, dt)
    
    return new_particles, key