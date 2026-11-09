import jax
import jax.numpy as jnp
from jax import jit, grad
from pricer.batch_pricer import evaluate_particles_option_surface

@jit
def surface_loss_fn(calib_params: jnp.ndarray, spot: float, strikes: jnp.ndarray, market_mids: jnp.ndarray, T: float, r: float, option_type: float):
    """
    Computes loss between model surface and empirical market quotes.
    calib_params: [v0, theta, kappa]
    """
    v0, theta, kappa = calib_params[0], calib_params[1], calib_params[2]
    
    # Enforce lower bound on candidate variance
    v0_clamped = jnp.maximum(v0, 0.005)
    candidate_particle = jnp.array([[spot, v0_clamped]])
    
    # Surface evaluation across option chain
    model_prices = evaluate_particles_option_surface(candidate_particle, strikes, T, r, option_type)[0]
    
    # Mean Squared Relative Error Loss (weights all strikes evenly)
    rel_errors = (model_prices - market_mids) / (market_mids + 1e-4)
    return jnp.mean(rel_errors ** 2)

class HestonSurfaceCalibrator:
    """
    Gradient-descent surface calibrator with parameter-wise gradient scaling.
    """
    def __init__(self, num_steps: int = 200):
        self.num_steps = num_steps
        self.grad_loss = jit(grad(surface_loss_fn))

    def calibrate(self, spot: float, strikes: jnp.ndarray, market_mids: jnp.ndarray, T: float = 0.25, r: float = 0.05, option_type: float = 1.0):
        # Start centered near 20% volatility (v0 = 0.04)
        calib_params = jnp.array([0.04, 0.04, 2.0])
        
        # Parameter-specific learning rates: v0, theta, kappa
        step_sizes = jnp.array([1e-4, 1e-4, 1e-2])
        
        for step in range(self.num_steps):
            grads = self.grad_loss(calib_params, spot, strikes, market_mids, T, r, option_type)
            
            # Gradient update
            calib_params = calib_params - step_sizes * grads
            
            # Enforce physical market bounds (10% <= Vol <= 45%)
            calib_params = calib_params.at[0].set(jnp.clip(calib_params[0], 0.01, 0.20))
            calib_params = calib_params.at[1].set(jnp.clip(calib_params[1], 0.01, 0.20))
            calib_params = calib_params.at[2].set(jnp.clip(calib_params[2], 0.5, 5.0))

        v0_opt, theta_opt, kappa_opt = float(calib_params[0]), float(calib_params[1]), float(calib_params[2])
        print(f"[CALIBRATION] Optimized Surface Parameters -> Initial Var (v0): {v0_opt:.4f} (Vol: {jnp.sqrt(v0_opt)*100:.2f}%), Theta: {theta_opt:.4f}, Kappa: {kappa_opt:.2f}")
        
        return jnp.array([0.05, kappa_opt, theta_opt, 0.1, -0.7]), v0_opt