import jax.numpy as jnp
from pricer.batch_pricer import evaluate_particles_option_surface
from jax import jit

@jit
def update_weights(particles, current_weights, market_mids, bid_ask_spreads, strikes, z_scores, T, r, option_type=1.0):
    """
    Updates particle weights using a vector of option chain quotes Y_t and dynamic noise R_t.
    """
    # 1. Dynamic Observation Noise Variance R_t per contract: (Spread / 2)^2
    obs_variances = jnp.maximum((bid_ask_spreads / 2.0)**2, 1e-6)  # Shape: (M,)
    
    # 2. Contract Masking: If a contract has |Z| >= 4.0, set its noise variance to infinity
    # so it contributes 0 to the log-likelihood sum without breaking state updates for healthy contracts.
    mask = jnp.abs(z_scores) < 4.0
    effective_variances = jnp.where(mask, obs_variances, 1e12)
    
    # 3. Model predictions across surface: Shape (N, M)
    expected_surface = evaluate_particles_option_surface(particles, strikes, T, r, option_type)
    
    # 4. Innovations E = Y_t - H(x_t): Shape (N, M)
    innovations = market_mids - expected_surface
    
    # 5. Multivariate Log-Likelihood: Sum of squared errors weighted by dynamic noise
    log_likelihoods = -0.5 * jnp.sum((innovations**2) / effective_variances, axis=1)  # Shape: (N,)
    
    # 6. Log-Sum-Exp Trick to prevent exponential underflow
    max_log_lik = jnp.max(log_likelihoods)
    likelihoods = jnp.exp(log_likelihoods - max_log_lik)
    likelihoods = jnp.maximum(likelihoods, 1e-12)
    
    # 7. Normalize weights
    unnormalized_weights = current_weights * likelihoods
    weight_sum = jnp.sum(unnormalized_weights)
    safe_sum = jnp.where(weight_sum < 1e-12, 1e-12, weight_sum)
    
    return unnormalized_weights / safe_sum