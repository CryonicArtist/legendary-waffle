import jax.numpy as jnp
from jax import jit
from pricer.batch_pricer import evaluate_particles_option_surface

@jit
def update_weights(particles, current_weights, market_mids, bid_ask_spreads, strikes, z_scores, T, r, option_type=1.0):
    """
    Updates particle weights with model risk variance floor and contract-normalized likelihoods.
    """
    # 1. Add Model Risk Variance Floor (0.002) to Bid-Ask Noise R_t
    model_risk_var = 0.002
    obs_variances = jnp.maximum((bid_ask_spreads / 2.0)**2, 1e-6) + model_risk_var
    
    # 2. Mask outliers (|Z| >= 4.0)
    mask = jnp.abs(z_scores) < 4.0
    effective_variances = jnp.where(mask, obs_variances, 1e12)
    
    # 3. Model predictions and innovations: Shape (N, M)
    expected_surface = evaluate_particles_option_surface(particles, strikes, T, r, option_type)
    innovations = market_mids - expected_surface
    
    # 4. Average Log-Likelihood per Contract (Normalized by M = active strikes)
    num_active_contracts = jnp.maximum(jnp.sum(mask), 1.0)
    log_likelihoods = -0.5 * jnp.sum((innovations**2) / effective_variances, axis=1) / num_active_contracts
    
    # 5. Log-Sum-Exp Trick
    max_log_lik = jnp.max(log_likelihoods)
    likelihoods = jnp.exp(log_likelihoods - max_log_lik)
    likelihoods = jnp.maximum(likelihoods, 1e-12)
    
    # 6. Weight Normalization
    unnormalized_weights = current_weights * likelihoods
    weight_sum = jnp.sum(unnormalized_weights)
    safe_sum = jnp.where(weight_sum < 1e-12, 1e-12, weight_sum)
    
    return unnormalized_weights / safe_sum