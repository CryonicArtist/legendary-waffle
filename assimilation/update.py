import jax.numpy as jnp
from jax import jit
from pricer.batch_pricer import evaluate_particles_option_surface

@jit
def update_weights(particles, current_weights, market_mids, bid_ask_spreads, strikes, z_scores, T, r, option_type=1.0, gamma=0.05):
    """
    Updates particle weights with likelihood tempering (gamma) to maintain ensemble diversity.
    """
    model_risk_var = 0.015
    obs_variances = jnp.maximum((bid_ask_spreads / 2.0)**2, 1e-6) + model_risk_var
    
    mask = jnp.abs(z_scores) < 4.0
    effective_variances = jnp.where(mask, obs_variances, 1e12)
    
    expected_surface = evaluate_particles_option_surface(particles, strikes, T, r, option_type)
    innovations = market_mids - expected_surface
    
    # Raw log-likelihood across active contracts
    log_likelihoods = -0.5 * jnp.sum((innovations**2) / effective_variances, axis=1)
    
    # Apply tempering factor gamma to prevent exponent collapse
    tempered_log_lik = gamma * log_likelihoods
    
    max_log_lik = jnp.max(tempered_log_lik)
    likelihoods = jnp.exp(tempered_log_lik - max_log_lik)
    likelihoods = jnp.maximum(likelihoods, 1e-12)
    
    unnormalized_weights = current_weights * likelihoods
    weight_sum = jnp.sum(unnormalized_weights)
    safe_sum = jnp.where(weight_sum < 1e-12, 1e-12, weight_sum)
    
    return unnormalized_weights / safe_sum