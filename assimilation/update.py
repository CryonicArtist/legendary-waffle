import jax.numpy as jnp
from pricer.batch_pricer import evaluate_particles_option_prices

def update_weights(particles, current_weights, market_option_price, observation_noise_variance, K, T, r, option_type=1.0):
    """
    Updates particle weights based on live option contract prices.
    """
    # 1. Map particle latent states [S_i, v_i] -> Expected Option Price H(x_i)
    expected_option_prices = evaluate_particles_option_prices(particles, K, T, r, option_type)
    
    # 2. Innovation: Difference between live market option tick and particle predictions
    innovations = market_option_price - expected_option_prices
    
    # 3. Gaussian likelihood calculation
    likelihoods = jnp.exp(-0.5 * (innovations**2) / observation_noise_variance)
    
    # 4. Multiply and normalize posterior weights
    unnormalized_weights = current_weights * likelihoods
    weight_sum = jnp.sum(unnormalized_weights)
    
    # Prevent divide-by-zero numerical instability
    safe_sum = jnp.where(weight_sum < 1e-12, 1e-12, weight_sum)
    new_weights = unnormalized_weights / safe_sum
    
    return new_weights