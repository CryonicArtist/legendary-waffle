import jax.numpy as jnp

def update_weights(particles, current_weights, market_price, observation_noise_variance):
    """
    Updates particle weights based on live market observations.
    (Note: Using a placeholder observation operator H(x) = S for now.
    In production, this would be your fast option pricer).
    """
    # Placeholder: Assuming we are observing the underlying spot directly for this example.
    # Replace `expected_prices` with your actual Option Pricer H(particles)
    expected_prices = particles[:, 0] 
    
    # Calculate Gaussian likelihood of the market observation given particle states
    innovations = market_price - expected_prices
    likelihoods = jnp.exp(-0.5 * (innovations**2) / observation_noise_variance)
    
    # Update and normalize weights
    unnormalized_weights = current_weights * likelihoods
    new_weights = unnormalized_weights / jnp.sum(unnormalized_weights)
    
    return new_weights