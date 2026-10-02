import jax.numpy as jnp
from pricer.batch_pricer import evaluate_particles_option_prices

def evaluate_arbitrage_signal(particles, weights, market_option_price, K, T, r, option_type=1.0, z_threshold=2.0):
    """
    Evaluates statistical mispricing of an incoming market option tick.
    
    Returns:
    - expected_price: Model ensemble's posterior mean price
    - std_dev: Model posterior price uncertainty
    - z_score: Statistical distance of market quote from model expectation
    - signal: +1.0 (BUY / Underpriced), -1.0 (SELL / Overpriced), 0.0 (HOLD / Fair)
    """
    particle_prices = evaluate_particles_option_prices(particles, K, T, r, option_type)
    
    # Weighted mean and variance of theoretical model prices
    expected_price = jnp.sum(particle_prices * weights)
    variance = jnp.sum(weights * (particle_prices - expected_price)**2)
    std_dev = jnp.sqrt(jnp.maximum(variance, 1e-12))
    
    # Calculate Z-score: positive if market is priced above model
    z_score = (market_option_price - expected_price) / std_dev
    
    # Generate signal based on standard deviations away from ensemble expectation
    # Market < Model by threshold -> Market option is CHEAP -> BUY (+1)
    # Market > Model by threshold -> Market option is EXPENSIVE -> SELL (-1)
    signal = jnp.where(z_score < -z_threshold, 1.0, 
                jnp.where(z_score > z_threshold, -1.0, 0.0))
    
    return expected_price, std_dev, z_score, signal