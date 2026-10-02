import jax
import jax.numpy as jnp
from pricer.black_scholes import black_scholes_price

def single_particle_greeks(state, K, T, r, option_type=1.0):
    """
    Computes exact Delta, Gamma, and Vega for a single particle state [S, v] 
    using JAX Automatic Adjoint Differentiation (AAD).
    """
    # Helper mapping spot S -> price for double differentiation (Gamma)
    def price_from_S(S_val):
        s_arr = jnp.array([S_val, state[1]])
        return black_scholes_price(s_arr, K, T, r, option_type)

    # Helper mapping state [S, v] -> price for gradients (Delta & Vega)
    def price_from_state(st_arr):
        return black_scholes_price(st_arr, K, T, r, option_type)

    # 1. First derivatives: dC/dS (Delta) and dC/dv
    grad_state = jax.grad(price_from_state)(state)
    delta = grad_state[0]
    dC_dv = grad_state[1]

    # Convert dC/dv -> Vega (dC/d_sigma): dC/d_sigma = dC/dv * 2 * sqrt(v)
    sigma = jnp.sqrt(jnp.maximum(state[1], 1e-8))
    vega = dC_dv * 2.0 * sigma

    # 2. Second derivative: d^2C / dS^2 (Gamma)
    gamma = jax.grad(jax.grad(price_from_S))(state[0])

    return jnp.array([delta, gamma, vega])