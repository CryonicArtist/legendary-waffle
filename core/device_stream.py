import jax
import jax.numpy as jnp
from jax import device_put

class CUDAStreamManager:
    """
    Manages non-blocking JAX device allocations for low-latency host-to-device transfers.
    Safely falls back to CPU if CUDA runtime is not present.
    """
    def __init__(self):
        try:
            # Check for NVIDIA CUDA accelerators
            self.gpu_device = jax.devices("cuda")[0]
        except RuntimeError:
            try:
                # Check for AMD ROCm accelerators
                self.gpu_device = jax.devices("rocm")[0]
            except RuntimeError:
                # Fall back to CPU host execution
                self.gpu_device = jax.devices("cpu")[0]

        print(f">> Initialized Execution Target Device: {self.gpu_device}")

    def transfer_to_device(self, host_tensor: jnp.ndarray) -> jax.Array:
        """Transfers host memory tensors asynchronously to target GPU/CPU memory."""
        return device_put(host_tensor, self.gpu_device)