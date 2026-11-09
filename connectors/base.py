from abc import ABC, abstractmethod
import jax.numpy as jnp

class BaseExchangeConnector(ABC):
    """
    Abstract Base Class for live broker/exchange connections (REST / WebSocket / FIX).
    """
    @abstractmethod
    async def connect(self):
        """Establishes WebSocket data streams and REST session authentication."""
        pass

    @abstractmethod
    async def subscribe_quotes(self, strikes: jnp.ndarray):
        """Subscribes to live L2/L3 quote streams for target strikes."""
        pass

    @abstractmethod
    async def send_order(self, symbol: str, strike: float, side: str, qty: float, order_type: str = "LIMIT", price: float = None) -> dict:
        """Submits an order to the venue execution gateway."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancels an active working order."""
        pass