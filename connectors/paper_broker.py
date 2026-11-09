import asyncio
import time
import random
from connectors.base import BaseExchangeConnector

class LivePaperBroker(BaseExchangeConnector):
    """
    Simulates a live brokerage/exchange execution venue with network RTT latency.
    """
    def __init__(self, simulated_rtt_ms: float = 15.0):
        self.rtt_ms = simulated_rtt_ms
        self.active_orders = {}
        self.order_counter = 0

    async def connect(self):
        print(f">> Connecting to Live Paper Broker Gateway (Simulated RTT: {self.rtt_ms:.1f}ms)...")
        await asyncio.sleep(0.05)  # Simulate auth handshake
        print(">> Broker Gateway Connected & Authenticated.")

    async def subscribe_quotes(self, strikes):
        print(f">> Subscribed to live WebSocket quote feeds for strikes: {strikes}")

    async def send_order(self, symbol: str, strike: float, side: str, qty: float, order_type: str = "LIMIT", price: float = None) -> dict:
        """
        Simulates sending an order over wire with network RTT latency.
        """
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:06d}"
        
        # Simulate Network Latency to Exchange (Inbound RTT)
        await asyncio.sleep((self.rtt_ms / 2.0) / 1000.0)
        
        order_record = {
            "order_id": order_id,
            "symbol": symbol,
            "strike": strike,
            "side": side,
            "qty": qty,
            "price": price,
            "status": "FILLED",
            "timestamp": time.time()
        }
        
        # Simulate Exchange Matching Engine ACK (Outbound RTT)
        await asyncio.sleep((self.rtt_ms / 2.0) / 1000.0)
        
        self.active_orders[order_id] = order_record
        print(f"   [BROKER ACK] Order {order_id} FILLED: {side} {qty:.0f}x {symbol} K={strike:.1f} @ ${price:.2f}")
        return order_record

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.active_orders:
            self.active_orders[order_id]["status"] = "CANCELLED"
            await asyncio.sleep((self.rtt_ms) / 1000.0)
            return True
        return False