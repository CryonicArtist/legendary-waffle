import asyncio
import json
import time
import websockets
import jax.numpy as jnp
from ingest.ring_buffer import MarketDataBuffer

def build_occ_symbol(underlying: str, expiry_yymmdd: str, strike: float, option_type: str = "C") -> str:
    """
    Formats option parameters into standard OCC ticker format.
    Example: build_occ_symbol("SPY", "261218", 500.0, "C") -> "O:SPY261218C00500000"
    """
    strike_int = int(round(strike * 1000))
    return f"O:{underlying.upper()}{expiry_yymmdd}{option_type.upper()}{strike_int:08d}"

class PolygonWebSocketStreamer:
    """
    Asynchronous WebSocket tick stream worker for Polygon.io Options Quotes.
    Pushes real-time multi-strike bid/ask vectors into MarketDataBuffer.
    """
    def __init__(
        self,
        buffer: MarketDataBuffer,
        api_key: str,
        underlying: str,
        expiry_yymmdd: str,
        strikes: jnp.ndarray,
        option_type: str = "C",
        ws_url: str = "wss://socket.polygon.io/options"
    ):
        self.buffer = buffer
        self.api_key = api_key
        self.ws_url = ws_url
        self.strikes = strikes
        self.num_strikes = len(strikes)
        
        # Build symbol mapping: OCC_Symbol -> strike_index
        self.symbol_to_idx = {}
        self.subscribe_params = []
        
        for idx, k in enumerate(strikes):
            occ_sym = build_occ_symbol(underlying, expiry_yymmdd, float(k), option_type)
            self.symbol_to_idx[occ_sym] = idx
            # Polygon quote prefix is 'Q.'
            self.subscribe_params.append(f"Q.{occ_sym}")
            
        # Real-time state cache vector
        self.latest_bids = [0.0] * self.num_strikes
        self.latest_asks = [0.0] * self.num_strikes
        self.is_running = False

    async def start_stream(self):
        """
        Main async loop: maintains WebSocket connection, authenticates,
        subscribes to quote streams, and pushes updates to ring buffer.
        """
        self.is_running = True
        print(f">> Connecting to Polygon.io WebSocket Feed ({len(self.subscribe_params)} streams)...")

        async for websocket in websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10):
            try:
                # 1. Receive Connection ACK
                conn_msg = await websocket.recv()
                print(f"   [WS ACK] {conn_msg}")

                # 2. Authenticate
                auth_payload = {"action": "auth", "params": self.api_key}
                await websocket.send(json.dumps(auth_payload))
                auth_resp = await websocket.recv()
                print(f"   [AUTH] {auth_resp}")

                # 3. Subscribe to Multi-Strike Quote Stream
                sub_payload = {"action": "subscribe", "params": ",".join(self.subscribe_params)}
                await websocket.send(json.dumps(sub_payload))
                sub_resp = await websocket.recv()
                print(f"   [SUB ACK] {sub_resp}\n>> Real-Time Market Data Stream Active.")

                # 4. Process Live Quotes Loop
                while self.is_running:
                    msg = await websocket.recv()
                    data_frames = json.loads(msg)

                    for frame in data_frames:
                        # Event type 'Q' = Quote Update
                        if frame.get("ev") == "Q":
                            sym = frame.get("sym")
                            if sym in self.symbol_to_idx:
                                idx = self.symbol_to_idx[sym]
                                bid = frame.get("bp", 0.0)
                                ask = frame.get("ap", 0.0)

                                if bid > 0 and ask > 0:
                                    self.latest_bids[idx] = bid
                                    self.latest_asks[idx] = ask

                                    # Compute mid-price vector and spread vector
                                    mids = []
                                    spreads = []
                                    for i in range(self.num_strikes):
                                        b, a = self.latest_bids[i], self.latest_asks[i]
                                        if b > 0 and a > 0:
                                            mids.append((b + a) / 2.0)
                                            spreads.append(max(a - b, 0.01))
                                        else:
                                            # Default placeholder for unquoted strikes
                                            mids.append(0.0)
                                            spreads.append(0.10)

                                    # Non-blocking push to engine ring buffer
                                    mids_tensor = jnp.array(mids)
                                    spreads_tensor = jnp.array(spreads)
                                    await self.buffer.push_tick(mids_tensor, spreads_tensor)

            except websockets.ConnectionClosed:
                print("[WS WARNING] Connection closed. Reconnecting in 2s...")
                await asyncio.sleep(2.0)
            except Exception as e:
                print(f"[WS ERROR] Streaming exception: {e}")
                await asyncio.sleep(1.0)

    def stop_stream(self):
        self.is_running = False