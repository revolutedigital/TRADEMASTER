"""WebSocket endpoint routes for real-time data streaming to frontend."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.websocket.hub import ws_hub
from app.core.security import authenticate_websocket
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


async def _authenticate_ws(websocket: WebSocket) -> bool:
    """Authenticate WebSocket connection. Closes with 4001 if unauthorized."""
    payload = await authenticate_websocket(websocket)
    if payload is None:
        await websocket.close(code=4001, reason="Authentication required")
        return False
    return True


@router.websocket("/ws/market")
async def market_stream_all(websocket: WebSocket):
    """Real-time market data stream for all symbols + portfolio + signals."""
    if not await _authenticate_ws(websocket):
        return
    await ws_hub.connect(websocket, "market:BTCUSDT", accept=True)
    await ws_hub.connect(websocket, "market:ETHUSDT", accept=False)
    await ws_hub.connect(websocket, "portfolio", accept=False)
    await ws_hub.connect(websocket, "signals", accept=False)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_hub.disconnect(websocket, "market:BTCUSDT")
        ws_hub.disconnect(websocket, "market:ETHUSDT")
        ws_hub.disconnect(websocket, "portfolio")
        ws_hub.disconnect(websocket, "signals")


@router.websocket("/ws/market/{symbol}")
async def market_stream(websocket: WebSocket, symbol: str):
    """Real-time market data stream for a symbol."""
    if not await _authenticate_ws(websocket):
        return
    channel = f"market:{symbol.upper()}"
    await ws_hub.connect(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_hub.disconnect(websocket, channel)


@router.websocket("/ws/portfolio")
async def portfolio_stream(websocket: WebSocket):
    """Real-time portfolio updates."""
    if not await _authenticate_ws(websocket):
        return
    await ws_hub.connect(websocket, "portfolio")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_hub.disconnect(websocket, "portfolio")


@router.websocket("/ws/signals")
async def signals_stream(websocket: WebSocket):
    """Real-time AI trading signals."""
    if not await _authenticate_ws(websocket):
        return
    await ws_hub.connect(websocket, "signals")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_hub.disconnect(websocket, "signals")
