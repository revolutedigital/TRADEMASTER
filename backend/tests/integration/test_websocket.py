"""Integration tests for WebSocket endpoints."""

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


class TestWebSocketAuth:
    def test_ws_market_without_token_is_rejected(self):
        """WebSocket connection without auth token should be rejected."""
        from app.main import create_app

        app = create_app()
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/market") as ws:
                ws.receive()
        assert exc_info.value.code == 4001

    def test_ws_portfolio_without_token_is_rejected(self):
        """Portfolio WebSocket without auth should be rejected."""
        from app.main import create_app

        app = create_app()
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/portfolio") as ws:
                ws.receive()
        assert exc_info.value.code == 4001

    def test_ws_with_valid_token(self, auth_token):
        """WebSocket with valid token should connect."""
        from app.main import create_app

        app = create_app()
        client = TestClient(app)
        with client.websocket_connect(f"/ws/market?token={auth_token}") as ws:
            assert ws.accepted_subprotocol is None
