"""FIX 4.4 Protocol adapter for institutional-grade exchange connectivity.

Provides ultra-low latency order routing via the Financial Information eXchange protocol.
This is a simulation/framework - production use requires a proper FIX engine (QuickFIX).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class FIXMsgType(str, Enum):
    NEW_ORDER = "D"
    CANCEL = "F"
    REPLACE = "G"
    EXECUTION_REPORT = "8"
    HEARTBEAT = "0"
    LOGON = "A"
    LOGOUT = "5"
    REJECT = "3"


class FIXOrderType(str, Enum):
    MARKET = "1"
    LIMIT = "2"
    STOP = "3"
    STOP_LIMIT = "4"


class FIXSide(str, Enum):
    BUY = "1"
    SELL = "2"


@dataclass
class FIXMessage:
    msg_type: FIXMsgType
    sender_comp_id: str
    target_comp_id: str
    msg_seq_num: int
    fields: dict = field(default_factory=dict)
    sending_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_fix_string(self) -> str:
        """Serialize to FIX message format."""
        parts = [
            f"8=FIX.4.4",
            f"35={self.msg_type.value}",
            f"49={self.sender_comp_id}",
            f"56={self.target_comp_id}",
            f"34={self.msg_seq_num}",
            f"52={self.sending_time.strftime('%Y%m%d-%H:%M:%S.%f')[:-3]}",
        ]
        for tag, value in self.fields.items():
            parts.append(f"{tag}={value}")
        
        body = "\x01".join(parts) + "\x01"
        body_length = len(body)
        full_msg = f"8=FIX.4.4\x019={body_length}\x01{body}"
        
        # Calculate checksum
        checksum = sum(ord(c) for c in full_msg) % 256
        full_msg += f"10={checksum:03d}\x01"
        
        return full_msg


class FIXAdapter:
    """FIX 4.4 protocol adapter for institutional exchange connectivity."""

    def __init__(self, sender_comp_id: str = "TRADEMASTER", target_comp_id: str = "EXCHANGE"):
        self._sender = sender_comp_id
        self._target = target_comp_id
        self._seq_num = 0
        self._connected = False
        self._pending_orders: dict[str, FIXMessage] = {}

    def _next_seq(self) -> int:
        self._seq_num += 1
        return self._seq_num

    async def connect(self, host: str = "localhost", port: int = 9878) -> bool:
        """Establish FIX session with logon."""
        logon = FIXMessage(
            msg_type=FIXMsgType.LOGON,
            sender_comp_id=self._sender,
            target_comp_id=self._target,
            msg_seq_num=self._next_seq(),
            fields={"98": "0", "108": "30"},  # EncryptMethod=None, HeartBtInt=30s
        )
        logger.info("fix_logon_sent", target=f"{host}:{port}")
        self._connected = True
        return True

    async def disconnect(self):
        """Logout from FIX session."""
        if self._connected:
            logout = FIXMessage(
                msg_type=FIXMsgType.LOGOUT,
                sender_comp_id=self._sender,
                target_comp_id=self._target,
                msg_seq_num=self._next_seq(),
            )
            logger.info("fix_logout_sent")
            self._connected = False

    async def send_new_order(self, cl_ord_id: str, symbol: str, side: str, quantity: Decimal, order_type: str = "MARKET", price: Decimal | None = None) -> FIXMessage:
        """Send a new order via FIX."""
        fix_side = FIXSide.BUY if side.upper() == "BUY" else FIXSide.SELL
        fix_type = FIXOrderType.LIMIT if order_type == "LIMIT" else FIXOrderType.MARKET

        fields = {
            "11": cl_ord_id,       # ClOrdID
            "55": symbol,          # Symbol
            "54": fix_side.value,  # Side
            "38": str(quantity),   # OrderQty
            "40": fix_type.value,  # OrdType
            "60": datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S"),  # TransactTime
        }
        if price and fix_type == FIXOrderType.LIMIT:
            fields["44"] = str(price)  # Price

        msg = FIXMessage(
            msg_type=FIXMsgType.NEW_ORDER,
            sender_comp_id=self._sender,
            target_comp_id=self._target,
            msg_seq_num=self._next_seq(),
            fields=fields,
        )
        self._pending_orders[cl_ord_id] = msg
        logger.info("fix_order_sent", cl_ord_id=cl_ord_id, symbol=symbol, side=side, qty=str(quantity))
        return msg

    async def send_cancel(self, cl_ord_id: str, orig_cl_ord_id: str, symbol: str, side: str) -> FIXMessage:
        """Send order cancel request."""
        fix_side = FIXSide.BUY if side.upper() == "BUY" else FIXSide.SELL
        msg = FIXMessage(
            msg_type=FIXMsgType.CANCEL,
            sender_comp_id=self._sender,
            target_comp_id=self._target,
            msg_seq_num=self._next_seq(),
            fields={
                "11": cl_ord_id,
                "41": orig_cl_ord_id,
                "55": symbol,
                "54": fix_side.value,
                "60": datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S"),
            },
        )
        logger.info("fix_cancel_sent", cl_ord_id=cl_ord_id, orig=orig_cl_ord_id)
        return msg

    def get_session_status(self) -> dict:
        return {
            "connected": self._connected,
            "sender": self._sender,
            "target": self._target,
            "seq_num": self._seq_num,
            "pending_orders": len(self._pending_orders),
        }


fix_adapter = FIXAdapter()
