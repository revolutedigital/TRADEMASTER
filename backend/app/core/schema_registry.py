"""Schema registry for data validation and versioning."""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaVersion:
    name: str
    version: int
    schema: dict
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SchemaRegistry:
    """Register and validate data schemas for pipeline events."""

    def __init__(self):
        self._schemas: dict[str, list[SchemaVersion]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register("ohlcv", 1, {
            "type": "object",
            "required": ["symbol", "open_time", "open", "high", "low", "close", "volume"],
            "properties": {
                "symbol": {"type": "string"},
                "open_time": {"type": "string", "format": "date-time"},
                "open": {"type": "number"},
                "high": {"type": "number"},
                "low": {"type": "number"},
                "close": {"type": "number"},
                "volume": {"type": "number"},
            },
        })
        self.register("trade_event", 1, {
            "type": "object",
            "required": ["order_id", "symbol", "side", "quantity", "avg_price"],
            "properties": {
                "order_id": {"type": "integer"},
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["BUY", "SELL"]},
                "quantity": {"type": "number"},
                "avg_price": {"type": "number"},
                "commission": {"type": "number"},
            },
        })
        self.register("ml_prediction", 1, {
            "type": "object",
            "required": ["symbol", "action", "confidence"],
            "properties": {
                "symbol": {"type": "string"},
                "action": {"type": "string", "enum": ["BUY", "HOLD", "SELL"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "strength": {"type": "number"},
            },
        })

    def register(self, name: str, version: int, schema: dict) -> None:
        if name not in self._schemas:
            self._schemas[name] = []
        self._schemas[name].append(SchemaVersion(name=name, version=version, schema=schema))
        logger.info("schema_registered", name=name, version=version)

    def validate(self, data: dict, schema_name: str) -> tuple[bool, list[str]]:
        schema_versions = self._schemas.get(schema_name)
        if not schema_versions:
            return False, [f"Unknown schema: {schema_name}"]
        latest = schema_versions[-1]
        errors = []
        for req_field in latest.schema.get("required", []):
            if req_field not in data:
                errors.append(f"Missing required field: {req_field}")
        return len(errors) == 0, errors

    def get_latest(self, name: str) -> dict | None:
        versions = self._schemas.get(name)
        return versions[-1].schema if versions else None

    def list_schemas(self) -> list[dict]:
        result = []
        for name, versions in self._schemas.items():
            latest = versions[-1]
            result.append({"name": name, "version": latest.version, "fields": list(latest.schema.get("properties", {}).keys())})
        return result


schema_registry = SchemaRegistry()
