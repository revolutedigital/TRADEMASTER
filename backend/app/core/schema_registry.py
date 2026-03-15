"""Schema registry for data validation and versioning."""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaVersion:
    name: str
    version: int
    schema: dict
    hash: str
    created_at: str


class SchemaRegistry:
    """Registry for validating data schemas across the pipeline."""

    def __init__(self, storage_dir: str = "ml_artifacts/schemas"):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._schemas: dict[str, SchemaVersion] = {}
        self._load_schemas()

    def _schema_hash(self, schema: dict) -> str:
        canonical = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _load_schemas(self) -> None:
        """Load all registered schemas from disk."""
        for path in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                name = data["name"]
                self._schemas[name] = SchemaVersion(
                    name=name,
                    version=data["version"],
                    schema=data["schema"],
                    hash=data["hash"],
                    created_at=data["created_at"],
                )
            except Exception as e:
                logger.warning("schema_load_failed", path=str(path), error=str(e))

    def register(self, name: str, schema: dict) -> SchemaVersion:
        """Register or update a schema."""
        new_hash = self._schema_hash(schema)
        existing = self._schemas.get(name)

        if existing and existing.hash == new_hash:
            return existing  # No change

        version = (existing.version + 1) if existing else 1
        sv = SchemaVersion(
            name=name,
            version=version,
            schema=schema,
            hash=new_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Save to disk
        path = self._storage_dir / f"{name}.json"
        path.write_text(json.dumps({
            "name": sv.name,
            "version": sv.version,
            "schema": sv.schema,
            "hash": sv.hash,
            "created_at": sv.created_at,
        }, indent=2))

        self._schemas[name] = sv
        logger.info("schema_registered", name=name, version=version, hash=new_hash)
        return sv

    def validate(self, name: str, data: dict) -> tuple[bool, list[str]]:
        """Validate data against a registered schema.

        Returns (is_valid, list_of_errors).
        """
        sv = self._schemas.get(name)
        if sv is None:
            return False, [f"Schema '{name}' not registered"]

        errors = []
        schema = sv.schema

        # Validate required fields
        for field in schema.get("required", []):
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Validate field types
        for field, expected_type in schema.get("fields", {}).items():
            if field in data:
                value = data[field]
                if expected_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Field '{field}' expected float, got {type(value).__name__}")
                elif expected_type == "str" and not isinstance(value, str):
                    errors.append(f"Field '{field}' expected str, got {type(value).__name__}")
                elif expected_type == "int" and not isinstance(value, int):
                    errors.append(f"Field '{field}' expected int, got {type(value).__name__}")

        # Validate ranges
        for field, constraints in schema.get("ranges", {}).items():
            if field in data and isinstance(data[field], (int, float)):
                val = data[field]
                if "min" in constraints and val < constraints["min"]:
                    errors.append(f"Field '{field}' value {val} below min {constraints['min']}")
                if "max" in constraints and val > constraints["max"]:
                    errors.append(f"Field '{field}' value {val} above max {constraints['max']}")

        return len(errors) == 0, errors

    def get_schema(self, name: str) -> SchemaVersion | None:
        return self._schemas.get(name)

    def list_schemas(self) -> list[dict]:
        return [
            {"name": sv.name, "version": sv.version, "hash": sv.hash, "created_at": sv.created_at}
            for sv in self._schemas.values()
        ]


schema_registry = SchemaRegistry()
