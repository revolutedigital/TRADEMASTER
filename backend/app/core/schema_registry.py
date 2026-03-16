"""Schema registry for data validation and versioning.

Persists schemas to both filesystem (fast reads) and PostgreSQL
(survives Railway ephemeral filesystem redeployments).
On startup, if the filesystem is empty, schemas are restored from DB.
"""
import asyncio
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
    """Registry for validating data schemas across the pipeline.

    Dual persistence:
    - Primary: filesystem at ``ml_artifacts/schemas/`` (fast, local reads)
    - Secondary: ``schema_versions`` PostgreSQL table (durable across deploys)
    """

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

    async def load_from_db(self) -> int:
        """Restore schemas from PostgreSQL if filesystem is empty.

        Called during application startup. Only loads from DB when the
        filesystem cache has no schemas (e.g. after Railway redeploy).

        Returns the number of schemas loaded from DB.
        """
        if self._schemas:
            logger.info("schema_registry_has_filesystem_cache", count=len(self._schemas))
            return 0

        try:
            from sqlalchemy import text
            from app.models.base import async_session_factory

            loaded = 0
            async with async_session_factory() as session:
                result = await session.execute(
                    text("SELECT name, version, schema, hash, created_at FROM schema_versions")
                )
                rows = result.fetchall()
                for row in rows:
                    name = row[0]
                    sv = SchemaVersion(
                        name=name,
                        version=int(row[1]),
                        schema=json.loads(row[2]),
                        hash=row[3],
                        created_at=row[4],
                    )
                    self._schemas[name] = sv

                    # Restore to filesystem for fast future reads
                    try:
                        path = self._storage_dir / f"{name}.json"
                        path.write_text(json.dumps({
                            "name": sv.name,
                            "version": sv.version,
                            "schema": sv.schema,
                            "hash": sv.hash,
                            "created_at": sv.created_at,
                        }, indent=2))
                    except Exception as e:
                        logger.debug("schema_fs_restore_failed", name=name, error=str(e))

                    loaded += 1

            logger.info("schemas_loaded_from_db", count=loaded)
            return loaded

        except Exception as exc:
            logger.warning("schema_load_from_db_failed", error=str(exc))
            return 0

    def _persist_to_db_bg(self, sv: SchemaVersion) -> None:
        """Schedule best-effort DB persistence for a schema version."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._save_schema_to_db(sv))
        except RuntimeError:
            pass  # No running event loop (tests, sync context)

    async def _save_schema_to_db(self, sv: SchemaVersion) -> None:
        """Persist a single schema version to PostgreSQL."""
        try:
            from sqlalchemy import text
            from app.models.base import async_session_factory

            async with async_session_factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO schema_versions
                            (name, version, schema, hash, created_at)
                        VALUES
                            (:name, :version, :schema, :hash, :created_at)
                        ON CONFLICT (name) DO UPDATE SET
                            version = EXCLUDED.version,
                            schema = EXCLUDED.schema,
                            hash = EXCLUDED.hash,
                            created_at = EXCLUDED.created_at,
                            updated_at = NOW()
                    """),
                    {
                        "name": sv.name,
                        "version": sv.version,
                        "schema": json.dumps(sv.schema),
                        "hash": sv.hash,
                        "created_at": sv.created_at,
                    },
                )
                await session.commit()
                logger.debug("schema_persisted_to_db", name=sv.name, version=sv.version)
        except Exception as exc:
            logger.debug("schema_db_persist_failed", name=sv.name, error=str(exc))

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

        # Save to filesystem
        path = self._storage_dir / f"{name}.json"
        path.write_text(json.dumps({
            "name": sv.name,
            "version": sv.version,
            "schema": sv.schema,
            "hash": sv.hash,
            "created_at": sv.created_at,
        }, indent=2))

        self._schemas[name] = sv

        # Also persist to DB (best-effort, non-blocking)
        self._persist_to_db_bg(sv)

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
