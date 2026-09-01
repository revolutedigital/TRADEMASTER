"""Regression coverage for PostgreSQL instances without TimescaleDB."""

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_timescaledb_migration() -> ModuleType:
    migration_path = (
        Path(__file__).parents[2] / "alembic" / "versions" / "002_timescaledb_hypertables.py"
    )
    spec = importlib.util.spec_from_file_location("timescaledb_migration", migration_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _UnavailableExtensionResult:
    def scalar_one(self) -> bool:
        return False


class _UnavailableExtensionConnection:
    def execute(self, _statement: object) -> _UnavailableExtensionResult:
        return _UnavailableExtensionResult()


def test_timescaledb_migration_keeps_plain_postgres_compatible(monkeypatch) -> None:
    migration = _load_timescaledb_migration()
    executed_sql: list[str] = []

    monkeypatch.setattr(migration.op, "get_bind", lambda: _UnavailableExtensionConnection())
    monkeypatch.setattr(migration.op, "execute", executed_sql.append)

    migration.upgrade()

    assert executed_sql == []
