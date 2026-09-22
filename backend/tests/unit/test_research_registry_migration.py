"""Migration 020 is additive, constrained, and reversible."""

import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

from app.models import research_experiment  # noqa: F401
from app.models.base import Base


TABLES = (
    "research_experiments",
    "research_data_uses",
    "research_hypothesis_attempts",
    "research_experiment_events",
)


def _load_migration():
    path = (
        Path(__file__).parents[2] / "alembic" / "versions" / "020_research_experiment_registry.py"
    )
    spec = importlib.util.spec_from_file_location("research_registry_migration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(connection, direction: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(_load_migration(), direction)()


@pytest.fixture
def connection():
    engine = sa.create_engine("sqlite://")
    with engine.begin() as database_connection:
        database_connection.execute(sa.text("CREATE TABLE trades (id INTEGER PRIMARY KEY)"))
        yield database_connection


def test_migration_chains_after_019() -> None:
    migration = _load_migration()
    assert (migration.revision, migration.down_revision) == ("020", "019")


def test_migration_matches_models_and_rolls_back(connection) -> None:
    _run(connection, "upgrade")
    inspector = sa.inspect(connection)
    for table_name in TABLES:
        migrated = {column["name"] for column in inspector.get_columns(table_name)}
        modelled = {column.name for column in Base.metadata.tables[table_name].columns}
        assert migrated == modelled

    _run(connection, "downgrade")
    assert set(sa.inspect(connection).get_table_names()) == {"trades"}


def test_database_rejects_invalid_status_role_and_time_window(connection) -> None:
    _run(connection, "upgrade")
    base_insert = (
        "INSERT INTO research_experiments "
        "(id, name, status, code_revision, protocol_sha256, product_json, "
        "cost_profile_json, approval_gate_json) VALUES "
        "('{experiment_id}', 'experiment', '{status}', '{revision}', '{protocol}', "
        "'{{}}', '{{}}', '{{}}')"
    )
    connection.execute(
        sa.text(
            base_insert.format(
                experiment_id="e1",
                status="DRAFT",
                revision="1" * 40,
                protocol="2" * 64,
            )
        )
    )
    with pytest.raises(sa.exc.IntegrityError), connection.begin_nested():
        connection.execute(
            sa.text(
                base_insert.format(
                    experiment_id="e2",
                    status="ACTIVE",
                    revision="3" * 40,
                    protocol="4" * 64,
                )
            )
        )

    partition_insert = (
        "INSERT INTO research_data_uses "
        "(experiment_id, role, start_at, end_at, manifest_sha256) VALUES "
        "('e1', '{role}', '2026-01-02', '2026-01-01', '{manifest}')"
    )
    with pytest.raises(sa.exc.IntegrityError), connection.begin_nested():
        connection.execute(sa.text(partition_insert.format(role="TRAINING", manifest="5" * 64)))
    with pytest.raises(sa.exc.IntegrityError), connection.begin_nested():
        connection.execute(
            sa.text(
                partition_insert.replace(
                    "'2026-01-02', '2026-01-01'", "'2026-01-01', '2026-01-02'"
                ).format(role="LIVE", manifest="6" * 64)
            )
        )
