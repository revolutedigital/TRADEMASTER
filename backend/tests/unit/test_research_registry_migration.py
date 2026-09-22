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


def _load_migration(filename: str = "020_research_experiment_registry.py"):
    path = Path(__file__).parents[2] / "alembic" / "versions" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(
    connection,
    direction: str,
    filename: str = "020_research_experiment_registry.py",
) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(_load_migration(filename), direction)()


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
        if table_name == "research_experiment_events":
            modelled -= {"event_sha256", "previous_event_sha256"}
        assert migrated == modelled

    _run(connection, "downgrade")
    assert set(sa.inspect(connection).get_table_names()) == {"trades"}


def test_event_hash_chain_migration_backfills_and_rolls_back(connection) -> None:
    _run(connection, "upgrade")
    connection.execute(
        sa.text(
            """
            INSERT INTO research_experiments
            (id, name, status, code_revision, protocol_sha256, product_json,
             cost_profile_json, approval_gate_json)
            VALUES ('e1', 'experiment', 'FROZEN', :revision, :protocol, '{}', '{}', '{}')
            """
        ),
        {"revision": "1" * 40, "protocol": "2" * 64},
    )
    connection.execute(
        sa.text(
            """
            INSERT INTO research_experiment_events
            (experiment_id, kind, payload_json, occurred_at)
            VALUES
            ('e1', 'DRAFT_CREATED', '{}', '2026-01-01T00:00:00+00:00'),
            ('e1', 'EXPERIMENT_FROZEN', '{"experiment_sha256":"3"}', '2026-01-01T00:00:01+00:00')
            """
        )
    )

    _run(connection, "upgrade", "023_research_event_hash_chain.py")
    rows = connection.execute(
        sa.text(
            """
            SELECT previous_event_sha256, event_sha256
            FROM research_experiment_events
            ORDER BY occurred_at, id
            """
        )
    ).mappings().all()

    assert rows[0]["previous_event_sha256"] is None
    assert len(rows[0]["event_sha256"]) == 64
    assert rows[1]["previous_event_sha256"] == rows[0]["event_sha256"]
    assert len(rows[1]["event_sha256"]) == 64

    _run(connection, "downgrade", "023_research_event_hash_chain.py")
    columns = {column["name"] for column in sa.inspect(connection).get_columns("research_experiment_events")}
    assert "event_sha256" not in columns
    assert "previous_event_sha256" not in columns


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
