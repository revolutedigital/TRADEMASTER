"""Migration 019 is additive, matches the models, enforces its rules, and rolls back cleanly."""

import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

from app.models import fx_bot  # noqa: F401  (registers the tables on the metadata)
from app.models.base import Base

TABLES = ("fx_bots", "fx_trades", "fx_equity_snapshots", "fx_bot_events")


def load_migration():
    path = Path(__file__).parents[2] / "alembic" / "versions" / "019_fx_bots.py"
    spec = importlib.util.spec_from_file_location("fx_bots_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(connection, direction: str) -> None:
    migration = load_migration()
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, direction)()


@pytest.fixture
def connection():
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(sa.text("CREATE TABLE trades (id INTEGER PRIMARY KEY, symbol TEXT)"))
        connection.execute(sa.text("INSERT INTO trades (symbol) VALUES ('BTCUSDT')"))
        yield connection


def test_it_chains_after_revision_018() -> None:
    migration = load_migration()

    assert (migration.revision, migration.down_revision) == ("019", "018")


def test_upgrade_creates_only_the_four_tables_and_leaves_existing_data_alone(connection) -> None:
    run(connection, "upgrade")

    names = set(sa.inspect(connection).get_table_names())
    assert set(TABLES) <= names and names - set(TABLES) == {"trades"}
    assert connection.execute(sa.text("SELECT symbol FROM trades")).scalar() == "BTCUSDT"


def test_the_migration_and_the_models_describe_the_same_columns(connection) -> None:
    run(connection, "upgrade")
    inspector = sa.inspect(connection)

    for name in TABLES:
        migrated = {column["name"] for column in inspector.get_columns(name)}
        modelled = {column.name for column in Base.metadata.tables[name].columns}
        assert migrated == modelled, name


def test_downgrade_removes_the_tables_and_is_repeatable(connection) -> None:
    run(connection, "upgrade")
    run(connection, "downgrade")

    assert set(sa.inspect(connection).get_table_names()) == {"trades"}
    assert connection.execute(sa.text("SELECT COUNT(*) FROM trades")).scalar() == 1
    run(connection, "upgrade")
    assert set(TABLES) <= set(sa.inspect(connection).get_table_names())


def test_the_database_itself_refuses_nonsense_rows(connection) -> None:
    run(connection, "upgrade")
    insert_bot = (
        "INSERT INTO fx_bots (key, name, strategy_key, symbol, timeframe_seconds, params_json, mode, "
        "risk_fraction, max_daily_loss_fraction) VALUES ('{key}', 'n', 'F1a', 'EURUSD', 900, '[]', '{mode}', {risk}, 0.01)"
    )
    connection.execute(sa.text(insert_bot.format(key="ok", mode="demo", risk=0.0025)))

    for key, mode, risk in (("a", "paper", 0.0025), ("b", "live", 0), ("c", "live", 2)):
        with pytest.raises(sa.exc.IntegrityError), connection.begin_nested():
            connection.execute(sa.text(insert_bot.format(key=key, mode=mode, risk=risk)))
    with pytest.raises(sa.exc.IntegrityError), connection.begin_nested():
        connection.execute(sa.text(insert_bot.format(key="ok", mode="demo", risk=0.0025)))  # duplicate key
    with pytest.raises(sa.exc.IntegrityError), connection.begin_nested():
        connection.execute(sa.text(
            "INSERT INTO fx_trades (bot_id, client_order_id, symbol, side, units, status) "
            "VALUES (1, 'x', 'EURUSD', 0, 1000, 'open')"
        ))
