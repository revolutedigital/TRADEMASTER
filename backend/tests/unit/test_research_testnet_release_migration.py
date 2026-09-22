"""Migration 022 adds only research-only Testnet release metadata."""

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

from app.models import research_experiment  # noqa: F401
from app.models.base import Base


def load_migration(filename: str):
    path = Path(__file__).parents[2] / "alembic" / "versions" / filename
    spec = importlib.util.spec_from_file_location(filename, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run(connection, filename: str, direction: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(load_migration(filename), direction)()


def test_testnet_release_migration_chains_matches_model_and_rolls_back() -> None:
    migration = load_migration("022_research_testnet_releases.py")
    assert (migration.revision, migration.down_revision) == ("022", "021")
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        run(connection, "020_research_experiment_registry.py", "upgrade")
        run(connection, "021_research_shadow_signals.py", "upgrade")
        run(connection, "022_research_testnet_releases.py", "upgrade")
        migrated = {
            column["name"]
            for column in sa.inspect(connection).get_columns("research_testnet_releases")
        }
        modelled = {
            column.name for column in Base.metadata.tables["research_testnet_releases"].columns
        }
        assert migrated == modelled
        run(connection, "022_research_testnet_releases.py", "downgrade")
        assert "research_testnet_releases" not in sa.inspect(connection).get_table_names()
